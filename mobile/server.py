"""The customer app: a phone-sized web app.

The coach finds the customer's archetype, and fixed rules their risk level (strategy). The customer then sees their
strategy's reference portfolio and every ETF's sheet, with what the agents read in it, places their own orders and
follows what they hold against the reference. ReBel is execution-only (Belfius session of 27 Oct, slide 6): the app
informs, the customer decides every order.

The app computes nothing slow. The reference portfolios are fixed rules; the ETF sheets and the charts' points come
from the morning run (graph/morning.py), done once for everyone. What a customer may buy and what they hold follow
mobile/wallet.py.

Run:  python -m mobile.server   then open http://localhost:8600
"""
import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from agents import etf_sheets, model_portfolios, profiler
from data import config as settings
from data.lessons import LESSONS
from data.profiles import ARCHETYPES, SLEEVE_ETFS, SLEEVES, STRATEGIES
from data.schemas import (CUSTOMER_LOG, LIST_SEP, split_list, validate_chart_history, validate_customer_log,
                          validate_etf_sheets, validate_etf_views)
from data.universe import load_universe
from graph.morning import HISTORY, SHEETS
from mobile import wallet
from mobile.customers import ACCOUNT, CUSTOMERS, known

PORT = 8600
STATIC = Path(__file__).resolve().parent / "static"
LOG_NAME = "customer_log.csv"
VIEWS = "etf_views.csv"  # the theme analysts' view of each ETF (Macro/News agent)
RETURNS = ("1 month", "1 year")  # the returns shown next to an ETF in a list


def customer_logs_dir() -> Path:
    """Where the app writes what its customers do. They are synthetic demo customers, so their clicks stay out of
    the committed logs: logs/mock/ is ignored by git."""
    return settings.LOGS_DIR / "mock"


def read_log(logs: Path) -> pd.DataFrame:
    """The customer log, or an empty table before the first customer."""
    path = logs / LOG_NAME
    if not path.exists():
        return pd.DataFrame(columns=CUSTOMER_LOG)
    return validate_customer_log(pd.read_csv(path, dtype=str, keep_default_na=False))


def record(logs: Path, customer: str, event: str, state: dict | None = None, detail: dict | None = None) -> None:
    """Appends one event to the customer log. `state` carries the archetype, the strategy, their reasons and the
    flags; `detail` what the event is about (the answers, an order, the check's answers)."""
    state = state or {}
    why = {"archetype": state.get("why", ""), "strategy": state.get("strategy_why", [])} if state else {}
    row = pd.DataFrame([[datetime.now().isoformat(timespec="seconds"), customer, event, state.get("archetype") or "",
                         state.get("strategy") or "", json.dumps(why), json.dumps(detail or {}),
                         LIST_SEP.join(state.get("flags", []))]], columns=CUSTOMER_LOG)
    logs.mkdir(parents=True, exist_ok=True)
    validate_customer_log(pd.concat([read_log(logs), row], ignore_index=True)).to_csv(logs / LOG_NAME, index=False)


def customer_state(logs: Path, customer: str) -> dict | None:
    """Where a customer stands: archetype, strategy and why, answers, flags, the orders they placed, whether they
    opened complex products and the lessons they got right (lesson -> True once answered right, False before).
    None before the coach has placed them, or after they started over."""
    state = None
    log = read_log(logs)
    for row in log[log["customer"] == customer].itertuples():
        if row.event == "reset":
            state = None
        elif row.event == "profiled":
            answers, why = profiler.clean(json.loads(row.detail)), json.loads(row.why)
            state = {"archetype": row.archetype, "strategy": row.strategy, "why": why.get("archetype", ""),
                     "strategy_why": why.get("strategy", []), "answers": answers,
                     "sheet": [[profiler.QUESTIONS[q]["name"], profiler.shown(q, v)] for q, v in answers.items()],
                     "flags": split_list(row.flags), "orders": [], "unlocked": False, "lessons": {}}
        elif state is not None and row.event == "order":
            state["orders"].append({**json.loads(row.detail), "time": row.time})
        elif state is not None and row.event == "unlocked":
            state["unlocked"] = True
        elif state is not None and row.event == "lesson":
            done = json.loads(row.detail)
            state["lessons"][done["lesson"]] = state["lessons"].get(done["lesson"], False) or bool(done["right"])
    return state


@lru_cache(maxsize=1)
def universe() -> dict[str, dict]:
    """ticker -> what the app shows of it: name, theme (without its note in brackets), family, ISIN."""
    return {row.ticker: {"ticker": row.ticker, "name": row.name, "theme": row.theme.split(" (")[0],
                         "family": row.family, "isin": row.isin} for row in load_universe().itertuples()}


def latest_sheets(shared: Path) -> pd.DataFrame:
    """Every line of the latest morning run's sheets, or an empty table before the first run."""
    path = shared / SHEETS
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", "block", "name", "value", "state", "detail"])
    sheets = validate_etf_sheets(pd.read_csv(path, keep_default_na=False))
    return sheets[sheets["date"] == sheets["date"].max()]


def chart_points(shared: Path) -> pd.DataFrame:
    """The points of the charts from the latest morning run, or an empty table before it."""
    path = shared / HISTORY
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", "value"])
    return validate_chart_history(pd.read_csv(path)).sort_values("date")


def points(history: pd.DataFrame, ticker: str) -> list[dict]:
    """One ETF's chart points, [{"date", "value"}] in date order."""
    rows = history[history["ticker"] == ticker]
    return [{"date": str(r.date), "value": round(float(r.value), 6)} for r in rows.itertuples()]


def last_prices(shared: Path) -> dict[str, dict]:
    """ticker -> {"price": last adjusted close in EUR, "as_of": its date}, from the charts' points."""
    history = chart_points(shared)
    etfs = history.groupby("ticker").tail(1)
    return {r.ticker: {"price": float(r.value), "as_of": str(r.date)} for r in etfs.itertuples()}


def returns_by_ticker(sheets: pd.DataFrame) -> dict[str, dict[str, float]]:
    """ticker -> {"1 month": ..., "1 year": ...} from the sheets."""
    rows = sheets[(sheets["block"] == "returns") & sheets["name"].isin(RETURNS)]
    out: dict[str, dict[str, float]] = {}
    for r in rows.itertuples():
        out.setdefault(r.ticker, {})[r.name] = float(r.value)
    return out


def reference(strategy: str, amount: float, rules: list[str] | None = None, shared: Path | None = None,
              chart: bool = True) -> dict:
    """The strategy's reference portfolio today (its latest version, agents/model_portfolios.py) in euros for this
    amount: one line per ETF, sleeve by sleeve, with what each did, the chart of this mix, the date of the version
    (None: the starting mix) and what changed from the version before. An ETF the customer's rules leave out gives
    its share to the largest core equity ETF; `left_out` says which. Locks are added by mobile/wallet.py."""
    etfs = universe()
    held, version = model_portfolios.current(shared, strategy)
    rows = [(ticker, sleeve, weight) for ticker, (sleeve, weight) in held.items()]
    rows.sort(key=lambda r: (list(SLEEVES).index(r[1]), -r[2]))
    left_out = []
    for ticker, _, weight in rows:
        rule = wallet.ruled_out_by(etfs[ticker], rules)
        if rule:
            left_out.append({"ticker": ticker, "rule": wallet.rule_name(rule), "weight": weight})
    gone = {out["ticker"] for out in left_out}
    freed = sum(out["weight"] for out in left_out)
    kept = [r for r in rows if r[0] not in gone]
    returns = returns_by_ticker(latest_sheets(shared)) if shared else {}
    lines = [{**etfs[ticker], "sleeve": sleeve, "weight": round(weight + (freed if i == 0 else 0), 6),
              "eur": round(amount * (weight + (freed if i == 0 else 0)), 2), "returns": returns.get(ticker, {})}
             for i, (ticker, sleeve, weight) in enumerate(kept)]
    sleeves = [{"key": key, "name": name, "weight": round(sum(l["weight"] for l in lines if l["sleeve"] == key), 6)}
               for key, name in SLEEVES.items()]
    mix = {ticker: weight for ticker, (_, weight) in held.items()}
    history = []
    if shared and chart:
        found = chart_points(shared)
        history = wallet.mix_curve(mix, {t: points(found, t) for t in mix})
    before = model_portfolios.previous(shared, strategy, version) if version else {}
    moved = wallet.changes({t: w for t, (_, w) in before.items()}, mix) if version else []
    return {"strategy": strategy, "amount": amount, "lines": lines, "left_out": left_out,
            "sleeves": [s for s in sleeves if s["weight"] > 0], "history": history, "version": version,
            "changes": moved}


def catalogue(shared: Path) -> list[dict]:
    """Every ETF of the universe, in its order, with its two returns. Locks are added by mobile/wallet.py."""
    returns = returns_by_ticker(latest_sheets(shared))
    return [{**etf, "returns": returns.get(ticker, {})} for ticker, etf in universe().items()]


def news_view(shared: Path, ticker: str, as_of: str) -> dict | None:
    """The news analyst's latest view of an ETF up to `as_of`, or None."""
    path = shared / VIEWS
    if not path.exists():
        return None
    views = validate_etf_views(pd.read_csv(path, keep_default_na=False))
    views = views[(views["ticker"] == ticker) & (views["date"].astype(str) <= as_of)]
    if views.empty:
        return None
    view = views.sort_values("date").iloc[-1]
    return {"date": str(view["date"]), "direction": float(view["direction"]), "confidence": float(view["confidence"]),
            "rationale": str(view["rationale"]), "counter": str(view["counter_argument"]),
            "sources": [url for url in split_list(view["sources"]) if url.startswith(("https://", "http://"))]}


def etf_sheet(shared: Path, ticker: str) -> dict | None:
    """The latest sheet of one ETF from the morning run, with its chart, the news analyst's view and the agents'
    reading in plain words. None when there is no sheet yet."""
    sheets = latest_sheets(shared)
    sheets = sheets[sheets["ticker"] == ticker]
    if sheets.empty:
        return None
    as_of = str(sheets["date"].iloc[0])
    lines = [{"block": r.block, "name": r.name, "label": etf_sheets.label({"block": r.block, "name": r.name}),
              "value": float(r.value), "state": r.state, "detail": r.detail} for r in sheets.itertuples()]
    news = news_view(shared, ticker, as_of)
    return {**universe()[ticker], "as_of": as_of, "rows": sheets, "lines": lines, "news": news,
            "reading": etf_sheets.reading(lines, news), "history": points(chart_points(shared), ticker)}


def quiz_questions() -> list[dict]:
    """The check on complex products as the phone shows it: the questions and the answers, not which is right."""
    return [{"id": key, "text": spec["text"], "choices": [{"value": value, "label": label}
                                                          for value, (label, _) in spec["answers"].items()]}
            for key, spec in profiler.QUIZ.items()]


def lessons() -> dict[str, dict]:
    """The lessons as the phone shows them: the text and the question, not which answer is right."""
    return {key: {"title": spec["title"], "text": spec["text"], "question": spec["question"],
                  "choices": [{"value": value, "label": label} for value, (label, _) in spec["answers"].items()]}
            for key, spec in LESSONS.items()}


def start(shared: Path | None = None) -> dict:
    """What the phone needs before the first screen: the demo customers, the strategies, the archetypes, the coach,
    and the strategy the team trades today (chosen in the cockpit, logged in `shared`)."""
    shared = settings.LOGS_DIR if shared is None else shared
    people = [{"id": key, **{field: c[field] for field in ("first_name", "age", "city", "status")}}
              for key, c in CUSTOMERS.items()]
    return {"customers": people, "strategies": STRATEGIES, "archetypes": ARCHETYPES,
            "traded": model_portfolios.traded(shared, date.today()),
            "account_data": list(ACCOUNT.values()), "coach": profiler.COACH, "questions": profiler.MAX_QUESTIONS,
            "sleeves": SLEEVES, "core_etfs": {t: sleeve for sleeve, etfs in SLEEVE_ETFS.items() for t in etfs},
            "quiz": quiz_questions(), "lessons": lessons()}


def portfolio(state: dict, shared: Path) -> dict:
    """What the customer holds, valued at the last close, against their strategy's reference portfolio, and the
    orders that would follow the reference (`proposal`): the customer decides on each. `proposal["new"]`: the
    reference changed since the customer's last order."""
    last = {ticker: p["price"] for ticker, p in last_prices(shared).items()}
    rules = state["answers"].get("rules")
    ref = reference(state["strategy"], 1, rules, shared, chart=False)
    target = wallet.decorate(ref["lines"], state["strategy"], state["unlocked"], rules)
    history = chart_points(shared)
    held = {order["ticker"] for order in state["orders"]}
    book = wallet.holdings(state["orders"], last, universe(), target)
    last_order = max((order.get("time", "")[:10] for order in state["orders"]), default="")
    return {**book, "history": wallet.mix_history(state["orders"], {t: points(history, t) for t in held}),
            "proposal": {"version": ref["version"], "changes": ref["changes"],
                         "new": bool(state["orders"]) and bool(ref["version"]) and ref["version"] >= last_order,
                         "orders": wallet.rebalance(book["lines"], target) if book["lines"] else []}}


def create_app(client=None, logs: Path | None = None, shared: Path | None = None) -> Starlette:
    """The web app. `client` replaces the model's client, `logs` the folder of the customer log and `shared` the
    folder the morning run writes to (tests)."""
    logs = customer_logs_dir() if logs is None else logs
    shared = settings.LOGS_DIR if shared is None else shared

    def error(message: str, status: int) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    def placed(request) -> tuple[str | None, dict | None]:
        customer = request.query_params.get("customer")
        return customer, customer_state(logs, customer) if customer in CUSTOMERS else None

    async def at_start(request):
        return JSONResponse(start(shared))

    async def state(request):
        customer = request.query_params.get("customer")
        if customer not in CUSTOMERS:
            return error("unknown customer", 404)
        return JSONResponse({"state": customer_state(logs, customer), "known": known(customer, True)})

    async def chat(request):
        body = await request.json()
        customer = body.get("customer")
        if customer not in CUSTOMERS:
            return error("unknown customer", 404)
        try:
            turn = await run_in_threadpool(profiler.step, CUSTOMERS[customer], bool(body.get("consent")),
                                           body.get("answers"), body.get("reply"), client)
        except Exception as exc:  # a typed reply and no model to read it: the app says so, tapping still works
            print(f"  coach: {type(exc).__name__}: {exc}")
            return error(f"{profiler.COACH} cannot read typed answers right now: tap one instead.", 502)
        if turn["archetype"]:
            record(logs, customer, "profiled", turn, turn["answers"])
        return JSONResponse(turn)

    async def customer_reference(request):
        _, current = placed(request)
        if current is None:
            return error("this customer has no risk level yet", 404)
        answers = current["answers"]
        out = reference(current["strategy"], float(answers.get("amount", 0)), answers.get("rules"), shared)
        out["lines"] = wallet.decorate(out["lines"], current["strategy"], current["unlocked"], answers.get("rules"))
        return JSONResponse(out)

    async def etfs(request):
        _, current = placed(request)
        if current is None:
            return error("this customer has no risk level yet", 404)
        return JSONResponse({"etfs": wallet.decorate(catalogue(shared), current["strategy"], current["unlocked"],
                                                     current["answers"].get("rules"))})

    async def etf(request):
        ticker = request.query_params.get("ticker")
        sheet = etf_sheet(shared, ticker) if ticker in universe() else None
        if sheet is None:
            return error("No sheet for this ETF yet: the morning run has not been done.", 404)
        return JSONResponse({key: value for key, value in sheet.items() if key != "rows"})

    async def why(request):
        body = await request.json()
        sheet = etf_sheet(shared, body.get("ticker")) if body.get("ticker") in universe() else None
        line = next((l for l in (sheet or {}).get("lines", []) if l["block"] == body.get("block") and l["name"] == body.get("name")), None)
        if line is None:
            return error("unknown line", 404)
        question = f"Why does the {line['block']} line '{line['name']}' of this sheet read {line['state'] or 'this value'}? What does it mean?"
        try:
            answer = await run_in_threadpool(etf_sheets.explain, sheet["rows"], question, client)
        except Exception as exc:  # no key, quota used: the numbers stay on screen, only the explanation is missing
            print(f"  why: {type(exc).__name__}: {exc}")
            return error(f"{profiler.COACH} cannot explain right now.", 502)
        return JSONResponse({"answer": answer})

    async def order(request):
        body = await request.json()
        customer = body.get("customer")
        current = customer_state(logs, customer) if customer in CUSTOMERS else None
        orders = body.get("orders")
        if current is None:
            return error("this customer has no risk level yet", 404)
        if not isinstance(orders, list) or not orders or not all(isinstance(o, dict) for o in orders):
            return error("no order to place", 400)
        rules = current["answers"].get("rules")
        held = {line["ticker"]: line["value"] for line in portfolio(current, shared)["lines"]} if current["orders"] else {}
        for item in orders:
            side = item.get("side", "BUY")
            problem = wallet.check_order(universe().get(item.get("ticker")), item.get("eur"), current["strategy"],
                                         current["unlocked"], rules, side, held.get(item.get("ticker"), 0.0))
            if problem:
                return error(f"{item.get('ticker')}: {problem}", 400)
        prices = last_prices(shared)
        for item in sorted(orders, key=lambda o: o.get("side", "BUY") != "SELL"):  # sells first: they fund the buys
            price = prices.get(item["ticker"])
            eur = round(float(item["eur"]), 2)
            detail = {"ticker": item["ticker"], "side": item.get("side", "BUY"), "eur": eur,
                      "units": round(eur / price["price"], 6) if price else None,
                      "price": round(price["price"], 4) if price else None, "as_of": price["as_of"] if price else None}
            record(logs, customer, "order", current, detail)
        now = customer_state(logs, customer)
        return JSONResponse({"state": now, "portfolio": portfolio(now, shared)})

    async def customer_portfolio(request):
        _, current = placed(request)
        if current is None:
            return error("this customer has no risk level yet", 404)
        return JSONResponse(portfolio(current, shared))

    async def check(request):
        body = await request.json()
        customer = body.get("customer")
        current = customer_state(logs, customer) if customer in CUSTOMERS else None
        if current is None:
            return error("this customer has no risk level yet", 404)
        if not STRATEGIES[current["strategy"]]["complex"]:
            return error(f"Complex products are not part of the {STRATEGIES[current['strategy']]['label']} risk level.", 400)
        answers = body.get("answers") if isinstance(body.get("answers"), dict) else {}
        result = profiler.quiz(answers)
        if result["passed"] and not current["unlocked"]:
            record(logs, customer, "unlocked", current, {key: answers.get(key) for key in profiler.QUIZ})
        return JSONResponse({**result, "state": customer_state(logs, customer)})

    async def learn(request):
        body = await request.json()
        customer = body.get("customer")
        current = customer_state(logs, customer) if customer in CUSTOMERS else None
        if current is None:
            return error("this customer has no risk level yet", 404)
        if body.get("lesson") not in LESSONS:
            return error("unknown lesson", 404)
        result = profiler.lesson(body["lesson"], body.get("answer"))
        record(logs, customer, "lesson", current, {"lesson": body["lesson"], "answer": body.get("answer"),
                                                   "right": result["right"]})
        return JSONResponse({**result, "state": customer_state(logs, customer)})

    async def reset(request):
        customer = (await request.json()).get("customer")
        if customer not in CUSTOMERS:
            return error("unknown customer", 404)
        record(logs, customer, "reset")
        return JSONResponse({"state": None})

    return Starlette(routes=[
        Route("/api/start", at_start), Route("/api/state", state), Route("/api/reference", customer_reference),
        Route("/api/etfs", etfs), Route("/api/etf", etf), Route("/api/portfolio", customer_portfolio),
        Route("/api/chat", chat, methods=["POST"]), Route("/api/why", why, methods=["POST"]),
        Route("/api/order", order, methods=["POST"]), Route("/api/quiz", check, methods=["POST"]),
        Route("/api/lesson", learn, methods=["POST"]),
        Route("/api/reset", reset, methods=["POST"]),
        Mount("/", StaticFiles(directory=STATIC, html=True)),
    ])


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
