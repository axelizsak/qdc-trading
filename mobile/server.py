"""The customer app: a phone-sized web app where a customer answers the coach, gets one of the five investor
profiles, reads the model portfolio that goes with it and each ETF's sheet, and approves the plan.

The app computes nothing slow. The model portfolios are fixed rules, and the ETF sheets come from the morning run
(graph/morning.py), which is done once for everyone.

Run:  python -m mobile.server   then open http://localhost:8600
"""
import json
from datetime import date, datetime
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
from data.profiles import PROFILES, SLEEVES
from data.schemas import CUSTOMER_LOG, LIST_SEP, split_list, validate_customer_log, validate_etf_sheets
from data.universe import load_universe
from graph.morning import SHEETS
from mobile.customers import ACCOUNT, CUSTOMERS, known

PORT = 8600
STATIC = Path(__file__).resolve().parent / "static"
LOG_NAME = "customer_log.csv"
# The customer's rules (the last question of the coach): which ETFs each one leaves out of a plan.
RULED_OUT = {
    "weapons": lambda ticker, family: ticker == "DEFENSE",
    "crypto": lambda ticker, family: family == "crypto",
    "leverage": lambda ticker, family: family == "leveraged_inverse",
    "fossil": lambda ticker, family: ticker in ("WORLD ENERGY", "OIL & GAS E&P"),
}


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


def record(logs: Path, customer: str, event: str, state: dict | None = None) -> None:
    """Appends one event to the customer log. `state` carries the profile, the reason, the answers and the flags."""
    state = state or {}
    row = pd.DataFrame([[datetime.now().isoformat(timespec="seconds"), customer, event, state.get("profile") or "",
                         state.get("why", ""), json.dumps(state.get("answers", {})),
                         LIST_SEP.join(state.get("flags", []))]], columns=CUSTOMER_LOG)
    logs.mkdir(parents=True, exist_ok=True)
    validate_customer_log(pd.concat([read_log(logs), row], ignore_index=True)).to_csv(logs / LOG_NAME, index=False)


def customer_state(logs: Path, customer: str) -> dict | None:
    """Where a customer stands: their profile, answers and flags, and their decision on the plan if they took
    one. None before the coach has placed them, or after they started over."""
    state = None
    log = read_log(logs)
    for row in log[log["customer"] == customer].itertuples():
        if row.event == "reset":
            state = None
        elif row.event == "profiled":
            answers = profiler.clean(json.loads(row.sheet))
            state = {"profile": row.profile, "why": row.why, "answers": answers,
                     "sheet": [[profiler.QUESTIONS[q]["name"], profiler.shown(q, v)] for q, v in answers.items()],
                     "flags": split_list(row.flags), "decision": None, "decided_at": None}
        elif state is not None:
            state.update(decision=row.event, decided_at=row.time)
    return state


def plan(profile: str, amount: float, rules: list[str] | None = None) -> dict:
    """The model portfolio of a profile, in euros for this customer's amount: one line per ETF, sleeve by sleeve.
    An ETF the customer's rules leave out gives its share to the largest core equity ETF; `left_out` says which."""
    universe = load_universe().set_index("ticker")
    rows = [(ticker, sleeve, weight, universe.loc[ticker]) for ticker, (sleeve, weight) in model_portfolios.weights(profile).items()]
    rows.sort(key=lambda r: (list(SLEEVES).index(r[1]), -r[2]))
    left_out = [{"ticker": ticker, "rule": profiler.QUESTIONS["rules"]["answers"][rule][0], "weight": weight}
                for ticker, _, weight, etf in rows for rule in rules or [] if RULED_OUT[rule](ticker, etf["family"])]
    gone = {line["ticker"] for line in left_out}
    freed = sum(weight for ticker, _, weight, _ in rows if ticker in gone)
    lines = [{"ticker": ticker, "name": etf["name"], "theme": etf["theme"].split(" (")[0], "isin": etf["isin"], "sleeve": sleeve,
              "weight": weight + (freed if i == 0 else 0)}
             for i, (ticker, sleeve, weight, etf) in enumerate(r for r in rows if r[0] not in gone)]
    for line in lines:
        line["weight"] = round(line["weight"], 6)
        line["eur"] = round(amount * line["weight"], 2)
    return {"profile": profile, "amount": amount, "lines": lines, "left_out": left_out}


def etf_sheet(shared: Path, ticker: str) -> dict | None:
    """The latest sheet of one ETF from the morning run, or None when there is none yet."""
    path = shared / SHEETS
    if not path.exists():
        return None
    sheets = validate_etf_sheets(pd.read_csv(path, keep_default_na=False))
    sheets = sheets[(sheets["ticker"] == ticker) & (sheets["date"] == sheets["date"].max())]
    if sheets.empty:
        return None
    etf = load_universe().set_index("ticker").loc[ticker]
    return {"ticker": ticker, "name": etf["name"], "theme": etf["theme"].split(" (")[0], "as_of": str(sheets["date"].iloc[0]),
            "rows": sheets, "lines": [{"block": r.block, "name": r.name, "value": float(r.value), "state": r.state, "detail": r.detail}
                                      for r in sheets.itertuples()]}


def create_app(client=None, logs: Path | None = None, shared: Path | None = None) -> Starlette:
    """The web app. `client` replaces the model's client, `logs` the folder of the customer log and `shared` the
    folder the morning run writes to (tests)."""
    logs = customer_logs_dir() if logs is None else logs
    shared = settings.LOGS_DIR if shared is None else shared
    tickers = set(load_universe()["ticker"])

    def error(message: str, status: int) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def start(request):
        people = [{"id": key, **{field: c[field] for field in ("first_name", "age", "city", "status")}}
                  for key, c in CUSTOMERS.items()]
        return JSONResponse({"customers": people, "profiles": PROFILES, "account_data": list(ACCOUNT.values()),
                             "coach": profiler.COACH, "questions": len(profiler.ASKED) + 1, "sleeves": SLEEVES})

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
        if turn["profile"]:
            record(logs, customer, "profiled", turn)
        return JSONResponse(turn)

    async def customer_plan(request):
        customer = request.query_params.get("customer")
        current = customer_state(logs, customer) if customer in CUSTOMERS else None
        if current is None:
            return error("this customer has no profile yet", 404)
        answers = current["answers"]
        return JSONResponse(plan(current["profile"], float(answers.get("amount", 0)), answers.get("rules")))

    async def etf(request):
        ticker = request.query_params.get("ticker")
        sheet = etf_sheet(shared, ticker) if ticker in tickers else None
        if sheet is None:
            return error("No sheet for this ETF yet: the morning run has not been done.", 404)
        return JSONResponse({key: value for key, value in sheet.items() if key != "rows"})

    async def why(request):
        body = await request.json()
        sheet = etf_sheet(shared, body.get("ticker")) if body.get("ticker") in tickers else None
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

    async def decision(request):
        body = await request.json()
        customer, choice = body.get("customer"), body.get("decision")
        current = customer_state(logs, customer) if customer in CUSTOMERS else None
        if current is None or choice not in ("approved", "declined"):
            return error("no plan to decide on", 400)
        record(logs, customer, choice, current)
        return JSONResponse({"state": customer_state(logs, customer)})

    async def reset(request):
        customer = (await request.json()).get("customer")
        if customer not in CUSTOMERS:
            return error("unknown customer", 404)
        record(logs, customer, "reset")
        return JSONResponse({"state": None})

    return Starlette(routes=[
        Route("/api/start", start), Route("/api/state", state), Route("/api/plan", customer_plan), Route("/api/etf", etf),
        Route("/api/chat", chat, methods=["POST"]), Route("/api/why", why, methods=["POST"]),
        Route("/api/decision", decision, methods=["POST"]), Route("/api/reset", reset, methods=["POST"]),
        Mount("/", StaticFiles(directory=STATIC, html=True)),
    ])


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
