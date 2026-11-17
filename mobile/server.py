"""The customer app: a phone-sized web app where a customer chats with the coach, gets one of the five investor
profiles and approves the plan that goes with it.

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

from agents import profiler
from data import config as settings
from data import mock
from data.profiles import PROFILES
from data.schemas import CUSTOMER_LOG, LIST_SEP, split_list, validate_customer_log
from data.universe import load_universe
from mobile.customers import ACCOUNT, CUSTOMERS, known

PORT = 8600
STATIC = Path(__file__).resolve().parent / "static"
LOG_NAME = "customer_log.csv"
# The plans are still an illustration (data/mock.py). Until the engine builds one portfolio per profile, the app
# writes to logs/mock/ (gitignored), like the pipeline does while a step is on a mock; after that, to logs/.
ILLUSTRATION = True
# ETF family -> what the plan calls it. Every other family is shares.
ASSETS = {"bonds": "Bonds", "cash": "Cash", "gold": "Gold", "crypto": "Crypto"}
# The customer's rules (the last question of the coach): which ETFs each one leaves out of a plan.
RULED_OUT = {
    "weapons": lambda etf: etf.ticker == "DEFENSE",
    "crypto": lambda etf: etf.family == "crypto",
    "leverage": lambda etf: etf.family == "leveraged_inverse",
    "fossil": lambda etf: etf.ticker in ("WORLD ENERGY", "OIL & GAS E&P"),
}


def read_log(logs: Path) -> pd.DataFrame:
    """logs/customer_log.csv, or an empty table before the first customer."""
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


def default_logs_dir() -> Path:
    return settings.LOGS_DIR / "mock" if ILLUSTRATION else settings.LOGS_DIR


def plan(profile: str, amount: float, as_of: date, rules: list[str] | None = None) -> dict:
    """The portfolio of a profile, in euros for this customer's amount: one line per ETF, the core first.
    An ETF the customer's rules leave out gives its share to the largest core ETF; `left_out` says which."""
    portfolios = mock.mock_profile_portfolios(as_of.isoformat())
    rows = portfolios[portfolios["profile"] == profile].merge(load_universe(), on="ticker")
    rows = rows.sort_values(["sleeve", "target_weight"], ascending=[True, False]).reset_index(drop=True)
    left_out = [{"ticker": etf.ticker, "rule": profiler.QUESTIONS["rules"]["answers"][rule][0],
                 "weight": float(etf.target_weight)}
                for etf in rows.itertuples() for rule in rules or [] if RULED_OUT[rule](etf)]
    gone = rows["ticker"].isin([line["ticker"] for line in left_out])
    if gone.any():
        freed = rows.loc[gone, "target_weight"].sum()
        rows = rows[~gone].reset_index(drop=True)
        rows.loc[0, "target_weight"] += freed  # the first row is the largest core ETF
    lines = [{"ticker": r.ticker, "name": r.name, "theme": r.theme.split(" (")[0], "isin": r.isin, "sleeve": r.sleeve,
              "asset": ASSETS.get(r.family, "Shares"), "weight": float(r.target_weight),
              "eur": round(amount * float(r.target_weight), 2)} for r in rows.itertuples()]
    return {"profile": profile, "as_of": as_of.isoformat(), "amount": amount, "illustration": ILLUSTRATION,
            "lines": lines, "left_out": left_out}


def create_app(client=None, logs: Path | None = None) -> Starlette:
    """The web app. `client` replaces the the LLM client and `logs` the logs folder (tests)."""
    logs = default_logs_dir() if logs is None else logs

    def error(message: str, status: int) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def start(request):
        people = [{"id": key, **{field: c[field] for field in ("first_name", "age", "city", "status")}}
                  for key, c in CUSTOMERS.items()]
        return JSONResponse({"customers": people, "profiles": PROFILES, "account_data": list(ACCOUNT.values()),
                             "coach": profiler.COACH, "questions": len(profiler.ASKED) + 1})

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
        return JSONResponse(plan(current["profile"], float(answers.get("amount", 0)), date.today(), answers.get("rules")))

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
        Route("/api/start", start), Route("/api/state", state), Route("/api/plan", customer_plan),
        Route("/api/chat", chat, methods=["POST"]), Route("/api/decision", decision, methods=["POST"]),
        Route("/api/reset", reset, methods=["POST"]),
        Mount("/", StaticFiles(directory=STATIC, html=True)),
    ])


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
