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


def read_log(logs: Path) -> pd.DataFrame:
    """logs/customer_log.csv, or an empty table before the first customer."""
    path = logs / LOG_NAME
    if not path.exists():
        return pd.DataFrame(columns=CUSTOMER_LOG)
    return validate_customer_log(pd.read_csv(path, dtype=str, keep_default_na=False))


def record(logs: Path, customer: str, event: str, state: dict | None = None) -> None:
    """Appends one event to the customer log. `state` carries the profile, the reason, the sheet and the flags."""
    state = state or {}
    row = pd.DataFrame([[datetime.now().isoformat(timespec="seconds"), customer, event, state.get("profile") or "",
                         state.get("why", ""), json.dumps(state.get("sheet", {})),
                         LIST_SEP.join(state.get("flags", []))]], columns=CUSTOMER_LOG)
    logs.mkdir(parents=True, exist_ok=True)
    validate_customer_log(pd.concat([read_log(logs), row], ignore_index=True)).to_csv(logs / LOG_NAME, index=False)


def customer_state(logs: Path, customer: str) -> dict | None:
    """Where a customer stands: their profile, sheet and flags, and their decision on the plan if they took one.
    None before the coach has placed them, or after they started over."""
    state = None
    log = read_log(logs)
    for row in log[log["customer"] == customer].itertuples():
        if row.event == "reset":
            state = None
        elif row.event == "profiled":
            state = {"profile": row.profile, "why": row.why, "sheet": json.loads(row.sheet),
                     "flags": split_list(row.flags), "decision": None, "decided_at": None}
        elif state is not None:
            state.update(decision=row.event, decided_at=row.time)
    return state


def default_logs_dir() -> Path:
    return settings.LOGS_DIR / "mock" if ILLUSTRATION else settings.LOGS_DIR


def plan(profile: str, amount: float, as_of: date) -> dict:
    """The portfolio of a profile, in euros for this customer's amount: one line per ETF, the core first."""
    portfolios = mock.mock_profile_portfolios(as_of.isoformat())
    rows = portfolios[portfolios["profile"] == profile].merge(load_universe(), on="ticker")
    rows = rows.sort_values(["sleeve", "target_weight"], ascending=[True, False])
    lines = [{"ticker": r.ticker, "name": r.name, "theme": r.theme.split(" (")[0], "isin": r.isin, "sleeve": r.sleeve,
              "asset": ASSETS.get(r.family, "Shares"), "weight": float(r.target_weight),
              "eur": round(amount * float(r.target_weight), 2)} for r in rows.itertuples()]
    return {"profile": profile, "as_of": as_of.isoformat(), "amount": amount, "illustration": ILLUSTRATION,
            "lines": lines}


def create_app(client=None, logs: Path | None = None) -> Starlette:
    """The web app. `client` replaces the Claude client and `logs` the logs folder (tests)."""
    logs = default_logs_dir() if logs is None else logs

    def error(message: str, status: int) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def start(request):
        people = [{"id": key, **{field: c[field] for field in ("first_name", "age", "city", "status")}}
                  for key, c in CUSTOMERS.items()]
        return JSONResponse({"customers": people, "profiles": PROFILES, "account_data": list(ACCOUNT.values()),
                             "coach": profiler.COACH,
                             "sheet": {field: label for field, (label, _, _) in profiler.SHEET.items()}})

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
            reply = await run_in_threadpool(profiler.turn, known(customer, bool(body.get("consent"))),
                                            body.get("messages") or [], body.get("sheet"), client)
        except Exception as exc:  # no key, no network, a refusal: the app says so and offers to try again
            print(f"  coach: {type(exc).__name__}: {exc}")
            return error(f"{profiler.COACH} cannot answer right now ({type(exc).__name__}).", 502)
        if reply["profile"]:
            record(logs, customer, "profiled", reply)
        return JSONResponse(reply)

    async def customer_plan(request):
        customer = request.query_params.get("customer")
        current = customer_state(logs, customer) if customer in CUSTOMERS else None
        if current is None:
            return error("this customer has no profile yet", 404)
        return JSONResponse(plan(current["profile"], float(current["sheet"].get("amount_eur", 0)), date.today()))

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
