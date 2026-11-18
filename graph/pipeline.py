"""LangGraph orchestration of one trading day. Owner: Axel.

    data -> learning -> (technical | macro_news) -> portfolio -> orders review (human)
         -> risk -> risk review (human) -> executive summary -> summary review (human) -> order log

- Each step calls its real module, or its mock while `USE_MOCK[step]` is True in data/config.py.
- Every output is checked against data/schemas.py. A step that fails is replaced by a safe
  fallback (neutral signals with confidence 0, no orders, no risk report), the failure goes to
  <logs>/incidents.csv, and the day carries on to the human.
- The day starts with the learning step (graph/review_loop.py): one rule re-weights the technical agent's three
  answers, the four agents and the LLM's portfolio against the code's from what each earned, and it writes the
  lessons the agents then read. The technical agent and the Portfolio Manager use those weights as given.
- The graph pauses at three human reviews. At each one the human approves, sends the day back to
  the Portfolio Manager with feedback (at most MAX_REVISIONS times a day), or stops it: no trade today.
  Between reviews the state is saved in SQLite (<logs>/checkpoints.sqlite, one thread per date).
- The trading rules (at most MAX_ORDERS_PER_DAY orders, enough cash, no selling more than held)
  are checked on the Portfolio Manager's orders, and again on the orders the human approves.
- Each round of the Portfolio Manager writes its memo to <logs>/memo/<date>.md, and the risk step its board and
  reading to <logs>/risk/<date>.md. When the day ends: <logs>/orders.csv (AI suggestion next to the human decision),
  <logs>/reviews.csv (every review with its feedback) and <logs>/summary/<date>.md.
- A day is computed once: the logs are its record, whoever opens it and on whichever machine. Signals and the
  macro summary are written to <logs>/ and read back; each round of the Portfolio Manager is frozen in
  <logs>/proposals/<date>-round<n>.json and read back, so the same day always shows the same orders and the agents
  (and the LLM API) are never called twice for it. A day whose reviews are in <logs>/reviews.csv is finished: it is
  read, never run again. The saved graph state (checkpoints.sqlite) is only a convenience of one machine.
- Any day can be run: days before the trading window are practice, to collect signals and test
  orders. Practice orders never count in the competition, which starts from START_CAPITAL_EUR.
- <logs> is logs/mock/ (gitignored) while any step is on a mock, else logs/ (graded evidence).

Usage:
    python -m graph.pipeline                                        # run today until the first review
    python -m graph.pipeline --date 2026-12-07                      # run that day until the first review
    python -m graph.pipeline --date 2026-12-07 --answer a.json      # answer the review waiting
    python -m graph.pipeline --date 2026-12-07 --approve-all        # approve every review left

Answer file for the review waiting (feedback is required to send the day back or to stop it):
    {"decision": "approved"}                                          # risk or summary review
    {"decision": "revised", "feedback": "too much in leveraged ETFs"} # back to the Portfolio Manager
    {"decision": "stopped", "feedback": "Fed tonight, no trade"}      # no trade today
At the orders review, approving also takes one decision per proposed order, plus any order you add:
    {"decision": "approved", "orders": [
        {"ticker": "WORLD", "decision": "approved"},
        {"ticker": "GOLD", "decision": "modified", "units": 4, "reason": "..."},
        {"ticker": "BITCOIN", "decision": "rejected", "reason": "..."},
        {"ticker": "DEFENSE", "decision": "added", "side": "BUY", "units": 5, "reason": "..."}]}
"""
import argparse
import json
import operator
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Iterator, TypedDict

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents import learning, logbook, macro_news, model_portfolios, portfolio, risk, technical
from data import config as settings
from data import mock
from data.schemas import (
    EX_DIVIDENDS, FEATURES, HOLDINGS, INCIDENTS, ORDER_LOG, ORDERS, PORTFOLIO, PRICES, REVIEW_DECISIONS, REVIEWS, RISK_REPORT,
    SIDES, SIGNALS, validate_ex_dividends, validate_features, validate_holdings, validate_incidents, validate_order_log,
    validate_orders, validate_portfolio, validate_prices, validate_reviews, validate_risk_report, validate_signals,
)
from data.profiles import STRATEGIES
from data.universe import load_universe, tickers
from graph import review_loop

TECHNICAL_AGENTS = ["technical_det", "technical_llm"]
MACRO_AGENTS = ["macro_market", "macro_etf"]  # the views through the exposure table, the ETF analysts
_FILE_LOCK = threading.Lock()  # technical and macro_news run in parallel and share log files


class DayState(TypedDict, total=False):
    date: str
    prices: list[dict]
    indicators: list[dict]  # Features format
    holdings: list[dict]
    ex_dividends: list[dict] | None  # Ex-dividends format; None when the dates could not be read
    cash_eur: float
    signals: Annotated[list[dict], operator.add]  # technical and macro_news both add rows
    macro_summary: str
    lessons: str
    technical_weights: dict[str, float]  # today's weights of the technical agent's three answers (learning step)
    agent_multipliers: dict[str, float]  # what each agent's record makes of its starting weight (learning step)
    trust: float  # how much the LLM's portfolio counts against the code's (learning step)
    learning: list[dict]  # the learning scoreboard: level, name, start, weight, sessions, mean_result, right_share
    agent_weights: dict[str, float]  # today's weights over the agents that scored (portfolio step)
    round: int  # 1, then +1 each time the human sends the day back to the Portfolio Manager
    target_weights: list[dict]
    memo: str
    proposed_orders: list[dict]
    human_decisions: list[dict]  # one per proposed or added order, from the orders review
    approved_orders: list[dict]  # Orders format: what the human approved at the orders review
    risk_report: list[dict]
    risk_board: str  # Markdown: the Risk Manager's one-screen view of the report
    risk_reading: str  # Markdown: its independent reading of the memo, the events and three scenarios (LLM text)
    summary: str
    reviews: Annotated[list[dict], operator.add]  # Reviews format
    outcome: str  # "approved" or "stopped", once the day is done
    incidents: Annotated[list[dict], operator.add]


# --- Steps: the mock and the real module behind each switch -------------------------------

def _real_data(as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    from data import data_agent
    prices = data_agent.fetch_prices(as_of - timedelta(days=400), as_of)
    return prices, data_agent.compute_features(prices)


def _mock_portfolio(signals, weights, lessons, holdings, cash_eur, prices, as_of, feedback, **extra):
    revision = len(feedback.splitlines())  # another proposal after each piece of feedback
    memo = "Mock investment memo." + (f" Revised after: {feedback.splitlines()[-1]}" if feedback else "")
    return mock.mock_portfolio(seed=revision), mock.mock_orders(as_of.isoformat(), seed=revision), memo


def _real_ex_dividends(as_of: date) -> pd.DataFrame:
    from data import dividends
    return dividends.expected_ex_dividends(as_of, settings.TRADING_DAYS[-1] + timedelta(days=7), today=as_of)


# The ex-dividend dates come with the data step: its mock while USE_MOCK["data"] is on.
EX_DIVIDENDS_SOURCE = {True: lambda as_of: mock.mock_ex_dividends(as_of.isoformat()), False: _real_ex_dividends}

MOCK = {
    "data": lambda as_of: (mock.mock_prices(end=as_of.isoformat()), mock.mock_features(as_of.isoformat())),
    "technical": lambda prices, features, as_of, lessons, **extra: mock.mock_signals(as_of.isoformat(), agents=TECHNICAL_AGENTS),
    "macro_news": lambda as_of, lessons, **extra: ("Mock macro summary.",
                                                   mock.mock_signals(as_of.isoformat(), agents=MACRO_AGENTS)),
    "portfolio": _mock_portfolio,
    "risk": lambda orders, holdings, cash_eur, prices, as_of, **extra: (mock.mock_risk_report(as_of.isoformat()), {}),
}
REAL = {
    "data": lambda *args: _real_data(*args),
    "technical": lambda *args, **extra: technical.run(*args, **extra),
    "macro_news": lambda *args, **extra: macro_news.run(*args, **extra),
    "portfolio": lambda *args, **extra: portfolio.run(*args, **extra),
    "risk": lambda *args, **extra: risk.analyse(*args, **extra),  # (report, the details behind it)
    # Optional LLM text on top of the report (one LLM call). It never fails the day.
    "risk_reading": lambda report, *args, **extra: risk.render_assessment(risk.assess(report, *args, **extra)),
}


def _call(step: str, *args, **extra):
    """`extra`: the optional keyword inputs of a step (the technical weights, the logs folder, the macro views...)."""
    return (MOCK if settings.USE_MOCK[step] else REAL)[step](*args, **extra)


# --- Helpers -------------------------------------------------------------------------------

def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> plain rows for the saved state (dates as ISO strings, NaN as None)."""
    out = df.copy()
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    return out.astype(object).where(out.notna(), None).to_dict("records")


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _day(state: DayState, config: RunnableConfig) -> tuple[date, Path]:
    return date.fromisoformat(state["date"]), Path(config["configurable"]["logs_dir"])


def _replace_rows(path: Path, new: pd.DataFrame, keep_old=None) -> None:
    """Rewrite `path` with the old rows for which keep_old(old) is True (all by default), then `new`."""
    with _FILE_LOCK:
        if path.exists():
            old = pd.read_csv(path)
            if keep_old is not None:
                old = old[keep_old(old)]
            if len(old):
                new = pd.concat([old, new], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False)


def _incident(logs: Path, as_of: date, step: str, exc: Exception) -> dict:
    row = {"date": as_of.isoformat(), "step": step, "error": f"{type(exc).__name__}: {exc}"}
    _replace_rows(logs / "incidents.csv", validate_incidents(pd.DataFrame([row], columns=INCIDENTS)))
    print(f"incident in {step}: {row['error']}")
    return row


def neutral_signals(as_of: date, agents: list[str], reason: str) -> pd.DataFrame:
    """Fallback when a signal agent fails: score 0 and confidence 0 for every ETF."""
    rows = [{"ticker": t, "date": as_of.isoformat(), "agent": a, "score": 0.0, "confidence": 0.0,
             "rationale": f"agent failed: {reason}"} for a in agents for t in tickers()]
    return validate_signals(pd.DataFrame(rows, columns=SIGNALS))


def _check_signals(signals: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    validate_signals(signals)
    unexpected = set(signals["agent"]) - set(agents)
    if unexpected:
        raise ValueError(f"signals: unexpected agents {sorted(unexpected)}")
    return signals


def _read_signals(logs: Path, as_of: date, agents: list[str], step: str) -> pd.DataFrame | None:
    """Signals already logged for this date and these agents, or None.
    Mock signals are never reused once the step runs its real module."""
    path = logs / "signals.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df[(df["date"].astype(str) == as_of.isoformat()) & df["agent"].isin(agents)]
    if not settings.USE_MOCK[step] and (rows["rationale"] == "mock").any():
        return None
    return rows if len(rows) else None  # an agent may give only some of its scores (no technical_llm yet)


def _write_signals(logs: Path, signals: pd.DataFrame, as_of: date, agents: list[str]) -> None:
    _replace_rows(logs / "signals.csv", signals[SIGNALS],
                  lambda old: ~((old["date"].astype(str) == as_of.isoformat()) & old["agent"].isin(agents)))


def scoring_agents(signals: pd.DataFrame) -> list[str]:
    """The known agents that gave a real score in `signals`: not the neutral rows of an agent that failed."""
    real = signals[~signals["rationale"].astype(str).str.startswith("agent failed")]
    return [a for a in settings.SIGNAL_AGENTS if a in set(real["agent"])]


def same_book(log: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """The order-log rows of the portfolio `as_of` belongs to. Days before the trading window are
    practice and share a practice book; the competition starts again from START_CAPITAL_EUR on the
    first trading day, so practice orders never count in it."""
    start = settings.TRADING_DAYS[0]
    dates = pd.to_datetime(log["date"]).dt.date
    return log[dates >= start] if as_of >= start else log[dates < start]


def load_book(logs: Path, prices: pd.DataFrame, as_of: date,
              with_the_day: bool = False) -> tuple[pd.DataFrame, float]:
    """Holdings and cash before `as_of` (or after its orders, `with_the_day`), rebuilt from the executed
    orders of the same book (see same_book) in <logs>/orders.csv.

    An order fills at the price of fill_prices (the rule is one setting, FILL_AT_PREVIOUS_CLOSE), as in the
    Portfolio Manager, the Risk Manager and the Logbook.
    """
    cash = float(settings.START_CAPITAL_EUR)
    path = logs / "orders.csv"
    empty = pd.DataFrame(columns=HOLDINGS)
    if not path.exists():
        return empty, cash
    log = same_book(validate_order_log(pd.read_csv(path)), as_of)
    day = log["date"].astype(str)
    before = (day <= as_of.isoformat()) if with_the_day else (day < as_of.isoformat())
    done = log[(log["decision"] != "rejected") & before]
    if done.empty:
        return empty, cash
    units = done["units"].astype(float)
    signed = units.where(done["side"] == "BUY", -units)
    close = (prices.assign(date=prices["date"].astype(str))
             .pivot(index="date", columns="ticker", values="close_eur").sort_index())
    fills = [close.loc[settings.fills_from(close.index, d), t].dropna().iloc[-1]
             for d, t in zip(done["date"].astype(str), done["ticker"])]
    cash -= float((signed * fills).sum())
    position = signed.groupby(done["ticker"]).sum()
    holdings = position[position > 0].rename("units").rename_axis("ticker").reset_index()
    return validate_holdings(holdings), cash


GATE_NAMES = {"orders": "orders review", "risk": "risk review", "summary": "summary review"}


def fill_prices(prices: pd.DataFrame, as_of: date) -> pd.Series:
    """The price an order of `as_of` fills at, per ticker, on the ETF's own exchange: the last close before
    that day, or the close of that day itself (FILL_AT_PREVIOUS_CLOSE in data/config.py). Every module values
    an order this way. While the day's close is not known yet, both rules give the last close known."""
    day = prices["date"].astype(str).str[:10]
    p = prices[settings.fills_from(day, as_of.isoformat())].assign(_day=day).sort_values("_day")
    return p.groupby("ticker")["close_eur"].last().astype(float)


def cash_after(orders: pd.DataFrame, cash_eur: float, close: pd.Series) -> float:
    """Cash left once the orders are filled at `close` (orders without a price are left out)."""
    value = orders["units"].astype(float) * orders["ticker"].map(close)
    return cash_eur - float(value.where(orders["side"] == "BUY", -value).sum())


def check_orders(orders: pd.DataFrame, holdings: pd.DataFrame, cash_eur: float, prices: pd.DataFrame,
                 as_of: date) -> list[str]:
    """The trading rules, in plain words (empty when the orders pass): at most MAX_ORDERS_PER_DAY orders,
    one per ETF, units with at most ORDER_UNIT_DECIMALS decimals, no selling more than held, enough cash at
    the fill prices (sells fund buys)."""
    problems = []
    if len(orders) > settings.MAX_ORDERS_PER_DAY:
        problems.append(f"{len(orders)} orders, the limit is {settings.MAX_ORDERS_PER_DAY} a day")
    twice = sorted(set(orders.loc[orders["ticker"].duplicated(), "ticker"]))
    if twice:
        problems.append("More than one order on " + ", ".join(twice))
    held = dict(zip(holdings["ticker"], holdings["units"].astype(float)))
    step = 10 ** settings.ORDER_UNIT_DECIMALS
    scaled = orders["units"].astype(float) * step
    exits = (orders["side"] == "SELL") & ((orders["units"].astype(float) - orders["ticker"].map(held).fillna(-1.0)).abs() < 1e-9)
    finer = sorted(set(orders.loc[((scaled - scaled.round()).abs() > 1e-6) & ~exits, "ticker"]))  # selling all one holds is fine
    if finer:
        problems.append(("Whole units only: " if settings.ORDER_UNIT_DECIMALS == 0 else
                         f"At most {settings.ORDER_UNIT_DECIMALS} decimals in the units: ") + ", ".join(finer))
    for row in orders[orders["side"] == "SELL"].itertuples():
        if float(row.units) > held.get(row.ticker, 0) + 1e-9:
            problems.append(f"Selling {float(row.units):g} {row.ticker}, but only {held.get(row.ticker, 0):g} held")
    close = fill_prices(prices, as_of)
    missing = sorted(set(orders.loc[orders["ticker"].map(close).isna(), "ticker"]))
    if missing:
        problems.append("No price for " + ", ".join(missing))
    left = cash_after(orders, cash_eur, close)
    if left < -0.005:  # half a cent: the Portfolio Manager sizes its orders to the last cent of cash
        problems.append(f"Not enough cash: the orders need €{-left:,.0f} more than available")
    return problems


def executed_orders(log: pd.DataFrame) -> pd.DataFrame:
    """The rows of an order log that go to the Belfius form, in the Orders format."""
    done = log[log["decision"] != "rejected"]
    return pd.DataFrame({"date": done["date"], "side": done["side"], "ticker": done["ticker"],
                         "units": done["units"].astype(float)}, columns=ORDERS)


def approve(pending: dict) -> dict:
    """The answer that approves the review waiting, with every proposed order as it is."""
    answer = {"decision": "approved"}
    if pending["gate"] == "orders":
        answer["orders"] = [{"ticker": o["ticker"], "decision": "approved"} for o in pending["proposed_orders"]]
    return answer


def check_answer(state: dict, pending: dict, answer: dict) -> None:
    """Refuse an answer the day cannot take, before resuming the graph (so the day stays paused)."""
    decision = answer.get("decision")
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(REVIEW_DECISIONS)}")
    if decision != "approved" and not str(answer.get("feedback") or "").strip():
        raise ValueError("feedback is required to send the day back or to stop it")
    if decision == "revised" and pending["revisions_left"] <= 0:
        raise ValueError(f"no revision left today (at most {settings.MAX_REVISIONS})")
    if pending["gate"] == "orders" and decision == "approved":
        as_of = date.fromisoformat(state["date"])
        log = build_order_log(as_of, _frame(state["proposed_orders"], ORDERS), answer.get("orders", []))
        problems = check_orders(executed_orders(log), _frame(state["holdings"], HOLDINGS), state["cash_eur"],
                                _frame(state["prices"], PRICES), as_of)
        if problems:
            raise ValueError("; ".join(problems))


def feedback_for_portfolio(reviews: list[dict]) -> str:
    """The day's feedback for the Portfolio Manager: one line per time the human sent the day back."""
    return "\n".join(f"Round {r['round']}, {GATE_NAMES[r['gate']]}: {r['feedback']}"
                     for r in reviews if r["decision"] == "revised")


def order_notes(decisions: list[dict]) -> str:
    """The human changes to the proposed orders, in words: 'rejected BITCOIN: too volatile; ...'."""
    notes = []
    for d in decisions:
        if d.get("decision", "approved") == "approved":
            continue
        text = f"{d['decision']} {d['ticker']}"
        if d["decision"] == "modified" and d.get("units") is not None:
            text += f" to {fmt_units(d['units'])} units"
        if d["decision"] == "added":
            text += f" ({d.get('side')} {fmt_units(d.get('units') or 0)})"
        if str(d.get("reason") or "").strip():
            text += f": {d['reason']}"
        notes.append(text)
    return "; ".join(notes)


def fmt_units(units) -> str:
    """Units as the Belfius form takes them: up to ORDER_UNIT_DECIMALS decimals, no trailing zeros
    ('3', '2.5', '0.1234')."""
    text = f"{float(units):.{settings.ORDER_UNIT_DECIMALS}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def executive_summary(state: DayState, as_of: date) -> str:
    """The day on one page for the last human review: orders, why, risk, feedback. Built by code: the only
    LLM text in it is quoted (the Portfolio Manager's memo and the Risk Manager's independent reading)."""
    orders = _frame(state.get("approved_orders", []), ORDERS)
    close = fill_prices(_frame(state["prices"], PRICES), as_of)
    cash = float(state["cash_eur"])
    round_ = state.get("round", 1)
    lines = [f"# Executive summary · {as_of:%a %d %b %Y}" + (f" · round {round_}" if round_ > 1 else ""),
             "", "## Orders to submit", ""]
    if orders.empty:
        lines.append("No orders: the portfolio stays as it is.")
    else:
        lines += ["| Side | ETF | Units | Fill price | Value |", "|---|---|---:|---:|---:|"]
        for o in orders.itertuples():
            price = close.get(o.ticker)
            known = price is not None and pd.notna(price)
            lines.append(f"| {o.side} | {o.ticker} | {fmt_units(o.units)} | {f'€{price:,.2f}' if known else 'no price'} "
                         f"| {f'€{float(o.units) * price:,.0f}' if known else '—'} |")
        lines += ["", f"An order fills at {settings.fill_rule()}."]
        lines += ["", f"Cash: €{cash:,.0f} now, €{cash_after(orders, cash, close):,.0f} after the orders."]
    changes = order_notes(state.get("human_decisions", []))
    if changes:
        lines += ["", "## Your changes to the AI proposal", ""] + [f"- {c}" for c in changes.split("; ")]
    memo = "\n".join("#" + line if line.startswith("#") else line  # its headings, one level down
                     for line in (state.get("memo") or "No memo.").splitlines())
    lines += ["", "## Why: the Portfolio Manager's memo", "", memo]
    report = state.get("risk_report", [])
    if state.get("risk_board"):
        lines += ["", state["risk_board"].rstrip(), "", "### Every risk metric", ""]
    else:
        lines += ["", "## Risk after the orders", ""]
    if report:
        lines += ["| Metric | Value | Reference level | Status | Comment |", "|---|---:|---:|---|---|"]
        for r in report:
            limit = "—" if r.get("limit") is None or pd.isna(r["limit"]) else f"{float(r['limit']):g}"
            comment = str(r.get("comment") or "").replace("|", "/")
            lines.append(f"| {r['metric']} | {float(r['value']):g} | {limit} | {r['status']} | {comment} |")
    else:
        lines.append("The Risk Manager gave no assessment (see the incidents).")
    if state.get("risk_reading"):
        lines += ["", state["risk_reading"].rstrip()]
    sent_back = feedback_for_portfolio(state.get("reviews", []))
    if sent_back:
        lines += ["", "## Your feedback so far", ""] + [f"- {line}" for line in sent_back.splitlines()]
    if state.get("incidents"):
        lines += ["", "## Incidents", ""] + [f"- {i['step']}: {i['error']}" for i in state["incidents"]]
    return "\n".join(lines) + "\n"


def build_order_log(as_of: date, proposed: pd.DataFrame, decisions: list[dict]) -> pd.DataFrame:
    """One row per proposed order with its human decision, plus the orders the human added."""
    by_ticker: dict[str, dict] = {}
    for d in decisions:
        if d["ticker"] in by_ticker:
            raise ValueError(f"two decisions for {d['ticker']}")
        by_ticker[d["ticker"]] = d
    rows = []
    for order in proposed.itertuples():
        d = by_ticker.pop(order.ticker, None)
        if d is None:
            raise ValueError(f"no decision for the proposed order on {order.ticker}")
        decision = d.get("decision")
        if decision == "approved":
            side, units = order.side, order.units
        elif decision == "modified":
            side, units = d.get("side", order.side), d.get("units", order.units)
        elif decision == "rejected":
            side, units = None, None
        else:
            raise ValueError(f"{order.ticker}: a proposed order is approved, modified or rejected")
        rows.append([as_of.isoformat(), order.ticker, order.side, order.units, decision, side, units,
                     d.get("reason", "")])
    for ticker, d in by_ticker.items():
        if d.get("decision") != "added":
            raise ValueError(f"{ticker}: no order was proposed, so the decision must be 'added'")
        rows.append([as_of.isoformat(), ticker, None, None, "added", d.get("side"), d.get("units"),
                     d.get("reason", "")])
    return validate_order_log(pd.DataFrame(rows, columns=ORDER_LOG))


# --- Graph nodes ---------------------------------------------------------------------------

def data_step(state: DayState, config: RunnableConfig) -> dict:
    as_of, logs = _day(state, config)
    incidents = []
    try:
        prices, features = _call("data", as_of)
        validate_prices(prices)
        validate_features(features)
        holdings, cash = load_book(logs, prices, as_of)
    except Exception as exc:
        incidents.append(_incident(logs, as_of, "data", exc))
        prices, features = pd.DataFrame(columns=PRICES), pd.DataFrame(columns=FEATURES)
        holdings, cash = pd.DataFrame(columns=HOLDINGS), float(settings.START_CAPITAL_EUR)
    try:  # without the dates the Risk Manager warns about every distributing ETF: a failure here costs nothing else
        ex_dividends = _records(validate_ex_dividends(EX_DIVIDENDS_SOURCE[settings.USE_MOCK["data"]](as_of)))
    except Exception as exc:
        ex_dividends = None
        incidents.append(_incident(logs, as_of, "dividends", exc))
    lessons_path = logs / "lessons.md"
    return {
        "prices": _records(prices),
        "indicators": _records(features),
        "holdings": _records(holdings),
        "ex_dividends": ex_dividends,
        "cash_eur": cash,
        "lessons": lessons_path.read_text(encoding="utf-8") if lessons_path.exists() else "",
        "incidents": incidents,
    }


def _signals_home(step: str, logs: Path) -> Path:
    """Where a signal step keeps its signals: logs/ (graded evidence, and the data the review loop learns
    from) once the step is real, even while other steps are still on mocks; else <logs>."""
    return logs if settings.USE_MOCK[step] else settings.LOGS_DIR


def learning_step(state: DayState, config: RunnableConfig) -> dict:
    """Performance monitoring and learning (graph/review_loop.py): one rule re-weights the technical answers, the
    agents and the LLM's portfolio from what each earned, and the lessons are written. A failure is an incident: the
    day goes on with the starting weights and the lessons already on file."""
    as_of, logs = _day(state, config)
    try:
        result = review_loop.review(as_of, _frame(state["prices"], PRICES), logs, _frame(state["indicators"], FEATURES),
                                    replay_technical=not settings.USE_MOCK["technical"],  # a mock has no rules to replay
                                    fetch=None if settings.USE_MOCK["data"] else review_loop.long_history)
        return {"lessons": result["journal"], "technical_weights": result["technical_weights"],
                "agent_multipliers": result["agent_multipliers"], "trust": result["trust"],
                "learning": _records(result["board"])}
    except Exception as exc:
        return {"incidents": [_incident(logs, as_of, "learning", exc)]}


def technical_step(state: DayState, config: RunnableConfig) -> dict:
    as_of, logs = _day(state, config)
    home = _signals_home("technical", logs)
    cached = _read_signals(home, as_of, TECHNICAL_AGENTS, "technical")
    if cached is not None:
        return {"signals": _records(cached)}
    try:
        signals = _check_signals(_call("technical", _frame(state["prices"], PRICES),
                                       _frame(state["indicators"], FEATURES), as_of, state["lessons"],
                                       weights=state.get("technical_weights")),
                                 TECHNICAL_AGENTS)
        _write_signals(home, signals, as_of, TECHNICAL_AGENTS)
        return {"signals": _records(signals)}
    except Exception as exc:
        incident = _incident(logs, as_of, "technical", exc)
        return {"signals": _records(neutral_signals(as_of, TECHNICAL_AGENTS, incident["error"])),
                "incidents": [incident]}


def macro_news_step(state: DayState, config: RunnableConfig) -> dict:
    """The real agent reads and writes logs/ (graded evidence) even while other steps are on mocks, so the
    day's run from `python -m agents.macro_news` is reused and the model is never called twice for a date.
    It reads the news of the day: it runs on the day itself, never for another date."""
    as_of, logs = _day(state, config)
    real = not settings.USE_MOCK["macro_news"]
    home = _signals_home("macro_news", logs)
    summary_path = home / "macro" / f"{as_of.isoformat()}.md"
    cached = _read_signals(home, as_of, MACRO_AGENTS, "macro_news")
    if cached is not None:
        summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        return {"signals": _records(cached), "macro_summary": summary}
    try:
        if real and as_of != date.today():
            raise ValueError(f"no Macro/News run for {as_of}: it reads the news of the day, so it only runs "
                             f"on {as_of} itself (in the cockpit or with python -m agents.macro_news)")
        # prices: the analysts read what each ETF already did, and how their past calls turned out
        summary, signals = _call("macro_news", as_of, state["lessons"], prices=_frame(state["prices"], PRICES))
        _check_signals(signals, MACRO_AGENTS)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8", newline="\n")
        _write_signals(home, signals, as_of, MACRO_AGENTS)
        return {"signals": _records(signals), "macro_summary": summary}
    except Exception as exc:
        incident = _incident(logs, as_of, "macro_news", exc)
        return {"signals": _records(neutral_signals(as_of, MACRO_AGENTS, incident["error"])),
                "macro_summary": f"Macro/News agent failed: {incident['error']}", "incidents": [incident]}


def _book(logs: Path, as_of: date, record, *args, **kwargs) -> list[dict]:
    """Archive in the logbook. A failure is an incident: the logbook never stops the day."""
    try:
        record(logs, as_of, *args, **kwargs)
        return []
    except Exception as exc:
        return [_incident(logs, as_of, "logbook", exc)]


def proposal_path(logs: Path, as_of: date, round_: int) -> Path:
    """Where a round of the Portfolio Manager is frozen."""
    return logs / "proposals" / f"{as_of.isoformat()}-round{round_}.json"


def _logged_proposal(logs: Path, as_of: date) -> dict | None:
    """The first round of a day run before proposals were frozen, rebuilt from what the Portfolio Manager logged:
    its final blend (pm_portfolios.csv), its orders (recommendations.csv, the latest run) and its memo. None when
    the logs hold none."""
    day = as_of.isoformat()
    paths = [logs / "pm_portfolios.csv", logs / "recommendations.csv", logs / "memo" / f"{day}.md"]
    if not all(path.exists() for path in paths):
        return None
    blend, recs = pd.read_csv(paths[0]), pd.read_csv(paths[1])
    blend = blend[(blend["date"].astype(str) == day) & (blend["source"] == "final")]
    recs = recs[recs["date"].astype(str) == day]
    if blend.empty or recs.empty:
        return None
    recs = recs[(recs["created_at"] == recs["created_at"].max()) & recs["side"].isin(sorted(SIDES))]
    target = pd.DataFrame({"ticker": blend["ticker"], "target_weight": blend["weight"]}, columns=PORTFOLIO)
    orders = pd.DataFrame({"date": day, "side": recs["side"], "ticker": recs["ticker"], "units": recs["units"]},
                          columns=ORDERS)
    return {"round": 1, "feedback": "", "target": _records(target), "orders": _records(orders),
            "memo": paths[2].read_text(encoding="utf-8")}


def frozen_proposal(logs: Path, as_of: date, round_: int, feedback: str) -> dict | None:
    """The round as it was first computed, if it was: same day, same round, same feedback from the human."""
    path = proposal_path(logs, as_of, round_)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else (
        _logged_proposal(logs, as_of) if round_ == 1 else None)
    return saved if saved is not None and saved.get("feedback", "") == feedback else None


def portfolio_step(state: DayState, config: RunnableConfig) -> dict:
    as_of, logs = _day(state, config)
    fresh = {"human_decisions": [], "approved_orders": [], "risk_report": [], "risk_board": "", "risk_reading": "",
             "summary": ""}  # a new round
    holdings, prices = _frame(state["holdings"], HOLDINGS), _frame(state["prices"], PRICES)
    signals = _frame(state["signals"], SIGNALS)
    # today's weights: each agent that scored today, its starting weight times what its record made of it
    weights = learning.weights_now(learning.starting_weights(scoring_agents(signals)), state.get("agent_multipliers") or {})
    fresh["agent_weights"] = weights
    round_, feedback = state.get("round", 1), feedback_for_portfolio(state.get("reviews", []))
    strategy = model_portfolios.traded(logs, as_of)  # chosen in the cockpit, logged: the same on every machine
    saved = frozen_proposal(logs, as_of, round_, feedback)
    if saved is not None:  # this round was already computed: the logs say what it proposed
        book = _book(logs, as_of, logbook.record_round, round_, target=_frame(saved["target"], PORTFOLIO),
                     memo=saved["memo"], proposed=_frame(saved["orders"], ORDERS))
        return {**fresh, "target_weights": saved["target"], "proposed_orders": saved["orders"], "memo": saved["memo"],
                "incidents": book}
    try:
        # logs_dir: where it logs its portfolios and proposals (the learning step reads them)
        target, orders, memo = _call("portfolio", signals, weights,
                                     state["lessons"], holdings, state["cash_eur"], prices, as_of, feedback,
                                     macro_summary=state.get("macro_summary", ""), logs_dir=logs,
                                     trust=state.get("trust"), learning=pd.DataFrame(state.get("learning") or []),
                                     strategy=strategy)
        validate_portfolio(target)
        validate_orders(orders)
        problems = check_orders(orders, holdings, state["cash_eur"], prices, as_of)
        if problems:
            raise ValueError("the proposed orders break the trading rules: " + "; ".join(problems))
    except Exception as exc:
        incident = _incident(logs, as_of, "portfolio", exc)
        return {**fresh, "target_weights": [], "proposed_orders": [],
                "memo": f"Portfolio Manager failed: {incident['error']}", "incidents": [incident]}
    page = logs / "memo" / f"{as_of.isoformat()}.md"  # the latest round's memo, for the cockpit
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(memo, encoding="utf-8", newline="\n")
    frozen = proposal_path(logs, as_of, round_)  # from now on, this round is read back, never computed again
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps({"round": round_, "strategy": strategy, "feedback": feedback,
                                  "target": _records(target[PORTFOLIO]),
                                  "orders": _records(orders[ORDERS]), "memo": memo}, indent=1, default=str),
                      encoding="utf-8", newline="\n")
    book = _book(logs, as_of, logbook.record_round, state.get("round", 1), target=target[PORTFOLIO], memo=memo,
                 proposed=orders[ORDERS])
    return {**fresh, "target_weights": _records(target[PORTFOLIO]), "proposed_orders": _records(orders[ORDERS]),
            "memo": memo, "incidents": book}


def _review(state: DayState, gate: str, show: dict) -> tuple[dict, dict]:
    """Pause for the human at `gate`. Returns (their answer, the state update)."""
    round_ = state.get("round", 1)
    answer = interrupt({"gate": gate, "date": state["date"], "round": round_,
                        "revisions_left": settings.MAX_REVISIONS - (round_ - 1),
                        "incidents": state.get("incidents", []), **show})
    feedback = str(answer.get("feedback") or "").strip()
    if gate == "orders" and answer["decision"] == "revised":
        notes = order_notes(answer.get("orders", []))
        feedback = f"{feedback} [{notes}]" if notes else feedback
    update = {"reviews": [{"date": state["date"], "round": round_, "gate": gate, "decision": answer["decision"],
                           "feedback": feedback}]}
    if answer["decision"] == "revised":
        update["round"] = round_ + 1
    elif answer["decision"] == "stopped":
        update["outcome"] = "stopped"
    return answer, update


def orders_review_step(state: DayState) -> dict:
    answer, update = _review(state, "orders", {
        "proposed_orders": state["proposed_orders"], "target_weights": state.get("target_weights", []),
        "memo": state.get("memo", ""), "macro_summary": state.get("macro_summary", ""),
    })
    if answer["decision"] == "approved":
        log = build_order_log(date.fromisoformat(state["date"]), _frame(state["proposed_orders"], ORDERS),
                              answer.get("orders", []))
        update.update(human_decisions=list(answer.get("orders", [])), approved_orders=_records(executed_orders(log)))
    return update


def risk_step(state: DayState, config: RunnableConfig) -> dict:
    """The risk report, its board and (real module only) the independent reading, also saved as
    <logs>/risk/<date>.md. A failed board or reading is an incident: the report still reaches the human."""
    as_of, logs = _day(state, config)
    orders, holdings = _frame(state["approved_orders"], ORDERS), _frame(state["holdings"], HOLDINGS)
    prices, cash = _frame(state["prices"], PRICES), state["cash_eur"]
    macro_home = _signals_home("macro_news", logs)  # where the Macro/News agent keeps its views and evidence
    views_path = macro_home / "macro_views.csv"
    try:
        dated = state.get("ex_dividends")
        report, details = _call("risk", orders, holdings, cash, prices, as_of, logs_dir=logs,
                                macro_views=pd.read_csv(views_path) if views_path.exists() else None,
                                ex_dividends=None if dated is None else _frame(dated, EX_DIVIDENDS))
        validate_risk_report(report)
    except Exception as exc:
        return {"risk_report": [], "risk_board": "", "risk_reading": "",
                "incidents": [_incident(logs, as_of, "risk", exc)]}
    board, reading, incidents = "", "", []
    try:
        board = risk.board(report, details)
        if not settings.USE_MOCK["risk"]:
            reading = REAL["risk_reading"](report, orders, holdings, cash, prices, as_of, memo=state.get("memo", ""),
                                           macro_summary=state.get("macro_summary", ""), logs_dir=macro_home)
            page = logs / "risk" / f"{as_of.isoformat()}.md"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(f"{board}\n{reading}", encoding="utf-8", newline="\n")
    except Exception as exc:
        incidents.append(_incident(logs, as_of, "risk", exc))
    incidents += _book(logs, as_of, logbook.record_round, state.get("round", 1), risk_report=report[RISK_REPORT])
    return {"risk_report": _records(report[RISK_REPORT]), "risk_board": board, "risk_reading": reading,
            "incidents": incidents}


def risk_review_step(state: DayState) -> dict:
    return _review(state, "risk", {"approved_orders": state["approved_orders"],
                                   "risk_report": state.get("risk_report", []),
                                   "risk_board": state.get("risk_board", ""),
                                   "risk_reading": state.get("risk_reading", "")})[1]


def summary_step(state: DayState, config: RunnableConfig) -> dict:
    as_of, _ = _day(state, config)
    return {"summary": executive_summary(state, as_of)}


def summary_review_step(state: DayState) -> dict:
    return _review(state, "summary", {"summary": state["summary"], "approved_orders": state["approved_orders"]})[1]


def publish_references(state: DayState, as_of: date, logs: Path) -> list[dict]:
    """Once the team has signed the day off: each strategy's reference for the customers follows the engine. The
    code's portfolio of the Portfolio Manager (no LLM) is built for each strategy from the day's scores and learned
    weights, inside its limits, and published as a new version when enough of it changes (agents/model_portfolios.py).
    Returns the incidents (a failure never stops the day)."""
    try:
        signals, prices = _frame(state["signals"], SIGNALS), _frame(state["prices"], PRICES)
        targets = {}
        for strategy in STRATEGIES:
            target, _, _ = _call("portfolio", signals, state.get("agent_weights") or {}, "",
                                 pd.DataFrame(columns=HOLDINGS), settings.START_CAPITAL_EUR, prices, as_of, "",
                                 use_llm=False, strategy=strategy)
            targets[strategy] = target.set_index("ticker")["target_weight"]
        families = dict(zip(load_universe()["ticker"], load_universe()["family"]))
        model_portfolios.publish(logs, as_of, targets, families)
        return []
    except Exception as exc:
        return [_incident(logs, as_of, "references", exc)]


def order_log_step(state: DayState, config: RunnableConfig) -> dict:
    as_of, logs = _day(state, config)
    proposed = _frame(state["proposed_orders"], ORDERS)
    outcome = state.get("outcome") or "approved"
    if outcome == "stopped":
        last = state["reviews"][-1]
        reason = f"no trade today ({GATE_NAMES[last['gate']]}): {last['feedback']}"
        decisions = [{"ticker": t, "decision": "rejected", "reason": reason} for t in proposed["ticker"]]
    else:
        decisions = state["human_decisions"]
    other_days = lambda old: old["date"].astype(str) != as_of.isoformat()  # noqa: E731
    _replace_rows(logs / "orders.csv", build_order_log(as_of, proposed, decisions), other_days)
    _replace_rows(logs / "reviews.csv", validate_reviews(_frame(state["reviews"], REVIEWS)), other_days)
    incidents = []
    if outcome == "approved":
        path = logs / "summary" / f"{as_of.isoformat()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state["summary"], encoding="utf-8", newline="\n")
        incidents = publish_references(state, as_of, logs)
    # last: the logbook joins everything the day wrote into the audit trail, and computes the fills and the value
    return {"outcome": outcome,
            "incidents": incidents + _book(logs, as_of, logbook.record_day, _frame(state["prices"], PRICES))}


def _after(next_step: str):
    """Route a review: approved -> next step, revised -> Portfolio Manager, stopped -> order log."""
    def route(state: DayState) -> str:
        decision = state["reviews"][-1]["decision"]
        return {"approved": next_step, "revised": "portfolio", "stopped": "order_log"}[decision]
    return route


def build_graph() -> StateGraph:
    g = StateGraph(DayState)
    g.add_node("data", data_step)
    g.add_node("learning", learning_step)
    g.add_node("technical", technical_step)
    g.add_node("macro_news", macro_news_step)
    g.add_node("portfolio", portfolio_step)
    g.add_node("orders_review", orders_review_step)
    g.add_node("risk", risk_step)
    g.add_node("risk_review", risk_review_step)
    g.add_node("summary", summary_step)
    g.add_node("summary_review", summary_review_step)
    g.add_node("order_log", order_log_step)
    g.add_edge(START, "data")
    g.add_edge("data", "learning")
    g.add_edge("learning", "technical")
    g.add_edge("learning", "macro_news")
    g.add_edge(["technical", "macro_news"], "portfolio")
    g.add_edge("portfolio", "orders_review")
    g.add_conditional_edges("orders_review", _after("risk"), ["risk", "portfolio", "order_log"])
    g.add_edge("risk", "risk_review")
    g.add_conditional_edges("risk_review", _after("summary"), ["summary", "portfolio", "order_log"])
    g.add_edge("summary", "summary_review")
    g.add_conditional_edges("summary_review", _after("order_log"), ["order_log", "portfolio"])
    g.add_edge("order_log", END)
    return g


# --- Python API (used by the CLI below and by the Streamlit app) --------------------------

def default_logs_dir() -> Path:
    return settings.LOGS_DIR / "mock" if any(settings.USE_MOCK.values()) else settings.LOGS_DIR


@contextmanager
def open_graph(logs: Path) -> Iterator:
    logs.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(logs / "checkpoints.sqlite", check_same_thread=False)
    try:
        yield build_graph().compile(checkpointer=SqliteSaver(conn))
    finally:
        conn.close()


def thread(as_of: date, logs: Path) -> RunnableConfig:
    return {"configurable": {"thread_id": as_of.isoformat(), "logs_dir": str(logs)}}


def _pending(graph, cfg: RunnableConfig) -> dict | None:
    snapshot = graph.get_state(cfg)
    return snapshot.interrupts[0].value if snapshot.interrupts else None


def _outdated(snapshot) -> bool:
    """A day paused by an older version of the graph (a single approval step, no review gate)."""
    return bool(snapshot.interrupts) and "gate" not in snapshot.interrupts[0].value


def finished_in_logs(as_of: date, logs: Path) -> bool:
    """Whether the day's reviews are in the logs: written once, when the day ends (approved or no trade)."""
    path = logs / "reviews.csv"
    return path.exists() and (pd.read_csv(path)["date"].astype(str) == as_of.isoformat()).any()


def day_status(as_of: date, logs: Path | None = None) -> tuple[str, dict, dict | None]:
    """Read-only: ("not started" | "waiting for approval" | "done", saved state, pause payload or None).
    The payload's "gate" says which review is waiting: orders, risk or summary. A day finished in the logs is done,
    even on a machine that never ran it (its state is then empty: read the logs)."""
    logs = Path(logs) if logs else default_logs_dir()
    with open_graph(logs) as graph:
        snapshot = graph.get_state(thread(as_of, logs))
    if not snapshot.values or _outdated(snapshot):
        return ("done" if finished_in_logs(as_of, logs) else "not started"), {}, None
    if snapshot.interrupts:
        return "waiting for approval", snapshot.values, snapshot.interrupts[0].value
    return "done", snapshot.values, None


def run_day(as_of: date, logs: Path | None = None) -> dict | None:
    """Run a day up to its first review and return what the human must look at.

    A day is run once. A day already paused here is not run again; a day finished in the logs (by anyone) is not run
    at all; a day another machine started replays from the logs, to the same orders. Returns None once it is done.
    """
    logs = Path(logs) if logs else default_logs_dir()
    with open_graph(logs) as graph:
        cfg = thread(as_of, logs)
        if _outdated(graph.get_state(cfg)):
            graph.checkpointer.delete_thread(as_of.isoformat())
        if not graph.get_state(cfg).values and finished_in_logs(as_of, logs):
            return None
        if not graph.get_state(cfg).values:
            graph.invoke({"date": as_of.isoformat(), "round": 1}, cfg)
        return _pending(graph, cfg)


def resume_day(as_of: date, answer: dict, logs: Path | None = None) -> dict | None:
    """Answer the review waiting for `as_of`. Returns the next review, or None once the day is done
    (<logs>/orders.csv, reviews.csv and summary/<date>.md are then written)."""
    logs = Path(logs) if logs else default_logs_dir()
    with open_graph(logs) as graph:
        cfg = thread(as_of, logs)
        snapshot = graph.get_state(cfg)
        if not snapshot.interrupts:
            raise ValueError(f"{as_of}: no review waiting, run the day first")
        check_answer(snapshot.values, snapshot.interrupts[0].value, answer)  # fail before resuming
        graph.invoke(Command(resume=answer), cfg)
        return _pending(graph, cfg)


def approve_day(as_of: date, logs: Path | None = None) -> None:
    """Run the day if needed, then approve every review left with the orders as proposed."""
    pending = run_day(as_of, logs)
    while pending is not None:
        pending = resume_day(as_of, approve(pending), logs)


def _print_review(as_of: date, pending: dict, logs: Path) -> None:
    gate = pending["gate"]
    print(f"{as_of}: waiting for the {GATE_NAMES[gate]} (round {pending['round']}, "
          f"{pending['revisions_left']} revision(s) left, state saved in {logs / 'checkpoints.sqlite'})\n")
    if gate == "orders":
        print(pending["memo"], "\n")
        orders = pd.DataFrame(pending["proposed_orders"], columns=ORDERS)
        print(orders.to_string(index=False) if len(orders) else "No orders proposed.")
    elif gate == "risk":
        report = pd.DataFrame(pending["risk_report"], columns=RISK_REPORT)
        print(report.to_string(index=False) if len(report) else "No risk assessment.")
    else:
        print(pending["summary"])
    for incident in pending["incidents"]:
        print(f"incident in {incident['step']}: {incident['error']}")
    print("\nAnswer with --answer <file.json>, or --approve-all")


def _print_done(as_of: date, logs: Path) -> None:
    log = pd.read_csv(logs / "orders.csv")
    log = log[log["date"].astype(str) == as_of.isoformat()]
    print(log.to_string(index=False, na_rep="") if len(log) else "No orders.")
    print(f"Written to {logs / 'orders.csv'} and {logs / 'reviews.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one trading day through the agent pipeline.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD, default today")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--approve-all", action="store_true", help="approve every review left, orders as proposed")
    resume.add_argument("--answer", type=Path, help="answer the review waiting with this JSON file")
    parser.add_argument("--logs-dir", type=Path, help="default: logs/mock/ while any step is on a mock, else logs/")
    args = parser.parse_args(argv)
    logs = args.logs_dir or default_logs_dir()

    pending = run_day(args.date, logs)
    if pending is None:
        print(f"{args.date}: already done, see {logs / 'orders.csv'}. A day is run once: its logs are the record.")
        return
    if args.approve_all:
        approve_day(args.date, logs)
        pending = None
    elif args.answer:
        pending = resume_day(args.date, json.loads(args.answer.read_text(encoding="utf-8")), logs)
    if pending is None:
        _print_done(args.date, logs)
    else:
        _print_review(args.date, pending, logs)


if __name__ == "__main__":
    main()
