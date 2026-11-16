"""The day as a diagram: every step with its place, what it does, its files and the code behind it.

The same flow as the team's diagram (README): data, learning, the two agents, the Portfolio Manager, then the
three human reviews around the Risk Manager and the executive summary, and the logbook. A review can send the day
back to the Portfolio Manager; the next morning, the learning step reads what the day wrote.
"""
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from agents import learning, logbook, macro_analysts, macro_news, portfolio, risk, technical
from app import data as cockpit
from data import data_agent, features
from data.schemas import REVIEW_GATES
from graph import pipeline

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Step:
    key: str
    name: str
    label: str  # on the box, lines split by "\n"
    kind: str  # code | llm | you
    x: float
    y: float
    what: str
    inputs: str
    outputs: str
    files: list[str]
    code: list[Callable] = field(default_factory=list)


STEPS = [
    Step("data", "Data", "Data", "code", 0.0, 0.0,
         "Real prices of the 50 ETFs on their Belfius exchange, 37 indicators each, and the ex-dividend dates.",
         "Yahoo Finance, EODHD", "prices, indicators", ["data/data_agent.py", "data/features.py", "data/dividends.py"],
         [data_agent.fetch_prices, features.compute_features]),
    Step("learning", "Learning", "Learning", "code", 1.15, 0.0,
         "One rule re-weights every expert each morning: a weight × e^(2 × what following it earned in the "
         "last session). Also writes the lessons the agents read.",
         "the logs, the prices", "the weights of the day, lessons.md",
         ["agents/learning.py", "graph/review_loop.py"], [learning.daily_weights, learning.call_results]),
    Step("technical", "Technical agent", "Technical", "llm", 2.3, 0.55,
         "Three answers per ETF from prices (trend, mean reversion, momentum ranking), weighted by what each earned "
         "since 2022: technical_det. Claude weighs the same answers when they disagree: technical_llm.",
         "prices, indicators, its weights", "2 scores per ETF",
         ["agents/technical.py", "agents/technical_llm.py", "agents/strategy_config.py"],
         [technical.suggest_signal, technical.signal_scores, technical.blend]),
    Step("macro_news", "Macro / News agent", "Macro /\nNews", "llm", 2.3, -0.55,
         "Searches the news and data sources, gives a view on 15 macro factors (macro_market, through the exposure "
         "table), and 20 theme analysts score their ETFs (macro_etf).",
         "web searches, data collectors", "2 scores per ETF, a report",
         ["agents/macro_news.py", "agents/macro_llm.py", "agents/macro_analysts.py", "data/macro_exposure.csv"],
         [macro_news.score_etfs, macro_analysts.etf_signals]),
    Step("portfolio", "Portfolio Manager", "Portfolio\nManager", "llm", 3.45, 0.0,
         "Combines the four scores with their weights, builds a target portfolio (code) and asks Claude for its "
         "own, blends both, then picks at most 5 orders.",
         "4 scores, weights, holdings, cash", "target, up to 5 orders, a memo",
         ["agents/portfolio.py", "agents/portfolio_config.py", "agents/portfolio_llm.py"],
         [portfolio.combine_signals, portfolio.blend, portfolio.select_trades]),
    Step("review_orders", "Your review: the orders", "You:\norders", "you", 4.6, 0.0,
         "You approve, change, reject or add orders. The trading rules are checked again on what you approve.",
         "the proposed orders", "approved orders, or the day sent back", ["app/views/today.py", "graph/pipeline.py"],
         [pipeline.check_orders]),
    Step("risk", "Risk Manager", "Risk\nManager", "llm", 5.75, 0.0,
         "Measures the portfolio after your orders on 22 metrics against reference levels. Claude writes an "
         "independent reading. It reports, it never advises.",
         "approved orders, holdings, prices", "risk report, board, reading",
         ["agents/risk.py", "agents/risk_config.py", "agents/risk_llm.py"], [risk.status_of, risk.board]),
    Step("review_risk", "Your review: the risk", "You:\nrisk", "you", 6.9, 0.0,
         "You accept the risk, or send the day back to the Portfolio Manager with the limits you want.",
         "the risk report", "go on, or the day sent back", ["app/views/today.py", "graph/pipeline.py"],
         [pipeline.check_answer]),
    Step("summary", "Executive summary", "Executive\nsummary", "code", 8.05, 0.0,
         "The day on one page, built by code: orders, cash, the memo, the risk, your feedback.",
         "the whole day", "one page to sign", ["graph/pipeline.py"], [pipeline.executive_summary]),
    Step("review_summary", "Your review: the summary", "You:\nsummary", "you", 9.2, 0.0,
         "You sign off the summary. Approving writes the order log for the Belfius form.",
         "the summary", "the order log", ["app/views/today.py", "graph/pipeline.py"], [pipeline.order_log_step]),
    Step("logbook", "Orders and logbook", "Orders &\nlogbook", "code", 10.35, 0.0,
         "Writes the orders, your reviews and an audit trail of the whole day, each row hashed with the one before.",
         "everything the day wrote", "orders.csv, logbook.csv", ["agents/logbook.py", "graph/pipeline.py"],
         [logbook.record_day, logbook.verify]),
]
BY_KEY = {s.key: s for s in STEPS}
HALF_W, HALF_H = 0.42, 0.31
LANE_BACK, LANE_NEXT = -1.1, 1.15


def _right(a: Step, b: Step) -> list[tuple[float, float]]:
    """From the right side of a to the left side of b, with an elbow when they are not level."""
    start, end = (a.x + HALF_W, a.y), (b.x - HALF_W, b.y)
    if a.y == b.y:
        return [start, end]
    middle = (start[0] + end[0]) / 2
    return [start, (middle, a.y), (middle, b.y), end]


def edges() -> list[dict]:
    """The arrows: the flow, the three "send back" loops to the Portfolio Manager, the next morning's learning."""
    s = BY_KEY
    flow = [("data", "learning"), ("learning", "technical"), ("learning", "macro_news"), ("technical", "portfolio"),
            ("macro_news", "portfolio"), ("portfolio", "review_orders"), ("review_orders", "risk"),
            ("risk", "review_risk"), ("review_risk", "summary"), ("summary", "review_summary"),
            ("review_summary", "logbook")]
    out = [{"points": _right(s[a], s[b]), "style": "flow", "from": a, "to": b} for a, b in flow]
    pm = s["portfolio"]
    for gate in REVIEW_GATES:
        review = s[f"review_{gate}"]
        out.append({"points": [(review.x, review.y - HALF_H), (review.x, LANE_BACK), (pm.x, LANE_BACK),
                               (pm.x, pm.y - HALF_H)], "style": "back", "from": review.key, "to": "portfolio"})
    end, start = s["logbook"], s["learning"]
    out.append({"points": [(end.x, end.y + HALF_H), (end.x, LANE_NEXT), (start.x, LANE_NEXT),
                           (start.x, start.y + HALF_H)], "style": "next", "from": "logbook", "to": "learning"})
    return out


def nodes() -> pd.DataFrame:
    return pd.DataFrame([{"step": s.key, "name": s.name, "label": s.label, "kind": s.kind, "x": s.x, "y": s.y,
                          "hint": "click to see what it does"} for s in STEPS])


def code_of(fn: Callable) -> tuple[str, str]:
    """(file:line, source) of a function."""
    path = Path(inspect.getsourcefile(fn)).resolve().relative_to(ROOT)
    return f"{path}:{inspect.getsourcelines(fn)[1]}", inspect.getsource(fn)


def today(key: str, status: str, state: dict, pending: dict | None, logs: Path, day) -> str:
    """One line on what the step gave for the day, from the day's saved state and the logs."""
    if status == "not started" and key not in ("learning", "logbook"):
        return "Not run yet for this day: run it from the dashboard."
    signals = pd.DataFrame(state.get("signals", []))
    if key == "data":
        prices = pd.DataFrame(state.get("prices", []))
        return f"{prices['ticker'].nunique()} ETFs priced, last close {prices['date'].max()}" if len(prices) else "No prices."
    if key == "learning":
        found = cockpit.weights_on(logs, day)
        if not found:
            return "No weights written yet for this day."
        return " · ".join(f"{level}: " + ", ".join(f"{n.replace('_', ' ')} {w:.0%}" for n, w in sorted(
            weights.items(), key=lambda kv: -kv[1])) for level, weights in found.items())
    if key in ("technical", "macro_news"):
        agents = pipeline.TECHNICAL_AGENTS if key == "technical" else pipeline.MACRO_AGENTS
        mine = signals[signals["agent"].isin(agents)] if len(signals) else signals
        if mine.empty:
            return "No score."
        first = mine[mine["agent"] == agents[0]].sort_values("score")
        return (f"{mine['ticker'].nunique()} ETFs scored by {mine['agent'].nunique()} scores · {agents[0]} strongest "
                f"{first['ticker'].iloc[-1]} {first['score'].iloc[-1]:+.2f}, weakest {first['ticker'].iloc[0]} "
                f"{first['score'].iloc[0]:+.2f}")
    if key == "portfolio":
        orders = state.get("proposed_orders", [])
        return f"{len(orders)} order{'s' if len(orders) != 1 else ''} proposed, round {state.get('round', 1)}"
    if key.startswith("review_"):
        gate = key.removeprefix("review_")
        answers = [r for r in state.get("reviews", []) if r["gate"] == gate]
        if answers:
            return f"Your last answer: {answers[-1]['decision']} (round {answers[-1]['round']})"
        return "Waiting for you." if pending and pending["gate"] == gate else "Not reached yet."
    if key == "risk":
        report = pd.DataFrame(state.get("risk_report", []))
        if report.empty:
            return "Not reached yet: it runs after your orders review."
        counts = report["status"].value_counts()
        return f"{len(report)} metrics: {counts.get('warning', 0)} warnings, {counts.get('breach', 0)} breaches"
    if key == "summary":
        return "Written: waiting for your sign-off." if state.get("summary") else "Not reached yet."
    if key == "logbook":
        ledger = logbook.load(logs)
        mine = ledger[ledger["date"].astype(str) == day.isoformat()] if len(ledger) else ledger
        return f"{len(mine)} events for this day · hash chain {'intact' if not logbook.verify(logs) else 'BROKEN'}"
    return ""
