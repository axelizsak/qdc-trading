"""Orders: the three human reviews of the day, then the export for the Belfius form.

1. Orders review: approve, modify or reject each proposed order and add your own. A reason is
   required to change an order. Checks: at most 5 orders, enough cash, enough units to sell.
2. Risk review: the Risk Manager's view of the portfolio once those orders are filled.
3. Summary review: the executive summary. Approving it writes the order log.
At every review you can also send the day back to the Portfolio Manager with your feedback,
or stop it: no trade today.
"""
import pandas as pd
import streamlit as st

from app import data, ui
from data import config as settings
from data.schemas import HOLDINGS, ORDERS, REVIEWS, RISK_REPORT
from data.universe import load_universe
from graph import pipeline

DECISION = {"Approve": "approved", "Modify": "modified", "Reject": "rejected"}
STATUS = {"ok": "✓ ok", "warning": "▲ warning", "breach": "✕ breach"}

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
universe = load_universe()
ui.page("Orders", "Three checks before anything is traded: the orders, the risk, then the executive summary.")


def belfius_export(log: pd.DataFrame) -> bytes:
    executed = log[log["decision"] != "rejected"].merge(universe[["ticker", "isin"]], on="ticker", how="left")
    form = pd.DataFrame({"Date": executed["date"], "Type": executed["side"], "Units": executed["units"].astype(int),
                         "Ticker": executed["ticker"], "ISIN": executed["isin"]})
    return form.to_csv(index=False).encode()


def compact(summary: str) -> str:
    """The executive summary without its title and with small headings, to fit in the page."""
    lines = summary.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join("#### " + line[3:] if line.startswith("## ") else line for line in lines)


def show_reviews() -> None:
    reviews = pd.DataFrame(state.get("reviews", []), columns=REVIEWS)
    if len(reviews):
        ui.label("Your reviews", "every answer of the day, with the feedback sent back")
        st.dataframe(reviews.drop(columns="date"), hide_index=True, width="stretch")


def review_actions(approve_label: str, extra: dict | None = None, blocked: bool = False) -> None:
    """Feedback box and the three answers. Feedback is required to send the day back or to stop it."""
    gate, round_, left = pending["gate"], pending["round"], pending["revisions_left"]
    feedback = st.text_area("Feedback", key=f"feedback_{gate}_{round_}", height=90,
                            placeholder="Required to send the day back or to stop it: what should change, and why")
    has_feedback = bool(feedback.strip())
    go, back, stop = st.columns([2, 2, 1.4])
    answer = None
    if go.button(approve_label, type="primary", disabled=blocked, key="submit", icon=":material/check:",
                 width="stretch"):
        answer = {"decision": "approved", "feedback": feedback, **(extra or {})}
    if back.button("Send back to the Portfolio Manager", disabled=not has_feedback or left <= 0, key="revise",
                   icon=":material/undo:", width="stretch",
                   help=f"{left} of {settings.MAX_REVISIONS} left today" if left > 0 else "No revision left today"):
        answer = {"decision": "revised", "feedback": feedback, **(extra or {})}
    if stop.button("No trade today", disabled=not has_feedback, key="stop", icon=":material/block:",
                   width="stretch"):
        answer = {"decision": "stopped", "feedback": feedback}
    if answer is not None:
        try:
            pipeline.resume_day(as_of, answer, logs)
        except ValueError as exc:
            st.error(str(exc), icon=":material/block:")
        else:
            st.rerun()


if status == "not started":
    ui.kpis([("Status", "Not started"), ("Orders", "—")])
    st.write("")
    st.page_link("views/home.py", label="Run the day first: it pauses here for your first review",
                 icon=":material/arrow_back:")
    st.stop()

if status == "done":
    log = data.order_log(logs)
    day_log = log[log["date"].astype(str) == as_of.isoformat()] if log is not None else pd.DataFrame()
    if state.get("outcome") == "stopped":
        last = state["reviews"][-1]
        ui.kpis([("Status", "No trade today", None, "down"), ("Stopped at", pipeline.GATE_NAMES[last["gate"]]),
                 ("Round", last["round"])])
        st.info(f"Your reason: {last['feedback']}", icon=":material/block:")
    else:
        executed = day_log[day_log["decision"] != "rejected"] if len(day_log) else day_log
        ui.kpis([("Status", "Approved", None, "up"), ("Orders to submit", len(executed)),
                 ("Changed by you", int((day_log["decision"] != "approved").sum()) if len(day_log) else 0),
                 ("Rounds", state.get("round", 1))])
        ui.label("Order log", "AI suggestion next to your decision")
        st.dataframe(day_log, hide_index=True, width="stretch")
        st.download_button("Export for the Belfius form", belfius_export(day_log), f"orders_{as_of}.csv",
                           "text/csv", icon=":material/download:", type="primary")
        with st.expander("Executive summary you approved"):
            st.markdown(compact(state.get("summary", "")))
    show_reviews()
    st.stop()

# --- Waiting for one of your reviews ----------------------------------------------------------
gate = pending["gate"]
ui.review_steps(pending, state)
if pending["round"] > 1:
    st.caption(f"Round {pending['round']}: the Portfolio Manager revised its proposal after your feedback · "
               f"{pending['revisions_left']} revision(s) left today")
st.write("")
for incident in pending["incidents"]:
    st.warning(f"{ui.STEP_LABELS[incident['step']]} failed and was replaced by its fallback: {incident['error']}",
               icon=":material/error:")

if gate == "risk":
    orders = pd.DataFrame(pending["approved_orders"], columns=ORDERS)
    report = pd.DataFrame(pending["risk_report"], columns=RISK_REPORT)
    counts = report["status"].value_counts()
    breaches, warnings = int(counts.get("breach", 0)), int(counts.get("warning", 0))
    ui.kpis([("Review", "Risk", None, "hot"), ("Orders you approved", len(orders)),
             ("Breaches", breaches, None, "down" if breaches else None),
             ("Warnings", warnings, None, "hot" if warnings else None)])
    ui.label("Risk after the orders", "the Risk Manager's view of the portfolio once your orders are filled")
    if report.empty:
        st.info("No risk assessment today: the Risk Manager failed (see above).", icon=":material/info:")
    else:
        st.dataframe(report.drop(columns="date").assign(status=report["status"].map(STATUS)), hide_index=True,
                     width="stretch", column_config={
                         "metric": st.column_config.TextColumn("Metric"),
                         "value": st.column_config.NumberColumn("Value", format="%g"),
                         "limit": st.column_config.NumberColumn("Limit", format="%g"),
                         "status": st.column_config.TextColumn("Status"),
                         "comment": st.column_config.TextColumn("Comment"),
                     })
    ui.label("Orders you approved")
    st.dataframe(orders.drop(columns="date"), hide_index=True, width="stretch")
    ui.label("Your answer", "accept the risk, or send the day back with the limits you want")
    review_actions("Accept the risk")
    show_reviews()
    st.stop()

if gate == "summary":
    orders = pd.DataFrame(pending["approved_orders"], columns=ORDERS)
    ui.kpis([("Review", "Executive summary", None, "hot"), ("Orders to submit", len(orders)),
             ("Round", pending["round"])])
    ui.label("Executive summary", "what you are about to sign off")
    with st.container(border=True):
        st.markdown(compact(pending["summary"]))
    ui.label("Your answer", "approving writes the order log for the Belfius form")
    review_actions("Approve and write the order log")
    show_reviews()
    st.stop()

# --- Orders review --------------------------------------------------------------------------
proposed = pd.DataFrame(pending["proposed_orders"], columns=ORDERS)
prices, _ = data.day_prices(as_of, state)
close = data.last_close(prices, as_of)
holdings = pd.DataFrame(state.get("holdings", []), columns=HOLDINGS)
cash = float(state.get("cash_eur", settings.START_CAPITAL_EUR))
figures = st.empty()

ui.label("Investment memo", "the Portfolio Manager's reasoning")
with st.container(border=True):
    st.markdown(pending["memo"])

ui.label("Proposed orders", "change the decision, side or units · a reason is required to modify or reject")
editor = proposed.rename(columns={"side": "ai_side", "units": "ai_units"})[["ticker", "ai_side", "ai_units"]]
editor = editor.assign(decision="Approve", side=editor["ai_side"], units=editor["ai_units"], reason="")
edited = st.data_editor(
    editor, hide_index=True, width="stretch", key=f"decisions_{pending['round']}",
    disabled=["ticker", "ai_side", "ai_units"],
    column_config={
        "ticker": st.column_config.TextColumn("ETF"),
        "ai_side": st.column_config.TextColumn("AI side"),
        "ai_units": st.column_config.NumberColumn("AI units"),
        "decision": st.column_config.SelectboxColumn("Decision", options=list(DECISION), required=True),
        "side": st.column_config.SelectboxColumn("Side", options=["BUY", "SELL"]),
        "units": st.column_config.NumberColumn("Units", min_value=1, step=1),
        "reason": st.column_config.TextColumn("Reason"),
    },
)
with st.expander("Add an order of your own"):
    added = st.data_editor(
        pd.DataFrame({"ticker": pd.Series(dtype=str), "side": pd.Series(dtype=str),
                      "units": pd.Series(dtype=float), "reason": pd.Series(dtype=str)}),
        num_rows="dynamic", hide_index=True, width="stretch", key=f"added_{pending['round']}",
        column_config={
            "ticker": st.column_config.SelectboxColumn("ETF", options=universe["ticker"].tolist()),
            "side": st.column_config.SelectboxColumn("Side", options=["BUY", "SELL"]),
            "units": st.column_config.NumberColumn("Units", min_value=1, step=1),
            "reason": st.column_config.TextColumn("Reason (required)"),
        },
    )

decisions = []
for row in edited.itertuples():
    decision = DECISION[row.decision]
    item = {"ticker": row.ticker, "decision": decision}
    if decision != "approved":
        item["reason"] = row.reason or ""
    if decision == "modified":
        item.update(side=row.side, units=row.units)
    decisions.append(item)
for row in added.dropna(subset=["ticker"]).itertuples():
    decisions.append({"ticker": row.ticker, "decision": "added", "side": row.side, "units": row.units,
                      "reason": row.reason if isinstance(row.reason, str) else ""})

problems, executed = [], proposed.iloc[0:0]
for item in decisions:
    if item["decision"] != "approved" and not str(item.get("reason") or "").strip():
        verb = {"modified": "modify", "rejected": "reject", "added": "add"}[item["decision"]]
        problems.append(f"{item['ticker']}: give a reason to {verb} this order")
try:
    executed = pipeline.executed_orders(pipeline.build_order_log(as_of, proposed, decisions))
    problems += pipeline.check_orders(executed, holdings, cash, prices, as_of)
except (ValueError, KeyError, TypeError) as exc:
    if not problems:  # missing reasons are reported above in plain words
        problems.append(str(exc).removeprefix("order_log: "))

cash_after = pipeline.cash_after(executed, cash, close)
change = cash_after - cash
with figures.container():
    ui.kpis([
        ("Review", "Orders", None, "hot"),
        ("Orders to submit", f"{len(executed)} / {settings.MAX_ORDERS_PER_DAY}"),
        ("Cash now", f"€{cash:,.0f}"),
        ("Cash after", f"€{cash_after:,.0f}", f"{'−' if change < 0 else '+'}€{abs(change):,.0f} at last close"),
        ("Checks", "OK" if not problems else f"{len(problems)} to fix", None, None if not problems else "down"),
    ])

ui.label("Your answer", "approve to send these orders to the Risk Manager")
for problem in problems:
    st.error(problem, icon=":material/block:")
review_actions("Approve the orders", {"orders": decisions}, blocked=bool(problems))
show_reviews()
