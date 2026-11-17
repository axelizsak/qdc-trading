"""Today: your three reviews of the day, one at a time, then the export for the Belfius form.

1. Orders: approve, change, reject or add orders (a reason is needed to change one). The trading rules are checked
   as you edit: at most 5 orders, enough cash, enough units to sell.
2. Risk: the Risk Manager's status per area and its warnings, for the portfolio once your orders are filled.
3. Summary: what you are about to sign off. Approving writes the order log and the logbook.
At every review you can send the day back to the Portfolio Manager with your feedback, or stop it: no trade today.
"""
import pandas as pd
import streamlit as st

from app import data, ui
from data import config as settings
from data.schemas import HOLDINGS, ORDERS, REVIEWS, RISK_REPORT
from data.universe import load_universe
from graph import pipeline

DECISION = {"Approve": "approved", "Modify": "modified", "Reject": "rejected"}
UNIT_STEP = float(10 ** -settings.ORDER_UNIT_DECIMALS)  # 0.0001 for fractional units, 1 for whole units
UNIT_FORMAT = f"%.{settings.ORDER_UNIT_DECIMALS}f"

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
universe = load_universe()
ui.page("Today", "Three checks before anything is traded: the orders, the risk, then the summary.")


def belfius_export(log: pd.DataFrame) -> bytes:
    executed = log[log["decision"] != "rejected"].merge(universe[["ticker", "isin"]], on="ticker", how="left")
    form = pd.DataFrame({"Date": executed["date"], "Type": executed["side"],
                         "Units": executed["units"].map(pipeline.fmt_units),  # fractional units, never rounded
                         "Ticker": executed["ticker"], "ISIN": executed["isin"]})
    return form.to_csv(index=False).encode()


def your_reviews() -> None:
    reviews = pd.DataFrame(state.get("reviews", []), columns=REVIEWS)
    if len(reviews):
        ui.label("Your answers today")
        st.dataframe(reviews.drop(columns="date"), hide_index=True, width="stretch", column_config={
            "round": "Round", "gate": "Review", "decision": "Your answer",
            "feedback": st.column_config.TextColumn("Feedback", width="large")})


def review_actions(approve_label: str, extra: dict | None = None, blocked: bool = False) -> None:
    """Feedback, then the three answers. Feedback is needed to send the day back or to stop it."""
    gate, round_, left = pending["gate"], pending["round"], pending["revisions_left"]
    ui.label("Your answer")
    feedback = st.text_area("Feedback", key=f"feedback_{gate}_{round_}", height=80, label_visibility="collapsed",
                            placeholder="Feedback: needed to send the day back or to stop it")
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
    if stop.button("No trade today", disabled=not has_feedback, key="stop", icon=":material/block:", width="stretch"):
        answer = {"decision": "stopped", "feedback": feedback}
    if answer is not None:
        try:
            pipeline.resume_day(as_of, answer, logs)
        except ValueError as exc:
            st.error(str(exc), icon=":material/block:")
        else:
            st.rerun()


def orders_table(orders: pd.DataFrame) -> None:
    prices, _ = data.day_prices(as_of, state)
    close = pipeline.fill_prices(prices, as_of)
    table = orders.drop(columns="date").assign(price=orders["ticker"].map(close))
    table["value"] = table["units"].astype(float) * table["price"]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "ticker": st.column_config.TextColumn("ETF"), "side": st.column_config.TextColumn("Side"),
        "units": st.column_config.NumberColumn("Units", format=UNIT_FORMAT),
        "price": st.column_config.NumberColumn("Fill price", format="€%.2f"),
        "value": st.column_config.NumberColumn("Value", format="€%.0f")})


def show_record(record: dict) -> None:
    """A finished day as the logs hold it: the outcome, the orders against the AI's proposal, your answers."""
    day_log, reviews = record["order_log"], record["reviews"]
    outcome = data.day_outcome(record)
    if outcome == "no trade today":
        last = reviews.iloc[-1]
        ui.kpis([("Today", "No trade", None, "down"), ("Stopped at", pipeline.GATE_NAMES[last["gate"]]),
                 ("Round", int(last["round"]))])
        st.info(f"The reason: {last['feedback']}", icon=":material/block:")
    else:
        executed = day_log[day_log["decision"] != "rejected"] if len(day_log) else day_log
        ui.kpis([("Today", "Approved", None, "up"), ("Orders to submit", len(executed)),
                 ("Changed by you", int((day_log["decision"] != "approved").sum()) if len(day_log) else 0),
                 ("Rounds", int(reviews["round"].max()) if len(reviews) else 1)])
        st.write("")
        left, right = st.columns([1, 1])
        left.download_button("Export for the Belfius form", belfius_export(day_log), f"orders_{as_of}.csv", "text/csv",
                             icon=":material/download:", type="primary", width="stretch")
        with right:
            ui.document_download("The summary that was signed", record["summary"], f"summary_{as_of}", deck=True, width="stretch")
        ui.label("Order log", "the AI's proposal next to the human decision")
        text = {"ai_side": "", "side": "", "reason": ""}  # empty, not "None", where the AI or the human left nothing
        st.dataframe(day_log.drop(columns="date").astype(dict.fromkeys(text, object)).fillna(text), hide_index=True,
                     width="stretch", column_config={
            "ticker": "ETF", "ai_side": "AI side", "ai_units": st.column_config.NumberColumn("AI units", format=UNIT_FORMAT),
            "decision": "Decision", "side": "Side", "units": st.column_config.NumberColumn("Units", format=UNIT_FORMAT),
            "reason": st.column_config.TextColumn("Reason", width="large")})
    if len(reviews):
        ui.label("The answers at each review")
        st.dataframe(reviews.drop(columns="date").astype({"feedback": object}).fillna({"feedback": ""}), hide_index=True,
                     width="stretch", column_config={
            "round": "Round", "gate": "Review", "decision": "Answer",
            "feedback": st.column_config.TextColumn("Feedback", width="large")})


record = data.day_record(as_of, logs)
if status == "done" or data.day_outcome(record) in ("approved", "no trade today"):  # finished: what the logs hold
    show_record(record)
    st.stop()

if status == "not started":
    st.info("The day has not run yet. Run it from the dashboard: it stops here for your first review.",
            icon=":material/info:")
    st.page_link("views/dashboard.py", label="Go to the dashboard", icon=":material/arrow_back:")
    st.stop()

# --- A review is waiting -----------------------------------------------------------------------
gate = pending["gate"]
ui.review_steps(pending, state)
if pending["round"] > 1:
    st.caption(f"Round {pending['round']}: the Portfolio Manager revised its proposal after your feedback · "
               f"{pending['revisions_left']} revision(s) left today")
for incident in pending["incidents"]:
    st.warning(ui.incident_text(incident), icon=":material/error:")

if gate == "risk":
    orders = pd.DataFrame(pending["approved_orders"], columns=ORDERS)
    report = pd.DataFrame(pending["risk_report"], columns=RISK_REPORT)
    counts = report["status"].value_counts()
    ui.kpis([("Orders you approved", len(orders)), ("Breaches", int(counts.get("breach", 0)), None,
                                                     "down" if counts.get("breach", 0) else None),
             ("Warnings", int(counts.get("warning", 0)), None, "wait" if counts.get("warning", 0) else None),
             ("Inside their levels", int(counts.get("ok", 0)))])
    areas = data.md_table(pending.get("risk_board") or "", "| Area")
    if report.empty:
        st.info("No risk assessment today: the Risk Manager failed (see above).", icon=":material/info:")
    elif len(areas):
        ui.label("Risk by area", "once your orders are filled")
        ui.chips([(f"{a.Area}: {a.Status}", a.Status.lower()) for a in areas.itertuples()])
        st.write("")
        st.dataframe(areas.rename(columns={"In one line": "Why"}), hide_index=True, width="stretch",
                     column_config={"Why": st.column_config.TextColumn("Why", width="large")})
    st.page_link("views/risk.py", label="Every metric and the independent reading", icon=":material/arrow_forward:")
    ui.label("Orders you approved")
    orders_table(orders)
    review_actions("Accept the risk")
    your_reviews()
    st.stop()

if gate == "summary":
    orders = pd.DataFrame(pending["approved_orders"], columns=ORDERS)
    cash = float(state.get("cash_eur", settings.START_CAPITAL_EUR))
    prices, _ = data.day_prices(as_of, state)
    after = pipeline.cash_after(orders, cash, pipeline.fill_prices(prices, as_of))
    report = pd.DataFrame(state.get("risk_report", []), columns=RISK_REPORT)
    changes = pipeline.order_notes(state.get("human_decisions", []))
    ui.kpis([("Orders to submit", len(orders)), ("Cash after", f"€{after:,.0f}"),
             ("Risk warnings", int((report["status"] != "ok").sum())),
             ("Your changes", len(changes.split("; ")) if changes else 0)])
    ui.label("What you sign off")
    orders_table(orders)
    if changes:
        st.caption("Your changes to the AI's proposal: " + changes)
    ui.document_download("Read the full summary", pending["summary"], f"summary_{as_of}", deck=True)
    review_actions("Approve and write the order log")
    your_reviews()
    st.stop()

# --- Orders review --------------------------------------------------------------------------
proposed = pd.DataFrame(pending["proposed_orders"], columns=ORDERS)
prices, _ = data.day_prices(as_of, state)
close = pipeline.fill_prices(prices, as_of)  # an order fills at the last close before the day
holdings = pd.DataFrame(state.get("holdings", []), columns=HOLDINGS)
cash = float(state.get("cash_eur", settings.START_CAPITAL_EUR))
figures = st.empty()

summary = data.memo_summary(pending.get("memo", ""))
if summary:
    st.info(summary, icon=":material/smart_toy:")
st.page_link("views/portfolio.py", label="The Portfolio Manager's target and full memo", icon=":material/arrow_forward:")

ui.label("Proposed orders", "change the decision, side or units · a reason is needed to modify or reject")
editor = proposed.rename(columns={"side": "ai_side", "units": "ai_units"})[["ticker", "ai_side", "ai_units"]]
editor["ai_units"] = editor["ai_units"].astype(float)  # the Portfolio Manager sizes orders in fractional units
editor = editor.assign(decision="Approve", side=editor["ai_side"], units=editor["ai_units"], reason="")
edited = st.data_editor(
    editor, hide_index=True, width="stretch", key=f"decisions_{pending['round']}",
    disabled=["ticker", "ai_side", "ai_units"],
    column_config={
        "ticker": st.column_config.TextColumn("ETF"),
        "ai_side": st.column_config.TextColumn("AI side"),
        "ai_units": st.column_config.NumberColumn("AI units", format=UNIT_FORMAT),
        "decision": st.column_config.SelectboxColumn("Decision", options=list(DECISION), required=True),
        "side": st.column_config.SelectboxColumn("Side", options=["BUY", "SELL"]),
        "units": st.column_config.NumberColumn("Units", min_value=UNIT_STEP, step=UNIT_STEP, format=UNIT_FORMAT),
        "reason": st.column_config.TextColumn("Reason"),
    },
)
added = pd.DataFrame({"ticker": pd.Series(dtype=str), "side": pd.Series(dtype=str),
                      "units": pd.Series(dtype=float), "reason": pd.Series(dtype=str)})
if st.toggle("Add an order of your own", key=f"add_{pending['round']}"):
    added = st.data_editor(
        added, num_rows="dynamic", hide_index=True, width="stretch", key=f"added_{pending['round']}",
        column_config={
            "ticker": st.column_config.SelectboxColumn("ETF", options=universe["ticker"].tolist()),
            "side": st.column_config.SelectboxColumn("Side", options=["BUY", "SELL"]),
            "units": st.column_config.NumberColumn("Units", min_value=UNIT_STEP, step=UNIT_STEP, format=UNIT_FORMAT),
            "reason": st.column_config.TextColumn("Reason (needed)"),
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
with figures.container():
    ui.kpis([
        ("Orders to submit", f"{len(executed)} / {settings.MAX_ORDERS_PER_DAY}"),
        ("Cash now", f"€{cash:,.0f}"),
        ("Cash after", f"€{cash_after:,.0f}", "at the fill prices"),
        ("Checks", "OK" if not problems else f"{len(problems)} to fix", None, None if not problems else "down"),
    ])
for problem in problems:
    st.error(problem, icon=":material/block:")
review_actions("Approve the orders", {"orders": decisions}, blocked=bool(problems))
your_reviews()
