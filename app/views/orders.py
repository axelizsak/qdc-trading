"""Orders: approve, modify or reject each proposed order, add your own, then resume the paused day.

A reason is required to modify, reject or add an order. Checks: at most 5 executed orders, enough
cash and enough units to sell. Approval writes the order log; the export matches the Belfius form.
"""
import pandas as pd
import streamlit as st

from app import data, ui
from data import config as settings
from data.schemas import ORDERS
from data.universe import load_universe
from graph import pipeline

DECISION = {"Approve": "approved", "Modify": "modified", "Reject": "rejected"}

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
universe = load_universe()
ui.page("Orders", "Approve, change or reject what the agents propose. Nothing is traded without you.")


def belfius_export(log: pd.DataFrame) -> bytes:
    executed = log[log["decision"] != "rejected"].merge(universe[["ticker", "isin"]], on="ticker", how="left")
    form = pd.DataFrame({"Date": executed["date"], "Type": executed["side"], "Units": executed["units"].astype(int),
                         "Ticker": executed["ticker"], "ISIN": executed["isin"]})
    return form.to_csv(index=False).encode()


if status == "not started":
    ui.kpis([("Status", "Not started"), ("Orders", "—")])
    st.write("")
    st.page_link("views/home.py", label="Run the day first: it pauses here for your approval",
                 icon=":material/arrow_back:")
    st.stop()

if status == "done":
    log = data.order_log(logs)
    day_log = log[log["date"].astype(str) == as_of.isoformat()] if log is not None else pd.DataFrame()
    executed = day_log[day_log["decision"] != "rejected"] if len(day_log) else day_log
    ui.kpis([("Status", "Approved", None, "up"), ("Orders to submit", len(executed)),
             ("Changed by you", int((day_log["decision"] != "approved").sum()) if len(day_log) else 0)])
    ui.label("Order log", "AI suggestion next to your decision")
    st.dataframe(day_log, hide_index=True, width="stretch")
    st.download_button("Export for the Belfius form", belfius_export(day_log), f"orders_{as_of}.csv", "text/csv",
                       icon=":material/download:", type="primary")
    st.stop()

# --- Waiting for approval ---------------------------------------------------------------------
proposed = pd.DataFrame(pending["proposed_orders"], columns=ORDERS)
prices, _ = data.day_prices(as_of, state)
close = data.last_close(prices, as_of)
held = {h["ticker"]: float(h["units"]) for h in state.get("holdings", [])}
cash = float(state.get("cash_eur", settings.START_CAPITAL_EUR))
figures = st.empty()

for incident in pending["incidents"]:
    st.warning(f"{incident['step']} failed and was replaced by its fallback: {incident['error']}",
               icon=":material/error:")

ui.label("Investment memo", "the Portfolio Manager's reasoning")
with st.container(border=True):
    st.markdown(pending["memo"])

ui.label("Proposed orders", "change the decision, side or units · a reason is required to modify or reject")
editor = proposed.rename(columns={"side": "ai_side", "units": "ai_units"})[["ticker", "ai_side", "ai_units"]]
editor = editor.assign(decision="Approve", side=editor["ai_side"], units=editor["ai_units"], reason="")
edited = st.data_editor(
    editor, hide_index=True, width="stretch", key="decisions", disabled=["ticker", "ai_side", "ai_units"],
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
        num_rows="dynamic", hide_index=True, width="stretch", key="added",
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

problems, log = [], None
for item in decisions:
    if item["decision"] != "approved" and not str(item.get("reason") or "").strip():
        verb = {"modified": "modify", "rejected": "reject", "added": "add"}[item["decision"]]
        problems.append(f"{item['ticker']}: give a reason to {verb} this order")
try:
    log = pipeline.build_order_log(as_of, proposed, decisions)
except (ValueError, KeyError, TypeError) as exc:
    if not problems:  # reasons are reported above in plain words
        problems.append(str(exc).removeprefix("order_log: "))

executed_count, cash_after = 0, cash
if log is not None:
    executed = log[log["decision"] != "rejected"]
    executed_count = len(executed)
    units = executed["units"].astype(float)
    px = executed["ticker"].map(close)
    cash_after = cash - float((units * px).where(executed["side"] == "BUY", -(units * px)).sum())
    for row in executed[executed["side"] == "SELL"].itertuples():
        if float(row.units) > held.get(row.ticker, 0):
            problems.append(f"Selling {row.units:g} {row.ticker}, but only {held.get(row.ticker, 0):g} held")
    if px.isna().any():
        problems.append("No price for " + ", ".join(executed.loc[px.isna(), "ticker"]))
    if cash_after < 0:
        problems.append(f"Not enough cash: the orders need €{-cash_after:,.0f} more than available")

change = cash_after - cash
with figures.container():
    ui.kpis([
        ("Status", "Waiting for you", None, "hot"),
        ("Orders to submit", f"{executed_count} / {settings.MAX_ORDERS_PER_DAY}"),
        ("Cash now", f"€{cash:,.0f}"),
        ("Cash after", f"€{cash_after:,.0f}", f"{'−' if change < 0 else '+'}€{abs(change):,.0f} at last close"),
        ("Checks", "OK" if not problems else f"{len(problems)} to fix", None, None if not problems else "down"),
    ])

ui.label("Approve")
for problem in problems:
    st.error(problem, icon=":material/block:")
if st.button("Approve and write the order log", type="primary", disabled=bool(problems), key="submit",
             icon=":material/check:"):
    pipeline.resume_day(as_of, decisions, logs)
    st.rerun()
