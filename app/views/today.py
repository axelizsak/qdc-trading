"""Today: run or reload the trading day, status of each agent and each source, cost."""
import pandas as pd
import streamlit as st

from app import data, ui
from data import config as settings
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
st.title("Today")
st.caption(f"Trading day {as_of:%A %d %B %Y}" + ("" if as_of in settings.TRADING_DAYS else " (not a competition day)"))

status, state, pending = pipeline.day_status(as_of, logs)
report = data.macro_report(as_of, ui.macro_logs(), logs)
incidents = state.get("incidents", [])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Day", status.capitalize())
c2.metric("Orders proposed", len(pending["proposed_orders"]) if pending else len(state.get("proposed_orders", [])))
c3.metric("Incidents", len(incidents))
c4.metric("Macro run cost", data.macro_cost(report) or "n/a", help="From the Macro/News report of this day")

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("Run the day", type="primary", disabled=status != "not started", icon=":material/play_arrow:",
             help="Runs every agent, then pauses for your approval on the Orders page"):
    with st.spinner("Agents running..."):
        pipeline.run_day(as_of, logs)
    st.rerun()
if b2.button("Run again", disabled=status == "not started", icon=":material/replay:",
             help="Starts the day again. Signals already logged are read back, so the agents are not called again"):
    with st.spinner("Running again from the logs..."):
        pipeline.run_day(as_of, logs, rerun=True)
    st.rerun()
if status == "waiting for approval":
    st.page_link("views/orders.py", label="Review the proposed orders", icon=":material/fact_check:")

st.subheader("Agents")
failed = {i["step"]: i["error"] for i in incidents}
for step in ui.STEPS:
    name, mode, state_col, detail = st.columns([2, 1, 1, 6])
    name.markdown(f"**{step}**")
    mode.markdown(":gray-badge[mock]" if settings.USE_MOCK[step] else ":blue-badge[real]")
    if status == "not started":
        state_col.markdown(":gray-badge[not run]")
    elif step in failed:
        state_col.markdown(":red-badge[:material/error: failed]")
        detail.caption(failed[step])
    else:
        state_col.markdown(":green-badge[:material/check: ok]")

st.subheader("Macro sources")
registry, registry_mock = data.source_registry(ui.macro_logs())
evidence, _ = data.evidence(ui.macro_logs())
ui.mock_note(registry_mock, "No source registry yet")
today_counts = evidence[evidence["date"].astype(str) == as_of.isoformat()].groupby("source").size()
table = registry.assign(items_today=registry["source"].map(today_counts).fillna(0).astype(int))
st.dataframe(
    table[["source", "kind", "status", "items_today", "n_uses", "hit_rate", "factors", "first_seen"]],
    hide_index=True, width="stretch",
    column_config={
        "status": st.column_config.TextColumn("Status", help="active, testing (on trial) or dropped"),
        "items_today": st.column_config.NumberColumn("Items today"),
        "n_uses": st.column_config.NumberColumn("Graded uses"),
        "hit_rate": st.column_config.ProgressColumn("Hit rate", min_value=0, max_value=1, format="percent"),
    },
)
if report:
    failures = report.split("## Failed collectors and searches")
    if len(failures) > 1:
        st.warning("Failed collectors and searches in this day's macro run:\n" + failures[1].strip())
    with st.expander("Macro/News report of the day"):
        st.markdown(report)
else:
    st.caption("No Macro/News report for this day. Run `python -m agents.macro_news --date "
               f"{as_of.isoformat()}`.")
