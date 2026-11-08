"""Day: where the trading day stands and what to do next."""
import streamlit as st

from app import data, ui
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
report = data.macro_report(as_of, ui.macro_logs(), logs)
incidents = state.get("incidents", [])
proposed = pending["proposed_orders"] if pending else state.get("proposed_orders", [])

ui.page("Day", "Where the trading day stands, and what to do next.")
ui.kpis([
    ("Status", status.capitalize(), None, "hot" if status == "waiting for approval" else None),
    ("Orders proposed", len(proposed) if status != "not started" else "—"),
    ("Incidents", len(incidents), "every step ran" if status != "not started" and not incidents else None,
     "down" if incidents else None),
    ("Macro run cost", data.macro_cost(report) or "—", "Claude API" if data.macro_cost(report) else None),
])

ui.label("Pipeline")
ui.pipeline_strip(status, incidents)
for incident in incidents:
    st.error(f"{ui.STEP_LABELS[incident['step']]} failed and was replaced by its fallback: {incident['error']}")

ui.label("Next step")
if status == "not started":
    if st.button("Run the day", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Agents running..."):
            pipeline.run_day(as_of, logs)
        st.rerun()
    st.caption("Runs every agent, then pauses for your approval on the Orders page.")
else:
    go, again, _ = st.columns([2, 1, 3])
    if status == "waiting for approval":
        go.page_link("views/orders.py", label=f"Review the {len(proposed)} proposed orders",
                     icon=":material/arrow_forward:")
    else:
        go.page_link("views/orders.py", label="Approved · export for the Belfius form", icon=":material/task_alt:")
    if again.button("Run again", icon=":material/replay:",
                    help="Starts the day again. Signals already logged are read back: no new API calls"):
        with st.spinner("Running again from the logs..."):
            pipeline.run_day(as_of, logs, rerun=True)
        st.rerun()

registry, registry_mock = data.source_registry(ui.macro_logs())
counts = registry["status"].value_counts()
ui.label("Macro sources", f"{len(registry)} sources · {counts.get('active', 0)} active · "
                          f"{counts.get('testing', 0)} on trial · {counts.get('dropped', 0)} dropped")
ui.mock_note(registry_mock, "no source registry yet")
if report and "## Failed collectors and searches" in report:
    st.warning("Some collectors or searches failed in this day's macro run:\n"
               + report.split("## Failed collectors and searches")[1].strip())
with st.expander("Source registry"):
    st.dataframe(registry[["source", "kind", "status", "n_uses", "hit_rate", "factors", "first_seen"]],
                 hide_index=True, width="stretch",
                 column_config={"n_uses": st.column_config.NumberColumn("graded uses"),
                                "hit_rate": st.column_config.ProgressColumn("hit rate", min_value=0, max_value=1,
                                                                            format="percent")})
if report:
    with st.expander("Macro/News report of the day"):
        st.markdown(report)
