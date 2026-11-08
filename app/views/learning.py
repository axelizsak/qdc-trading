"""Learning: how the system re-weights its agents and sources after each trading day."""
import pandas as pd
import streamlit as st

from app import charts, data, ui

logs = ui.macro_logs()
weights, weights_mock = data.agent_weights(logs)
weights = weights.assign(date=pd.to_datetime(weights["date"].astype(str)))
registry, registry_mock = data.source_registry(logs)
graded_sources = registry[registry["n_uses"] > 0]
graded_agents = weights.dropna(subset=["hit_rate"])
latest = graded_agents[graded_agents["date"] == graded_agents["date"].max()] if len(graded_agents) else graded_agents

ui.page("Learning", "How the system re-weights its agents and sources after each trading day.")
best = latest.loc[latest["hit_rate"].idxmax()] if len(latest) else None
ui.kpis([
    ("Reviews", weights["date"].nunique()),
    ("Best agent", best["agent"] if best is not None else "—",
     f"hit rate {best['hit_rate']:.0%}" if best is not None else None),
    ("Graded sources", len(graded_sources), f"of {len(registry)}"),
    ("Dropped sources", int((registry["status"] == "dropped").sum())),
])

ui.label("Agent weights", "share of each agent in the combined score")
ui.mock_note(weights_mock, "the review loop has not written logs/agent_weights.csv yet")
st.altair_chart(charts.lines(weights, "date", "weight", "agent", "weight", ".0%", zero_based=True), width="stretch")

agents_col, sources_col = st.columns(2, gap="large")
with agents_col:
    ui.label("Agent hit rates", "latest review · line = 50%")
    if latest.empty:
        st.caption("Appears after the first review.")
    else:
        st.altair_chart(charts.hbars(latest, "hit_rate", "agent", ".0%", tooltip=["agent", "weight", "hit_rate"],
                                     rule_at=0.5), width="stretch")
with sources_col:
    ui.label("Source hit rates", "line = 50%")
    ui.mock_note(registry_mock, "no source registry yet")
    if graded_sources.empty:
        st.caption("Appears once the review loop has scored a trading day.")
    else:
        st.altair_chart(charts.hbars(graded_sources, "hit_rate", "source", ".0%",
                                     tooltip=["source", "status", "n_uses", "hit_rate"], rule_at=0.5),
                        width="stretch")

ui.label("Lessons journal", "what the agents read before each decision")
journal = data.lessons(logs)
if journal:
    with st.container(border=True):
        st.markdown(journal)
else:
    st.caption("Empty: the review loop writes a lesson after each trading day.")
