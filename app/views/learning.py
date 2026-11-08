"""Learning: agent weights over time, hit rates of agents and sources, lessons journal."""
import pandas as pd
import streamlit as st

from app import charts, data, ui

st.title("Learning")
logs = ui.macro_logs()

st.subheader("Agent weights")
weights, weights_mock = data.agent_weights(logs)
ui.mock_note(weights_mock, "The review loop has not written logs/agent_weights.csv yet")
weights = weights.assign(date=pd.to_datetime(weights["date"].astype(str)))
st.altair_chart(charts.lines(weights, "date", "weight", "agent", ui.tokens(), "Weight", ".0%", zero_based=True),
                width="stretch")

st.subheader("Agent hit rates")
graded = weights.dropna(subset=["hit_rate"])
if graded.empty:
    st.caption("No hit rate yet: it appears after the first review.")
else:
    latest = graded[graded["date"] == graded["date"].max()]
    st.altair_chart(charts.hbars(latest, "hit_rate", "agent", ui.tokens(), ".0%",
                                 tooltip=["agent", "weight", "hit_rate"], rule_at=0.5), width="stretch")
with st.expander("Table view"):
    st.dataframe(weights, hide_index=True, width="stretch")

st.subheader("Source hit rates")
registry, registry_mock = data.source_registry(logs)
ui.mock_note(registry_mock, "No source registry yet")
st.dataframe(registry.sort_values("hit_rate", ascending=False), hide_index=True, width="stretch",
             column_config={"hit_rate": st.column_config.ProgressColumn("Hit rate", min_value=0, max_value=1,
                                                                        format="percent")})

st.subheader("Lessons journal")
journal = data.lessons(logs)
if journal:
    st.markdown(journal)
else:
    st.caption("logs/lessons.md is empty. The review loop writes a lesson after each trading day.")
