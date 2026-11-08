"""Signals: heat map ETF x source, filtered by theme."""
import pandas as pd
import streamlit as st

from agents.portfolio import combine_scores
from app import charts, data, ui
from data.config import SIGNAL_AGENTS
from data.universe import load_universe
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
st.title("Signals")
_, state, _ = pipeline.day_status(as_of, logs)
signals, is_mock = data.day_signals(as_of, logs, state)
ui.mock_note(is_mock, "This day has not run yet")

universe = load_universe()
weights = state.get("agent_weights") or pipeline.load_agent_weights(logs, as_of)
combined = combine_scores(signals, weights)

themes = st.multiselect("Themes", universe["theme"].unique().tolist(), placeholder="All themes")
shown = universe if not themes else universe[universe["theme"].isin(themes)]

agents = [a for a in SIGNAL_AGENTS if a in set(signals["agent"])]
grid = signals.merge(universe[["ticker", "theme"]], on="ticker").rename(columns={"agent": "source"})
combined_rows = pd.DataFrame({"ticker": combined.index, "source": "combined", "score": combined.values,
                              "confidence": None, "rationale": "Σ weight x score"}).merge(
    universe[["ticker", "theme"]], on="ticker")
grid = pd.concat([grid, combined_rows], ignore_index=True)
grid = grid[grid["ticker"].isin(shown["ticker"])]
grid["order"] = grid["ticker"].map(combined)
grid = grid.sort_values("order", ascending=False)

st.caption("Blue = buy signal, red = sell, gray = neutral. The last column combines the agents with their "
           "current weights: " + ", ".join(f"{a} {w:.0%}" for a, w in weights.items()) + ".")
st.altair_chart(charts.heatmap(grid, agents + ["combined"], ui.tokens()), width="stretch")

with st.expander("Table view"):
    wide = grid.pivot_table(index=["ticker", "theme"], columns="source", values="score").reset_index()
    wide = wide.sort_values("combined", ascending=False)
    st.dataframe(wide, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.2f") for c in agents + ["combined"]})
