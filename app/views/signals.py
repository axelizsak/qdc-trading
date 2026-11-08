"""Signals: which ETFs the agents want to buy or sell."""
import pandas as pd
import streamlit as st

from agents.portfolio import combine_scores
from app import charts, data, ui
from data.config import SIGNAL_AGENTS
from data.universe import load_universe
from graph import pipeline

TOP = 8
as_of, logs = ui.as_of(), ui.pipeline_logs()
_, state, _ = pipeline.day_status(as_of, logs)
signals, is_mock = data.day_signals(as_of, logs, state)
universe = load_universe()
weights = state.get("agent_weights") or pipeline.load_agent_weights(logs, as_of)
combined = combine_scores(signals, weights).rename("score").reset_index().merge(universe[["ticker", "theme"]])

ui.page("Signals", "Which ETFs the agents want to buy or sell, once their votes are weighted.")
ui.mock_note(is_mock, "this day has not run yet")
themes = st.multiselect("Themes", universe["theme"].unique().tolist(), placeholder="All themes",
                        label_visibility="collapsed")
shown = combined if not themes else combined[combined["theme"].isin(themes)]

strongest = shown.loc[shown["score"].abs().idxmax()] if len(shown) else None
ui.kpis([
    ("ETFs", len(shown)),
    ("Buy side", int((shown["score"] > 0.2).sum()), "score above +0.2"),
    ("Sell side", int((shown["score"] < -0.2).sum()), "score below -0.2"),
    ("Strongest", strongest["ticker"] if strongest is not None else "—",
     f"{strongest['score']:+.2f}" if strongest is not None else None,
     ("up" if strongest["score"] > 0 else "down") if strongest is not None else None),
])
st.caption("Weights: " + " · ".join(f"{a} {w:.0%}" for a, w in weights.items()))

buy, sell = st.columns(2, gap="large")
with buy:
    ui.label("Strongest buy")
    top = shown[shown["score"] > 0].nlargest(TOP, "score")
    if top.empty:
        st.caption("No buy signal.")
    else:
        st.altair_chart(charts.ranked(top, "score", "ticker", ui.T["positive"], [0, 1], "+.2f",
                                      ["ticker", "theme", "score"]), width="stretch")
with sell:
    ui.label("Strongest sell")
    bottom = shown[shown["score"] < 0].nsmallest(TOP, "score")
    if bottom.empty:
        st.caption("No sell signal.")
    else:
        st.altair_chart(charts.ranked(bottom, "score", "ticker", ui.T["negative"], [-1, 0], "+.2f",
                                      ["ticker", "theme", "score"]), width="stretch")

ui.label("All ETFs", "each agent's score, then the weighted combination")
with st.expander("Heat map"):
    agents = [a for a in SIGNAL_AGENTS if a in set(signals["agent"])]
    grid = signals.merge(universe[["ticker", "theme"]], on="ticker").rename(columns={"agent": "source"})
    combo = shown.assign(source="combined", confidence=None, rationale="weighted sum of the agents")
    grid = pd.concat([grid, combo], ignore_index=True)
    grid = grid[grid["ticker"].isin(shown["ticker"])]
    grid["order"] = grid["ticker"].map(shown.set_index("ticker")["score"])
    st.altair_chart(charts.heatmap(grid.sort_values("order", ascending=False), agents + ["combined"]),
                    width="stretch")
with st.expander("Table"):
    wide = grid.pivot_table(index=["ticker", "theme"], columns="source", values="score").reset_index()
    st.dataframe(wide.sort_values("combined", ascending=False), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.2f") for c in agents + ["combined"]})
