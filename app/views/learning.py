"""Learning: how every weight of the system moved, and what the backtest since 2022 says."""
import pandas as pd
import streamlit as st

from app import charts, data, ui
from data import config as settings

logs = ui.pipeline_logs()
ui.page("Learning", "One rule for every weight: after each session, a weight × e^"
                    f"({settings.LEARNING_SPEED:g} × what following it earned), kept between "
                    f"{settings.WEIGHT_MIN:.0%} and {settings.WEIGHT_MAX:.0%}.")

summary = data.backtest_summary()
if summary is not None:
    rows = summary.set_index("variant")
    first = rows.iloc[0]
    other = rows.loc["learned_three"] if "learned_three" in rows.index else None
    ui.kpis([("Sharpe · the system", f"{first['sharpe']:.2f}",
              f"trend rule · worst fall {first['max_drawdown']:.0%}"),
             ("Sharpe · 3 answers learned", f"{other['sharpe']:.2f}" if other is not None else "—",
              "16 to 18 Nov · worst fall " + (f"{other['max_drawdown']:.0%}" if other is not None else "—")),
             (f"Sharpe · {data.BENCHMARK_LABEL}", f"{first['bench_sharpe']:.2f}", f"worst fall {first['bench_max_drawdown']:.0%}"),
             ("Years tested", f"{first['years']:.1f}", "daily, since January 2022")])

learned, learned_mock = data.weights(logs)
learned = learned.assign(date=pd.to_datetime(learned["date"].astype(str)))
ui.mock_note(learned_mock, "the learning step has not written logs/weights.csv yet")
LEVELS = [("technical", "The technical agent's answers", "the trend rule alone since 19 Nov; one point a week"),
          ("agents", "The four agents", "since their first score"),
          ("portfolio", "The LLM's portfolio against the code's", "in the Portfolio Manager's blend")]
for level, title, aside in LEVELS:
    rows = learned[learned["level"] == level]
    if level == "technical":
        rows = rows.groupby([pd.Grouper(key="date", freq="W"), "name"])["weight"].last().reset_index()
    ui.label(title, aside)
    if rows["name"].nunique() < 2:  # one answer counts: nothing to learn between answers
        ui.chips([(f"{n.replace('_', ' ')} 100%", "none") for n in rows["name"].unique()] or [("none yet", "none")])
        st.caption("Team choice after the backtest: the trend rule alone (as good, with a smaller worst fall, and "
                   "simpler). Mean reversion and momentum are still recorded every session, for the LLM and to check "
                   "the choice again.")
    elif rows["date"].nunique() < 2:  # one set so far: the weights as numbers, the curve needs two
        now = rows[rows["date"] == rows["date"].max()].sort_values("weight", ascending=False)
        ui.chips([(f"{r.name.replace('_', ' ')} {r.weight:.0%}", "none") for r in now.itertuples()] or [("none yet", "none")])
    else:
        st.altair_chart(charts.lines(rows.assign(name=rows["name"].str.replace("_", " ")), "date", "weight", "name",
                                     "weight", ".0%", zero_based=True, x_format="%b %Y" if level == "technical" else "%d %b",
                                     height=230), width="stretch")

ui.label("Backtest", "the code-only system replayed day by day since 2022, the learning included")
curves = data.backtest_curves()
if curves is None or summary is None:
    st.caption("Run it once: python -m backtest.system_backtest --daily")
    st.stop()
st.altair_chart(charts.lines(curves, "date", "value_eur", "series", "€", ",.0f", x_format="%b %Y", height=320),
                width="stretch")
table = summary[["variant", "what", "sharpe", "sharpe_h1", "sharpe_h2", "max_drawdown", "sharpe_vs_first",
                 "t_sharpe_vs_first", "holds"]].sort_values("sharpe", ascending=False)
table = table.assign(max_drawdown=table["max_drawdown"] * 100)
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "variant": st.column_config.TextColumn("Variant"), "what": st.column_config.TextColumn("What it changes", width="large"),
    "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
    "sharpe_h1": st.column_config.NumberColumn("1st half", format="%.2f"),
    "sharpe_h2": st.column_config.NumberColumn("2nd half", format="%.2f"),
    "max_drawdown": st.column_config.NumberColumn("Worst fall", format="%.0f%%"),
    "sharpe_vs_first": st.column_config.NumberColumn("vs the system", format="%+.2f"),
    "t_sharpe_vs_first": st.column_config.NumberColumn("t", format="%+.1f"),
    "holds": st.column_config.CheckboxColumn("Ahead in both halves")})
st.caption("A t-value under 2 means the gap is not distinguishable from luck. The LLM scores cannot be replayed on the "
           "past (the LLM already knows it): they learn live, from their first day.")
