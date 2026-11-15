"""Research: what the agents say (signals), why (macro views), and how far to trust them (track record)."""
from datetime import date

import pandas as pd
import streamlit as st

from agents import macro_news
from agents.portfolio import combine_scores
from app import charts, data, ui
from data.config import MACRO_FACTOR_MEANING, SIGNAL_AGENTS
from data.schemas import split_list
from data.universe import load_universe
from graph import pipeline

TOP = 8
as_of, logs, macro_logs = ui.as_of(), ui.pipeline_logs(), ui.macro_logs()
ui.page("Research", "What the agents want to trade, why, and how far to trust them.")

# --- Signals ---------------------------------------------------------------------------------
_, state, _ = pipeline.day_status(as_of, logs)
signals, signals_mock = data.day_signals(as_of, logs, state)
universe = load_universe()
# what the review loop learned so far, over the agents that gave a score today (else the starting weights)
weights = pipeline.load_agent_weights(logs, as_of, pipeline.scoring_agents(signals))
if not weights:  # a day logged before the agents had their current names
    weights = {agent: 1 / signals["agent"].nunique() for agent in sorted(signals["agent"].unique())}
combined = combine_scores(signals, weights).rename("score").reset_index().merge(universe[["ticker", "theme"]])

ui.label("Signals", "agents' votes, weighted by the review loop: "
                    + " · ".join(f"{a} {w:.0%}" for a, w in weights.items()))
ui.mock_note(signals_mock, "this day has not run yet")
themes = st.multiselect("Themes", universe["theme"].unique().tolist(), placeholder="All themes",
                        label_visibility="collapsed")
shown = combined if not themes else combined[combined["theme"].isin(themes)]
buy, sell = st.columns(2, gap="large")
with buy:
    st.caption("Strongest buy")
    top = shown[shown["score"] > 0].nlargest(TOP, "score")
    if top.empty:
        st.caption("No buy signal.")
    else:
        st.altair_chart(charts.ranked(top, "score", "ticker", ui.T["positive"], [0, 1], "+.2f",
                                      ["ticker", "theme", "score"]), width="stretch")
with sell:
    st.caption("Strongest sell")
    bottom = shown[shown["score"] < 0].nsmallest(TOP, "score")
    if bottom.empty:
        st.caption("No sell signal.")
    else:
        st.altair_chart(charts.ranked(bottom, "score", "ticker", ui.T["negative"], [-1, 0], "+.2f",
                                      ["ticker", "theme", "score"]), width="stretch")
with st.expander("All 50 ETFs, agent by agent"):
    agents = [a for a in SIGNAL_AGENTS if a in set(signals["agent"])]
    grid = signals.merge(universe[["ticker", "theme"]], on="ticker").rename(columns={"agent": "source"})
    grid = pd.concat([grid, shown.assign(source="combined", confidence=None, rationale="weighted sum of the agents")],
                     ignore_index=True)
    grid = grid[grid["ticker"].isin(shown["ticker"])]
    grid["order"] = grid["ticker"].map(shown.set_index("ticker")["score"])
    st.altair_chart(charts.heatmap(grid.sort_values("order", ascending=False), agents + ["combined"]),
                    width="stretch")

# --- Macro views -----------------------------------------------------------------------------
views, views_mock = data.macro_views(macro_logs)
evidence, _ = data.evidence(macro_logs)
dates = sorted(views["date"].astype(str).unique(), reverse=True)
earlier = [d for d in dates if d <= as_of.isoformat()]
day = earlier[0] if earlier else dates[0]
day_views = views[views["date"].astype(str) == day].copy()
day_views["name"] = day_views["factor"].map(ui.FACTOR_LABELS)
day_views["meaning"] = day_views["factor"].map(MACRO_FACTOR_MEANING)
day_evidence = evidence[evidence["date"].astype(str) == day]
report = data.macro_report(date.fromisoformat(day), macro_logs)
cost = data.macro_cost(report)

ui.label("Macro views", f"run of {date.fromisoformat(day):%a %d %b} · {len(day_evidence)} facts from "
                        f"{day_evidence['source'].nunique()} sources" + (f" · cost {cost}" if cost else ""))
ui.mock_note(views_mock, "no macro run yet")
st.caption("Most bullish at the top · blue = up, red = down · faded = low confidence")
st.altair_chart(charts.factor_bars(day_views), width="stretch")

ranked = day_views.assign(weight=day_views["direction"].abs() * day_views["confidence"]).sort_values(
    "weight", ascending=False)
factor = st.selectbox("Why", ranked["factor"].tolist(), format_func=lambda f: f"Why · {ui.FACTOR_LABELS[f]}",
                      label_visibility="collapsed")
view = ranked.set_index("factor").loc[factor]


def source_links(sources: str) -> str:
    lines = []
    for source in split_list(sources):
        rows = day_evidence[(day_evidence["source"] == source) & (day_evidence["factor"] == factor)]
        rows = rows if len(rows) else day_evidence[day_evidence["source"] == source]
        urls = [u for u in dict.fromkeys(rows["url"]) if isinstance(u, str) and u.startswith("http")][:3]
        lines.append(f"- `{source}` " + " · ".join(f"[{i + 1}]({u})" for i, u in enumerate(urls)))
    return "\n".join(lines)


with st.container(border=True):
    tone = "blue" if view["direction"] >= 0 else "red"
    st.markdown(f"#### {ui.FACTOR_LABELS[factor]} :{tone}[{view['direction']:+.2f}] "
                f"<span style='color:{ui.T['muted']};font-size:.8em'>confidence {view['confidence']:.0%}</span>",
                unsafe_allow_html=True)
    st.caption(f"+1 means {MACRO_FACTOR_MEANING[factor]}")
    st.markdown(view["rationale"])
    st.markdown(f"**Counter-argument** · {view['counter_argument']}")
    st.markdown("**Sources**\n" + source_links(view["sources"]))

etf_views, etf_mock = data.etf_views(macro_logs)
day_etf = etf_views[etf_views["date"].astype(str) == day]
ui.label("ETF analysts", "one hedge-fund-style analyst per theme · their calls are the macro_etf score")
ui.mock_note(etf_mock, "no analyst run yet")
if day_etf.empty:
    st.caption("No analyst views in this run.")
else:
    ahead = macro_news.catalysts(day_etf)
    if ahead:  # the dated events the analysts found in their searches, soonest first
        st.caption("Events ahead · " + " · ".join(
            f"**{date.fromisoformat(d):%a %d %b}** {event} ({len(tickers)} ETF{'s' if len(tickers) > 1 else ''})"
            for d, event, tickers in ahead[:6]))
    theme_of = universe.set_index("ticker")["theme"]
    ranked_etf = day_etf.assign(weight=day_etf["direction"].abs() * day_etf["confidence"]).sort_values(
        "weight", ascending=False).set_index("ticker")
    pick = st.selectbox("ETF analyst", ranked_etf.index.tolist(), label_visibility="collapsed",
                        format_func=lambda t: f"Analyst · {t} {ranked_etf.loc[t, 'direction']:+.2f}")
    etf_view = ranked_etf.loc[pick]
    with st.container(border=True):
        tone = "blue" if etf_view["direction"] >= 0 else "red"
        st.markdown(f"#### {pick} :{tone}[{etf_view['direction']:+.2f}] "
                    f"<span style='color:{ui.T['muted']};font-size:.8em'>confidence {etf_view['confidence']:.0%}</span>",
                    unsafe_allow_html=True)
        st.caption(f"{theme_of.get(pick, '')} analyst · for leveraged and inverse ETFs, the view is on the underlying")
        st.markdown(etf_view["rationale"])
        if isinstance(etf_view["catalyst"], str) and etf_view["catalyst"].strip():
            st.markdown(f"**Next catalyst** · {date.fromisoformat(str(etf_view['catalyst_date'])[:10]):%a %d %b}: "
                        f"{etf_view['catalyst']}")
        st.markdown(f"**Counter-argument** · {etf_view['counter_argument']}")
        st.markdown("**Sources** · " + " · ".join(
            f"[{i + 1}]({u})" for i, u in enumerate(split_list(etf_view["sources"])[:6])))

found = day_evidence[day_evidence["kind"] == "exploration"]
with st.expander(f"New sources on trial · {len(found)} found by free exploration"):
    if found.empty:
        st.caption("None in this run.")
    for r in found.itertuples():
        what, _, why = r.summary.partition(" Why it matters: ")
        name = r.source.removeprefix("explore_").replace("_", " ").capitalize()
        st.markdown(f"**{name}** · {ui.FACTOR_LABELS.get(r.factor, r.factor)} · [source]({r.url})")
        st.caption(what + (f" Why it matters: {why}" if why else ""))
if report:
    with st.expander("Full macro report"):
        st.markdown(report)

# --- Track record ----------------------------------------------------------------------------
agent_weights, weights_mock = data.agent_weights(macro_logs)
agent_weights = agent_weights.assign(date=pd.to_datetime(agent_weights["date"].astype(str)))
registry, registry_mock = data.source_registry(macro_logs)
graded_agents = agent_weights.dropna(subset=["hit_rate"])
latest = graded_agents[graded_agents["date"] == graded_agents["date"].max()] if len(graded_agents) else graded_agents
graded_sources = registry[registry["n_uses"] > 0]

ui.label("Track record", "the review loop grades every score and re-weights the agents before each day")
ui.mock_note(weights_mock, "the review loop has not written logs/agent_weights.csv yet")
if agent_weights["date"].nunique() < 2:  # one review so far: the weights as numbers, the curve needs two
    now = agent_weights.sort_values("weight", ascending=False)
    st.caption("Agent weights · " + " · ".join(f"{r.agent} {r.weight:.0%}" for r in now.itertuples())
               + " · the curve starts with the second review")
else:
    st.caption("Agent weights")
    st.altair_chart(charts.lines(agent_weights, "date", "weight", "agent", "weight", ".0%", zero_based=True),
                    width="stretch")
agents_col, sources_col = st.columns(2, gap="large")
with agents_col:
    st.caption("Agent hit rates · line = 50%")
    if latest.empty:
        st.caption("Appears after the first review.")
    else:
        st.altair_chart(charts.hbars(latest, "hit_rate", "agent", ".0%", tooltip=["agent", "weight", "hit_rate"],
                                     rule_at=0.5), width="stretch")
with sources_col:
    counts = registry["status"].value_counts()
    st.caption(f"Source hit rates · {counts.get('active', 0)} active, {counts.get('testing', 0)} on trial, "
               f"{counts.get('dropped', 0)} dropped")
    if graded_sources.empty:
        st.caption("Appears once a macro view can be judged against the sessions that followed.")
    else:
        st.altair_chart(charts.hbars(graded_sources, "hit_rate", "source", ".0%",
                                     tooltip=["source", "status", "n_uses", "hit_rate"], rule_at=0.5),
                        width="stretch")
with st.expander("Source registry"):
    st.dataframe(registry, hide_index=True, width="stretch",
                 column_config={"hit_rate": st.column_config.ProgressColumn("hit rate", min_value=0, max_value=1,
                                                                            format="percent")})
journal = data.lessons(macro_logs)
with st.expander("Lessons journal · what the agents read before they decide"):
    if journal:
        st.markdown(ui.demote("\n".join(journal.splitlines()[1:])))
    else:
        st.caption("Empty: the review loop writes it when a day runs (or: python -m graph.review_loop).")
