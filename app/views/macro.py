"""Macro: views, counter-arguments, clickable sources, exploration findings, which sources were right."""
from datetime import date

import altair as alt
import streamlit as st

from app import charts, data, ui
from data.config import MACRO_FACTOR_MEANING, MACRO_FACTORS
from data.schemas import split_list

st.title("Macro")
logs = ui.macro_logs()
views, views_mock = data.macro_views(logs)
evidence, _ = data.evidence(logs)
ui.mock_note(views_mock, "No macro run yet")

dates = sorted(views["date"].astype(str).unique(), reverse=True)
earlier = [d for d in dates if d <= ui.as_of().isoformat()]
day = st.selectbox("Run of", dates, index=dates.index(earlier[0]) if earlier else 0)
day_views = views[views["date"].astype(str) == day].copy()
day_views["order"] = day_views["factor"].map(MACRO_FACTORS.index)
day_views = day_views.sort_values("order")
day_evidence = evidence[evidence["date"].astype(str) == day]
report = data.macro_report(date.fromisoformat(day), logs)

c1, c2, c3 = st.columns(3)
c1.metric("Cost of the run", data.macro_cost(report) or "n/a")
c2.metric("Evidence items", len(day_evidence))
c3.metric("Sources", day_evidence["source"].nunique())

st.subheader("Views")
st.caption("Direction over the next few trading days: blue = the factor goes up, red = down. "
           "Hover a bar for the confidence.")
st.altair_chart(charts.diverging_bars(
    day_views, "direction", "factor", ui.tokens(),
    tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("direction:Q", format="+.2f", title="Direction"),
             alt.Tooltip("confidence:Q", format=".0%", title="Confidence")]), width="stretch")


def source_links(factor: str, sources: str) -> str:
    links = []
    for source in split_list(sources):
        rows = day_evidence[(day_evidence["source"] == source) & (day_evidence["factor"] == factor)]
        rows = rows if len(rows) else day_evidence[day_evidence["source"] == source]
        urls = [u for u in dict.fromkeys(rows["url"]) if isinstance(u, str) and u.startswith("http")][:3]
        links.append(source + (": " + " · ".join(f"[{i + 1}]({u})" for i, u in enumerate(urls)) if urls else ""))
    return "\n".join(f"- {link}" for link in links)


for v in day_views.itertuples():
    with st.expander(f"**{v.factor}** {v.direction:+.2f} · confidence {v.confidence:.0%}"):
        st.caption(f"+1 means {MACRO_FACTOR_MEANING[v.factor]}")
        st.markdown(v.rationale)
        st.markdown(f"**Counter-argument:** {v.counter_argument}")
        st.markdown("**Sources**\n" + source_links(v.factor, v.sources))

st.subheader("Exploration findings")
found = day_evidence[day_evidence["kind"] == "exploration"]
if found.empty:
    st.caption("No exploration findings for this run.")
for r in found.itertuples():
    st.markdown(f"- **{r.source.removeprefix('explore_')}** ({r.factor}) {r.summary} [link]({r.url})")

st.subheader("Which sources were right")
registry, registry_mock = data.source_registry(logs)
ui.mock_note(registry_mock, "No source registry yet")
graded = registry[registry["n_uses"] > 0]
if graded.empty:
    st.caption("No graded views yet: hit rates appear once the review loop has scored a trading day.")
else:
    st.altair_chart(charts.hbars(graded, "hit_rate", "source", ui.tokens(), ".0%",
                                 tooltip=["source", "status", "n_uses", "hit_rate"], rule_at=0.5), width="stretch")
st.dataframe(registry.sort_values(["status", "hit_rate"], ascending=[True, False]), hide_index=True, width="stretch",
             column_config={"hit_rate": st.column_config.ProgressColumn("Hit rate", min_value=0, max_value=1,
                                                                        format="percent")})
