"""Macro: what the Macro/News agent expects, why, and which sources it trusts."""
from datetime import date

import streamlit as st

from app import charts, data, ui
from data.config import MACRO_FACTOR_MEANING
from data.schemas import split_list

logs = ui.macro_logs()
views, views_mock = data.macro_views(logs)
evidence, _ = data.evidence(logs)

ui.page("Macro", "What the Macro/News agent expects over the next few trading days, and why.")
ui.mock_note(views_mock, "no macro run yet")

dates = sorted(views["date"].astype(str).unique(), reverse=True)
earlier = [d for d in dates if d <= ui.as_of().isoformat()]
day = dates[0] if len(dates) == 1 else st.selectbox("Run", dates, index=dates.index(earlier[0]) if earlier else 0)
day_views = views[views["date"].astype(str) == day].copy()
day_views["name"] = day_views["factor"].map(ui.FACTOR_LABELS)
day_views["meaning"] = day_views["factor"].map(MACRO_FACTOR_MEANING)
day_evidence = evidence[evidence["date"].astype(str) == day]
report = data.macro_report(date.fromisoformat(day), logs)

ui.kpis([
    ("Run", date.fromisoformat(day).strftime("%a %d %b")),
    ("Cost", data.macro_cost(report) or "—", "Claude API"),
    ("Evidence", len(day_evidence), "facts and numbers"),
    ("Sources", day_evidence["source"].nunique()),
])

ui.label("Macro picture", "most bullish at the top · blue = up, red = down · faded = low confidence")
st.altair_chart(charts.factor_bars(day_views), width="stretch")

ui.label("Why")
ranked = day_views.assign(weight=day_views["direction"].abs() * day_views["confidence"]).sort_values(
    "weight", ascending=False)
factor = st.selectbox("Factor", ranked["factor"].tolist(), format_func=ui.FACTOR_LABELS.get,
                      label_visibility="collapsed")
view = ranked.set_index("factor").loc[factor]


def source_links(sources: str) -> str:
    lines = []
    for source in split_list(sources):
        rows = day_evidence[(day_evidence["source"] == source) & (day_evidence["factor"] == factor)]
        rows = rows if len(rows) else day_evidence[day_evidence["source"] == source]
        urls = [u for u in dict.fromkeys(rows["url"]) if isinstance(u, str) and u.startswith("http")][:3]
        links = " · ".join(f"[{i + 1}]({u})" for i, u in enumerate(urls))
        lines.append(f"- `{source}` {links}")
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

found = day_evidence[day_evidence["kind"] == "exploration"]
ui.label("New sources on trial", f"{len(found)} found by free exploration · they weigh less until proven")
if found.empty:
    st.caption("None in this run.")
for r in found.itertuples():
    name = r.source.removeprefix("explore_").replace("_", " ").capitalize()
    with st.expander(f"{name} · {ui.FACTOR_LABELS.get(r.factor, r.factor)}"):
        what, _, why = r.summary.partition(" Why it matters: ")
        st.markdown(what)
        if why:
            st.markdown(f"**Why it matters** · {why}")
        st.markdown(f"[Open the source]({r.url})")

registry, registry_mock = data.source_registry(logs)
graded = registry[registry["n_uses"] > 0]
ui.label("Source track record", "hit rate of each source after the trading days")
ui.mock_note(registry_mock, "no source registry yet")
if graded.empty:
    st.caption("No graded views yet: hit rates appear once the review loop has scored a trading day.")
else:
    st.altair_chart(charts.hbars(graded, "hit_rate", "source", ".0%",
                                 tooltip=["source", "status", "n_uses", "hit_rate"], rule_at=0.5), width="stretch")
