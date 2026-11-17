"""Macro / News agent: what the news says, factor by factor and ETF by ETF."""
from datetime import date

import pandas as pd
import streamlit as st

from agents import macro_news
from app import charts, data, ui
from data.config import MACRO_FACTOR_MEANING
from data.schemas import split_list

as_of, logs = ui.as_of(), ui.macro_logs()
ui.page("Macro / News agent", "What the news says: a view on 15 macro factors (macro_market) and 20 theme analysts "
                              "scoring their ETFs (macro_etf).")

views, views_mock = data.macro_views(logs)
evidence, _ = data.evidence(logs)
etf_views, _ = data.etf_views(logs)
runs = sorted(views["date"].astype(str).unique())
day = max([d for d in runs if d <= as_of.isoformat()] or runs[:1])
day_views = views[views["date"].astype(str) == day].assign(
    name=lambda v: v["factor"].map(ui.FACTOR_LABELS), meaning=lambda v: v["factor"].map(MACRO_FACTOR_MEANING))
day_evidence = evidence[evidence["date"].astype(str) == day]
day_etf = etf_views[etf_views["date"].astype(str) == day]
report = data.macro_report(date.fromisoformat(day), logs)
ui.mock_note(views_mock, "no macro run yet")
ui.kpis([("Run of", f"{date.fromisoformat(day):%a %d %b}", "the latest on or before the day"),
         ("Facts collected", len(day_evidence), f"from {day_evidence['source'].nunique()} sources"),
         ("ETF calls", int((day_etf["direction"].abs() >= 0.1).sum()) if len(day_etf) else 0, "an analyst view of 0.1 or more"),
         ("Cost", data.macro_cost(report) or "—")])

left, right = st.columns(2, gap="large")
with left:
    ui.label("Macro views", "+1 to -1 · faded = low confidence")
    st.altair_chart(charts.diverging(day_views, "direction", "name", ["name", "direction", "confidence", "meaning"],
                                     opacity="confidence"), width="stretch")
with right:
    ui.label("Strongest ETF calls", "the theme analysts")
    if day_etf.empty:
        st.caption("No analyst view in this run.")
    else:
        strongest = day_etf.reindex(day_etf["direction"].abs().sort_values(ascending=False).index).head(15)
        st.altair_chart(charts.diverging(strongest, "direction", "ticker", ["ticker", "direction", "confidence"],
                                         opacity="confidence"), width="stretch")

ahead = macro_news.catalysts(day_etf) if len(day_etf) else []
if ahead:
    ui.label("Events ahead", "found by the analysts")
    st.dataframe(pd.DataFrame([{"date": date.fromisoformat(d), "event": e.split(". ")[0], "ETFs": ", ".join(t[:5])}
                               for d, e, t in ahead[:8]]), hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("Date", format="ddd D MMM"),
                                "event": st.column_config.TextColumn("Event", width="large"),
                                "ETFs": st.column_config.TextColumn("ETFs", width="medium")})

ui.label("Why", "the reasoning behind one view")
options = [("factor", f) for f in day_views.reindex(day_views["direction"].abs().sort_values(ascending=False).index)["factor"]]
options += [("etf", t) for t in (day_etf.reindex(day_etf["direction"].abs().sort_values(ascending=False).index)["ticker"]
                                 if len(day_etf) else [])]
kind, key = st.selectbox("Why", options, label_visibility="collapsed",
                         format_func=lambda o: f"Factor · {ui.FACTOR_LABELS[o[1]]}" if o[0] == "factor" else f"ETF · {o[1]}")
view = (day_views.set_index("factor") if kind == "factor" else day_etf.set_index("ticker")).loc[key]
tone = "blue" if view["direction"] >= 0 else "red"
title = ui.FACTOR_LABELS[key] if kind == "factor" else key
st.markdown(f"**{title}** :{tone}[{view['direction']:+.2f}] · confidence {view['confidence']:.0%}"
            + (f" · +1 means {MACRO_FACTOR_MEANING[key]}" if kind == "factor" else ""))
st.markdown(view["rationale"])
st.markdown(f"**Counter-argument** · {view['counter_argument']}")
links = [s for s in split_list(view["sources"]) if s.startswith("http")] or [
    u for u in day_evidence[day_evidence["source"].isin(split_list(view["sources"]))]["url"] if isinstance(u, str)]
if links:
    st.markdown("**Sources** · " + " · ".join(f"[{i + 1}]({u})" for i, u in enumerate(dict.fromkeys(links[:6]))))
if report:
    ui.document_download("The full report of the run", report, f"macro_{day}")
