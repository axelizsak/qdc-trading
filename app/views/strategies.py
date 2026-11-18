"""Strategies: the three reference portfolios customers see in the app, side by side, and the one the team trades.

The team chooses here which strategy the agents trade: from the chosen day on, every day not run yet stays within
that strategy's limits (agents/portfolio.py `within_strategy`). The choice is logged (logs/strategy_choices.csv), so
every machine trades the same one. A day already run keeps the strategy it was run with."""
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from agents import model_portfolios
from app import charts, data, ui
from data import config as settings
from data.profiles import ARCHETYPES, SLEEVES, STRATEGIES
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
_, state, _ = pipeline.day_status(as_of, logs)
with st.spinner("Loading the prices..."):
    prices, prices_mock = data.day_prices(as_of, state)
TRADED = model_portfolios.traded(logs, as_of)  # the strategy the team trades on this day
ui.page("Strategies", "The three reference portfolios customers see in the app, and the one the agents trade for us.")
ui.mock_note(prices_mock, "no prices for this day yet: run the day")

for column, (key, spec) in zip(st.columns(len(STRATEGIES)), STRATEGIES.items()):
    traded = " · <span class='wait'>traded by the team</span>" if key == TRADED else ""
    column.markdown(f"<div class='qdc-kpi'><div class='l'>Risk level {spec['level']} of {len(STRATEGIES)}{traded}</div>"
                    f"<div class='v'>{escape(spec['label'])}</div><div class='d'>{escape(spec['tagline'])}</div></div>",
                    unsafe_allow_html=True)

ui.label("The strategy we trade", f"on {as_of:%a %d %b} and every day after, until the next choice")
limits = {key: spec["limits"] for key, spec in STRATEGIES.items()}
st.dataframe(pd.DataFrame([{"Strategy": STRATEGIES[key]["label"], "Shares at most": lim["equity_max"],
                            "Themes at most": lim["themes_max"], "Complex products at most": lim["complex_max"]}
                           for key, lim in limits.items()]), hide_index=True, width="stretch", column_config={
    name: st.column_config.NumberColumn(name, format="percent")
    for name in ("Shares at most", "Themes at most", "Complex products at most")})
st.caption("The agents trade freely inside these limits; the code cuts back what goes beyond them, and the memo says "
           "what it cut. What the themes limit cuts goes to the core ETFs, what the shares limit cuts goes to bonds.")
with st.form("choose_strategy", border=True):
    pick_traded = st.radio("Strategy", list(STRATEGIES), index=list(STRATEGIES).index(TRADED), horizontal=True,
                           format_func=lambda key: STRATEGIES[key]["label"])
    reason = st.text_input("Why", placeholder="Needed: why this strategy from this day on")
    if st.form_submit_button(f"Trade it from {as_of:%a %d %b}", type="primary", icon=":material/check:"):
        if not reason.strip():
            st.error("Say why: the choice is logged with its reason.", icon=":material/block:")
        else:
            model_portfolios.choose(logs, as_of, pick_traded, reason)
            st.rerun()
if pipeline.proposal_path(logs, as_of, 1).exists():
    st.caption(f"{as_of:%d %b} is already run: it keeps the strategy it was run with. A new choice applies to the "
               "days not run yet.")
made = model_portfolios.choices(logs)
if len(made):
    st.dataframe(made.assign(strategy=made["strategy"].map(lambda k: STRATEGIES[k]["label"])).iloc[::-1],
                 hide_index=True, width="stretch", column_config={
        "date": "From", "strategy": "Strategy", "reason": st.column_config.TextColumn("Why", width="large")})

curves = {key: model_portfolios.curve(prices, key) for key in STRATEGIES}
start = settings.START_CAPITAL_EUR
sessions = len(next(iter(curves.values())))
table = pd.DataFrame([{
    "Strategy": spec["label"] + (" (traded)" if key == TRADED else ""), "Risk level": spec["level"],
    **{name: spec["sleeves"][sleeve] for sleeve, name in SLEEVES.items()},
    "Return": curves[key].iloc[-1] - 1,
    "Volatility a year": curves[key].pct_change().std() * np.sqrt(model_portfolios.YEAR),
    "Worst fall": (curves[key] / curves[key].cummax() - 1).min(),
} for key, spec in STRATEGIES.items()])

ui.label("Side by side", f"shares per sleeve, and what each did over the last {sessions} sessions")
share = {name: st.column_config.ProgressColumn(name, min_value=0, max_value=1, format="percent", width="small")
         for name in SLEEVES.values()}  # narrow, so that the three results on the right stay in view
st.dataframe(table, hide_index=True, width="stretch", column_config={
    **share, "Risk level": st.column_config.NumberColumn("Risk level", format="%d of 3"),
    "Return": st.column_config.NumberColumn("Return", format="percent"),
    "Volatility a year": st.column_config.NumberColumn("Volatility a year", format="percent"),
    "Worst fall": st.column_config.NumberColumn("Worst fall", format="percent")})

ui.label("Value over time", f"€{start:,.0f} in each reference portfolio, put back to its weights each month")
history = pd.concat([pd.DataFrame({"date": pd.to_datetime(curve.index), "value_eur": curve.values * start,
                                   "series": STRATEGIES[key]["label"]}) for key, curve in curves.items()])
st.altair_chart(charts.lines(history, "date", "value_eur", "series", "€", ",.0f", x_format="%b %y", height=380),
                width="stretch")

ui.label("Archetypes", "the strategy each kind of customer most often ends in")
st.dataframe(pd.DataFrame([{"Archetype": spec["label"], "Who": spec["who"],
                            "Typical strategy": STRATEGIES[spec["typical"]]["label"]} for spec in ARCHETYPES.values()]),
             hide_index=True, width="stretch",
             column_config={"Who": st.column_config.TextColumn("Who", width="large")})
st.caption("The customer's own answers set the strategy, by fixed rules. The archetype only changes how the coach "
           "helps.")

ui.label("Inside one strategy", "its ETFs and their share of the portfolio")
pick = st.selectbox("Strategy", list(STRATEGIES), index=list(STRATEGIES).index(TRADED),
                    format_func=lambda key: STRATEGIES[key]["label"], label_visibility="collapsed")
held = pd.DataFrame([{"ticker": ticker, "sleeve": SLEEVES[sleeve], "target_weight": weight}
                     for ticker, (sleeve, weight) in model_portfolios.weights(pick).items()])
left, right = st.columns([3, 2], gap="large")
with left:
    st.altair_chart(charts.hbars(held, "target_weight", "ticker", ".0%", ["ticker", "sleeve", "target_weight"],
                                 domain=(0, max(0.5, float(held["target_weight"].max())))), width="stretch")
with right:
    spec = STRATEGIES[pick]
    traded = " · traded by the team" if pick == TRADED else ""
    st.markdown(f"**{spec['label']}** · risk level {spec['level']} of 3{traded}\n\n"
                f"{spec['tagline']} Fits {spec['fits']}.\n\n"
                + ("Complex products (leveraged, inverse, crypto) open after the customer's 3-question check."
                   if spec["complex"] else "No complex products (leveraged, inverse, crypto) at this level."))
