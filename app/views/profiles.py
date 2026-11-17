"""Profiles: the five model portfolios of the customer app side by side, to choose the one the team trades."""
import numpy as np
import pandas as pd
import streamlit as st

from agents import model_portfolios
from app import charts, data, ui
from data import config as settings
from data.profiles import PROFILES, SLEEVES
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
_, state, _ = pipeline.day_status(as_of, logs)
with st.spinner("Loading the prices..."):
    prices, prices_mock = data.day_prices(as_of, state)
ui.page("Profiles", "The five model portfolios of the customer app, side by side. Each is a fixed split, nothing in "
                    "it follows the market: compare them here to choose the one the team trades.")
ui.mock_note(prices_mock, "no prices for this day yet: run the day")

curves = {key: model_portfolios.curve(prices, key) for key in PROFILES}
start = settings.START_CAPITAL_EUR
sessions = len(next(iter(curves.values())))
table = pd.DataFrame([{
    "Profile": spec["label"], "Risk level": spec["level"],
    **{name: spec["sleeves"][sleeve] for sleeve, name in SLEEVES.items()},
    "Return": curves[key].iloc[-1] - 1,
    "Volatility a year": curves[key].pct_change().std() * np.sqrt(model_portfolios.YEAR),
    "Worst fall": (curves[key] / curves[key].cummax() - 1).min(),
} for key, spec in PROFILES.items()])

ui.label("The five profiles", f"shares per sleeve, and what each did over the last {sessions} sessions")
share = {name: st.column_config.ProgressColumn(name, min_value=0, max_value=1, format="percent", width="small")
         for name in SLEEVES.values()}  # narrow, so that the three results on the right stay in view
st.dataframe(table, hide_index=True, width="stretch", column_config={
    **share, "Risk level": st.column_config.NumberColumn("Risk level", format="%d of 5"),
    "Return": st.column_config.NumberColumn("Return", format="percent"),
    "Volatility a year": st.column_config.NumberColumn("Volatility a year", format="percent"),
    "Worst fall": st.column_config.NumberColumn("Worst fall", format="percent")})

ui.label("Value over time", f"€{start:,.0f} in each model portfolio, put back to its weights each month")
history = pd.concat([pd.DataFrame({"date": pd.to_datetime(curve.index), "value_eur": curve.values * start,
                                   "series": PROFILES[key]["label"]}) for key, curve in curves.items()])
st.altair_chart(charts.lines(history, "date", "value_eur", "series", "€", ",.0f", x_format="%b %y", height=420),
                width="stretch")

ui.label("Inside one profile", "its ETFs and their share of the portfolio")
pick = st.selectbox("Profile", list(PROFILES), format_func=lambda key: PROFILES[key]["label"], label_visibility="collapsed")
held = pd.DataFrame([{"ticker": ticker, "sleeve": SLEEVES[sleeve], "target_weight": weight}
                     for ticker, (sleeve, weight) in model_portfolios.weights(pick).items()])
left, right = st.columns([3, 2], gap="large")
with left:
    st.altair_chart(charts.hbars(held, "target_weight", "ticker", ".0%", ["ticker", "sleeve", "target_weight"],
                                 domain=(0, max(0.5, float(held["target_weight"].max())))), width="stretch")
with right:
    spec = PROFILES[pick]
    st.markdown(f"**{spec['label']}** · risk level {spec['level']} of 5\n\n{spec['who'].capitalize()}.\n\n"
                f"Goal: {spec['goal']}. Horizon: {spec['horizon']}.\n\n" + "\n".join(f"- {rule}" for rule in spec["rules"]))
