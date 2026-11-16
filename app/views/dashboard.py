"""Dashboard: how the portfolio does against the benchmark, where the day stands, what is held."""
import streamlit as st

from app import charts, data, ui
from data import config as settings
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
with st.spinner("Loading the prices..."):
    prices, prices_mock = data.day_prices(as_of, state)
log = data.order_log(logs)
held, cash = data.positions(as_of, logs, prices)
total = float(held["value_eur"].sum()) + cash
history = data.portfolio_history(pipeline.same_book(log, as_of) if log is not None else None, prices)
start = settings.START_CAPITAL_EUR
gain = total / start - 1

ui.page("Dashboard", f"Our €{start:,.0f} against the same amount in the {data.BENCHMARK_LABEL}, and what the day "
                     "needs from you.")
ui.mock_note(prices_mock, "no prices for this day yet: run the day")
edge = None
if len(history):
    last = history.sort_values("date").groupby("series")["value_eur"].last()
    edge = (last["Portfolio"] - last[data.BENCHMARK_LABEL]) / start
ui.kpis([
    ("Value", f"€{total:,.0f}", f"{gain:+.2%} since the start", "up" if gain >= 0 else "down"),
    (f"Against the {data.BENCHMARK_LABEL}", "—" if edge is None else f"{edge:+.2%}",
     "no order yet" if edge is None else "ahead" if edge >= 0 else "behind",
     None if edge is None else "up" if edge >= 0 else "down"),
    ("Cash", f"€{cash:,.0f}", f"{cash / total:.0%} of the portfolio" if total else None),
    ("Today", (data.day_outcome(data.day_record(as_of, logs)) if ui.READ_ONLY
               else ui.day_label(status, state, pending)).capitalize(), None,
     "wait" if status == "waiting for approval" and not ui.READ_ONLY else None),
])

if ui.READ_ONLY:
    st.caption("Read-only: this page shows what the team's logs hold. The team runs the days and reviews the orders "
               "on its own machine.")
else:
    ui.label("Today", f"{as_of:%A %d %B}")
    ui.day_steps(status, state, pending)
for incident in state.get("incidents", []):
    st.error(ui.incident_text(incident), icon=":material/error:")
if not ui.READ_ONLY:
    st.write("")
    go, again, _ = st.columns([2, 1, 2])
    if status == "not started":
        if go.button("Run the day", type="primary", icon=":material/play_arrow:", width="stretch",
                     help="Runs every agent, then pauses for your first review on the Today page"):
            with st.spinner("The agents are running (about a minute)..."):
                pipeline.run_day(as_of, logs)
            st.rerun()
    else:
        if status == "waiting for approval":
            what = {"orders": f"Review the {len(state.get('proposed_orders', []))} proposed orders",
                    "risk": "Review the risk", "summary": "Sign off the summary"}[pending["gate"]]
            go.page_link("views/today.py", label=what, icon=":material/arrow_forward:")
        elif state.get("outcome") == "stopped":
            go.page_link("views/today.py", label="No trade today: see why", icon=":material/block:")
        else:
            go.page_link("views/today.py", label="Approved: export the orders", icon=":material/task_alt:")
        if again.button("Run again", icon=":material/replay:",
                        help="Starts the day again. Scores already logged are read back: no new Claude calls"):
            with st.spinner("Running again from the logs..."):
                pipeline.run_day(as_of, logs, rerun=True)
            st.rerun()

ui.label("Performance", f"€ · an order fills at {settings.fill_rule()}")
if history.empty:
    st.caption("No executed order yet: the portfolio is all cash.")
elif history["date"].nunique() < 2:
    st.caption(f"The curve starts after the next trading day: €{history['value_eur'].iloc[0]:,.0f} on "
               f"{history['date'].iloc[0]:%a %d %b}.")
else:
    st.altair_chart(charts.lines(history, "date", "value_eur", "series", "€", ",.0f"), width="stretch")

ui.label("Positions", f"{len(held)} held")
if held.empty:
    st.caption("No positions.")
else:
    st.dataframe(held, hide_index=True, width="stretch", column_config={
        "ticker": st.column_config.TextColumn("ETF"),
        "units": st.column_config.NumberColumn("Units", format="%g"),
        "last_close_eur": st.column_config.NumberColumn("Last close", format="€%.2f"),
        "value_eur": st.column_config.NumberColumn("Value", format="€%.0f"),
        "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1, format="percent"),
    })
