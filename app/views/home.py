"""Portfolio (home): performance against WORLD first, then where the trading day stands, then positions."""
import streamlit as st

from app import charts, data, ui
from data import config as settings
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
prices, prices_mock = data.day_prices(as_of, state)
log = data.order_log(logs)
held, cash = data.positions(as_of, logs, prices)
total = float(held["value_eur"].sum()) + cash
history = data.portfolio_history(log, prices)
start = settings.START_CAPITAL_EUR
gain = total / start - 1

ui.page("Portfolio", "How we are doing against €10,000 put in WORLD, and what the day needs from you.")
ui.mock_note(prices_mock, "no prices for this day yet")
figures = [("Value", f"€{total:,.0f}", f"{'▲' if gain >= 0 else '▼'} {gain:+.2%} since start",
            "up" if gain >= 0 else "down")]
if len(history):
    last = history.sort_values("date").groupby("series")["value_eur"].last()
    edge = (last["Portfolio"] - last["WORLD"]) / start
    figures.append(("vs WORLD", f"{edge:+.2%}", "ahead of the market" if edge >= 0 else "behind the market",
                    "up" if edge >= 0 else "down"))
else:
    figures.append(("vs WORLD", "—", "no order yet"))
figures += [("Cash", f"€{cash:,.0f}", f"{cash / total:.0%} of the portfolio" if total else None),
            ("Today", status.capitalize(), None, "hot" if status == "waiting for approval" else None)]
ui.kpis(figures)

ui.label("Performance", "EUR · fills at the close of the order date")
if history.empty:
    st.caption("No executed orders yet: the portfolio is all cash.")
elif history["date"].nunique() < 2:
    st.caption(f"The curve starts after the next trading day: one point so far, "
               f"€{history['value_eur'].iloc[0]:,.0f} on {history['date'].iloc[0]:%a %d %b}.")
else:
    st.altair_chart(charts.lines(history, "date", "value_eur", "series", "EUR", ",.0f"), width="stretch")

incidents = state.get("incidents", [])
proposed = pending["proposed_orders"] if pending else state.get("proposed_orders", [])
ui.label("Today", f"{as_of:%a %d %b}")
ui.pipeline_strip(status, incidents)
for incident in incidents:
    st.error(f"{ui.STEP_LABELS[incident['step']]} failed and was replaced by its fallback: {incident['error']}")
st.write("")
if status == "not started":
    if st.button("Run the day", type="primary", icon=":material/play_arrow:",
                 help="Runs every agent, then pauses for your approval on the Orders page"):
        with st.spinner("Agents running..."):
            pipeline.run_day(as_of, logs)
        st.rerun()
else:
    go, again, _ = st.columns([2, 1, 3])
    if status == "waiting for approval":
        go.page_link("views/orders.py", label=f"Review the {len(proposed)} proposed orders",
                     icon=":material/arrow_forward:")
    else:
        go.page_link("views/orders.py", label="Approved · export for the Belfius form", icon=":material/task_alt:")
    if again.button("Run again", icon=":material/replay:",
                    help="Starts the day again. Signals already logged are read back: no new API calls"):
        with st.spinner("Running again from the logs..."):
            pipeline.run_day(as_of, logs, rerun=True)
        st.rerun()

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
if log is not None:
    with st.expander("Order log"):
        st.dataframe(log, hide_index=True, width="stretch")
