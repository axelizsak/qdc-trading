"""Portfolio: positions, cash, performance against WORLD."""
import streamlit as st

from app import charts, data, ui
from data import config as settings
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
st.title("Portfolio")
_, state, _ = pipeline.day_status(as_of, logs)
prices, prices_mock = data.day_prices(as_of, state)
ui.mock_note(prices_mock, "No prices for this day yet")
log = data.order_log(logs)

held, cash = data.positions(as_of, logs, prices)
invested = float(held["value_eur"].sum())
total = invested + cash
history = data.portfolio_history(log, prices)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio value", f"€{total:,.0f}", delta=f"{total / settings.START_CAPITAL_EUR - 1:+.2%} since start")
c2.metric("Cash", f"€{cash:,.0f}", help=f"{cash / total:.0%} of the portfolio" if total else None)
c3.metric("Positions", len(held))
if len(history):
    last = history.sort_values("date").groupby("series")["value_eur"].last()
    c4.metric("vs WORLD", f"{(last['Portfolio'] - last['WORLD']) / settings.START_CAPITAL_EUR:+.2%}",
              help="Difference with €10,000 put in WORLD on the day of the first order")
else:
    c4.metric("vs WORLD", "n/a")

st.subheader("Performance")
if history.empty:
    st.caption("No executed orders yet: the portfolio is all cash.")
else:
    st.caption("Value in EUR of the portfolio and of €10,000 put in WORLD on the day of the first order. "
               "Fills at the close of the order date.")
    st.altair_chart(charts.lines(history, "date", "value_eur", "series", ui.tokens(), "Value (EUR)", ",.0f"),
                    width="stretch")

st.subheader("Positions")
if held.empty:
    st.caption("No positions.")
else:
    st.dataframe(held, hide_index=True, width="stretch", column_config={
        "units": st.column_config.NumberColumn("Units", format="%g"),
        "last_close_eur": st.column_config.NumberColumn("Last close", format="€%.2f"),
        "value_eur": st.column_config.NumberColumn("Value", format="€%.0f"),
        "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1, format="percent"),
    })
if log is not None:
    with st.expander("Order log"):
        st.dataframe(log, hide_index=True, width="stretch")
