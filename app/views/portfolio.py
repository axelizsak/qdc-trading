"""Portfolio: what we hold and how it compares with the market."""
import streamlit as st

from app import charts, data, ui
from data import config as settings
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
_, state, _ = pipeline.day_status(as_of, logs)
prices, prices_mock = data.day_prices(as_of, state)
log = data.order_log(logs)
held, cash = data.positions(as_of, logs, prices)
total = float(held["value_eur"].sum()) + cash
history = data.portfolio_history(log, prices)
start = settings.START_CAPITAL_EUR

ui.page("Portfolio", "What we hold, and how it compares with €10,000 put in WORLD.")
ui.mock_note(prices_mock, "no prices for this day yet")
gain = total / start - 1
figures = [
    ("Value", f"€{total:,.0f}", f"{'▲' if gain >= 0 else '▼'} {gain:+.2%} since start", "up" if gain >= 0 else "down"),
    ("Cash", f"€{cash:,.0f}", f"{cash / total:.0%} of the portfolio" if total else None),
    ("Positions", len(held)),
]
if len(history):
    last = history.sort_values("date").groupby("series")["value_eur"].last()
    edge = (last["Portfolio"] - last["WORLD"]) / start
    figures.append(("vs WORLD", f"{edge:+.2%}", "ahead of the market" if edge >= 0 else "behind the market",
                    "up" if edge >= 0 else "down"))
else:
    figures.append(("vs WORLD", "—", "no order yet"))
ui.kpis(figures)

ui.label("Performance", "EUR · fills at the close of the order date")
if history.empty:
    st.caption("No executed orders yet: the portfolio is all cash.")
else:
    st.altair_chart(charts.lines(history, "date", "value_eur", "series", "EUR", ",.0f"), width="stretch")

ui.label("Positions")
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
