"""Portfolio (home): performance against the MSCI World benchmark first, then the trading day, then positions."""
from pathlib import Path

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
history = data.portfolio_history(pipeline.same_book(log, as_of) if log is not None else None, prices)
start = settings.START_CAPITAL_EUR
gain = total / start - 1

ui.page("Portfolio", f"How our €{start:,.0f} is doing against the same amount in the {data.BENCHMARK_LABEL} "
                     f"benchmark ({settings.BENCHMARK}), and what the day needs from you.")
ui.mock_note(prices_mock, "no prices for this day yet")
figures = [("Value", f"€{total:,.0f}", f"{'▲' if gain >= 0 else '▼'} {gain:+.2%} since start",
            "up" if gain >= 0 else "down")]
if len(history):
    last = history.sort_values("date").groupby("series")["value_eur"].last()
    edge = (last["Portfolio"] - last[data.BENCHMARK_LABEL]) / start
    figures.append((f"vs {data.BENCHMARK_LABEL}", f"{edge:+.2%}",
                    "ahead of the benchmark" if edge >= 0 else "behind the benchmark", "up" if edge >= 0 else "down"))
else:
    figures.append((f"vs {data.BENCHMARK_LABEL}", "—", "no order yet"))
figures += [("Cash", f"€{cash:,.0f}", f"{cash / total:.0%} of the portfolio" if total else None),
            ("Today", ui.day_label(status, state, pending).capitalize(), None,
             "hot" if status == "waiting for approval" else None)]
ui.kpis(figures)

ui.label("Performance", "EUR · an order fills at the last close before its day")
if history.empty:
    st.caption("No executed orders yet: the portfolio is all cash.")
elif history["date"].nunique() < 2:
    st.caption(f"The curve starts after the next trading day: one point so far, "
               f"€{history['value_eur'].iloc[0]:,.0f} on {history['date'].iloc[0]:%a %d %b}.")
else:
    st.altair_chart(charts.lines(history, "date", "value_eur", "series", "EUR", ",.0f"), width="stretch")

incidents = state.get("incidents", [])
ui.label("Today", f"{as_of:%a %d %b}")
ui.pipeline_strip(status, state, pending)
for incident in incidents:
    st.error(ui.incident_text(incident))
st.write("")
if status == "not started":
    if st.button("Run the day", type="primary", icon=":material/play_arrow:",
                 help="Runs every agent, then pauses for your first review on the Orders page"):
        with st.spinner("Agents running..."):
            pipeline.run_day(as_of, logs)
        st.rerun()
else:
    go, again, _ = st.columns([2, 1, 3])
    if status == "waiting for approval":
        label = {"orders": f"Review the {len(state.get('proposed_orders', []))} proposed orders",
                 "risk": "Review the risk after the orders",
                 "summary": "Review the executive summary"}[pending["gate"]]
        go.page_link("views/orders.py", label=label, icon=":material/arrow_forward:")
    elif state.get("outcome") == "stopped":
        go.page_link("views/orders.py", label="No trade today · see your reviews", icon=":material/block:")
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
with st.expander("How the system works"):
    st.image(str(Path(__file__).resolve().parents[2] / "docs" / "workflow.png"), width=720)
