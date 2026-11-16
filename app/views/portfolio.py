"""Portfolio Manager: the portfolio it aims for today, the orders that move toward it, and why."""
import pandas as pd
import streamlit as st

from app import charts, data, ui
from data.schemas import HOLDINGS, ORDERS, PORTFOLIO
from graph import pipeline

as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, _ = pipeline.day_status(as_of, logs)
ui.page("Portfolio Manager", "It combines the four scores with their weights, builds a target portfolio with the code "
                             "and with Claude, blends both, then picks at most 5 orders.")
if status == "not started" or not state.get("memo"):
    st.info("The Portfolio Manager has not run for this day yet: run the day from the dashboard.", icon=":material/info:")
    st.stop()

target = pd.DataFrame(state.get("target_weights", []), columns=PORTFOLIO)
orders = pd.DataFrame(state.get("proposed_orders", []), columns=ORDERS)
weights = state.get("agent_weights", {})
trust = state.get("trust")
ui.kpis([("Invested", f"{target['target_weight'].sum():.0%}", "of the portfolio in the target"),
         ("Positions", len(target), "in the target"),
         ("Claude's share", "—" if trust is None else f"{trust:.0%}", "of the blend with the code's portfolio"),
         ("Orders proposed", len(orders), f"round {state.get('round', 1)}")])

summary = data.memo_summary(state["memo"])
if summary:
    st.info(summary, icon=":material/smart_toy:")

left, right = st.columns([3, 2], gap="large")
with left:
    ui.label("Target portfolio", "share of the portfolio value")
    if target.empty:
        st.caption("All cash.")
    else:
        st.altair_chart(charts.hbars(target, "target_weight", "ticker", ".0%", ["ticker", "target_weight"],
                                     domain=(0, max(0.3, float(target["target_weight"].max())))), width="stretch")
with right:
    ui.label("Weight of each agent", "today")
    agents = pd.DataFrame({"agent": list(weights), "weight": list(weights.values())})
    st.altair_chart(charts.hbars(agents, "weight", "agent", ".0%", ["agent", "weight"]), width="stretch")

held = pd.DataFrame(state.get("holdings", []), columns=HOLDINGS)
ui.label("Orders proposed", f"held before them: {len(held)} ETF{'s' if len(held) != 1 else ''}, "
                            f"€{float(state.get('cash_eur', 0)):,.0f} in cash")
st.dataframe(orders.drop(columns="date"), hide_index=True, width="stretch",
             column_config={"side": "Side", "ticker": st.column_config.TextColumn("ETF"),
                            "units": st.column_config.NumberColumn("Units", format="%g")})

flags = data.section(state["memo"], "### To consider")
if flags:
    ui.label("To consider", "what Claude flags")
    st.markdown(flags)
st.download_button("The full memo", state["memo"], f"memo_{as_of}.md", "text/markdown", icon=":material/description:")
