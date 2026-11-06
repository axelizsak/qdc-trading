"""Human approval interface. Owner: Axel.

Run:  streamlit run app/streamlit_app.py

Shows the proposed orders with each agent's rationale. A human approves, edits
or rejects each order. Approved orders are appended to logs/orders.csv with
both the AI suggestion and the human decision.
"""
import streamlit as st

st.set_page_config(page_title="QDC Trading", layout="wide")
st.title("QDC Trading: order approval")
st.info("Not wired yet. See graph/pipeline.py.")
