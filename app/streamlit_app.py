"""QDC Trading cockpit (local). Run:  streamlit run app/streamlit_app.py

Every page works on mock data until the real logs exist. Orders are approved here: the page
resumes the paused LangGraph day and writes logs/orders.csv (logs/mock/ while steps are on mocks).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

st.set_page_config(page_title="QDC Terminal", page_icon=":material/candlestick_chart:", layout="wide")
navigation = st.navigation([
    st.Page("views/home.py", title="Portfolio", icon=":material/show_chart:", default=True),
    st.Page("views/orders.py", title="Orders", icon=":material/fact_check:"),
    st.Page("views/research.py", title="Research", icon=":material/insights:"),
], position="top")
ui.header()
navigation.run()
