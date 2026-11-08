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
    st.Page("views/today.py", title="Day", icon=":material/today:", default=True),
    st.Page("views/signals.py", title="Signals", icon=":material/grid_view:"),
    st.Page("views/macro.py", title="Macro", icon=":material/public:"),
    st.Page("views/orders.py", title="Orders", icon=":material/fact_check:"),
    st.Page("views/portfolio.py", title="Portfolio", icon=":material/account_balance_wallet:"),
    st.Page("views/learning.py", title="Learning", icon=":material/school:"),
], position="top")
ui.header()
navigation.run()
