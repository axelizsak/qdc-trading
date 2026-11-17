"""QDC Trading cockpit (local). Run:  streamlit run app/streamlit_app.py

Every page works on mock data until the real logs exist. Your three reviews of the day happen on the Today page:
it resumes the paused LangGraph day and writes logs/orders.csv (logs/mock/ while steps are on mocks).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

st.set_page_config(page_title="QDC Trading", page_icon=":material/candlestick_chart:", layout="wide")
navigation = st.navigation({
    "Day": [st.Page("views/dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
            st.Page("views/today.py", title="Today", icon=":material/fact_check:"),
            st.Page("views/profiles.py", title="Profiles", icon=":material/groups:")],
    "Agents": [st.Page("views/technical.py", title="Technical", icon=":material/show_chart:"),
               st.Page("views/macro.py", title="Macro / News", icon=":material/public:"),
               st.Page("views/portfolio.py", title="Portfolio Manager", icon=":material/pie_chart:"),
               st.Page("views/risk.py", title="Risk Manager", icon=":material/shield:")],
    "System": [st.Page("views/learning.py", title="Learning", icon=":material/insights:"),
               st.Page("views/how_it_works.py", title="How it works", icon=":material/account_tree:"),
               st.Page("views/logbook.py", title="Logbook", icon=":material/menu_book:")],
}, position="top")
ui.header()
navigation.run()
