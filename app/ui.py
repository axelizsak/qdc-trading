"""Shared page context: trading day, log folders, colour tokens, small display helpers."""
from datetime import date
from pathlib import Path

import streamlit as st

from data import config as settings
from graph import pipeline

STEPS = ["data", "technical", "macro_news", "portfolio", "risk"]

# Colour tokens per mode (validated reference palette); text always uses the theme's own ink.
TOKENS = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a"], "negative": "#e34948", "positive": "#2a78d6",
              "neutral": "#f0efec", "rule": "#898781", "ink": "#52514e"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70"], "negative": "#e66767", "positive": "#3987e5",
             "neutral": "#383835", "rule": "#898781", "ink": "#c3c2b7"},
}


def default_day() -> date:
    today = date.today()
    upcoming = [d for d in settings.TRADING_DAYS if d >= today]
    return upcoming[0] if upcoming else settings.TRADING_DAYS[-1]


def as_of() -> date:
    return st.session_state.get("as_of", default_day())


def pipeline_logs() -> Path:
    """Where the pipeline reads and writes: logs/mock/ while any step is on a mock."""
    return pipeline.default_logs_dir()


def macro_logs() -> Path:
    """The Macro/News agent always writes its real outputs to logs/."""
    return settings.LOGS_DIR


def tokens() -> dict:
    theme = getattr(st.context, "theme", None)
    return TOKENS["dark" if theme is not None and theme.type == "dark" else "light"]


def mock_note(is_mock: bool, what: str = "No real data yet") -> None:
    if is_mock:
        st.caption(f":gray-badge[:material/science: mock data] {what}: showing mock data.")


def sidebar() -> None:
    st.session_state.setdefault("as_of", default_day())
    with st.sidebar:
        st.date_input("Trading day", key="as_of", format="YYYY-MM-DD",
                      help="Trading days: " + ", ".join(d.strftime("%d %b") for d in settings.TRADING_DAYS))
        mocks = [s for s in STEPS if settings.USE_MOCK[s]]
        folder = pipeline_logs()
        shown = folder.relative_to(settings.ROOT) if folder.is_relative_to(settings.ROOT) else folder
        st.caption(f"Pipeline logs: `{shown}/`")
        st.caption("Steps on mocks: " + (", ".join(mocks) if mocks else "none"))
