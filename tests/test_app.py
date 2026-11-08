"""The Streamlit cockpit on mocks: every page runs, and approving orders writes the order log."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from data import config as settings
from data.schemas import validate_order_log
from graph import pipeline

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
DAY = date(2026, 12, 7)
PAGES = ["today", "signals", "macro", "orders", "portfolio", "learning"]


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)  # pipeline logs go to tmp_path/mock
    return pipeline.default_logs_dir()


def open_page(name: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["as_of"] = DAY
    at.run()
    if name != "today":
        at.switch_page(f"views/{name}.py").run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("name", PAGES)
def test_page_runs_before_the_day(logs, name):
    open_page(name)


@pytest.mark.parametrize("name", PAGES)
def test_page_runs_on_a_paused_day(logs, name):
    pipeline.run_day(DAY, logs)
    open_page(name)


def test_today_runs_the_day_until_the_pause(logs):
    at = open_page("today")
    next(b for b in at.button if b.label == "Run the day").click().run()
    assert not at.exception
    assert pipeline.day_status(DAY, logs)[0] == "waiting for approval"


def test_approval_writes_the_order_log_and_offers_the_export(logs):
    pending = pipeline.run_day(DAY, logs)
    at = open_page("orders")
    at.button(key="submit").click().run()
    assert not at.exception

    log = validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert sorted(log["ticker"]) == sorted(o["ticker"] for o in pending["proposed_orders"])
    assert (log["decision"] == "approved").all()
    assert pipeline.day_status(DAY, logs)[0] == "done"
    assert at.success  # the page now shows the approved orders and the Belfius export

    portfolio = open_page("portfolio")
    assert any("Portfolio value" == m.label for m in portfolio.metric)
