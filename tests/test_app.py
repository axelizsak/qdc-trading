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
PAGES = ["home", "orders", "research", "logbook"]


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)  # pipeline logs go to tmp_path/mock
    return pipeline.default_logs_dir()


def open_page(name: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["as_of"] = DAY
    at.run()
    if name != "home":
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


def test_home_runs_the_day_until_the_pause(logs):
    at = open_page("home")
    next(b for b in at.button if b.label == "Run the day").click().run()
    assert not at.exception
    assert pipeline.day_status(DAY, logs)[0] == "waiting for approval"


def _to_gate(logs, gate: str) -> None:
    """Run the day and approve the reviews before `gate`."""
    pending = pipeline.run_day(DAY, logs)
    while pending["gate"] != gate:
        pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)


@pytest.mark.parametrize("gate", ["risk", "summary"])
@pytest.mark.parametrize("name", PAGES)
def test_page_runs_at_each_review(logs, name, gate):
    _to_gate(logs, gate)
    open_page(name)


def test_three_approvals_write_the_order_log_and_offer_the_export(logs):
    pending = pipeline.run_day(DAY, logs)
    at = open_page("orders")
    for gate in ("orders", "risk", "summary"):
        assert pipeline.day_status(DAY, logs)[2]["gate"] == gate
        at.button(key="submit").click().run()
        assert not at.exception, at.exception

    log = validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert sorted(log["ticker"]) == sorted(o["ticker"] for o in pending["proposed_orders"])
    assert (log["decision"] == "approved").all()
    assert pipeline.day_status(DAY, logs)[0] == "done"
    assert any(">Approved<" in m.value for m in at.markdown)  # the page now shows the log and the export

    home = open_page("home")
    assert not home.exception
    assert any(">vs MSCI World<" in m.value for m in home.markdown)  # the Belfius benchmark


def test_feedback_sends_the_day_back_from_the_page(logs):
    _to_gate(logs, "risk")
    at = open_page("orders")
    at.text_area(key="feedback_risk_1").input("cut the leveraged ETFs").run()
    at.button(key="revise").click().run()
    assert not at.exception, at.exception
    pending = pipeline.day_status(DAY, logs)[2]
    assert pending["gate"] == "orders" and pending["round"] == 2


def test_no_trade_today_from_the_page(logs):
    pipeline.run_day(DAY, logs)
    at = open_page("orders")
    at.text_area(key="feedback_orders_1").input("Fed tonight").run()
    at.button(key="stop").click().run()
    assert not at.exception, at.exception
    status, state, _ = pipeline.day_status(DAY, logs)
    assert status == "done" and state["outcome"] == "stopped"
    for name in PAGES:
        open_page(name)


def test_pages_run_on_the_real_portfolio_and_risk_managers(logs, monkeypatch):
    monkeypatch.setitem(settings.USE_MOCK, "portfolio", False)
    monkeypatch.setitem(settings.USE_MOCK, "risk", False)
    pipeline.run_day(DAY, logs)
    at = open_page("orders")
    for gate in ("orders", "risk", "summary"):  # the memo, then the risk board and reading, then the summary
        assert pipeline.day_status(DAY, logs)[2]["gate"] == gate
        for name in PAGES:
            open_page(name)
        at.button(key="submit").click().run()
        assert not at.exception, at.exception
    assert pipeline.day_status(DAY, logs)[0] == "done"
    book = open_page("logbook")
    assert any(">Intact<" in m.value for m in book.markdown)  # the hash chain of the day just recorded
    assert not (logs / "incidents.csv").exists()

