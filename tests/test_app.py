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
PAGES = ["dashboard", "today", "technical", "macro", "portfolio", "risk", "learning", "how_it_works", "logbook"]


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)  # pipeline logs go to tmp_path/mock
    return pipeline.default_logs_dir()


def open_page(name: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["as_of"] = DAY
    at.run()
    if name != "dashboard":
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


def test_the_dashboard_runs_the_day_until_the_pause(logs):
    at = open_page("dashboard")
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
    at = open_page("today")
    for gate in ("orders", "risk", "summary"):
        assert pipeline.day_status(DAY, logs)[2]["gate"] == gate
        at.button(key="submit").click().run()
        assert not at.exception, at.exception

    log = validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert sorted(log["ticker"]) == sorted(o["ticker"] for o in pending["proposed_orders"])
    assert (log["decision"] == "approved").all()
    assert pipeline.day_status(DAY, logs)[0] == "done"
    assert any(">Approved<" in m.value for m in at.markdown)  # the page now shows the log and the export

    dashboard = open_page("dashboard")
    assert any(">Against the MSCI World<" in m.value for m in dashboard.markdown)  # the Belfius benchmark


def test_feedback_sends_the_day_back_from_the_page(logs):
    _to_gate(logs, "risk")
    at = open_page("today")
    at.text_area(key="feedback_risk_1").input("cut the leveraged ETFs").run()
    at.button(key="revise").click().run()
    assert not at.exception, at.exception
    pending = pipeline.day_status(DAY, logs)[2]
    assert pending["gate"] == "orders" and pending["round"] == 2


def test_no_trade_today_from_the_page(logs):
    pipeline.run_day(DAY, logs)
    at = open_page("today")
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
    at = open_page("today")
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


def test_the_learning_page_reads_the_weights_of_the_day(tmp_path):
    from app import data
    pd.DataFrame({"date": ["2026-12-04"] * 4 + ["2026-12-08"] * 4, "level": "agents", "name": settings.SIGNAL_AGENTS * 2,
                  "weight": [0.4, 0.2, 0.3, 0.1] * 2, "result": None}).to_csv(tmp_path / "weights.csv", index=False)
    assert data.weights_on(tmp_path, date(2026, 12, 7)) == {"agents": dict(zip(settings.SIGNAL_AGENTS, [0.4, 0.2, 0.3, 0.1]))}
    assert data.weights_on(tmp_path, date(2026, 12, 1)) == {}  # nothing learned yet


def test_every_box_of_the_diagram_has_its_files_and_its_code(logs):
    from app import flow
    root = Path(__file__).resolve().parent.parent
    keys = set(flow.BY_KEY)
    for step in flow.STEPS:
        assert all((root / f).exists() for f in step.files), step.key
        assert step.code and all(flow.code_of(fn)[1].startswith("def ") for fn in step.code), step.key
    for edge in flow.edges():
        assert {edge["from"], edge["to"]} <= keys and len(edge["points"]) >= 2
    pipeline.run_day(DAY, logs)
    status, state, pending = pipeline.day_status(DAY, logs)
    for step in flow.STEPS:  # every card has a line for the day
        assert flow.today(step.key, status, state, pending, logs, DAY)
    at = open_page("how_it_works")
    assert any(m.value == "Data" for m in at.subheader)  # the first box is shown before any click


def test_a_finished_day_is_read_from_its_files(logs, monkeypatch):
    monkeypatch.setitem(settings.USE_MOCK, "portfolio", False)
    monkeypatch.setitem(settings.USE_MOCK, "risk", False)
    pipeline.approve_day(DAY, logs)  # a real Portfolio Manager and Risk Manager write their files
    assert (logs / "memo" / f"{DAY}.md").exists() and (logs / "risk" / f"{DAY}.md").exists()
    (logs / "checkpoints.sqlite").unlink()  # without the saved graph state, the pages read what the day wrote
    for name in PAGES:
        open_page(name)
    assert any(">Approved<" in m.value for m in open_page("today").markdown)  # from orders.csv and reviews.csv
    assert any(">Breaches<" in m.value for m in open_page("risk").markdown)  # from risk_reports.csv
    assert any("Target portfolio" in m.value for m in open_page("portfolio").markdown)  # from pm_portfolios.csv
