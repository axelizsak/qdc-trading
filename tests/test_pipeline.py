"""The LangGraph pipeline on mocks: pause, resume, fallbacks and replay from the logs."""
from datetime import date

import pandas as pd
import pytest

from data import config as settings
from data.schemas import validate_order_log
from graph import pipeline

DAY = date(2026, 12, 7)


@pytest.fixture
def logs(tmp_path):
    return tmp_path / "logs"


def _boom(*args):
    raise RuntimeError("API down")


def _state(logs):
    with pipeline.open_graph(logs) as graph:
        return graph.get_state(pipeline.thread(DAY, logs)).values


def test_day_pauses_for_approval_then_resume_writes_orders(logs):
    pending = pipeline.run_day(DAY, logs)
    proposed = pd.DataFrame(pending["proposed_orders"])
    assert 0 < len(proposed) <= settings.MAX_ORDERS_PER_DAY
    assert not (logs / "orders.csv").exists()

    pipeline.resume_day(DAY, pipeline.approve_all(pending), logs)

    log = validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert sorted(log["ticker"]) == sorted(proposed["ticker"])
    assert (log["decision"] == "approved").all()
    assert set(pd.read_csv(logs / "signals.csv")["agent"]) == set(settings.SIGNAL_AGENTS)
    assert pipeline.run_day(DAY, logs) is None  # the day is done


def test_paused_day_is_read_from_sqlite_not_recomputed(logs, monkeypatch):
    first = pipeline.run_day(DAY, logs)
    monkeypatch.setitem(pipeline.MOCK, "data", _boom)
    assert pipeline.run_day(DAY, logs) == first


def test_failing_agent_gives_neutral_signals_and_an_incident(logs, monkeypatch):
    monkeypatch.setitem(pipeline.MOCK, "macro_news", _boom)
    pending = pipeline.run_day(DAY, logs)

    assert pending is not None  # the day still reaches the human
    incidents = pd.read_csv(logs / "incidents.csv")
    assert incidents["step"].tolist() == ["macro_news"]
    assert "API down" in incidents["error"][0]
    macro = [s for s in _state(logs)["signals"] if s["agent"] == "macro_news"]
    assert len(macro) == 50
    assert all(s["score"] == 0 and s["confidence"] == 0 for s in macro)


def test_switch_off_calls_the_real_module(logs, monkeypatch):
    monkeypatch.setitem(settings.USE_MOCK, "technical", False)  # agents/technical.py is not written yet
    pipeline.run_day(DAY, logs)
    incidents = pd.read_csv(logs / "incidents.csv")
    assert incidents["step"].tolist() == ["technical"]
    assert "NotImplementedError" in incidents["error"][0]


def test_failing_risk_step_proposes_no_orders(logs, monkeypatch):
    monkeypatch.setitem(pipeline.MOCK, "risk", _boom)
    pending = pipeline.run_day(DAY, logs)
    assert pending["proposed_orders"] == []
    pipeline.resume_day(DAY, [], logs)
    assert pipeline.run_day(DAY, logs) is None


def test_rerun_reads_signals_from_logs_instead_of_calling_agents(logs, monkeypatch):
    pipeline.run_day(DAY, logs)
    monkeypatch.setitem(pipeline.MOCK, "technical", _boom)
    monkeypatch.setitem(pipeline.MOCK, "macro_news", _boom)

    pending = pipeline.run_day(DAY, logs, rerun=True)

    assert pending is not None
    assert not (logs / "incidents.csv").exists()
    assert _state(logs)["macro_summary"] == "Mock macro summary."


def test_bad_decisions_are_refused_and_the_day_stays_paused(logs):
    pending = pipeline.run_day(DAY, logs)
    decisions = pipeline.approve_all(pending)
    decisions[0] = {"ticker": decisions[0]["ticker"], "decision": "rejected"}  # no reason
    with pytest.raises(ValueError, match="reason"):
        pipeline.resume_day(DAY, decisions, logs)
    assert pipeline.run_day(DAY, logs) == pending


def test_next_day_starts_from_the_executed_orders(logs):
    pending = pipeline.run_day(DAY, logs)
    pipeline.resume_day(DAY, pipeline.approve_all(pending), logs)

    pipeline.run_day(date(2026, 12, 10), logs)

    with pipeline.open_graph(logs) as graph:
        state = graph.get_state(pipeline.thread(date(2026, 12, 10), logs)).values
    bought = {o["ticker"]: o["units"] for o in pending["proposed_orders"]}
    assert {h["ticker"]: h["units"] for h in state["holdings"]} == bought
    assert 0 < state["cash_eur"] < settings.START_CAPITAL_EUR


def test_build_order_log_covers_every_decision():
    proposed = pd.DataFrame({"date": [DAY] * 3, "side": ["BUY"] * 3,
                             "ticker": ["WORLD", "GOLD", "BITCOIN"], "units": [10, 8, 2]})
    log = pipeline.build_order_log(DAY, proposed, [
        {"ticker": "WORLD", "decision": "approved"},
        {"ticker": "GOLD", "decision": "modified", "units": 4, "reason": "smaller"},
        {"ticker": "BITCOIN", "decision": "rejected", "reason": "too volatile"},
        {"ticker": "DEFENSE", "decision": "added", "side": "BUY", "units": 5, "reason": "conviction"},
    ])
    units = log.set_index("ticker")["units"]
    assert units.dropna().to_dict() == {"WORLD": 10, "GOLD": 4, "DEFENSE": 5}
    assert pd.isna(units["BITCOIN"])
    with pytest.raises(ValueError, match="no decision"):
        pipeline.build_order_log(DAY, proposed, [{"ticker": "WORLD", "decision": "approved"}])


def test_cli_runs_then_resumes(logs, capsys):
    pipeline.main(["--date", "2026-12-07", "--logs-dir", str(logs)])
    assert "waiting for human approval" in capsys.readouterr().out
    pipeline.main(["--date", "2026-12-07", "--logs-dir", str(logs), "--approve-all"])
    assert (logs / "orders.csv").exists()
