"""The LangGraph pipeline on mocks: three human reviews, feedback loops, fallbacks and replay from the logs."""
from datetime import date

import pandas as pd
import pytest

from data import config as settings
from data.schemas import validate_order_log, validate_reviews
from graph import pipeline

DAY = date(2026, 12, 7)


@pytest.fixture
def logs(tmp_path):
    return tmp_path / "logs"


def _boom(*args, **extra):
    raise RuntimeError("API down")


def _state(logs, day=DAY):
    with pipeline.open_graph(logs) as graph:
        return graph.get_state(pipeline.thread(day, logs)).values


def test_day_goes_through_three_reviews_then_writes_the_logs(logs):
    pending = pipeline.run_day(DAY, logs)
    assert pending["gate"] == "orders" and pending["round"] == 1
    proposed = pd.DataFrame(pending["proposed_orders"])
    assert 0 < len(proposed) <= settings.MAX_ORDERS_PER_DAY

    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    assert pending["gate"] == "risk" and len(pending["risk_report"]) > 0
    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    assert pending["gate"] == "summary" and "## Orders to submit" in pending["summary"]
    assert not (logs / "orders.csv").exists()  # nothing is written before the last review
    assert pipeline.resume_day(DAY, pipeline.approve(pending), logs) is None

    log = validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert sorted(log["ticker"]) == sorted(proposed["ticker"])
    assert (log["decision"] == "approved").all()
    reviews = validate_reviews(pd.read_csv(logs / "reviews.csv"))
    assert reviews["gate"].tolist() == ["orders", "risk", "summary"]
    assert (logs / "summary" / f"{DAY}.md").exists()
    assert set(pd.read_csv(logs / "signals.csv")["agent"]) == set(settings.SIGNAL_AGENTS)
    assert pipeline.run_day(DAY, logs) is None  # the day is done


def test_feedback_sends_the_day_back_to_the_portfolio_manager(logs, monkeypatch):
    received = []

    def portfolio(*args, **extra):
        received.append(args[-1])
        return pipeline._mock_portfolio(*args, **extra)

    monkeypatch.setitem(pipeline.MOCK, "portfolio", portfolio)
    pending = pipeline.run_day(DAY, logs)
    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    pending = pipeline.resume_day(DAY, {"decision": "revised", "feedback": "too much leverage"}, logs)

    assert pending["gate"] == "orders" and pending["round"] == 2
    assert pending["revisions_left"] == settings.MAX_REVISIONS - 1
    assert received == ["", "Round 1, risk review: too much leverage"]
    assert _state(logs)["risk_report"] == []  # the new round starts clean

    pipeline.approve_day(DAY, logs)
    reviews = pd.read_csv(logs / "reviews.csv")
    assert reviews[["round", "gate", "decision"]].values.tolist() == [
        [1, "orders", "approved"], [1, "risk", "revised"],
        [2, "orders", "approved"], [2, "risk", "approved"], [2, "summary", "approved"]]


def test_revisions_are_capped(logs):
    pending = pipeline.run_day(DAY, logs)
    for _ in range(settings.MAX_REVISIONS):
        pending = pipeline.resume_day(DAY, {"decision": "revised", "feedback": "again"}, logs)
    assert pending["revisions_left"] == 0
    with pytest.raises(ValueError, match="no revision left"):
        pipeline.resume_day(DAY, {"decision": "revised", "feedback": "again"}, logs)


def test_stop_means_no_trade_today(logs):
    pending = pipeline.run_day(DAY, logs)
    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    assert pipeline.resume_day(DAY, {"decision": "stopped", "feedback": "Fed tonight"}, logs) is None

    log = validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert (log["decision"] == "rejected").all()
    assert log["reason"].str.contains("Fed tonight").all()
    assert _state(logs)["outcome"] == "stopped"
    assert not (logs / "summary" / f"{DAY}.md").exists()
    holdings, cash = pipeline.load_book(logs, pd.DataFrame(columns=["date", "ticker", "close_eur"]), date(2026, 12, 10))
    assert holdings.empty and cash == settings.START_CAPITAL_EUR


def test_feedback_is_required_to_send_back_or_stop(logs):
    pending = pipeline.run_day(DAY, logs)
    for decision in ("revised", "stopped"):
        with pytest.raises(ValueError, match="feedback is required"):
            pipeline.resume_day(DAY, {"decision": decision, "feedback": "  "}, logs)
    assert pipeline.run_day(DAY, logs) == pending


def test_bad_order_decisions_are_refused_and_the_day_stays_paused(logs):
    pending = pipeline.run_day(DAY, logs)
    answer = pipeline.approve(pending)
    answer["orders"][0] = {"ticker": answer["orders"][0]["ticker"], "decision": "rejected"}  # no reason
    with pytest.raises(ValueError, match="reason"):
        pipeline.resume_day(DAY, answer, logs)
    answer = pipeline.approve(pending)
    answer["orders"].append({"ticker": "WORLD", "decision": "added", "side": "SELL", "units": 3, "reason": "x"})
    with pytest.raises(ValueError, match="only 0 held"):
        pipeline.resume_day(DAY, answer, logs)
    assert pipeline.run_day(DAY, logs) == pending


def test_orders_breaking_the_trading_rules_become_an_incident(logs, monkeypatch):
    def too_many(signals, weights, lessons, holdings, cash, prices, as_of, feedback, **extra):
        target, _, memo = pipeline._mock_portfolio(signals, weights, lessons, holdings, cash, prices, as_of, feedback)
        return target, pd.DataFrame({"date": as_of, "side": "BUY", "ticker": ["WORLD", "GOLD", "BITCOIN", "SEMIS",
                                                                             "INDIA", "KOREA"], "units": 1}), memo

    monkeypatch.setitem(pipeline.MOCK, "portfolio", too_many)
    pending = pipeline.run_day(DAY, logs)
    assert pending["proposed_orders"] == []
    assert "6 orders, the limit is 5" in pd.read_csv(logs / "incidents.csv")["error"][0]


def test_check_orders_in_plain_words():
    before = date(2026, 12, 4)  # an order of DAY fills at the last close before DAY
    prices = pd.DataFrame({"date": [before, before, DAY, DAY], "ticker": ["WORLD", "GOLD"] * 2,
                           "close_eur": [100.0, 50.0, 999.0, 999.0]})
    holdings = pd.DataFrame({"ticker": ["GOLD"], "units": [2]})
    orders = pd.DataFrame({"date": DAY, "side": ["BUY", "SELL", "BUY"], "ticker": ["WORLD", "GOLD", "SEMIS"],
                           "units": [200, 3, 1]})
    problems = pipeline.check_orders(orders, holdings, 1_000, prices, DAY)
    assert "Selling 3 GOLD, but only 2 held" in problems
    assert "No price for SEMIS" in problems
    assert any(p.startswith("Not enough cash") for p in problems)
    assert pipeline.check_orders(orders.head(1).assign(units=5), holdings, 1_000, prices, DAY) == []


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
    def not_written(*args, **extra):  # a module that is not written yet: the switch must reach it
        raise NotImplementedError

    monkeypatch.setitem(settings.USE_MOCK, "portfolio", False)
    monkeypatch.setitem(pipeline.REAL, "portfolio", not_written)
    pipeline.run_day(DAY, logs)
    incidents = pd.read_csv(logs / "incidents.csv")
    assert incidents["step"].tolist() == ["portfolio"]
    assert "NotImplementedError" in incidents["error"][0]


def test_failing_portfolio_manager_proposes_no_orders(logs, monkeypatch):
    monkeypatch.setitem(pipeline.MOCK, "portfolio", _boom)
    pending = pipeline.run_day(DAY, logs)
    assert pending["proposed_orders"] == [] and "Portfolio Manager failed" in pending["memo"]
    pipeline.approve_day(DAY, logs)
    assert pipeline.run_day(DAY, logs) is None


def test_failing_risk_manager_still_reaches_the_risk_review(logs, monkeypatch):
    monkeypatch.setitem(pipeline.MOCK, "risk", _boom)
    pending = pipeline.resume_day(DAY, pipeline.approve(pipeline.run_day(DAY, logs)), logs)
    assert pending["gate"] == "risk" and pending["risk_report"] == []
    assert [i["step"] for i in pending["incidents"]] == ["risk"]
    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    assert "The Risk Manager gave no assessment" in pending["summary"]


def test_rerun_reads_signals_from_logs_instead_of_calling_agents(logs, monkeypatch):
    pipeline.run_day(DAY, logs)
    monkeypatch.setitem(pipeline.MOCK, "technical", _boom)
    monkeypatch.setitem(pipeline.MOCK, "macro_news", _boom)

    pending = pipeline.run_day(DAY, logs, rerun=True)

    assert pending is not None
    assert not (logs / "incidents.csv").exists()
    assert _state(logs)["macro_summary"] == "Mock macro summary."


def test_next_day_starts_from_the_executed_orders(logs):
    pending = pipeline.run_day(DAY, logs)
    bought = {o["ticker"]: o["units"] for o in pending["proposed_orders"]}
    pipeline.approve_day(DAY, logs)

    pipeline.run_day(date(2026, 12, 10), logs)

    state = _state(logs, date(2026, 12, 10))
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


def test_cli_runs_answers_then_approves_the_rest(logs, tmp_path, capsys):
    pipeline.main(["--date", "2026-12-07", "--logs-dir", str(logs)])
    assert "waiting for the orders review" in capsys.readouterr().out
    answer = tmp_path / "answer.json"
    answer.write_text('{"decision": "revised", "feedback": "smaller sizes"}')
    pipeline.main(["--date", "2026-12-07", "--logs-dir", str(logs), "--answer", str(answer)])
    assert "round 2" in capsys.readouterr().out
    pipeline.main(["--date", "2026-12-07", "--logs-dir", str(logs), "--approve-all"])
    assert (logs / "orders.csv").exists() and (logs / "reviews.csv").exists()


def test_real_macro_news_reuses_the_days_run_from_logs(logs, tmp_path, monkeypatch):
    real_logs = tmp_path / "real"
    monkeypatch.setattr(settings, "LOGS_DIR", real_logs)
    monkeypatch.setitem(settings.USE_MOCK, "macro_news", False)
    monkeypatch.setitem(pipeline.REAL, "macro_news", _boom)  # Claude must not be called
    run = pipeline.mock.mock_signals(DAY.isoformat(), agents=pipeline.MACRO_AGENTS).assign(rationale="real run")
    (real_logs / "macro").mkdir(parents=True)
    run.to_csv(real_logs / "signals.csv", index=False)
    (real_logs / "macro" / f"{DAY}.md").write_text("# Macro report of the day")

    pipeline.run_day(DAY, logs)

    assert not (logs / "incidents.csv").exists()
    assert _state(logs)["macro_summary"] == "# Macro report of the day"
    assert {s["rationale"] for s in _state(logs)["signals"] if s["agent"] == "macro_news"} == {"real run"}


def test_real_macro_news_never_runs_for_another_day(logs, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path / "real")
    monkeypatch.setitem(settings.USE_MOCK, "macro_news", False)
    monkeypatch.setitem(pipeline.REAL, "macro_news", _boom)

    pending = pipeline.run_day(DAY, logs)  # DAY is not today

    assert "only runs on 2026-12-07 itself" in pending["incidents"][0]["error"]
    assert not (tmp_path / "real" / "signals.csv").exists()


def test_practice_orders_never_count_in_the_competition(logs):
    practice = date(2026, 11, 20)
    pipeline.approve_day(practice, logs)
    prices = pd.DataFrame(_state(logs, practice)["prices"])

    held, cash = pipeline.load_book(logs, prices, date(2026, 11, 23))
    assert len(held) > 0 and cash < settings.START_CAPITAL_EUR  # the practice book goes on
    held, cash = pipeline.load_book(logs, prices, settings.TRADING_DAYS[0])
    assert held.empty and cash == settings.START_CAPITAL_EUR  # the competition starts from scratch


def test_real_signal_steps_keep_their_signals_in_logs(logs, tmp_path, monkeypatch):
    real_logs = tmp_path / "real"
    monkeypatch.setattr(settings, "LOGS_DIR", real_logs)
    monkeypatch.setitem(settings.USE_MOCK, "technical", False)
    monkeypatch.setitem(pipeline.REAL, "technical", lambda prices, features, as_of, lessons: pipeline.mock.mock_signals(
        as_of.isoformat(), agents=["technical_det"]).assign(rationale="real technical"))

    pipeline.run_day(DAY, logs)  # portfolio and risk are still on mocks

    saved = pd.read_csv(real_logs / "signals.csv")
    assert set(saved["agent"]) == {"technical_det"} and len(saved) == 50


def test_a_day_runs_end_to_end_on_the_real_portfolio_and_risk_managers(logs, monkeypatch):
    """Mock prices and signals, real Portfolio Manager, Risk Manager and Logbook (no Claude key: code alone)."""
    from agents import logbook
    from data import schemas

    monkeypatch.setitem(settings.USE_MOCK, "portfolio", False)
    monkeypatch.setitem(settings.USE_MOCK, "risk", False)

    pending = pipeline.run_day(DAY, logs)
    assert pending["incidents"] == []
    assert 0 < len(pending["proposed_orders"]) <= settings.MAX_ORDERS_PER_DAY
    assert "Portfolio Manager" in pending["memo"]

    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    assert pending["incidents"] == []
    assert "## Risk board" in pending["risk_board"] and "### Positions" in pending["risk_board"]
    assert "Not available" in pending["risk_reading"]  # no key in tests: the reading says so, the day goes on
    assert {"orders", "cash_reserve"} <= {row["metric"] for row in pending["risk_report"]}

    pending = pipeline.resume_day(DAY, pipeline.approve(pending), logs)
    assert "## Risk board" in pending["summary"] and "### Independent reading" in pending["summary"]
    assert pipeline.resume_day(DAY, pipeline.approve(pending), logs) is None

    for name in ("pm_portfolios", "recommendations", "risk_forecasts", "risk_calibration", "reviews"):
        getattr(schemas, f"validate_{name}")(pd.read_csv(logs / f"{name}.csv"))
    schemas.validate_risk_report(pd.read_csv(logs / "risk_reports.csv"))
    validate_order_log(pd.read_csv(logs / "orders.csv"))
    assert (logs / "risk" / f"{DAY}.md").exists() and (logs / "logbook" / f"{DAY}.md").exists()
    assert logbook.verify(logs) == []  # the hash chain is intact
    assert not (logs / "incidents.csv").exists()

    # the next day starts from the book those orders left, filled at the last close before the day
    held, cash = pipeline.load_book(logs, pd.DataFrame(_state(logs)["prices"]), date(2026, 12, 10))
    assert len(held) == len(pd.read_csv(logs / "orders.csv")) and 0 <= cash < settings.START_CAPITAL_EUR


def test_every_module_fills_an_order_at_the_same_price():
    from agents import portfolio, risk

    prices = pipeline.mock.mock_prices(end=DAY.isoformat())
    mine = pipeline.fill_prices(prices, DAY)
    assert len(mine) == 50
    pd.testing.assert_series_equal(mine, portfolio.last_closes(prices, DAY, before=True), check_names=False)
    pd.testing.assert_series_equal(mine, risk.fill_prices(prices, DAY), check_names=False)
    last_day = prices[prices["date"].astype(str) == prices["date"].astype(str).max()].set_index("ticker")["close_eur"]
    assert not mine.equals(last_day.reindex(mine.index))  # strictly before the day, never the day's own close


def test_a_failing_logbook_never_stops_the_day(logs, monkeypatch):
    monkeypatch.setattr(pipeline.logbook, "record_round", _boom)
    monkeypatch.setattr(pipeline.logbook, "record_day", _boom)
    pipeline.approve_day(DAY, logs)
    assert pipeline.run_day(DAY, logs) is None
    assert set(pd.read_csv(logs / "incidents.csv")["step"]) == {"logbook"}
    assert len(validate_order_log(pd.read_csv(logs / "orders.csv"))) > 0
