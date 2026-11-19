"""The learning step on made-up logs: one agent that is always right, one that is always wrong."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import learning, logbook, macro_tape
from agents import portfolio_config as pm_cfg
from agents import strategy_config as tech_cfg
from data import config as settings
from data.features import compute_features
from data.mock import mock_prices
from data.schemas import validate_weights
from graph import pipeline, review_loop

DAY = date(2026, 12, 7)


@pytest.fixture
def prices():
    return mock_prices(end=DAY.isoformat(), days=40, seed=3)


def _log_scores(logs, prices, days: int = 8) -> list[str]:
    """Log `days` past days of scores: technical_det knows each day's session, macro_market says the opposite."""
    table = macro_tape.closes(prices, DAY)
    dates = list(table.index[-days - 1:-1])  # each one's session is closed before DAY
    rows = []
    for day in dates:
        at = table.index.get_loc(day)
        session = table.iloc[at] / table.iloc[at - 1] - 1  # from the close before `day` to the close of `day`
        session = session - session.mean()
        for ticker, ret in session.items():
            for agent, sign in (("technical_det", 1), ("macro_market", -1)):
                rows.append({"ticker": ticker, "date": day, "agent": agent, "score": float(sign * np.sign(ret)) * 0.8,
                             "confidence": 0.5, "rationale": "made up"})
    rows.append({"ticker": "WORLD", "date": dates[-1], "agent": "macro_etf", "score": 0.0, "confidence": 0.0,
                 "rationale": "agent failed: API down"})  # a failed agent is no evidence
    logs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(logs / "signals.csv", index=False)
    return dates


def _agents_today(logs) -> dict[str, float]:
    weights = validate_weights(pd.read_csv(logs / review_loop.WEIGHTS_FILE))
    rows = weights[(weights["level"] == "agents") & (weights["date"] == DAY.isoformat())]
    return dict(zip(rows["name"], rows["weight"]))


def test_the_agent_that_was_right_gains_weight_and_it_is_logged(tmp_path, prices):
    _log_scores(tmp_path, prices)
    result = review_loop.review(DAY, prices, tmp_path, replay_technical=False)

    now = _agents_today(tmp_path)
    assert set(now) == {"technical_det", "macro_market"}  # the failed macro_etf row is no evidence
    assert now["technical_det"] > 0.55 > 0.45 > now["macro_market"]  # from 50/50
    assert learning.weights_now({"technical_det": 0.5, "macro_market": 0.5}, result["agent_multipliers"]) == pytest.approx(now)
    board = result["board"].set_index(["level", "name"])
    assert board.loc[("agents", "technical_det"), "sessions"] == 8 and board.loc[("agents", "technical_det"), "right_share"] == 1
    assert board.loc[("agents", "macro_market"), "right_share"] == 0

    lessons = result["lessons"]
    assert lessons.startswith(f"## Review of {DAY}")
    assert "- The four agents: " in lessons and "technical_det 50% -> " in lessons and "a session over 8, right 100%" in lessons
    assert "**Where each agent is right**" in lessons and "**Largest misses of the last judged day**" in lessons


def test_the_journal_keeps_the_newest_review_first_and_a_rerun_changes_nothing(tmp_path, prices):
    _log_scores(tmp_path, prices)
    first = review_loop.review(DAY, prices, tmp_path, replay_technical=False)
    ledger = len(logbook.load(tmp_path))
    again = review_loop.review(DAY, prices, tmp_path, replay_technical=False)
    assert again["journal"] == first["journal"] and again["journal"].count("## Review of") == 1
    pd.testing.assert_frame_equal(again["weights"], first["weights"])
    assert len(logbook.load(tmp_path)) == ledger and logbook.verify(tmp_path) == []
    assert logbook.load(tmp_path)["kind"].tolist() == ["reassessment", "lesson"]

    for offset in range(1, review_loop.REVIEWS_KEPT + 2):  # later days: the journal keeps the newest ones
        review_loop.save_journal(tmp_path, date(2026, 12, 7 + offset), [f"## Review of 2026-12-{7 + offset:02d}", "", "x"])
    journal = (tmp_path / "lessons.md").read_text(encoding="utf-8")
    assert journal.startswith("# Lessons journal") and journal.count("## Review of") == review_loop.REVIEWS_KEPT
    assert journal.index("## Review of 2026-12-13") < journal.index("## Review of 2026-12-12")
    assert f"## Review of {DAY}" not in journal  # the oldest ones leave the file (they stay in git and the logbook)


def test_todays_scores_are_not_judged_and_empty_logs_keep_the_starting_weights(tmp_path, prices):
    result = review_loop.review(DAY, prices, tmp_path / "empty", replay_technical=False)
    assert result["agent_multipliers"] == {} and result["trust"] == pm_cfg.TRUST_START
    assert "no session judged yet" in result["lessons"]

    today = pd.DataFrame([{"ticker": "WORLD", "date": DAY.isoformat(), "agent": "technical_det", "score": 1.0,
                           "confidence": 1.0, "rationale": "today"}])
    (tmp_path / "today").mkdir()
    today.to_csv(tmp_path / "today" / "signals.csv", index=False)
    board = review_loop.review(DAY, prices, tmp_path / "today", replay_technical=False)["board"]
    assert board[board["level"] == "agents"]["sessions"].tolist() == [0]


def test_the_technical_answers_are_recorded_once_and_learned_session_by_session(tmp_path, monkeypatch):
    prices = mock_prices(end=DAY.isoformat(), days=300, seed=4)
    features = compute_features(prices)
    monkeypatch.setattr(settings, "LEARNING_START", date(2026, 11, 23))
    monkeypatch.setattr(tech_cfg, "SIGNAL_START", {s: 1 / 3 for s in settings.TECHNICAL_SIGNALS})  # three to weigh
    result = review_loop.review(date(2026, 12, 3), prices, tmp_path, features)
    answers = review_loop.load_answers(tmp_path / review_loop.ANSWERS)
    assert sorted(answers["date"].unique()) == ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27",
                                                "2026-11-30", "2026-12-01", "2026-12-02"]
    technical = result["weights"][result["weights"]["level"] == "technical"]
    assert technical["date"].nunique() == 9 and set(technical["name"]) == set(settings.TECHNICAL_SIGNALS)
    assert sum(result["technical_weights"].values()) == pytest.approx(1)
    assert result["technical_weights"] != pytest.approx({s: 1 / 3 for s in settings.TECHNICAL_SIGNALS})  # it learned

    def no_more_sheets(*args, **kwargs):  # the next morning only computes the sessions not recorded yet
        raise AssertionError("an old session was computed again")

    seen = []
    original = review_loop.learning.technical_answers
    monkeypatch.setattr(review_loop.learning, "technical_answers",
                        lambda p, f, days, as_of: seen.append(days) or original(p, f, days, as_of))
    review_loop.review(date(2026, 12, 4), prices, tmp_path, features)
    assert seen == [[date(2026, 12, 3)]]


def test_a_technical_score_missing_from_the_logs_is_replayed_from_the_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LEARNING_START", date(2026, 11, 30))
    prices = mock_prices(end=DAY.isoformat(), days=300, seed=4)
    pd.DataFrame([{"ticker": "WORLD", "date": "2026-12-02", "agent": "macro_etf", "score": 0.5, "confidence": 0.5,
                   "rationale": "r"}]).to_csv(tmp_path / "signals.csv", index=False)
    answers = review_loop.update_answers(tmp_path, prices, compute_features(prices), DAY)
    frame, _ = review_loop.technical_weights(answers, DAY)
    calls = review_loop.agent_calls(tmp_path, answers, frame)
    replayed = calls[calls["agent"] == "technical_det"]
    assert sorted(replayed["date"].unique()) == ["2026-12-02", "2026-12-03", "2026-12-04"]  # from the first logged score
    assert set(replayed["rationale"]) == {"replayed"}


def test_claudes_portfolio_gains_trust_when_it_earns_more_than_the_codes(tmp_path, prices):
    table = macro_tape.closes(prices, DAY)
    rows = []
    for day in table.index[-9:-1]:
        at = table.index.get_loc(day)
        session = (table.iloc[at] / table.iloc[at - 1] - 1).dropna().sort_values()
        for source, names in (("llm", session.index[-5:]), (pm_cfg.ANCHOR, session.index[:5]), ("baseline", session.index[-5:])):
            rows += [{"date": day, "source": source, "ticker": t, "weight": 0.2} for t in names]
    pd.DataFrame(rows).to_csv(tmp_path / "pm_portfolios.csv", index=False)
    result = review_loop.review(DAY, prices, tmp_path, replay_technical=False)
    assert result["trust"] > pm_cfg.TRUST_START  # judged against the anchor portfolio, not the other code one
    assert "- The LLM's portfolio (llm) against the code's (code): code 40% -> " in result["lessons"]
    assert "llm 60% -> " in result["lessons"]


def test_the_day_starts_with_the_learning_and_the_agents_get_its_weights(tmp_path, prices, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setitem(pipeline.MOCK, "data", lambda as_of: (prices, pipeline.mock.mock_features(as_of.isoformat())))
    _log_scores(logs, prices)
    seen = {}

    def technical(prices, features, as_of, lessons, weights=None):
        seen.update(lessons=lessons, weights=weights)
        return pipeline.mock.mock_signals(as_of.isoformat(), agents=pipeline.TECHNICAL_AGENTS)

    monkeypatch.setitem(pipeline.MOCK, "technical", technical)
    pending = pipeline.run_day(DAY, logs)
    assert pending["incidents"] == []
    assert f"## Review of {DAY}" in seen["lessons"] and seen["weights"] == pytest.approx({"trend": 1.0})
    assert (logs / "lessons.md").exists() and (logs / review_loop.WEIGHTS_FILE).exists()
    used = pipeline.day_status(DAY, logs)[1]["agent_weights"]  # the four agents scored: 25% each to start
    assert used["technical_det"] > 0.25 > used["macro_market"] and sum(used.values()) == pytest.approx(1)


def test_a_failing_learning_step_is_an_incident_and_the_day_goes_on(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("review broke")

    monkeypatch.setattr(pipeline.review_loop, "review", broken)
    pending = pipeline.run_day(DAY, tmp_path)
    assert [i["step"] for i in pending["incidents"]] == ["learning"]
    assert len(pending["proposed_orders"]) > 0
    assert pipeline.day_status(DAY, tmp_path)[1]["agent_weights"] == dict.fromkeys(settings.SIGNAL_AGENTS, 0.25)
