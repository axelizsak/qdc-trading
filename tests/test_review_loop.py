"""The review loop on made-up logs: one agent that is always right, one that is always wrong."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import logbook, macro_tape
from data import config as settings
from data.mock import mock_prices
from data.schemas import validate_agent_weights
from graph import pipeline, review_loop

DAY = date(2026, 12, 7)


@pytest.fixture
def prices():
    return mock_prices(end=DAY.isoformat(), days=40, seed=3)


def _log_scores(logs, prices, days: int = 8) -> list[str]:
    """Log `days` past days of scores: technical_det knows the next 3 sessions, macro_market says the opposite."""
    table = macro_tape.closes(prices, DAY)
    dates = list(table.index[-days - 4:-4])  # early enough for 3 sessions to be known before DAY
    rows = []
    for day in dates:
        at = table.index.get_loc(day)
        future = table.iloc[at + 2] / table.iloc[at - 1] - 1  # from the close before `day`, 3 sessions on
        for ticker, ret in future.items():
            for agent, sign in (("technical_det", 1), ("macro_market", -1)):
                rows.append({"ticker": ticker, "date": day, "agent": agent, "score": float(sign * np.sign(ret)) * 0.8,
                             "confidence": 0.5, "rationale": "made up"})
    rows.append({"ticker": "WORLD", "date": dates[-1], "agent": "macro_etf", "score": 0.0, "confidence": 0.0,
                 "rationale": "agent failed: API down"})  # a failed agent is no evidence
    logs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(logs / "signals.csv", index=False)
    return dates


def test_starting_weights_split_each_family_between_its_agents():
    assert review_loop.starting_weights(["technical_det", "macro_market", "macro_etf"]) == {
        "technical_det": 0.5, "macro_market": 0.25, "macro_etf": 0.25}
    assert review_loop.starting_weights(["macro_etf"]) == {"macro_etf": 1.0}
    assert review_loop.starting_weights([]) == {}


def test_the_agent_that_was_right_gains_weight_and_it_is_logged(tmp_path, prices):
    _log_scores(tmp_path, prices)
    result = review_loop.review(DAY, prices, tmp_path, backfill=False)

    weights = result["weights"]
    assert set(weights) == {"technical_det", "macro_market"}  # the failed macro_etf row is no evidence
    assert weights["technical_det"] > 0.59 > 0.41 > weights["macro_market"]  # from 50/50, inside the 60% bound
    logged = validate_agent_weights(pd.read_csv(tmp_path / "agent_weights.csv")).set_index("agent")
    assert logged["weight"].to_dict() == pytest.approx(weights) and set(logged["date"]) == {DAY.isoformat()}
    # the logged hit rate is the Portfolio Manager's: the mean over 1 and 3 sessions (perfect on 3, not on 1)
    assert logged.loc["technical_det", "hit_rate"] > 0.8 and logged.loc["macro_market", "hit_rate"] < 0.2

    lessons = result["lessons"]
    assert lessons.startswith(f"## Review of {DAY}")
    assert "- technical_det: 50% -> 60% (8 days judged, rank correlation +" in lessons
    assert "- macro_market: calls up right 0% (n=" in lessons and "- technical_det: calls up right 100% (n=" in lessons
    assert "**Largest misses of the last judged day**" in lessons and "- macro_market " in lessons.split("Largest misses")[1]


def test_the_journal_keeps_the_newest_review_first_and_a_rerun_changes_nothing(tmp_path, prices):
    _log_scores(tmp_path, prices)
    first = review_loop.review(DAY, prices, tmp_path, backfill=False)
    ledger = len(logbook.load(tmp_path))
    again = review_loop.review(DAY, prices, tmp_path, backfill=False)
    assert again["journal"] == first["journal"] and again["journal"].count("## Review of") == 1
    assert len(logbook.load(tmp_path)) == ledger and logbook.verify(tmp_path) == []
    kinds = logbook.load(tmp_path)["kind"].tolist()
    assert kinds == ["reassessment", "lesson"]

    for offset in range(1, review_loop.REVIEWS_KEPT + 2):  # later days: the journal keeps the newest ones
        review_loop.save_journal(tmp_path, date(2026, 12, 7 + offset), [f"## Review of 2026-12-{7 + offset:02d}", "", "x"])
    journal = (tmp_path / "lessons.md").read_text(encoding="utf-8")
    assert journal.startswith("# Lessons journal") and journal.count("## Review of") == review_loop.REVIEWS_KEPT
    assert journal.index("## Review of 2026-12-13") < journal.index("## Review of 2026-12-12")
    assert f"## Review of {DAY}" not in journal  # the oldest ones leave the file (they stay in git and the logbook)


def test_todays_scores_are_not_judged_and_empty_logs_give_an_empty_review(tmp_path, prices):
    result = review_loop.review(DAY, prices, tmp_path / "empty", backfill=False)
    assert result["weights"] == {} and "No score can be judged yet" in result["lessons"]
    assert not (tmp_path / "empty" / "agent_weights.csv").exists()

    today = pd.DataFrame([{"ticker": "WORLD", "date": DAY.isoformat(), "agent": "technical_det", "score": 1.0,
                           "confidence": 1.0, "rationale": "today"}])
    (tmp_path / "today").mkdir()
    today.to_csv(tmp_path / "today" / "signals.csv", index=False)
    assert review_loop.evidence(tmp_path / "today", DAY).empty


def test_the_day_starts_with_the_review_and_the_agents_get_its_lessons(tmp_path, prices, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setitem(pipeline.MOCK, "data", lambda as_of: (prices, pipeline.mock.mock_features(as_of.isoformat())))
    _log_scores(logs, prices)
    seen = {}

    def technical(prices, features, as_of, lessons):
        seen["lessons"] = lessons
        return pipeline.mock.mock_signals(as_of.isoformat(), agents=pipeline.TECHNICAL_AGENTS)

    monkeypatch.setitem(pipeline.MOCK, "technical", technical)
    pending = pipeline.run_day(DAY, logs)
    assert pending["incidents"] == []
    assert f"## Review of {DAY}" in seen["lessons"] and "- technical_det: 50% -> 60%" in seen["lessons"]
    assert (logs / "lessons.md").exists() and (logs / "agent_weights.csv").exists()
    # the cockpit shows what the review learned, over the agents that scored today
    shown = pipeline.load_agent_weights(logs, DAY, ["technical_det", "macro_market"])
    assert shown["technical_det"] == pytest.approx(0.6, abs=0.01)


def test_a_failing_review_is_an_incident_and_the_day_goes_on(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("review broke")

    monkeypatch.setattr(pipeline.review_loop, "review", broken)
    pending = pipeline.run_day(DAY, tmp_path)
    assert [i["step"] for i in pending["incidents"]] == ["review"]
    assert len(pending["proposed_orders"]) > 0
