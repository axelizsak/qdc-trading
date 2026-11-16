"""The one learning rule (agents/learning.py) on made-up results and prices."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import learning, technical
from data import config as settings
from data.features import compute_features
from data.mock import mock_prices
from data.schemas import validate_technical_answers

DAY = date(2026, 12, 7)


def results(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(session, expert, result) -> the results table the rule reads."""
    return pd.DataFrame([{"session": s, "call_date": s, "name": n, "result": r} for s, n, r in rows],
                        columns=learning.RESULTS)


def sessions(n: int) -> list[str]:
    return [d.date().isoformat() for d in pd.bdate_range("2026-11-02", periods=n)]


def today(frame: pd.DataFrame) -> dict[str, float]:
    rows = frame[frame["date"] == DAY.isoformat()]
    return dict(zip(rows["name"], rows["weight"]))


def test_starting_weights_split_each_family_between_its_agents():
    assert learning.starting_weights(["technical_det", "macro_market", "macro_etf"]) == {
        "technical_det": 0.5, "macro_market": 0.25, "macro_etf": 0.25}
    assert learning.starting_weights(settings.SIGNAL_AGENTS) == dict.fromkeys(settings.SIGNAL_AGENTS, 0.25)
    assert learning.starting_weights(["macro_etf"]) == {"macro_etf": 1.0}
    assert learning.starting_weights([]) == {}


def test_without_evidence_the_weights_are_the_starting_ones():
    frame, multiplier = learning.daily_weights(results([]), learning.shares({"a": 0.6, "b": 0.4}), DAY, known=["a", "b"])
    assert today(frame) == pytest.approx({"a": 0.6, "b": 0.4}) and multiplier == {"a": 1.0, "b": 1.0}
    calls = results([(s, n, np.nan) for s in sessions(5) for n in ("a", "b")])  # present, but no call: no change
    assert today(learning.daily_weights(calls, learning.shares({"a": 0.6, "b": 0.4}), DAY)[0]) == pytest.approx(
        {"a": 0.6, "b": 0.4})


def test_each_session_the_expert_that_earned_more_gains_weight_inside_the_bounds():
    rows = [(s, "good", 0.01) for s in sessions(20)] + [(s, "bad", -0.01) for s in sessions(20)]
    frame, _ = learning.daily_weights(results(rows), learning.shares({"good": 0.5, "bad": 0.5}), DAY)
    good = frame[frame["name"] == "good"]["weight"].tolist()
    assert good[0] == pytest.approx(0.5) and good == sorted(good) and good[-1] > 0.6  # one step after each session
    first = np.exp(settings.LEARNING_SPEED * 0.01) / (np.exp(settings.LEARNING_SPEED * 0.01) + np.exp(-settings.LEARNING_SPEED * 0.01))
    assert good[1] == pytest.approx(first)  # e^(speed x result), then scaled to sum to 1
    assert frame.groupby("date")["weight"].sum().tolist() == pytest.approx([1.0] * frame["date"].nunique())
    big = [(s, "good", 0.5) for s in sessions(20)] + [(s, "bad", -0.5) for s in sessions(20)]
    weights = today(learning.daily_weights(results(big), learning.shares({"good": 0.5, "bad": 0.5}), DAY)[0])
    assert weights["good"] == pytest.approx(settings.WEIGHT_MAX) and weights["bad"] == pytest.approx(1 - settings.WEIGHT_MAX)


def test_a_weight_at_its_bound_can_recover_at_once():
    rows = [(s, "a", -0.5) for s in sessions(10)] + [(s, "b", 0.5) for s in sessions(10)]
    rows += [(s, "a", 0.05) for s in sessions(12)[10:]] + [(s, "b", -0.05) for s in sessions(12)[10:]]
    frame, _ = learning.daily_weights(results(rows), learning.shares({"a": 0.5, "b": 0.5}), DAY)
    a = frame[frame["name"] == "a"]["weight"].tolist()  # 12 sessions, then today
    assert a[10] == pytest.approx(1 - settings.WEIGHT_MAX)  # held at the floor after ten bad sessions
    assert a[12] > a[11] > a[10]  # and it rises from the first good one: no debt to pay back first


def test_a_newcomer_joins_at_its_starting_share_and_an_absent_expert_keeps_its_record():
    days = sessions(8)
    rows = [(s, "a", 0.02) for s in days[:4]] + [(s, "b", -0.02) for s in days]   # a wins, then is away 4 sessions
    rows += [(s, "c", np.nan) for s in days[4:]]                                    # c arrives on the 5th
    start = learning.shares({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    frame, multiplier = learning.daily_weights(results(rows), start, DAY)
    first_c = frame[(frame["name"] == "c")].iloc[0]
    b_then = frame[(frame["name"] == "b") & (frame["date"] == first_c["date"])]["weight"].iloc[0]
    assert first_c["date"] == days[4] and first_c["weight"] > b_then  # b had lost weight against a; c starts afresh
    assert multiplier["a"] > 1 and today(frame)["a"] > 1 / 3  # a was away, its record is still there today
    alone = learning.daily_weights(results([(s, "x", -0.5) for s in days]), learning.shares({"x": 1.0}), DAY)[1]
    assert alone == {"x": 1.0}  # an expert alone is compared with nobody


def test_calls_are_judged_on_the_session_an_order_fills_into_and_the_latest_counts():
    prices = mock_prices(end="2026-12-08", days=30, seed=2)  # Friday 4 and Monday 7 December are quote days
    table = pd.DataFrame(prices).pivot(index="date", columns="ticker", values="adj_close_eur")
    table.index = table.index.astype(str)
    monday = table.loc["2026-12-07"] / table.loc["2026-12-04"] - 1
    up, down = monday.idxmax(), monday.idxmin()
    calls = pd.DataFrame([
        {"date": "2026-12-05", "name": "x", "ticker": up, "score": -1.0},   # Saturday: overwritten by Sunday's call
        {"date": "2026-12-06", "name": "x", "ticker": up, "score": 1.0},    # Sunday
        {"date": "2026-12-06", "name": "x", "ticker": down, "score": -1.0},
        {"date": "2026-12-08", "name": "x", "ticker": up, "score": 1.0}])   # Tuesday: its close is not known yet
    out = learning.call_results(calls, prices, date(2026, 12, 8))
    assert out[["session", "call_date"]].values.tolist() == [["2026-12-07", "2026-12-06"]]
    pair = monday[[up, down]]
    assert out["result"].iloc[0] == pytest.approx(((pair[up] - pair.mean()) - (pair[down] - pair.mean())) / 2)
    assert learning.call_results(calls, prices, date(2026, 12, 7)).empty  # on Monday morning, Monday's close is unknown


def test_a_portfolio_earns_its_return_and_cash_earns_nothing():
    prices = mock_prices(end="2026-12-08", days=30, seed=2)
    table = prices.pivot(index="date", columns="ticker", values="adj_close_eur")
    table.index = table.index.astype(str)
    gold = table.loc["2026-12-07", "GOLD"] / table.loc["2026-12-04", "GOLD"] - 1
    portfolios = pd.DataFrame([{"date": "2026-12-07", "name": "llm", "ticker": "GOLD", "weight": 0.5}])
    out = learning.portfolio_results(portfolios, prices, date(2026, 12, 8))
    assert out["result"].iloc[0] == pytest.approx(0.5 * gold)


def test_the_technical_answers_never_see_the_day_they_are_dated():
    prices = mock_prices(days=300, end="2026-12-04", seed=7)
    features = compute_features(prices)
    last = date(2026, 12, 4)
    found = validate_technical_answers(learning.technical_answers(prices, features, [last], date(2026, 12, 5)))
    assert set(found["date"]) == {last.isoformat()} and found["session_return"].notna().all()
    table = prices.pivot(index="date", columns="ticker", values="adj_close_eur")
    table.index = table.index.astype(str)
    gold = found.set_index("ticker").loc["GOLD", "session_return"]
    assert gold == pytest.approx(table.loc["2026-12-04", "GOLD"] / table.loc["2026-12-03", "GOLD"] - 1)
    altered = prices.copy()
    on_the_day = pd.to_datetime(altered["date"]).dt.date >= last
    altered.loc[on_the_day, ["close_eur", "adj_close_eur"]] *= 3  # a different day itself must not change its answers
    again = learning.technical_answers(altered, compute_features(altered), [last], date(2026, 12, 5))
    keep = [c for c in found.columns if c != "session_return"]
    pd.testing.assert_frame_equal(found[keep].reset_index(drop=True), again[keep].reset_index(drop=True))


def test_the_answers_results_are_the_same_sum_as_for_any_score():
    answers = pd.DataFrame({"date": "2026-12-07", "ticker": ["A", "B", "C", "D"],
                            "suggested_signal": ["buy", "avoid", "neutral", "watch"], "mr_setup": ["none", "buy", "avoid", "n/a"],
                            "momentum_13612": [0.3, -0.1, 0.1, np.nan], "confirmations": 1, "trend_strength": "strong",
                            "session_return": [0.02, -0.01, 0.00, 0.01]})
    out = learning.answer_results(answers).set_index("name")["result"]
    scores = technical.signal_scores(answers)
    for name in settings.TECHNICAL_SIGNALS:
        assert out[name] == pytest.approx(learning._score_result(scores[name], answers["session_return"]))
    assert scores["momentum"].tolist() == [1.0, -1.0, 0.0, 0.0]  # ranked within the day; no momentum, no call
