"""The system backtest on made-up prices: the competition's rules hold, and a variant changes only what it says."""
from datetime import date

import pandas as pd
import pytest

from agents import learning, technical
from agents import portfolio_config as pm_cfg
from agents import strategy_config as tech_cfg
from backtest import system_backtest as sb
from data import config as settings
from data.features import compute_features
from data.mock import mock_prices

SINCE = date(2026, 11, 2)  # a Monday: two windows fit before the prices end


@pytest.fixture(scope="module")
def market():
    prices = mock_prices(end="2026-12-04", days=380, seed=5)
    return sb.prepare(prices, compute_features(prices), SINCE)


def test_windows_follow_the_competition_calendar(market):
    data, starts = market
    assert starts == [date(2026, 11, 2), date(2026, 11, 9), date(2026, 11, 16)] and all(d.weekday() == 0 for d in starts)
    first = [starts[0] + pd.Timedelta(days=offset) for offset in sb.DECISION_OFFSETS]
    assert [d.strftime("%a") for d in first] == ["Mon", "Thu", "Mon", "Thu", "Mon"]
    assert set(first) <= set(data.answers)  # every decision day has its technical answers
    assert set(data.sessions) <= set(data.answers) and min(data.sessions) >= SINCE  # and every session, for the learning
    # a decision only sees the days before it, and fills at the last close before it
    assert data.before(starts[0])["date"].max() < pd.Timestamp(starts[0])
    last_before = data.closes[data.closes.index < pd.Timestamp(starts[0])].iloc[-1]
    assert data.fill(starts[0])["WORLD"] == last_before["WORLD"]


def test_a_window_trades_within_the_rules_and_is_valued_against_the_benchmark(market):
    data, starts = market
    current = next(v for v in sb.VARIANTS if v.name == "current")
    row = sb.run_window(data, starts[0], current)
    assert row["broken_rules"] == 0 and row["failed_days"] == 0
    assert 0 < row["orders"] <= settings.MAX_ORDERS_PER_DAY * len(sb.DECISION_OFFSETS)
    assert row["largest"] <= pm_cfg.MAX_POSITION + 0.10  # the limit at the fill, plus what the price did since
    assert row["excess"] == pytest.approx(row["ret"] - row["bench_ret"])
    assert row["value"] == pytest.approx(settings.START_CAPITAL_EUR * (1 + row["ret"]), abs=0.01)
    world = data.close(date.fromisoformat(row["end"]))["WORLD"] / data.fill(starts[0])["WORLD"] - 1
    assert row["bench_ret"] == pytest.approx(world)


def test_a_variant_changes_its_setting_only_for_its_own_run(market):
    data, starts = market
    cash = sb.Variant("all_cash", "nothing may be bought", {"portfolio_config.MAX_POSITION": 0.0})
    before = (pm_cfg.MAX_POSITION, pm_cfg.VALUATION_END, list(settings.TRADING_DAYS))
    row = sb.run_window(data, starts[0], cash)
    assert row["orders"] == 0 and row["ret"] == 0.0 and row["invested"] == 0.0
    assert (pm_cfg.MAX_POSITION, pm_cfg.VALUATION_END, list(settings.TRADING_DAYS)) == before  # all put back


def test_the_technical_weights_are_learned_from_the_sessions_before_each_decision(market):
    data, starts = market
    current = next(v for v in sb.VARIANTS if v.name == "current")
    with sb.overrides(current.settings):
        frame = sb.technical_weights(data, current)
    day = starts[1]
    used = learning.weights_before(frame, day)
    first = frame["date"].min()
    assert first == min(data.sessions).isoformat() and sum(used.values()) == pytest.approx(1)
    assert set(frame[frame["date"] == first]["weight"].round(6)) == {round(1 / 3, 6)}  # 1/3 each on the first session
    signals = sb.technical_signals(data, current, day)
    answers = data.answers[day]
    assert signals["score"].tolist() == pytest.approx(technical.blend(technical.signal_scores(answers), used).tolist())
    trend_only = next(v for v in sb.VARIANTS if v.name == "trend_only")
    with sb.overrides(trend_only.settings):
        only = sb.technical_signals(data, trend_only, day)
    assert only["score"].tolist() == answers["suggested_signal"].map(tech_cfg.SIGNAL_SCORES).tolist()  # the old rule


def test_the_summary_has_one_row_per_variant(market):
    data, starts = market
    variants = [v for v in sb.VARIANTS if v.name in ("current", "trend_only")]
    results = sb.run(data, starts[:2], variants, workers=1)
    assert len(results) == 4 and results["broken_rules"].sum() == 0
    summary = sb.summarise(results, variants)
    assert summary["variant"].tolist() == ["current", "trend_only"] and (summary["windows"] == 2).all()
    assert summary["mean_excess"].tolist() == pytest.approx((summary["mean_ret"] - summary["mean_bench"]).tolist())
    assert "current" in sb.show(summary)


def test_the_fill_rule_study_measures_what_ordering_after_the_open_is_worth(market):
    data, _ = market
    study = sb.fill_rule_study(data)
    assert len(study) > 0 and set(pd.to_datetime(study["day"]).dt.day_of_week) <= {0, 3}  # Mondays and Thursdays
    # the 5 biggest opening gaps are above the fill price at the open, and the day's 5 best ETFs are the ceiling
    assert (study["morning_gap"] > 0).all()
    assert (study["close_picks"] >= study["morning_picks"] - 1e-12).all() and (study["close_picks"] >= study["average_etf"]).all()
    assert "trading days of the morning picks" in sb.show_fill_rule(study)


def test_the_daily_book_carries_over_and_is_valued_at_every_close(market):
    data, _ = market
    days = sb.decision_days(data.closes, SINCE)
    assert set(days) <= set(data.answers) and {d.weekday() for d in days} == {0, 3}
    book = sb.run_continuous(data, sb.VARIANTS[0])
    closes = data.closes[data.closes.index >= data.fill_day(days[0])]
    assert book["date"].tolist() == [d.date().isoformat() for d in closes.index]  # every close, from the first fill
    first = book.iloc[0]
    assert first["value"] == settings.START_CAPITAL_EUR and first["bench"] == settings.START_CAPITAL_EUR
    assert book["broken_rules"].sum() == 0 and book["failed_days"].sum() == 0
    assert 0 < book["orders"].sum() <= settings.MAX_ORDERS_PER_DAY * len(days)
    assert book["orders"].max() <= settings.MAX_ORDERS_PER_DAY * 2  # at most two decisions between two closes
    assert book["invested"].between(0, 1).all() and book["invested"].iloc[-1] > 0  # it holds ETFs across decisions
    world = closes["WORLD"].ffill()
    assert book["bench"].iloc[-1] == pytest.approx(settings.START_CAPITAL_EUR * world.iloc[-1] / world.iloc[0])


def test_the_daily_summary_ranks_on_the_sharpe_ratio(market):
    data, _ = market
    variants = [sb.VARIANTS[0], sb.Variant("idle_in_cash", "idle budget stays in cash",
                                           {"portfolio_config.IDLE_TO_BENCHMARK": 0.0})]
    daily = sb.run_daily(data, variants, workers=1)
    summary = sb.summarise_daily(daily, variants)
    assert summary["variant"].tolist() == ["current", "idle_in_cash"]
    assert summary["sharpe_vs_first"].iloc[0] == 0 and pd.isna(summary["t_sharpe_vs_first"].iloc[0])
    assert summary["sharpe"].notna().all() and (summary["max_drawdown"] <= 0).all()
    returns = daily[daily["variant"] == "current"]["value"].pct_change().dropna()
    riskfree = daily[daily["variant"] == "current"]["riskfree"].pct_change().dropna()
    excess = returns - riskfree
    assert summary["sharpe"].iloc[0] == pytest.approx(excess.mean() / excess.std() * 252 ** 0.5)
    assert "benchmark" in sb.show_daily(summary)


def test_the_other_fill_rule_is_a_variant(market):
    data, starts = market
    same_day = next(v for v in sb.VARIANTS if v.name == "fill_same_day")
    row = sb.run_window(data, starts[0], same_day)
    assert row["broken_rules"] == 0  # the larger cash cushion covers a close above the last one
    assert settings.FILL_AT_PREVIOUS_CLOSE is True  # put back after the run
    current = sb.run_window(data, starts[0], sb.VARIANTS[0])
    assert row["ret"] != current["ret"]  # a different fill price gives a different result
