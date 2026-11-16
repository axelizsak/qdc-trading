"""The system backtest on made-up prices: the competition's rules hold, and a variant changes only what it says."""
from datetime import date

import pandas as pd
import pytest

from agents import portfolio_config as pm_cfg
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
    assert set(first) <= set(data.sheets)  # every decision day has its technical sheet
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


def test_score_ideas_only_reread_the_sheet(market):
    data, starts = market
    sheet = data.sheets[starts[0]]
    rule = sb.rule_score(sheet)
    assert set(rule.unique()) <= {1.0, 0.5, 0.0, -1.0}
    skipped = sb.skip_overbought(sheet)
    overbought = (sheet["mr_setup"] == "avoid") & (rule > 0)
    assert (skipped[overbought] == 0).all() and (skipped[~overbought] == rule[~overbought]).all()
    dipped = sb.buy_the_dip(sheet)
    assert (dipped >= rule).all() and (dipped[sheet["mr_setup"] == "buy"] >= 0.5).all()
    ranked = sb.momentum_rank(sheet)
    assert (ranked[rule <= 0] == rule[rule <= 0]).all() and (ranked[rule > 0] <= rule[rule > 0]).all()
    assert set(v.score for v in sb.VARIANTS) <= set(sb.SCORES)


def test_the_summary_has_one_row_per_variant(market):
    data, starts = market
    variants = [v for v in sb.VARIANTS if v.name in ("current", "skip_overbought")]
    results = sb.run(data, starts[:2], variants, workers=1)
    assert len(results) == 4 and results["broken_rules"].sum() == 0
    summary = sb.summarise(results, data, variants)
    assert summary["variant"].tolist() == ["current", "skip_overbought"] and (summary["windows"] == 2).all()
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


def test_the_other_fill_rule_is_a_variant(market):
    data, starts = market
    same_day = next(v for v in sb.VARIANTS if v.name == "fill_same_day")
    row = sb.run_window(data, starts[0], same_day)
    assert row["broken_rules"] == 0  # the larger cash cushion covers a close above the last one
    assert settings.FILL_AT_PREVIOUS_CLOSE is True  # put back after the run
    current = sb.run_window(data, starts[0], sb.VARIANTS[0])
    assert row["ret"] != current["ret"]  # a different fill price gives a different result
