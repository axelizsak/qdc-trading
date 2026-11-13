"""Tests of the technical agent's momentum strategy (agents/technical.py)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import strategy_config as cfg
from agents import technical
from data.features import compute_features
from data.mock import mock_prices
from data.universe import load_universe


# ---------- Question 1: direction ------------------------------------------------------------

def test_trend_answer_up_down_none():
    assert technical.trend_answer(110, 100, 0.05) == "up"
    assert technical.trend_answer(90, 100, -0.05) == "down"
    assert technical.trend_answer(110, 100, -0.05) == "none"    # above its average but falling
    assert technical.trend_answer(90, 100, 0.05) == "none"


def test_trend_answer_missing_value():
    assert technical.trend_answer(110, np.nan, 0.05) is None


def test_reverse_answer():
    assert technical.reverse_answer("up") == "down"
    assert technical.reverse_answer("down") == "up"
    assert technical.reverse_answer("none") == "none"
    assert technical.reverse_answer(None) is None


def test_momentum_setup():
    assert technical.momentum_setup("up", "up", "up") == "buy"
    assert technical.momentum_setup("none", "up", "up") == "buy"      # short term only must not be down
    assert technical.momentum_setup("down", "up", "up") == "none"
    assert technical.momentum_setup("up", "none", "up") == "none"
    assert technical.momentum_setup("up", "down", "down") == "avoid"
    assert technical.momentum_setup(None, "up", "up") == "none"       # missing data: no setup


# ---------- Question 2: strength -------------------------------------------------------------

def test_adx_label():
    assert technical.adx_label(30) == "strong"
    assert technical.adx_label(22) == "moderate"
    assert technical.adx_label(15) == "absent"
    assert technical.adx_label(np.nan) is None


def test_confirmations_follow_the_direction():
    assert technical.momentum_confirms(60, "up") is True
    assert technical.momentum_confirms(60, "down") is False
    assert technical.volume_confirms(-1000, "down") is True
    assert technical.volume_confirms(1000, None) is None
    assert technical.count_confirmations(True, None) == 1


def test_trend_quality_of_a_perfect_rise():
    # +0.1 % every day, no noise: R squared = 1, annualized slope = 1.001 ** 252 - 1
    prices = pd.Series(100 * 1.001 ** np.arange(120))
    assert technical.trend_quality(prices) == pytest.approx(1.001 ** 252 - 1, rel=1e-6)


def test_trend_quality_is_lower_for_a_choppy_rise():
    days = np.arange(120)
    smooth = pd.Series(100 * 1.001 ** days)
    choppy = pd.Series(100 * 1.001 ** days * (1 + 0.05 * np.sin(days)))
    assert technical.trend_quality(choppy) < technical.trend_quality(smooth)


def test_trend_quality_needs_90_prices():
    assert np.isnan(technical.trend_quality(pd.Series(np.linspace(100, 110, 50))))


def test_momentum_13612_is_the_average_of_four_returns():
    prices = pd.Series(np.arange(1, 301, dtype=float))     # 1, 2, ..., 300
    expected = (300 / 279 - 1 + 300 / 237 - 1 + 300 / 174 - 1 + 300 / 48 - 1) / 4
    assert technical.momentum_13612(prices) == pytest.approx(expected)
    assert np.isnan(technical.momentum_13612(prices.iloc[:200]))


# ---------- Suggested signal -----------------------------------------------------------------

def test_suggest_signal():
    assert technical.suggest_signal("buy", "up", "strong") == "buy"
    assert technical.suggest_signal("buy", "up", None) == "buy"          # no ADX (BEL 20): not blocked
    assert technical.suggest_signal("buy", "down", "strong") == "watch"  # market below its SMA 200
    assert technical.suggest_signal("buy", "up", "absent") == "watch"    # no real trend
    assert technical.suggest_signal("none", "up", "strong") == "neutral"
    assert technical.suggest_signal("avoid", "up", "strong") == "avoid"


# ---------- The whole agent on prices made up for the test -----------------------------------

def made_up_prices() -> pd.DataFrame:
    """Random prices for every ETF, then a steady rise for WORLD and S&P 500 and a steady fall for GOLD."""
    prices = mock_prices(days=320, seed=1)
    days = np.arange(320)
    shapes = {"WORLD": 1.001 ** days, "S&P 500": 1.0015 ** days, "GOLD": 0.999 ** days}
    for ticker, shape in shapes.items():
        rows = prices["ticker"] == ticker
        close = 100 * shape * (1 + 0.002 * np.sin(days))      # small wiggle so ADX and RSI are defined
        prices.loc[rows, "close_eur"] = close
        prices.loc[rows, "adj_close_eur"] = close
        prices.loc[rows, "open_eur"] = close
        prices.loc[rows, "high_eur"] = close * 1.005
        prices.loc[rows, "low_eur"] = close * 0.995
    return prices


@pytest.fixture(scope="module")
def sheet_and_signals():
    prices = made_up_prices()
    features = compute_features(prices)
    as_of = pd.to_datetime(prices["date"]).max().date()
    sheet = technical.build_sheet(prices, features, as_of)
    signals = technical.run(prices, features, as_of)
    return sheet, signals


def test_rising_etf_gets_a_buy_setup(sheet_and_signals):
    sheet, _ = sheet_and_signals
    sp500 = sheet[sheet["ticker"] == "S&P 500"].iloc[0]
    assert sp500["momentum_setup"] == "buy"
    assert sp500["market_trend"] == "up"


def test_falling_etf_gets_an_avoid_setup(sheet_and_signals):
    sheet, _ = sheet_and_signals
    gold = sheet[sheet["ticker"] == "GOLD"].iloc[0]
    assert gold["momentum_setup"] == "avoid"
    assert gold["suggested_signal"] == "avoid"


def test_inverse_etf_is_read_on_its_index_and_reversed(sheet_and_signals):
    sheet, _ = sheet_and_signals
    short = sheet[sheet["ticker"] == "S&P 500 SHORT"].iloc[0]
    assert short["read_on"] == "S&P 500"
    assert short["momentum_setup"] == "avoid"       # the S&P 500 rises, so its inverse is avoided


def test_bonds_and_cash_are_not_scored(sheet_and_signals):
    sheet, _ = sheet_and_signals
    universe = load_universe()
    skipped = universe[(~universe["buyable"]) | (universe["family"] == "cash")]["ticker"]
    for ticker in skipped:
        assert ticker not in set(sheet["ticker"])


def test_run_returns_valid_signals(sheet_and_signals):
    _, signals = sheet_and_signals
    assert set(signals["agent"]) == {technical.AGENT_DET}
    for score in signals["score"]:
        assert score in cfg.SIGNAL_SCORES.values()
    assert signals["confidence"].between(0, 1).all()
