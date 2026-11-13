"""Data agent, step 3: indicators (data/features.py). Small made-up series whose result is known. No network."""
import numpy as np
import pandas as pd
import pytest

from data import data_agent, features
from data.schemas import FEATURES, PRICES


def make_prices(close, ticker="WORLD", volume=None, spread=0.01, start="2025-01-01", adj_close=None) -> pd.DataFrame:
    """Prices table of one ETF from a list of closes. High/low = close +/- spread %.
    adj_close: adjusted closes; by default equal to the closes (accumulating ETF)."""
    close = np.array(close, dtype=float)
    days = pd.bdate_range(start, periods=len(close))
    if volume is None:
        volume = np.full(len(close), 1000.0)
    if adj_close is None:
        adj_close = close
    return pd.DataFrame({
        "date": days.date,
        "ticker": ticker,
        "open_eur": close,
        "high_eur": close * (1 + spread),
        "low_eur": close * (1 - spread),
        "close_eur": close,
        "adj_close_eur": np.array(adj_close, dtype=float),
        "volume": np.array(volume, dtype=float),
    })


def indicators(close, **kwargs) -> pd.DataFrame:
    return features.add_indicators(make_prices(close, **kwargs))


RISING = 100 * 1.01 ** np.arange(300)            # +1 % every day, 300 days
FALLING = 100 * 0.99 ** np.arange(300)


# ---------- The table ----------

def test_indicators_are_added_on_the_right_of_the_prices():
    prices = make_prices(RISING)
    before = prices.copy()
    table = features.add_indicators(prices)
    assert list(table.columns) == PRICES + features.FEATURE_COLUMNS
    assert len(features.FEATURE_COLUMNS) == 37
    assert len(table) == len(prices)
    pd.testing.assert_frame_equal(prices, before)                   # the input is not changed


def test_compute_features_gives_the_shared_long_format_without_empty_values():
    long = data_agent.compute_features(make_prices(RISING))
    assert list(long.columns) == FEATURES
    assert long["value"].notna().all()
    assert set(long["feature"]) == set(features.FEATURE_COLUMNS)


def test_each_etf_is_computed_on_its_own_quotes():
    world = make_prices(RISING, ticker="WORLD")
    gold = make_prices(FALLING, ticker="GOLD")
    gold = gold.drop(index=[100, 101])                              # two days without a quote
    table = features.add_indicators(pd.concat([world, gold], ignore_index=True))
    gold_rows = table[table["ticker"] == "GOLD"]
    assert len(gold_rows) == 298                                    # nothing filled
    assert (gold_rows["ret_10d"].dropna() < 0).all()
    assert (table[table["ticker"] == "WORLD"]["ret_10d"].dropna() > 0).all()


def test_not_enough_history_gives_empty_values():
    table = indicators(RISING[:30])
    assert table["sma_20"].notna().sum() == 11
    assert table["sma_50"].isna().all()
    assert table["sma_200"].isna().all()
    assert table["ret_12m_ex_1m"].isna().all()


# ---------- Adjusted close ----------

def dividend_history():
    """A distributing ETF: flat at 100, then a 4 EUR dividend on day 80 (raw close drops to 96).
    The adjusted close is 96 before the payment (100 x 0.96) and 96 after: no move at all."""
    close = np.r_[np.full(80, 100.0), np.full(40, 96.0)]
    adj_close = np.full(120, 96.0)
    return close, adj_close


def test_indicators_use_the_adjusted_close_so_a_dividend_is_not_a_drop():
    close, adj_close = dividend_history()
    table = indicators(close, adj_close=adj_close)
    last = table.iloc[-1]
    assert last["worst_day_60"] == pytest.approx(0.0)              # the raw close fell 4 %, the market did not
    assert last["drawdown_60"] == pytest.approx(0.0)
    assert last["ret_3m"] == pytest.approx(0.0)
    assert last["sma_50"] == pytest.approx(96.0)                   # levels in adjusted euros
    assert table["close_eur"].iloc[0] == pytest.approx(100.0)      # the real close is kept for valuation


def test_high_and_low_are_adjusted_with_the_same_ratio():
    close, adj_close = dividend_history()
    table = indicators(close, adj_close=adj_close, spread=0.01)
    assert table["atr_14"].iloc[-1] == pytest.approx(96 * 0.02, rel=1e-6)   # no fake gap on the payment day


def test_without_adjusted_close_the_whole_etf_uses_the_raw_close():
    close, adj_close = dividend_history()
    adj_close = adj_close.copy()
    adj_close[5] = np.nan                                          # one missing day: never mix the two series
    table = indicators(close, adj_close=adj_close)
    assert table["worst_day_60"].iloc[-1] == pytest.approx(-0.04)


# ---------- Moving averages and momentum ----------

def test_sma_ema_and_slope():
    close = np.arange(1, 61, dtype=float)                           # 1, 2, ..., 60
    table = indicators(close)
    assert table["sma_5"].iloc[-1] == pytest.approx(np.mean(close[-5:]))
    assert table["sma_5"].iloc[3] != table["sma_5"].iloc[3]          # NA before 5 quotes
    assert table["ema_5"].iloc[:4].isna().all()
    assert table["ema_5"].iloc[-1] > table["sma_5"].iloc[-1] - 1e-9  # rising: the EMA follows closer
    assert table["sma_50_slope_10"].iloc[-1] == pytest.approx(table["sma_50"].iloc[-1] / table["sma_50"].iloc[-11] - 1)


def test_returns_on_several_horizons():
    table = indicators(RISING)
    last = table.iloc[-1]
    assert last["ret_10d"] == pytest.approx(1.01 ** 10 - 1)
    assert last["ret_1m"] == pytest.approx(1.01 ** 21 - 1)
    assert last["ret_3m"] == pytest.approx(1.01 ** 63 - 1)
    assert last["ret_6m"] == pytest.approx(1.01 ** 126 - 1)
    assert last["ret_12m_ex_1m"] == pytest.approx(1.01 ** (252 - 21) - 1)   # the last month is skipped


# ---------- Oscillators ----------

def test_rsi_by_hand():
    # changes +1, +1, -1. Wilder average over 2 days: gains 1, 1, 0.5 / losses 0, 0, 0.5
    table = indicators([1.0, 2.0, 3.0, 2.0])
    assert table["rsi_2"].iloc[:2].isna().all()
    assert table["rsi_2"].iloc[2] == pytest.approx(100.0)
    assert table["rsi_2"].iloc[3] == pytest.approx(50.0)


def test_rsi_extremes():
    assert indicators(RISING)["rsi_14"].iloc[-1] == pytest.approx(100.0)
    assert indicators(FALLING)["rsi_14"].iloc[-1] == pytest.approx(0.0)
    assert indicators(np.full(30, 50.0))["rsi_14"].iloc[-1] == pytest.approx(50.0)


def test_zscore_and_bollinger():
    close = np.r_[np.full(19, 100.0), 110.0]
    table = indicators(close)
    std = np.std(close)                                             # population std, as Bollinger
    assert table["zscore_20"].iloc[-1] == pytest.approx((110 - close.mean()) / std)
    assert table["bb_upper_20"].iloc[-1] == pytest.approx(close.mean() + 2 * std)
    assert table["bb_lower_20"].iloc[-1] == pytest.approx(close.mean() - 2 * std)
    assert np.isnan(indicators(np.full(25, 100.0))["zscore_20"].iloc[-1])   # flat price: no z-score


# ---------- Trend strength ----------

def test_efficiency_ratio_straight_line_vs_back_and_forth():
    assert indicators(RISING)["efficiency_ratio_20"].iloc[-1] == pytest.approx(1.0)
    zigzag = np.tile([100.0, 101.0], 30)
    assert indicators(zigzag)["efficiency_ratio_20"].iloc[-1] < 0.1


def test_adx_strong_trend_and_no_trend():
    assert indicators(RISING)["adx_14"].iloc[-1] > 50
    zigzag = np.tile([100.0, 101.0], 60)
    assert indicators(zigzag)["adx_14"].iloc[-1] < 20


def test_high_low_indicators_are_empty_for_bel_20():
    table = indicators(RISING, ticker="BEL 20")
    assert table["adx_14"].isna().all()
    assert table["atr_14"].isna().all()
    assert table["sma_20"].notna().any()                            # the others are computed


# ---------- Support and resistance ----------

def test_highs_and_lows_on_closes():
    close = np.r_[np.arange(100, 130, dtype=float), 120.0]
    table = indicators(close)
    last = table.iloc[-1]
    assert last["high_20"] == pytest.approx(129.0)
    assert last["low_20"] == pytest.approx(min(close[-20:]))
    assert table["high_20"].iloc[-2] == pytest.approx(table["close_eur"].iloc[-2])   # new high: close == high_20
    assert table["high_52w"].isna().all()                           # less than a year of quotes


# ---------- Volatility and risk ----------

def test_volatility_and_risk_measures():
    steady = indicators(RISING)
    assert steady["vol_20"].iloc[-1] == pytest.approx(0.0, abs=1e-9)    # same return every day
    assert steady["downside_vol_60"].iloc[-1] == pytest.approx(0.0)
    assert steady["drawdown_60"].iloc[-1] == pytest.approx(0.0)

    close = np.r_[np.full(60, 100.0), 90.0]                        # 61 closes = 60 daily returns
    table = indicators(close)
    assert table["drawdown_60"].iloc[-1] == pytest.approx(-0.10)
    assert table["worst_day_60"].iloc[-1] == pytest.approx(-0.10)
    expected_downside = np.sqrt(0.10 ** 2 / 60) * np.sqrt(252)
    assert table["downside_vol_60"].iloc[-1] == pytest.approx(expected_downside)


def test_atr_on_constant_bars():
    table = indicators(np.full(40, 100.0), spread=0.01)             # high 101, low 99 every day
    assert table["atr_14"].iloc[-1] == pytest.approx(2.0)


def test_volatility_ratio_rises_when_the_market_wakes_up():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.005, 100)
    agitated = rng.normal(0, 0.03, 20)
    close = 100 * np.exp(np.cumsum(np.r_[calm, agitated]))
    assert indicators(close)["vol_ratio_20_60"].iloc[-1] > 1.5


# ---------- Volume ----------

def test_relative_volume_and_volume_trend():
    volume = np.r_[np.full(59, 1000.0), 2000.0]
    table = indicators(np.full(60, 100.0), volume=volume)
    assert table["volume_rel_20"].iloc[-1] == pytest.approx(2000 / np.mean(volume[-20:]))
    assert table["volume_trend_20_60"].iloc[-1] == pytest.approx(np.mean(volume[-20:]) / np.mean(volume))


def test_obv_adds_up_days_and_subtracts_down_days():
    table = indicators([10.0, 11.0, 12.0, 11.0, 11.0], volume=[100, 200, 300, 400, 500])
    assert table["obv"].tolist() == [0.0, 200.0, 500.0, 100.0, 100.0]   # flat day: unchanged


def test_volume_indicators_stay_empty_without_volume():
    table = indicators(RISING, volume=np.full(300, np.nan))
    assert table["volume_rel_20"].isna().all()
    assert table["obv"].isna().all()
    assert table["volume_trend_20_60"].isna().all()
