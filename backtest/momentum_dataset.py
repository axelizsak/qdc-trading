"""Shared dataset for the momentum study: rule variants (backtest) and ML models. Owner: Person 2.

One row per (date, ETF) with the indicators known that day and the ETF's own return over the next
4 quotes (prices adjusted for dividends). Only buyable, non-leveraged ETFs: the first backtest showed
that leveraged and inverse ETFs need a separate treatment.

Period split, the same for every method:
  - choose (train): 2021-01-01 to 2024-12-31, minus the last 4 quotes so no 4-day window crosses into 2025
  - check  (test):  2025-01-01 to the last date with a known 4-day return
"""
import numpy as np
import pandas as pd

from agents import strategy_config as cfg
from agents import technical
from backtest.momentum_signal_test import HORIZON, START, etf_history, market_by_date
from data.universe import load_universe

TEST_START = "2025-01-01"


def trend_code(answer: str | None) -> float:
    """up = +1, none = 0, down = -1, missing = NA (so a model can read the answer as a number)."""
    if answer == "up":
        return 1.0
    if answer == "down":
        return -1.0
    if answer == "none":
        return 0.0
    return np.nan


def add_extra_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Columns built from the indicators of one ETF, oldest first (all use past data only)."""
    rows = rows.copy()
    price = rows["signal_close"]
    rows["ret_12m"] = price / price.shift(252) - 1
    returns_13612 = []
    for days in cfg.MOMENTUM_13612_DAYS:
        returns_13612.append(price / price.shift(days) - 1)
    rows["momentum_13612"] = sum(returns_13612) / len(returns_13612)
    rows["dist_sma20"] = price / rows["sma_20"] - 1
    rows["dist_sma50"] = price / rows["sma_50"] - 1
    rows["dist_sma200"] = price / rows["sma_200"] - 1
    rows["dist_high_52w"] = price / rows["high_52w"] - 1
    channel = rows["high_20"] - rows["low_20"]
    rows["range_position_20"] = (price - rows["low_20"]) / channel.replace(0, np.nan)
    rows["obv_trend_20"] = np.sign(rows["obv_change"])
    return rows


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """The study table: one row per (date, buyable non-leveraged ETF), from START, with a known 4-day return."""
    history = etf_history(prices)
    market = market_by_date(history)
    universe = load_universe()

    frames = []
    for _, etf in universe.iterrows():
        if not bool(etf["buyable"]) or etf["family"] in cfg.SKIP_FAMILIES or etf["leveraged"] != 1:
            continue
        rows = history[history["ticker"] == etf["ticker"]].sort_values("date")
        if len(rows) == 0:
            continue
        rows = add_extra_columns(rows)
        rows["family"] = etf["family"]
        frames.append(rows)
    data = pd.concat(frames, ignore_index=True)

    # Trend answers on the three horizons, as in the agent
    for horizon, columns in cfg.HORIZONS.items():
        codes = []
        for price, sma, ret in zip(data["signal_close"], data[columns["sma"]], data[columns["ret"]]):
            codes.append(trend_code(technical.trend_answer(price, sma, ret)))
        data[f"trend_{horizon}"] = codes

    market_codes = []
    for day in data["date"]:
        market_codes.append(trend_code(market.get(day)))
    data["market_up"] = market_codes

    data = data[(data["date"].astype(str) >= START) & data["forward_return"].notna() & data["sma_200"].notna()]
    data = data.copy()
    data["went_up"] = (data["forward_return"] > 0).astype(int)
    data["period"] = np.where(data["date"].astype(str) >= TEST_START, "test", "train")
    data = drop_gap(data)
    return data.reset_index(drop=True)


def drop_gap(data: pd.DataFrame) -> pd.DataFrame:
    """Remove the last HORIZON train dates: their 4-day return would end inside the test period."""
    train_dates = sorted(data.loc[data["period"] == "train", "date"].unique())
    gap_dates = set(train_dates[-HORIZON:])
    return data[~data["date"].isin(gap_dates)]


def non_overlapping(data: pd.DataFrame) -> pd.DataFrame:
    """One date out of HORIZON, so that two 4-day windows never overlap (used for every statistic)."""
    dates = sorted(data["date"].unique())
    kept = set()
    for i in range(0, len(dates), HORIZON):
        kept.add(dates[i])
    return data[data["date"].isin(kept)]
