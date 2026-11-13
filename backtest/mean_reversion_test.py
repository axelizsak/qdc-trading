"""Backtest of the mean reversion rules on equity ETFs. Owner: Person 2.

Same method as backtest/combo_backtest.py: for each rule, holding period (4, 8, 12 quotes) and period
(train 2021-2024, test 2025-2026), the ETFs the rule selects are compared, day by day, with the
average equity ETF of the same day. Only ETFs with use_timing = True, not leveraged, and with at least
500 train days, so both periods hold the same ETFs.

Usage:  python -m backtest.mean_reversion_test data/cache/prices_ohlc_YYYY-MM-DD.csv
"""
import sys

import pandas as pd

from agents import strategy_config as cfg
from backtest import combo_backtest as cb
from backtest.momentum_dataset import build_dataset
from data.universe import load_universe

MIN_TRAIN_DAYS = 500

MEAN_REVERSION_RULES = {
    # Our rule in agents/technical.py: trend not down, no strong trend, one of the three measures oversold
    "MR buy (our rule)": [("mr_buy", "==", 1)],
    # Connors: above the SMA 200 and RSI 2 below 10
    "Connors: >SMA200 & RSI2<10": [("dist_sma200", ">", 0), ("rsi_2", "<", cfg.RSI2_OVERSOLD)],
    # Bollinger: below the lower band, long-term trend not down
    "Bollinger: trend not down & z<=-2": [("trend_long", ">=", 0), ("zscore_20", "<=", cfg.ZSCORE_LOW)],
    # Donchian: bottom of the 20-day channel, long-term trend not down
    "Donchian: trend not down & bottom 10%": [("trend_long", ">=", 0),
                                              ("range_position_20", "<", cfg.DONCHIAN_LOW)],
    # Checks of the guard: oversold whatever the trend, and oversold in a downtrend
    "oversold WITHOUT guard": [("oversold_any", "==", 1)],
    "oversold in downtrend": [("oversold_any", "==", 1), ("trend_long", "==", -1)],
    # Our avoid rule: two of the three measures overbought
    "MR avoid (2 of 3 overbought)": [("mr_avoid", "==", 1)],
}


def equity_etfs(data: pd.DataFrame) -> list[str]:
    """Equity ETFs (use_timing, not leveraged) present at least MIN_TRAIN_DAYS in the train period."""
    universe = load_universe()
    equities = set()
    for _, etf in universe.iterrows():
        if bool(etf["use_timing"]) and int(etf["leveraged"]) == 1:
            equities.add(etf["ticker"])
    train_days = data[data["period"] == "train"].groupby("ticker").size()
    kept = []
    for ticker, days in train_days.items():
        if ticker in equities and days >= MIN_TRAIN_DAYS:
            kept.append(ticker)
    return sorted(kept)


def add_mean_reversion_columns(data: pd.DataFrame) -> pd.DataFrame:
    """The oversold / overbought answers and the two setups, with the thresholds of strategy_config.py."""
    data = data.copy()
    position = data["range_position_20"]
    oversold_rsi = data["rsi_2"] < cfg.RSI2_OVERSOLD
    oversold_z = data["zscore_20"] <= cfg.ZSCORE_LOW
    oversold_channel = position < cfg.DONCHIAN_LOW
    overbought_count = ((data["rsi_2"] > cfg.RSI2_OVERBOUGHT).astype(int)
                        + (data["zscore_20"] >= cfg.ZSCORE_HIGH).astype(int)
                        + (position > cfg.DONCHIAN_HIGH).astype(int))
    no_strong_trend = (data["adx_14"] <= cfg.MR_ADX_MAX) | data["adx_14"].isna()
    guard = (data["trend_long"] >= 0) & no_strong_trend
    data["oversold_any"] = (oversold_rsi | oversold_z | oversold_channel).astype(int)
    data["mr_buy"] = (guard & (data["oversold_any"] == 1)).astype(int)
    data["mr_avoid"] = (overbought_count >= cfg.MR_MIN_OVERBOUGHT).astype(int)
    return data


def main(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path)
    data = build_dataset(prices)
    data = data[data["ticker"].isin(equity_etfs(data))]
    data = add_mean_reversion_columns(data)
    data = cb.add_forward_returns(data, prices)

    cb.COMBOS.clear()
    cb.COMBOS.update(MEAN_REVERSION_RULES)
    cb.SEEN_TEST.clear()
    print(f"{data['ticker'].nunique()} equity ETFs: {', '.join(sorted(data['ticker'].unique()))}")
    results = cb.run_combos(data)
    cb.print_report(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.mean_reversion_test data/cache/prices_ohlc_YYYY-MM-DD.csv")
        sys.exit(1)
    main(sys.argv[1])
