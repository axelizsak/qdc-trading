"""Backtest, level 1: does the technical agent's momentum signal predict the next 4 days? Owner: Person 2.

For every quote date and every scored ETF, the signal is computed with the same rules as
agents/technical.py, using only data known that day. It is then compared with the ETF's own return
over the next 4 quotes (prices adjusted for dividends). No portfolio, no costs: only the signal.

Usage:  python -m backtest.momentum_signal_test data/cache/prices_ohlc_YYYY-MM-DD.csv
"""
import sys

import numpy as np
import pandas as pd

from agents import strategy_config as cfg
from agents import technical
from data.features import add_indicators, signal_prices
from data.universe import load_universe

HORIZON = 4                 # quotes ahead: we hold 3 to 4 days between two order dates
START = "2021-01-01"        # first year needed to build the 12-month and SMA 200 indicators


# ---------- Step 1: indicators, signal price and forward return for every ETF ----------------

def rolling_trend_quality(close: pd.Series) -> pd.Series:
    """Clenow trend quality for every day, on the last QUALITY_DAYS prices (same formula as the agent)."""
    def one_window(window: np.ndarray) -> float:
        return technical.trend_quality(pd.Series(window))
    return close.rolling(cfg.QUALITY_DAYS).apply(one_window, raw=True)


def etf_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Indicators of every ETF plus the columns the backtest needs, one row per (date, ticker)."""
    table = add_indicators(prices)
    frames = []
    for ticker, rows in table.groupby("ticker", sort=True):
        rows = rows.sort_values("date").reset_index(drop=True)
        close = signal_prices(rows)[0].reset_index(drop=True)
        rows["signal_close"] = close
        rows["obv_change"] = rows["obv"] - rows["obv"].shift(cfg.OBV_DAYS)
        rows["trend_quality_90"] = rolling_trend_quality(close)
        rows["forward_return"] = close.shift(-HORIZON) / close - 1
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


# ---------- Step 2: the signal of every ETF on every day --------------------------------------

def market_by_date(history: pd.DataFrame) -> pd.Series:
    """'up' / 'down' for each date: WORLD above or below its SMA 200 (last known value on days off)."""
    world = history[history["ticker"] == cfg.MARKET_TICKER].set_index("date")
    market = {}
    for day, row in world.iterrows():
        market[day] = technical.market_trend(row["signal_close"], row["sma_200"])
    all_dates = sorted(history["date"].unique())
    return pd.Series(market).reindex(all_dates).ffill()


def signal_for_row(row: pd.Series, index_row: pd.Series, inverse: bool, market: str | None) -> dict:
    """Same steps as technical.sheet_for_one_etf, for one ETF on one day."""
    price = index_row["signal_close"]
    answers = {}
    for horizon, columns in cfg.HORIZONS.items():
        answers[horizon] = technical.trend_answer(price, index_row[columns["sma"]], index_row[columns["ret"]])
    index_setup = technical.momentum_setup(answers["short"], answers["medium"], answers["long"])
    direction = technical.trend_direction(index_setup, answers["long"])

    strength = technical.adx_label(index_row["adx_14"])
    momentum_ok = technical.momentum_confirms(index_row["rsi_14"], direction)
    volume_ok = technical.volume_confirms(index_row["obv_change"], direction)
    quality = index_row["trend_quality_90"]

    if inverse:
        for horizon in answers:
            answers[horizon] = technical.reverse_answer(answers[horizon])
        quality = -quality
    setup = technical.momentum_setup(answers["short"], answers["medium"], answers["long"])

    return {
        "momentum_setup": setup,
        "trend_strength": strength,
        "confirmations": technical.count_confirmations(momentum_ok, volume_ok),
        "trend_quality_90": quality,
        "market_trend": market,
        "suggested_signal": technical.suggest_signal(setup, market, strength),
    }


def signal_history(history: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, scored ETF) with its signal and its return over the next 4 quotes."""
    market = market_by_date(history)
    by_ticker = {}
    for ticker, rows in history.groupby("ticker"):
        by_ticker[ticker] = rows.set_index("date")

    results = []
    for _, etf in load_universe().iterrows():
        ticker = etf["ticker"]
        if not bool(etf["buyable"]) or etf["family"] in cfg.SKIP_FAMILIES or ticker not in by_ticker:
            continue
        read_on = ticker
        if isinstance(etf["underlying"], str) and etf["underlying"] != "":
            read_on = etf["underlying"]
        if read_on not in by_ticker:
            continue
        own = by_ticker[ticker]
        index = by_ticker[read_on]
        inverse = etf["leveraged"] < 0

        for day, row in own.iterrows():
            if str(day) < START or pd.isna(row["forward_return"]) or day not in index.index:
                continue
            index_row = index.loc[day]
            if pd.isna(index_row["sma_200"]):
                continue                    # less than a year of history: not tested
            signal = signal_for_row(row, index_row, inverse, market.get(day))
            signal["date"] = day
            signal["ticker"] = ticker
            signal["family"] = etf["family"]
            signal["forward_return"] = row["forward_return"]
            results.append(signal)
    return pd.DataFrame(results)


# ---------- Step 3: results ------------------------------------------------------------------

def summarize(signals: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Number of cases, mean and median 4-day return (%), share of positive returns (%)."""
    grouped = signals.groupby(by)["forward_return"]
    table = pd.DataFrame({
        "cases": grouped.count(),
        "mean_return_pct": grouped.mean() * 100,
        "median_return_pct": grouped.median() * 100,
        "hit_rate_pct": grouped.apply(lambda r: (r > 0).mean() * 100),
    })
    return table.round(3)


def non_overlapping(signals: pd.DataFrame) -> pd.DataFrame:
    """Only one date out of HORIZON, so that two 4-day windows never overlap."""
    dates = sorted(signals["date"].unique())
    kept = set()
    for i in range(0, len(dates), HORIZON):
        kept.add(dates[i])
    return signals[signals["date"].isin(kept)]


def print_report(signals: pd.DataFrame) -> None:
    order = ["buy", "watch", "neutral", "avoid"]
    signals = signals.copy()
    signals["year"] = pd.to_datetime(signals["date"]).dt.year

    print(f"\n{signals['ticker'].nunique()} ETFs, {signals['date'].nunique()} dates, "
          f"{signals['date'].min()} to {signals['date'].max()}, {len(signals)} cases")
    print("\n1. By suggested signal (all days)")
    print(summarize(signals, ["suggested_signal"]).reindex(order).to_string())
    print("\n2. By suggested signal, non-overlapping windows (one date out of 4)")
    print(summarize(non_overlapping(signals), ["suggested_signal"]).reindex(order).to_string())
    print("\n3. By momentum setup")
    print(summarize(signals, ["momentum_setup"]).to_string())
    print("\n4. Mean 4-day return (%) by year and signal")
    print((signals.pivot_table(index="year", columns="suggested_signal", values="forward_return",
                               aggfunc="mean") * 100).reindex(columns=order).round(3).to_string())
    print("\n5. Buy setups only: by trend strength (ADX)")
    buys = signals[signals["momentum_setup"] == "buy"]
    print(summarize(buys, ["trend_strength"]).to_string())
    print("\n6. Buy setups only: by number of confirmations")
    print(summarize(buys, ["confirmations"]).to_string())
    print("\n7. Buy setups only: by trend quality quintile (1 = lowest)")
    buys = buys.copy()
    buys["quality_quintile"] = pd.qcut(buys["trend_quality_90"], 5, labels=[1, 2, 3, 4, 5])
    print(summarize(buys, ["quality_quintile"]).to_string())
    print("\n8. Buy minus all ETFs, mean 4-day return (%) by family")
    family = signals.pivot_table(index="family", columns="suggested_signal", values="forward_return",
                                 aggfunc="mean") * 100
    family["all"] = signals.groupby("family")["forward_return"].mean() * 100
    print(family.reindex(columns=order + ["all"]).round(3).to_string())
    print("\n9. Market context: by market trend and signal")
    print(summarize(signals, ["market_trend", "suggested_signal"]).to_string())


def main(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path)
    history = etf_history(prices)
    signals = signal_history(history)
    print_report(signals)
    return signals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.momentum_signal_test data/cache/prices_ohlc_YYYY-MM-DD.csv")
        sys.exit(1)
    main(sys.argv[1])
