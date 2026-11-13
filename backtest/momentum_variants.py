"""Backtest of momentum rule variants written by hand. Owner: Person 2.

Each variant is a published way of doing time-series momentum. The list is fixed BEFORE looking at
the results: it is not tuned. Every variant answers one question per ETF and per day:
"buy setup, yes or no?". We then look at what the ETF did over the next 4 quotes.

Variants = long-term signal (5 choices) x horizons required (4 choices) x market filter (2 choices) = 40.
The best variants are chosen on 2021-2024 (train) and checked on 2025-2026 (test).

Usage:  python -m backtest.momentum_variants data/cache/prices_ohlc_YYYY-MM-DD.csv
"""
import sys

import numpy as np
import pandas as pd

from backtest.momentum_dataset import build_dataset, non_overlapping

# Long-term signal: is the ETF in a long-term uptrend?
LONG_TERM_SIGNALS = {
    "faber_sma200": "price above its SMA 200 (Faber)",
    "moskowitz_12m_skip": "12-month return, last month skipped, positive (Moskowitz)",
    "ret_12m_no_skip": "12-month return positive, last month included",
    "keller_13612": "average of 1, 3, 6 and 12-month returns positive (Faber, Keller)",
    "ours_sma200_and_12m": "price above SMA 200 AND 12-1 month return positive (our rule)",
}

# Which horizons must agree
HORIZON_RULES = {
    "long_only": "long term only",
    "long_medium": "long term + medium term up",
    "long_medium_short_not_down": "long + medium up, short not down (our rule)",
    "all_up": "long + medium + short all up",
}

MARKET_FILTERS = {
    "no_filter": "no market filter",
    "world_above_sma200": "buy only when WORLD is above its SMA 200",
}


def long_term_up(data: pd.DataFrame, signal: str) -> pd.Series:
    """True where the long-term signal says 'uptrend'."""
    if signal == "faber_sma200":
        return data["dist_sma200"] > 0
    if signal == "moskowitz_12m_skip":
        return data["ret_12m_ex_1m"] > 0
    if signal == "ret_12m_no_skip":
        return data["ret_12m"] > 0
    if signal == "keller_13612":
        return data["momentum_13612"] > 0
    if signal == "ours_sma200_and_12m":
        return (data["dist_sma200"] > 0) & (data["ret_12m_ex_1m"] > 0)
    raise ValueError(f"unknown long-term signal {signal}")


def buy_setup(data: pd.DataFrame, signal: str, horizons: str, market: str) -> pd.Series:
    """True where the variant gives a buy setup."""
    buy = long_term_up(data, signal)
    if horizons == "long_medium":
        buy = buy & (data["trend_medium"] == 1)
    elif horizons == "long_medium_short_not_down":
        buy = buy & (data["trend_medium"] == 1) & (data["trend_short"] >= 0)
    elif horizons == "all_up":
        buy = buy & (data["trend_medium"] == 1) & (data["trend_short"] == 1)
    if market == "world_above_sma200":
        buy = buy & (data["market_up"] == 1)
    return buy


def score_variant(data: pd.DataFrame, buy: pd.Series) -> dict:
    """Statistics on non-overlapping windows: how often a buy setup was followed by a rise,
    compared with how often any ETF rose in the same period (the base rate)."""
    data = data.assign(buy=buy.to_numpy())
    data = non_overlapping(data)
    base_hit = data["went_up"].mean()
    base_return = data["forward_return"].mean()
    bought = data[data["buy"]]
    not_bought = data[~data["buy"]]
    if len(bought) == 0:
        return {"buy_cases": 0}
    hit = bought["went_up"].mean()
    z = (hit - base_hit) / np.sqrt(base_hit * (1 - base_hit) / len(bought))
    return {
        "buy_cases": len(bought),
        "buy_share_pct": 100 * len(bought) / len(data),
        "buy_hit_pct": 100 * hit,
        "base_hit_pct": 100 * base_hit,
        "hit_edge_pts": 100 * (hit - base_hit),
        "z": z,
        "buy_return_pct": 100 * bought["forward_return"].mean(),
        "not_buy_return_pct": 100 * not_bought["forward_return"].mean(),
        "base_return_pct": 100 * base_return,
    }


def run_variants(data: pd.DataFrame) -> pd.DataFrame:
    """All 40 variants, scored on train and on test."""
    rows = []
    for signal in LONG_TERM_SIGNALS:
        for horizons in HORIZON_RULES:
            for market in MARKET_FILTERS:
                row = {"long_term": signal, "horizons": horizons, "market": market}
                for period in ["train", "test"]:
                    part = data[data["period"] == period]
                    stats = score_variant(part, buy_setup(part, signal, horizons, market))
                    for key, val in stats.items():
                        row[f"{period}_{key}"] = val
                rows.append(row)
    return pd.DataFrame(rows)


def print_report(results: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["long_term", "horizons", "market", "train_buy_share_pct", "train_hit_edge_pts", "train_z",
            "train_buy_return_pct", "test_buy_share_pct", "test_hit_edge_pts", "test_z", "test_buy_return_pct"]
    ranked = results.sort_values("train_hit_edge_pts", ascending=False)
    print("\nAll 40 variants, ranked on train (2021-2024). Edge = buy hit rate minus base hit rate, in points.")
    print(ranked[show].round(2).to_string(index=False))

    ours = results[(results["long_term"] == "ours_sma200_and_12m")
                   & (results["horizons"] == "long_medium_short_not_down") & (results["market"] == "no_filter")]
    print("\nOur rule:")
    print(ours[show].round(2).to_string(index=False))

    rank_train = results["train_hit_edge_pts"].rank()
    rank_test = results["test_hit_edge_pts"].rank()
    print(f"\nDo variants that are good on train stay good on test? Rank correlation: "
          f"{rank_train.corr(rank_test):.2f} (1 = same order, 0 = no link)")
    print("\nAverage test edge by choice (robustness: does a choice help whatever the others are?)")
    for column in ["long_term", "horizons", "market"]:
        print(results.groupby(column)[["train_hit_edge_pts", "test_hit_edge_pts"]].mean().round(2).to_string())
        print()


def main(path: str) -> pd.DataFrame:
    data = build_dataset(pd.read_csv(path))
    results = run_variants(data)
    print_report(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.momentum_variants data/cache/prices_ohlc_YYYY-MM-DD.csv")
        sys.exit(1)
    main(sys.argv[1])
