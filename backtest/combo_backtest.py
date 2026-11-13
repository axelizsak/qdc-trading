"""Backtest of the indicator combinations written in backtest/combos.py. Owner: Person 2.

For each combination, each holding period (4, 8, 12 quotes) and each period (train 2021-2024,
test 2025-2026): when the combination says "yes" for an ETF, did the ETF rise more often, and earn
more, than the average ETF on the same day?

Statistics are computed day by day (one day = one observation, since ETFs move together),
on non-overlapping windows (one date out of the holding period).

Usage:  python -m backtest.combo_backtest data/cache/prices_ohlc_YYYY-MM-DD.csv
"""
import sys

import numpy as np
import pandas as pd

from backtest.combos import COMBOS, SEEN_TEST
from backtest.momentum_dataset import build_dataset
from data.features import signal_prices

HOLDING_PERIODS = [4, 8, 12]


def add_forward_returns(data: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Return of each ETF over the next 4, 8 and 12 quotes (prices adjusted for dividends)."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    frames = []
    for ticker, rows in prices.groupby("ticker"):
        rows = rows.sort_values("date").reset_index(drop=True)
        close = signal_prices(rows)[0].reset_index(drop=True)
        returns = pd.DataFrame({"date": rows["date"], "ticker": ticker})
        for days in HOLDING_PERIODS:
            returns[f"return_{days}d"] = close.shift(-days) / close - 1
        frames.append(returns)
    return data.merge(pd.concat(frames), on=["date", "ticker"], how="left")


def condition_true(data: pd.DataFrame, indicator: str, operator: str, threshold: float) -> pd.Series:
    """True where the condition holds (a missing value counts as False)."""
    column = data[indicator]
    if operator == ">":
        return column > threshold
    if operator == "<":
        return column < threshold
    if operator == ">=":
        return column >= threshold
    if operator == "<=":
        return column <= threshold
    if operator == "==":
        return column == threshold
    raise ValueError(f"unknown operator {operator}")


def combo_selected(data: pd.DataFrame, conditions: list) -> pd.Series:
    """True where all the conditions of the combination hold."""
    selected = pd.Series(True, index=data.index)
    for indicator, operator, threshold in conditions:
        selected = selected & condition_true(data, indicator, operator, threshold)
    return selected


def t_stat(values: list) -> float:
    values = np.array(values, dtype=float)
    if len(values) < 3 or values.std() == 0:
        return np.nan
    return values.mean() / (values.std() / np.sqrt(len(values)))


def score_combo(data: pd.DataFrame, selected: pd.Series, days: int) -> dict:
    """Day-by-day comparison of the selected ETFs with all ETFs of the same day."""
    column = f"return_{days}d"
    data = data.assign(selected=selected.to_numpy())
    data = data[data[column].notna()]
    dates = sorted(data["date"].unique())
    kept = set(dates[::days])                   # non-overlapping windows
    data = data[data["date"].isin(kept)]

    hit_edges = []
    return_edges = []
    selected_count = 0
    for _, day in data.groupby("date"):
        chosen = day[day["selected"]]
        if len(chosen) == 0:
            continue
        selected_count = selected_count + len(chosen)
        hit_edges.append((chosen[column] > 0).mean() - (day[column] > 0).mean())
        return_edges.append(chosen[column].mean() - day[column].mean())
    if len(hit_edges) == 0:
        return {"share_pct": 0.0}
    return {
        "share_pct": 100 * selected_count / len(data),
        "days": len(hit_edges),
        "hit_edge_pts": 100 * float(np.mean(hit_edges)),
        "hit_t": t_stat(hit_edges),
        "return_edge_pct": 100 * float(np.mean(return_edges)),
        "return_t": t_stat(return_edges),
    }


def run_combos(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, conditions in COMBOS.items():
        for days in HOLDING_PERIODS:
            row = {"combo": name, "days": days}
            for period in ["train", "test"]:
                part = data[data["period"] == period]
                if period == "train":
                    # drop the last train dates whose holding period would end in the test period
                    train_dates = sorted(part["date"].unique())
                    part = part[~part["date"].isin(set(train_dates[-days:]))]
                stats = score_combo(part, combo_selected(part, conditions), days)
                for key, val in stats.items():
                    row[f"{period}_{key}"] = val
            row["seen_test"] = name in SEEN_TEST
            rows.append(row)
    return pd.DataFrame(rows)


def print_report(results: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["combo", "train_share_pct", "train_hit_edge_pts", "train_hit_t", "train_return_edge_pct",
            "train_return_t", "test_hit_edge_pts", "test_hit_t", "test_return_edge_pct", "test_return_t",
            "seen_test"]
    for days in HOLDING_PERIODS:
        part = results[results["days"] == days]
        print(f"\n=== Holding {days} quotes. Edge = selected ETFs minus all ETFs of the same day ===")
        print(part[show].round(2).to_string(index=False))


def main(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path)
    data = add_forward_returns(build_dataset(prices), prices)
    results = run_combos(data)
    print_report(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.combo_backtest data/cache/prices_ohlc_YYYY-MM-DD.csv")
        sys.exit(1)
    main(sys.argv[1])
