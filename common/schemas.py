"""Shared data formats, fixed on day 1.

Every module reads and writes these tables and nothing else. `ticker` is always
the Belfius short name from data/etf_universe.csv, never a Yahoo symbol.
"""
import pandas as pd

PRICES = ["date", "ticker", "close_eur"]
SIGNALS = ["ticker", "date", "agent", "score", "confidence", "rationale"]
PORTFOLIO = ["ticker", "target_weight"]
ORDERS = ["date", "side", "ticker", "units"]  # matches the Belfius form

SIDES = {"BUY", "SELL"}


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PRICES, "prices")
    if (df["close_eur"] <= 0).any():
        raise ValueError("prices: close_eur must be > 0")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("prices: duplicate (date, ticker) rows")
    return df


def validate_signals(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, SIGNALS, "signals")
    if not df["score"].between(-1, 1).all():
        raise ValueError("signals: score must be in [-1, 1]")
    if not df["confidence"].between(0, 1).all():
        raise ValueError("signals: confidence must be in [0, 1]")
    return df


def validate_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PORTFOLIO, "portfolio")
    if (df["target_weight"] < 0).any():
        raise ValueError("portfolio: target_weight must be >= 0 (no shorting; use inverse ETFs)")
    if df["target_weight"].sum() > 1 + 1e-9:
        raise ValueError("portfolio: target weights sum to more than 1 (remainder is cash)")
    return df


def validate_orders(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ORDERS, "orders")
    if not df["side"].isin(SIDES).all():
        raise ValueError(f"orders: side must be one of {sorted(SIDES)}")
    if (df["units"] <= 0).any():
        raise ValueError("orders: units must be > 0")
    return df
