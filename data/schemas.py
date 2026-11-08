"""Shared data formats.

Every module reads and writes these tables and nothing else. `ticker` is always
the Belfius short name from data/etf_universe.csv, never a Yahoo symbol.
List columns (macro_views.sources, source_registry.factors) are stored as
strings joined with LIST_SEP, so they survive a round trip through CSV.
"""
import numpy as np
import pandas as pd

from data.config import MACRO_FACTORS, SIGNAL_AGENTS

PRICES = ["date", "ticker", "close_eur"]
SIGNALS = ["ticker", "date", "agent", "score", "confidence", "rationale"]
PORTFOLIO = ["ticker", "target_weight"]
ORDERS = ["date", "side", "ticker", "units"]  # matches the Belfius form
HOLDINGS = ["ticker", "units"]
FEATURES = ["date", "ticker", "feature", "value"]  # long format: one row per (date, ticker, feature)
AGENT_WEIGHTS = ["date", "agent", "weight", "hit_rate"]
EVIDENCE = ["date", "factor", "source", "kind", "value", "summary", "url"]
MACRO_VIEWS = ["date", "factor", "direction", "confidence", "rationale", "counter_argument", "sources"]
SOURCE_REGISTRY = ["source", "kind", "factors", "status", "first_seen", "n_uses", "hit_rate"]

SIDES = {"BUY", "SELL"}
EVIDENCE_KINDS = {"number", "text", "exploration"}
SOURCE_STATUSES = {"active", "testing", "dropped"}
LIST_SEP = ";"

# Proposed starting set of features, to validate with the data agent's owner. Not enforced.
STARTER_FEATURES = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "drawdown_60d", "corr_world_60d"]


def split_list(value: str) -> list[str]:
    """'a;b' -> ['a', 'b']. Empty or missing -> []."""
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(LIST_SEP) if item.strip()]


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _require_in(df: pd.DataFrame, col: str, allowed, name: str) -> None:
    bad = sorted(set(df.loc[~df[col].isin(allowed), col].astype(str)))
    if bad:
        raise ValueError(f"{name}: unknown {col} {bad}")


def _require_between(df: pd.DataFrame, col: str, lo: float, hi: float, name: str,
                     allow_missing: bool = False) -> None:
    values = pd.to_numeric(df[col], errors="coerce")
    ok = values.between(lo, hi)
    if allow_missing:
        ok |= df[col].isna()
    if not ok.all():
        raise ValueError(f"{name}: {col} must be in [{lo}, {hi}]")


def _require_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    if df.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate ({', '.join(keys)}) rows")


def _require_text(df: pd.DataFrame, col: str, name: str) -> None:
    if not df[col].map(lambda v: isinstance(v, str) and v.strip() != "").all():
        raise ValueError(f"{name}: {col} must be a non-empty string")


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


def validate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Current positions. Cash is tracked separately."""
    _require_columns(df, HOLDINGS, "holdings")
    if (df["units"] <= 0).any():
        raise ValueError("holdings: units must be > 0 (drop closed positions)")
    _require_unique(df, ["ticker"], "holdings")
    return df


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Data agent output, long format. Pivot with df.pivot_table(index=['date', 'ticker'], columns='feature')."""
    _require_columns(df, FEATURES, "features")
    if not np.isfinite(pd.to_numeric(df["value"], errors="coerce")).all():
        raise ValueError("features: value must be a finite number (drop rows that cannot be computed)")
    _require_unique(df, ["date", "ticker", "feature"], "features")
    return df


def validate_agent_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Portfolio Manager weights per agent, one set per review date. hit_rate is empty before the first review."""
    _require_columns(df, AGENT_WEIGHTS, "agent_weights")
    _require_in(df, "agent", SIGNAL_AGENTS, "agent_weights")
    _require_between(df, "weight", 0, 1, "agent_weights")
    _require_between(df, "hit_rate", 0, 1, "agent_weights", allow_missing=True)
    _require_unique(df, ["date", "agent"], "agent_weights")
    sums = df.groupby("date")["weight"].sum()
    if not np.allclose(sums, 1):
        raise ValueError("agent_weights: weights must sum to 1 on each date")
    return df


def validate_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Raw facts collected by the Macro/News agent. kind=number rows carry a numeric value."""
    _require_columns(df, EVIDENCE, "evidence")
    _require_in(df, "factor", MACRO_FACTORS, "evidence")
    _require_in(df, "kind", EVIDENCE_KINDS, "evidence")
    numbers = pd.to_numeric(df.loc[df["kind"] == "number", "value"], errors="coerce")
    if numbers.isna().any():
        raise ValueError("evidence: kind=number rows need a numeric value")
    _require_text(df, "source", "evidence")
    _require_text(df, "summary", "evidence")
    return df


def validate_macro_views(df: pd.DataFrame) -> pd.DataFrame:
    """One view per factor and date, with the sources (LIST_SEP-joined) that support it."""
    _require_columns(df, MACRO_VIEWS, "macro_views")
    _require_in(df, "factor", MACRO_FACTORS, "macro_views")
    _require_between(df, "direction", -1, 1, "macro_views")
    _require_between(df, "confidence", 0, 1, "macro_views")
    if not df["sources"].map(split_list).map(bool).all():
        raise ValueError("macro_views: every view needs at least one source")
    _require_unique(df, ["date", "factor"], "macro_views")
    return df


def validate_source_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Every source the Macro/News agent has used, with its track record."""
    _require_columns(df, SOURCE_REGISTRY, "source_registry")
    _require_unique(df, ["source"], "source_registry")
    _require_in(df, "kind", EVIDENCE_KINDS, "source_registry")
    _require_in(df, "status", SOURCE_STATUSES, "source_registry")
    unknown = sorted({f for fs in df["factors"].map(split_list) for f in fs} - set(MACRO_FACTORS))
    if unknown or not df["factors"].map(split_list).map(bool).all():
        raise ValueError(f"source_registry: every source needs known factors (unknown: {unknown})")
    if (pd.to_numeric(df["n_uses"], errors="coerce").fillna(-1) < 0).any():
        raise ValueError("source_registry: n_uses must be >= 0")
    _require_between(df, "hit_rate", 0, 1, "source_registry", allow_missing=True)
    return df
