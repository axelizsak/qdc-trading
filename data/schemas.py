"""Shared data formats.

Every module reads and writes these tables and nothing else. `ticker` is always
the Belfius short name from data/etf_universe.csv, never a Yahoo symbol.
List columns (macro_views.sources, source_registry.factors) are stored as
strings joined with LIST_SEP, so they survive a round trip through CSV.
"""
import numpy as np
import pandas as pd

from data.config import MACRO_FACTORS, MAX_ORDERS_PER_DAY, SIGNAL_AGENTS

# Real daily prices (not adjusted for dividends), in EUR, on the Belfius exchange.
# One row per day the ETF was quoted: a day without a quote is absent, never filled.
# adj_close_eur: close adjusted for dividends (and splits) by the source. Equal to close_eur for an
# accumulating ETF; lower before each payment for a distributing ETF. Used for indicators and model
# training (real market moves, no fake drop on ex-dividend days). close_eur stays the price Belfius uses.
# NA when the source gives none.
# volume: units traded on that exchange that day (on-exchange only, so partial for an ETF);
# NA when the source gives none, 0 when the ETF did not trade.
PRICES = ["date", "ticker", "open_eur", "high_eur", "low_eur", "close_eur", "adj_close_eur", "volume"]
SIGNALS = ["ticker", "date", "agent", "score", "confidence", "rationale"]
PORTFOLIO = ["ticker", "target_weight"]
ORDERS = ["date", "side", "ticker", "units"]  # matches the Belfius form
HOLDINGS = ["ticker", "units"]
FEATURES = ["date", "ticker", "feature", "value"]  # long format: one row per (date, ticker, feature)
AGENT_WEIGHTS = ["date", "agent", "weight", "hit_rate"]
EVIDENCE = ["date", "factor", "source", "kind", "value", "summary", "url"]
MACRO_VIEWS = ["date", "factor", "direction", "confidence", "rationale", "counter_argument", "sources"]
SOURCE_REGISTRY = ["source", "kind", "factors", "status", "first_seen", "n_uses", "hit_rate"]
# logs/orders.csv: each proposed order with the human decision, plus orders the human added.
ORDER_LOG = ["date", "ticker", "ai_side", "ai_units", "decision", "side", "units", "reason"]
INCIDENTS = ["date", "step", "error"]
# logs/source_changes.csv: when a Macro/News source was added or removed, for before/after comparisons.
SOURCE_CHANGES = ["date", "source", "change", "note"]
# data/macro_exposure.csv: how much each ETF moves with each macro factor (before leverage).
MACRO_EXPOSURE = ["ticker", "multiplier", *MACRO_FACTORS]

SIDES = {"BUY", "SELL"}
EVIDENCE_KINDS = {"number", "text", "exploration"}
SOURCE_STATUSES = {"active", "testing", "dropped"}
DECISIONS = {"approved", "modified", "rejected", "added"}
SOURCE_CHANGE_KINDS = {"added", "removed"}
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
    adjusted = pd.to_numeric(df["adj_close_eur"], errors="coerce")
    if (adjusted <= 0).any():
        raise ValueError("prices: adj_close_eur must be > 0 (or empty when the source gives none)")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    if (volume < 0).any():
        raise ValueError("prices: volume must be >= 0 (or empty when the source gives none)")
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


def validate_order_log(df: pd.DataFrame) -> pd.DataFrame:
    """AI suggestion (ai_side, ai_units) next to the human decision and the final order (side, units).

    approved: final order = AI suggestion. modified: final order differs, reason required.
    rejected: no final order, reason required. added: no AI suggestion, reason required.
    """
    _require_columns(df, ORDER_LOG, "order_log")
    _require_in(df, "decision", DECISIONS, "order_log")
    _require_unique(df, ["date", "ticker"], "order_log")
    proposed = df["decision"] != "added"
    executed = df["decision"] != "rejected"
    if not (df.loc[proposed, "ai_side"].isin(SIDES).all()
            and (pd.to_numeric(df.loc[proposed, "ai_units"], errors="coerce") > 0).all()):
        raise ValueError("order_log: proposed orders need an ai_side and ai_units > 0")
    if df.loc[~proposed, ["ai_side", "ai_units"]].notna().any().any():
        raise ValueError("order_log: added orders have no AI suggestion")
    if not (df.loc[executed, "side"].isin(SIDES).all()
            and (pd.to_numeric(df.loc[executed, "units"], errors="coerce") > 0).all()):
        raise ValueError("order_log: executed orders need a side and units > 0")
    if df.loc[~executed, ["side", "units"]].notna().any().any():
        raise ValueError("order_log: rejected orders have no side or units")
    approved = df[df["decision"] == "approved"]
    if not ((approved["side"] == approved["ai_side"]).all()
            and (approved["units"].astype(float) == approved["ai_units"].astype(float)).all()):
        raise ValueError("order_log: approved orders must match the AI suggestion")
    _require_text(df[df["decision"] != "approved"], "reason", "order_log")
    if (df[executed].groupby("date").size() > MAX_ORDERS_PER_DAY).any():
        raise ValueError(f"order_log: at most {MAX_ORDERS_PER_DAY} executed orders per day")
    return df


def validate_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """A pipeline step that failed and was replaced by its fallback."""
    _require_columns(df, INCIDENTS, "incidents")
    _require_text(df, "step", "incidents")
    _require_text(df, "error", "incidents")
    return df


def validate_macro_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Exposure of each ETF to each factor in [-1, 1]; multiplier = leverage (negative for inverse ETFs)."""
    _require_columns(df, MACRO_EXPOSURE, "macro_exposure")
    _require_unique(df, ["ticker"], "macro_exposure")
    for factor in MACRO_FACTORS:
        _require_between(df, factor, -1, 1, "macro_exposure")
    multiplier = pd.to_numeric(df["multiplier"], errors="coerce")
    if multiplier.isna().any() or (multiplier == 0).any():
        raise ValueError("macro_exposure: multiplier must be a non-zero number")
    return df


def validate_source_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Dated log of the sources added to or removed from the Macro/News agent."""
    _require_columns(df, SOURCE_CHANGES, "source_changes")
    _require_in(df, "change", SOURCE_CHANGE_KINDS, "source_changes")
    _require_text(df, "source", "source_changes")
    _require_unique(df, ["date", "source", "change"], "source_changes")
    return df
