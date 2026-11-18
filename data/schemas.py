"""Shared data formats.

Every module reads and writes these tables and nothing else. `ticker` is always
the Belfius short name from data/etf_universe.csv, never a Yahoo symbol.
List columns (macro_views.sources, source_registry.factors) are stored as
strings joined with LIST_SEP, so they survive a round trip through CSV.
"""
import json

import numpy as np
import pandas as pd

from data.config import MACRO_FACTORS, MAX_ORDERS_PER_DAY, SIGNAL_AGENTS, TECHNICAL_SIGNALS
from data.profiles import ARCHETYPES, SLEEVES, STRATEGIES

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
# logs/weights.csv: every weight of the system (agents/learning.py), one row per session and expert: the weight it had
# for that session's decisions and what following it earned in the session (empty until the session's close is known).
# level: technical (the technical agent's three answers), agents (the four scores), portfolio (the LLM's portfolio and
# the code's).
WEIGHTS = ["date", "level", "name", "weight", "result"]
# logs/technical_answers.csv: what the technical agent's rules answered for each ETF on the morning of every session
# since LEARNING_START, from the prices known that morning, and the ETF's return over that session (from the fill
# price, data/config.py fills_from, to the next close). The technical weights are learned from them.
TECHNICAL_ANSWERS = ["date", "ticker", "suggested_signal", "mr_setup", "momentum_13612", "confirmations", "trend_strength",
                     "session_return"]
EVIDENCE = ["date", "factor", "source", "kind", "value", "summary", "url"]
MACRO_VIEWS = ["date", "factor", "direction", "confidence", "rationale", "counter_argument", "sources"]
SOURCE_REGISTRY = ["source", "kind", "factors", "status", "first_seen", "n_uses", "hit_rate"]
# logs/orders.csv: each proposed order with the human decision, plus orders the human added.
ORDER_LOG = ["date", "ticker", "ai_side", "ai_units", "decision", "side", "units", "reason"]
INCIDENTS = ["date", "step", "error"]
# logs/etf_views.csv: one view per ETF and day from the theme analysts (direction of the ETF or its underlying).
# catalyst_date and catalyst: the next scheduled event that can move the ETF, when the analyst found one (else empty).
ETF_VIEWS = ["date", "ticker", "direction", "confidence", "rationale", "counter_argument", "sources",
             "catalyst_date", "catalyst"]
# logs/source_changes.csv: when a Macro/News source was added or removed, for before/after comparisons.
SOURCE_CHANGES = ["date", "source", "change", "note"]
# data/macro_exposure.csv: how much each ETF moves with each macro factor (before leverage).
MACRO_EXPOSURE = ["ticker", "multiplier", *MACRO_FACTORS]
# Risk Manager output: the portfolio after the approved orders, one row per metric
# (concentration, volatility, drawdown, exposure, leverage, liquidity...). limit is empty when there is none.
RISK_REPORT = ["date", "metric", "value", "limit", "status", "comment"]
# logs/reviews.csv: every human answer at the three review gates, with the feedback sent back.
REVIEWS = ["date", "round", "gate", "decision", "feedback"]
# Ex-dividend dates of the distributing ETFs (data/dividends.py), for the Risk Manager: dividends are not credited.
# basis says what a row rests on: "paid", or "expected: ..." when it is estimated from last year's schedule.
EX_DIVIDENDS = ["ticker", "ex_date", "amount_eur", "basis"]
# logs/pm_portfolios.csv: the portfolios the Portfolio Manager built each day (the code's baseline and optimiser,
# the LLM's own, and the final blend). It learns from them how far to trust the LLM. One set per date.
PM_PORTFOLIOS = ["date", "source", "ticker", "weight"]
# logs/recommendations.csv: every order the Portfolio Manager proposed, round by round, with its fill price and its
# reasons. Append-only: each round gets a new recommendation_id. A round without an order is one row with side NONE.
RECOMMENDATIONS = ["recommendation_id", "created_at", "date", "ticker", "side", "units", "price_eur", "price_date",
                   "value_eur", "weight_now", "weight_target", "impact", "timing", "why"]
# logs/risk_forecasts.csv: what each risk assessment was made on (kind weight, name = ticker) and what it forecast
# (kind forecast, name = vol_raw, p5_5d, prob_below_start...), so the Risk Manager can grade itself later.
RISK_FORECASTS = ["date", "kind", "name", "value"]
# logs/risk_calibration.csv: the correction the Risk Manager learned for its volatility forecast, one row per date.
RISK_CALIBRATION = ["date", "vol_scale", "days_graded", "mean_ratio", "floor_breaches", "floor_days"]
# logs/logbook.csv: the audit trail, one row per event (signal, recommendation, risk flag, human decision, fill...),
# append-only. Each row carries the hash of the one before it, so a change to an old row shows (agents/logbook.py).
# `data` holds the event's numbers as a JSON object (empty when there are none), so tables can be built from the ledger.
LOGBOOK = ["seq", "date", "kind", "actor", "ticker", "round", "gate", "title", "detail", "data", "source", "event_id",
           "recorded_at", "prev_hash", "hash"]

# logs/strategy_portfolios.csv: the reference portfolio of each strategy (data/profiles.py STRATEGIES), what the
# customer app shows a customer of that risk level. sleeve: one of data/profiles.py SLEEVES (core equity, bonds, gold,
# satellites, cash). One portfolio per date and strategy, built by agents/model_portfolios.py.
STRATEGY_PORTFOLIOS = ["date", "strategy", "ticker", "target_weight", "sleeve"]
# logs/strategy_choices.csv: the strategy the team trades, chosen in the cockpit, append-only. From `date` on, every
# day not run yet is traded within that strategy's limits, until the next choice. reason: why, in a few words.
STRATEGY_CHOICES = ["date", "strategy", "reason"]
# logs/chart_history.csv: the points of the app's charts, rewritten by each morning run (graph/morning.py). kind etf:
# `name` is a ticker and `value` its adjusted close in EUR; kind strategy: `name` is a strategy and `value` what 1
# invested in its reference portfolio became. Every session of the last weeks, one a week before.
CHART_HISTORY = ["date", "kind", "name", "value"]
# logs/etf_sheets.csv: the sheet of each ETF as the customer app shows it, one row per line of a sheet, written by the
# morning run (graph/morning.py). block: returns, trend or macro. state: up, neutral or down for a trend; supportive,
# watch or headwind for a macro factor; empty for a return. detail: the checks or the view behind the line.
ETF_SHEETS = ["date", "ticker", "block", "name", "value", "state", "detail"]
# logs/customer_log.csv: what happened to each customer of the app, append-only. event: profiled (the coach found
# their archetype and the rules their strategy), order (they bought an ETF themselves), unlocked (they passed the check
# on complex products), lesson (they answered a lesson's question), reset (they start over). `why`: the reasons for the
# archetype and for the strategy, as a JSON object. `detail`: a JSON object: the suitability answers (profiled), the
# order (ticker, eur, units, price, as_of), the check's answers (unlocked), or the lesson, the answer and whether it was
# right (lesson). `flags`: the contradictions the code found, joined with LIST_SEP.
CUSTOMER_LOG = ["time", "customer", "event", "archetype", "strategy", "why", "detail", "flags"]

SIDES = {"BUY", "SELL"}
LOGBOOK_KINDS = {"signals", "macro_report", "agent_weights", "recommendation", "risk_report", "risk_flag",
                 "human_review", "order_decision", "fill", "valuation", "summary", "incident", "superseded",
                 "target_allocation", "human_vs_ai", "lesson", "reassessment"}
EVIDENCE_KINDS = {"number", "text", "exploration"}
SOURCE_STATUSES = {"active", "testing", "dropped"}
DECISIONS = {"approved", "modified", "rejected", "added"}
SOURCE_CHANGE_KINDS = {"added", "removed"}
RISK_STATUSES = {"ok", "warning", "breach"}
REVIEW_GATES = ["orders", "risk", "summary"]  # in the order the human meets them
REVIEW_DECISIONS = {"approved", "revised", "stopped"}  # revised: back to the Portfolio Manager; stopped: no trade
PM_SOURCES = {"baseline", "optimizer", "llm", "final"}
RISK_FORECAST_KINDS = {"weight", "forecast"}
CUSTOMER_EVENTS = {"profiled", "order", "unlocked", "lesson", "reset"}
CHART_KINDS = {"etf", "strategy"}
SHEET_STATES = {"returns": {""}, "trend": {"up", "neutral", "down"}, "macro": {"supportive", "watch", "headwind"}}
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


WEIGHT_NAMES = {"technical": set(TECHNICAL_SIGNALS), "agents": set(SIGNAL_AGENTS), "portfolio": {"llm", "code"}}
SUGGESTED_SIGNALS = {"buy", "watch", "neutral", "avoid"}
MR_SETUPS = {"buy", "avoid", "none", "n/a"}


def validate_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Every weight of the system, one set per date and level, summing to 1. result is empty until it is known."""
    _require_columns(df, WEIGHTS, "weights")
    _require_in(df, "level", WEIGHT_NAMES, "weights")
    for level, names in WEIGHT_NAMES.items():
        _require_in(df[df["level"] == level], "name", names, f"weights ({level})")
    _require_between(df, "weight", 0, 1, "weights")
    _require_unique(df, ["date", "level", "name"], "weights")
    if len(df) and not np.allclose(df.groupby(["date", "level"])["weight"].sum(), 1):
        raise ValueError("weights: weights must sum to 1 on each date and level")
    if not np.isfinite(pd.to_numeric(df["result"], errors="coerce").dropna()).all():
        raise ValueError("weights: result must be a number or empty")
    return df


def validate_technical_answers(df: pd.DataFrame) -> pd.DataFrame:
    """The technical rules' answers per session and ETF. momentum_13612 is empty without a year of prices."""
    _require_columns(df, TECHNICAL_ANSWERS, "technical_answers")
    _require_in(df, "suggested_signal", SUGGESTED_SIGNALS, "technical_answers")
    _require_in(df[df["mr_setup"].notna()], "mr_setup", MR_SETUPS, "technical_answers")  # a plain CSV read turns 'n/a' empty
    _require_between(df, "confirmations", 0, 2, "technical_answers")
    _require_unique(df, ["date", "ticker"], "technical_answers")
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


def validate_etf_views(df: pd.DataFrame) -> pd.DataFrame:
    """Theme analysts' views: direction in [-1, 1] (of the underlying for leveraged ETFs), with sources."""
    _require_columns(df, ETF_VIEWS, "etf_views")
    _require_between(df, "direction", -1, 1, "etf_views")
    _require_between(df, "confidence", 0, 1, "etf_views")
    if not df["sources"].map(split_list).map(bool).all():
        raise ValueError("etf_views: every view needs at least one source")
    dated = df[df["catalyst_date"].notna() & (df["catalyst_date"].astype(str).str.strip() != "")]
    if pd.to_datetime(dated["catalyst_date"].astype(str), format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError("etf_views: catalyst_date must be a date (YYYY-MM-DD) or empty")
    _require_text(dated, "catalyst", "etf_views")
    _require_unique(df, ["date", "ticker"], "etf_views")
    return df


def validate_risk_report(df: pd.DataFrame) -> pd.DataFrame:
    """Post-trade risk metrics: a finite value, a status (ok, warning, breach), one row per (date, metric)."""
    _require_columns(df, RISK_REPORT, "risk_report")
    _require_text(df, "metric", "risk_report")
    if not np.isfinite(pd.to_numeric(df["value"], errors="coerce")).all():
        raise ValueError("risk_report: value must be a finite number")
    _require_in(df, "status", RISK_STATUSES, "risk_report")
    _require_unique(df, ["date", "metric"], "risk_report")
    return df


def validate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Human answers at the review gates. Feedback is required to send a day back or to stop it."""
    _require_columns(df, REVIEWS, "reviews")
    _require_in(df, "gate", REVIEW_GATES, "reviews")
    _require_in(df, "decision", REVIEW_DECISIONS, "reviews")
    if not (pd.to_numeric(df["round"], errors="coerce") >= 1).all():
        raise ValueError("reviews: round must be 1 or more")
    _require_text(df[df["decision"] != "approved"], "feedback", "reviews")
    _require_unique(df, ["date", "round", "gate"], "reviews")
    return df


def validate_logbook(df: pd.DataFrame) -> pd.DataFrame:
    """Logbook rows: known kinds, a title on each row, seq running 1, 2, 3... without gaps.
    The hash chain itself is checked by agents/logbook.verify."""
    _require_columns(df, LOGBOOK, "logbook")
    _require_in(df, "kind", LOGBOOK_KINDS, "logbook")
    _require_text(df, "title", "logbook")
    _require_text(df, "event_id", "logbook")
    for text in df["data"].dropna():
        if str(text).strip():
            try:
                ok = isinstance(json.loads(text), dict)
            except ValueError:
                ok = False
            if not ok:
                raise ValueError("logbook: data must be a JSON object or empty")
    seq = pd.to_numeric(df["seq"], errors="coerce")
    if not (seq.to_numpy() == np.arange(1, len(df) + 1)).all():
        raise ValueError("logbook: seq must run 1, 2, 3... without gaps")
    return df


def validate_pm_portfolios(df: pd.DataFrame) -> pd.DataFrame:
    """Portfolio Manager portfolios: known sources, weights in [0, 1] that add up to at most 100% per portfolio."""
    _require_columns(df, PM_PORTFOLIOS, "pm_portfolios")
    _require_in(df, "source", PM_SOURCES, "pm_portfolios")
    _require_between(df, "weight", 0, 1, "pm_portfolios")
    _require_unique(df, ["date", "source", "ticker"], "pm_portfolios")
    if (df.groupby([df["date"].astype(str), "source"])["weight"].sum() > 1 + 1e-6).any():
        raise ValueError("pm_portfolios: the weights of a portfolio must add up to at most 1")
    return df


def validate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Proposed orders with their reasons: an id on every row; a ticker, units > 0 and a price > 0 unless side is NONE."""
    _require_columns(df, RECOMMENDATIONS, "recommendations")
    _require_text(df, "recommendation_id", "recommendations")
    _require_in(df, "side", SIDES | {"NONE"}, "recommendations")
    orders = df[df["side"] != "NONE"]
    _require_text(orders, "ticker", "recommendations")
    for col in ("units", "price_eur"):
        if not (pd.to_numeric(orders[col], errors="coerce") > 0).all():
            raise ValueError(f"recommendations: {col} must be > 0 on every proposed order")
    return df


def validate_risk_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Weights and forecasts behind each risk assessment: a finite value, one row per (date, kind, name)."""
    _require_columns(df, RISK_FORECASTS, "risk_forecasts")
    _require_in(df, "kind", RISK_FORECAST_KINDS, "risk_forecasts")
    _require_text(df, "name", "risk_forecasts")
    if not np.isfinite(pd.to_numeric(df["value"], errors="coerce")).all():
        raise ValueError("risk_forecasts: value must be a finite number")
    _require_unique(df, ["date", "kind", "name"], "risk_forecasts")
    return df


def validate_risk_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """The learned volatility correction: vol_scale > 0, one row per date."""
    _require_columns(df, RISK_CALIBRATION, "risk_calibration")
    if not (pd.to_numeric(df["vol_scale"], errors="coerce") > 0).all():
        raise ValueError("risk_calibration: vol_scale must be > 0")
    if not (pd.to_numeric(df["days_graded"], errors="coerce") >= 0).all():
        raise ValueError("risk_calibration: days_graded must be 0 or more")
    _require_unique(df, ["date"], "risk_calibration")
    return df


def validate_ex_dividends(df: pd.DataFrame) -> pd.DataFrame:
    """Ex-dividend dates: a real date, an amount above 0 or empty, what the row rests on, one row per (ticker, ex_date)."""
    _require_columns(df, EX_DIVIDENDS, "ex_dividends")
    if pd.to_datetime(df["ex_date"].astype(str), format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError("ex_dividends: ex_date must be a date (YYYY-MM-DD)")
    amount = pd.to_numeric(df["amount_eur"], errors="coerce")
    if not (df["amount_eur"].isna() | (amount > 0)).all():
        raise ValueError("ex_dividends: amount_eur must be above 0, or empty when it is not known")
    _require_text(df, "basis", "ex_dividends")
    _require_unique(df, ["ticker", "ex_date"], "ex_dividends")
    return df


def validate_strategy_portfolios(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy portfolios: known strategies and sleeves, weights in [0, 1] that add up to at most 100% per portfolio."""
    _require_columns(df, STRATEGY_PORTFOLIOS, "strategy_portfolios")
    _require_in(df, "strategy", set(STRATEGIES), "strategy_portfolios")
    _require_in(df, "sleeve", set(SLEEVES), "strategy_portfolios")
    _require_between(df, "target_weight", 0, 1, "strategy_portfolios")
    _require_unique(df, ["date", "strategy", "ticker"], "strategy_portfolios")
    if (df.groupby([df["date"].astype(str), "strategy"])["target_weight"].sum() > 1 + 1e-6).any():
        raise ValueError("strategy_portfolios: the weights of a portfolio must add up to at most 1")
    return df


def validate_strategy_choices(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy choices: a real date, a known strategy, a reason."""
    _require_columns(df, STRATEGY_CHOICES, "strategy_choices")
    if pd.to_datetime(df["date"].astype(str), format="%Y-%m-%d", errors="coerce").isna().any():
        raise ValueError("strategy_choices: date must be a date (YYYY-MM-DD)")
    _require_in(df, "strategy", set(STRATEGIES), "strategy_choices")
    _require_text(df, "reason", "strategy_choices")
    return df


def validate_chart_history(df: pd.DataFrame) -> pd.DataFrame:
    """Chart history: a known kind, a known strategy for kind strategy, a value above 0, one point per (kind, name,
    date)."""
    _require_columns(df, CHART_HISTORY, "chart_history")
    _require_in(df, "kind", CHART_KINDS, "chart_history")
    _require_in(df[df["kind"] == "strategy"], "name", set(STRATEGIES), "chart_history")
    _require_text(df, "name", "chart_history")
    value = pd.to_numeric(df["value"], errors="coerce")
    if not (np.isfinite(value) & (value > 0)).all():
        raise ValueError("chart_history: value must be a finite number above 0")
    _require_unique(df, ["kind", "name", "date"], "chart_history")
    return df


def _json_objects(df: pd.DataFrame, col: str, name: str) -> None:
    for text in df[col]:
        try:
            ok = isinstance(json.loads(text), dict)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{name}: {col} must be a JSON object")


def validate_customer_log(df: pd.DataFrame) -> pd.DataFrame:
    """Customer log: a customer and a known event on every row, a known archetype and strategy unless the event is a
    reset, and `why` and `detail` as JSON objects."""
    _require_columns(df, CUSTOMER_LOG, "customer_log")
    _require_text(df, "customer", "customer_log")
    _require_in(df, "event", CUSTOMER_EVENTS, "customer_log")
    placed = df[df["event"] != "reset"]
    _require_in(placed, "archetype", set(ARCHETYPES), "customer_log")
    _require_in(placed, "strategy", set(STRATEGIES), "customer_log")
    _json_objects(df, "why", "customer_log")
    _json_objects(df, "detail", "customer_log")
    return df


def validate_etf_sheets(df: pd.DataFrame) -> pd.DataFrame:
    """ETF sheets: a known block, a state that block allows, a finite value, one row per (date, ticker, block, name)."""
    _require_columns(df, ETF_SHEETS, "etf_sheets")
    _require_in(df, "block", set(SHEET_STATES), "etf_sheets")
    states = df["state"].fillna("")
    if not all(state in SHEET_STATES[block] for block, state in zip(df["block"], states)):
        raise ValueError("etf_sheets: a state its block does not allow")
    if not np.isfinite(pd.to_numeric(df["value"], errors="coerce")).all():
        raise ValueError("etf_sheets: value must be a finite number")
    _require_unique(df, ["date", "ticker", "block", "name"], "etf_sheets")
    return df
