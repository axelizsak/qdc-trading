"""Mock data in the shared formats, so every module can be built before real data exists.

Usage:  python -m data.mock   (writes CSVs to data/mock/, which is gitignored)
"""
import numpy as np
import pandas as pd

from data.config import DATA_DIR, MACRO_FACTORS, SIGNAL_AGENTS
from data.schemas import (
    LIST_SEP,
    STARTER_FEATURES,
    validate_agent_weights,
    validate_evidence,
    validate_features,
    validate_holdings,
    validate_macro_views,
    validate_prices,
    validate_signals,
    validate_source_registry,
)
from data.universe import tickers


def mock_prices(days: int = 260, end: str = "2026-11-30", seed: int = 0) -> pd.DataFrame:
    """Geometric random walk per ETF, business days only. Format: Prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days)
    frames = []
    for t in tickers():
        vol = rng.uniform(0.005, 0.03)
        rets = rng.normal(0.0003, vol, size=days)
        close = 100 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({"date": dates.date, "ticker": t, "close_eur": close.round(4)}))
    return validate_prices(pd.concat(frames, ignore_index=True))


def mock_signals(as_of: str = "2026-11-30", seed: int = 0) -> pd.DataFrame:
    """Random scores for every (agent, ETF). Format: Signals."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "ticker": t,
            "date": pd.Timestamp(as_of).date(),
            "agent": agent,
            "score": round(float(rng.uniform(-1, 1)), 3),
            "confidence": round(float(rng.uniform(0, 1)), 3),
            "rationale": "mock",
        }
        for agent in SIGNAL_AGENTS
        for t in tickers()
    ]
    return validate_signals(pd.DataFrame(rows))


def mock_holdings(n: int = 4, seed: int = 0) -> pd.DataFrame:
    """A few random positions. Format: Holdings."""
    rng = np.random.default_rng(seed)
    picked = rng.choice(tickers(), size=n, replace=False)
    units = rng.integers(1, 40, size=n)
    return validate_holdings(pd.DataFrame({"ticker": picked, "units": units}))


def mock_features(as_of: str = "2026-11-30", seed: int = 0) -> pd.DataFrame:
    """Random values for STARTER_FEATURES on every ETF. Format: Features (long)."""
    rng = np.random.default_rng(seed)
    day = pd.Timestamp(as_of).date()
    rows = [
        {"date": day, "ticker": t, "feature": f, "value": round(float(rng.normal(0, 0.02)), 5)}
        for t in tickers()
        for f in STARTER_FEATURES
    ]
    return validate_features(pd.DataFrame(rows))


def mock_agent_weights(dates: tuple[str, ...] = ("2026-12-07", "2026-12-10"), seed: int = 0) -> pd.DataFrame:
    """Equal weights with no hit rate on the first date, then reviewed weights. Format: Agent weights."""
    rng = np.random.default_rng(seed)
    n = len(SIGNAL_AGENTS)
    frames = []
    for i, d in enumerate(dates):
        weights = np.full(n, 1 / n) if i == 0 else rng.dirichlet(np.ones(n))
        hit_rates = np.full(n, np.nan) if i == 0 else rng.uniform(0.3, 0.7, size=n).round(3)
        frames.append(pd.DataFrame({
            "date": pd.Timestamp(d).date(),
            "agent": SIGNAL_AGENTS,
            "weight": weights,
            "hit_rate": hit_rates,
        }))
    return validate_agent_weights(pd.concat(frames, ignore_index=True))


def mock_evidence(as_of: str = "2026-11-30") -> pd.DataFrame:
    """One row of each kind. Format: Evidence."""
    day = pd.Timestamp(as_of).date()
    rows = [
        {"date": day, "factor": "growth_recession", "source": "fred_t10y2y", "kind": "number",
         "value": 0.42, "summary": "mock: 10y-2y spread, in % points", "url": "https://fred.stlouisfed.org/series/T10Y2Y"},
        {"date": day, "factor": "ecb_rates", "source": "web_news", "kind": "text",
         "value": np.nan, "summary": "mock: ECB speakers lean dovish", "url": "https://example.com/ecb"},
        {"date": day, "factor": "uranium", "source": "mock_exploration_source", "kind": "exploration",
         "value": np.nan, "summary": "mock: unusual signal found by exploration", "url": "https://example.com/uranium"},
    ]
    return validate_evidence(pd.DataFrame(rows))


def mock_macro_views(as_of: str = "2026-11-30", seed: int = 0) -> pd.DataFrame:
    """One random view per macro factor. Format: Macro views."""
    rng = np.random.default_rng(seed)
    day = pd.Timestamp(as_of).date()
    rows = [
        {
            "date": day,
            "factor": f,
            "direction": round(float(rng.uniform(-1, 1)), 3),
            "confidence": round(float(rng.uniform(0, 1)), 3),
            "rationale": "mock",
            "counter_argument": "mock",
            "sources": LIST_SEP.join(["web_news", "fred_t10y2y"]),
        }
        for f in MACRO_FACTORS
    ]
    return validate_macro_views(pd.DataFrame(rows))


def mock_source_registry() -> pd.DataFrame:
    """One active, one testing and one dropped source. Format: Source registry."""
    rows = [
        {"source": "fred_t10y2y", "kind": "number", "factors": LIST_SEP.join(["growth_recession", "fed_us_rates"]),
         "status": "active", "first_seen": "2026-11-16", "n_uses": 5, "hit_rate": 0.6},
        {"source": "mock_exploration_source", "kind": "exploration", "factors": "uranium",
         "status": "testing", "first_seen": "2026-11-30", "n_uses": 0, "hit_rate": np.nan},
        {"source": "mock_dropped_source", "kind": "text", "factors": "crypto",
         "status": "dropped", "first_seen": "2026-11-17", "n_uses": 4, "hit_rate": 0.25},
    ]
    return validate_source_registry(pd.DataFrame(rows))


MOCKS = {
    "prices": mock_prices,
    "signals": mock_signals,
    "holdings": mock_holdings,
    "features": mock_features,
    "agent_weights": mock_agent_weights,
    "evidence": mock_evidence,
    "macro_views": mock_macro_views,
    "source_registry": mock_source_registry,
}


if __name__ == "__main__":
    out = DATA_DIR / "mock"
    out.mkdir(exist_ok=True)
    for name, make in MOCKS.items():
        make().to_csv(out / f"{name}.csv", index=False)
    print(f"Mock data written to {out}")
