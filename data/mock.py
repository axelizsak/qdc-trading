"""Mock data in the shared formats, so every module can be built before real data exists.

Usage:  python -m data.mock   (writes CSVs to data/mock/, which is gitignored)
"""
import numpy as np
import pandas as pd

from data.config import DATA_DIR, SIGNAL_AGENTS
from data.schemas import validate_prices, validate_signals
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


if __name__ == "__main__":
    out = DATA_DIR / "mock"
    out.mkdir(exist_ok=True)
    mock_prices().to_csv(out / "prices.csv", index=False)
    mock_signals().to_csv(out / "signals.csv", index=False)
    print(f"Mock data written to {out}")
