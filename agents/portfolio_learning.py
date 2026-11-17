"""What the Portfolio Manager keeps for learning, and the price helpers the learning uses. Owner: Person 3. No LLM.

  - return_table / forward_return: closes before a day, and the return from an order's fill price some quotes
    ahead (the fill rule is one setting: data/config.py, fills_from). Only quotes before `as_of` are used, so
    nothing leaks from the future.
  - load_history: every real signal logged so far.
  - record_portfolios / load_records: the portfolios the Portfolio Manager built each day (logs/pm_portfolios.csv).
How much each agent counts, and how much the LLM's portfolio counts against the code's, is learned from these by
the system's one learning rule (agents/learning.py), every morning in graph/review_loop.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from agents import portfolio_config as cfg
from data import config as settings
from data.schemas import PM_PORTFOLIOS, SIGNALS, validate_signals

SOURCES = {"baseline", "optimizer", "llm", "final"}


def return_table(prices: pd.DataFrame, as_of) -> pd.DataFrame:
    """Close per ISO date (rows) and ticker (columns), quotes before `as_of` only.
    Uses the close adjusted for dividends when the source gives one, so a dividend is not a fake loss."""
    close = prices["close_eur"].astype(float)
    if "adj_close_eur" in prices:
        close = pd.to_numeric(prices["adj_close_eur"], errors="coerce").fillna(close)
    table = (pd.DataFrame({"date": prices["date"].astype(str), "ticker": prices["ticker"], "close": close})
             .pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index())
    return table[table.index < str(as_of)]


def forward_return(table: pd.DataFrame, decision_date: str, horizon: int) -> pd.Series | None:
    """Return from the fill price of `decision_date` (last close before it) `horizon` quotes ahead.
    None when those quotes are not known yet."""
    before = np.flatnonzero(settings.fills_from(table.index, decision_date))
    if len(before) == 0 or before[-1] + horizon >= len(table):
        return None
    start = before[-1]
    return table.iloc[start + horizon] / table.iloc[start] - 1


def load_history(logs_dir: Path | str) -> pd.DataFrame:
    """Every real signal logged so far (Signals format), without the neutral rows of a failed agent."""
    path = Path(logs_dir) / "signals.csv"
    if not path.exists():
        return pd.DataFrame(columns=SIGNALS)
    signals = validate_signals(pd.read_csv(path))
    failed = signals["rationale"].astype(str).str.startswith(cfg.FAILED_MARKER)
    return signals[~failed].reset_index(drop=True)


def load_records(logs_dir: Path | str) -> pd.DataFrame:
    path = Path(logs_dir) / "pm_portfolios.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=PM_PORTFOLIOS)


def record_portfolios(logs_dir: Path | str, as_of, portfolios: dict[str, pd.Series]) -> None:
    """Log the day's portfolios (source -> weights), replacing the rows already logged for `as_of`."""
    unknown = set(portfolios) - SOURCES
    if unknown:
        raise ValueError(f"pm_portfolios: unknown source {sorted(unknown)}")
    new = pd.DataFrame([{"date": str(as_of), "source": s, "ticker": t, "weight": float(w)}
                        for s, weights in portfolios.items() for t, w in weights.items()], columns=PM_PORTFOLIOS)
    path = Path(logs_dir) / "pm_portfolios.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"].astype(str) != str(as_of)], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
