"""More evidence for the Portfolio Manager's learning. Owner: Person 3. No LLM.

The technical score needs only prices, so it can be recomputed for past days. For each decision day D the score is
built from prices and indicators strictly before D (what the live run sees in the morning of D), so nothing leaks
from the future. The macro score cannot be backfilled: it reads the news of its own day.
The result is only passed to `portfolio.run(extra_history=...)`: it is never written to logs/signals.csv,
which stays the record of what the agents really said.
"""
from datetime import date

import pandas as pd

from agents import technical
from data.features import compute_features
from data.schemas import SIGNALS


def quote_days(prices: pd.DataFrame, n: int, before: date) -> list[date]:
    """The last `n` days with a quote strictly before `before`, oldest first."""
    days = sorted(set(pd.to_datetime(prices["date"]).dt.date))
    return [d for d in days if d < before][-n:]


def backfill_technical(prices: pd.DataFrame, days: list[date], features: pd.DataFrame | None = None) -> pd.DataFrame:
    """technical_det signals (Signals format) for each day of `days`, as the live run would have given them."""
    prices = prices.assign(date=pd.to_datetime(prices["date"]).dt.date)
    features = compute_features(prices) if features is None else features
    features = features.assign(date=pd.to_datetime(features["date"]).dt.date)
    frames = []
    for day in days:
        known_prices = prices[prices["date"] < day]
        known_features = features[features["date"] < day]
        if known_prices.empty or known_features.empty:
            continue
        sheet = technical.build_sheet(known_prices, known_features, known_prices["date"].max())
        if sheet.empty:
            continue
        signals = technical.signals_from_sheet(sheet)
        signals["date"] = day.isoformat()
        frames.append(signals)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SIGNALS)
