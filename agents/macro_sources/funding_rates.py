"""Perpetual futures funding rates from Binance (public API, no key): how crowded crypto longs are."""
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from agents.macro_sources._common import context, evidence, http_client, ordinal

SOURCE = "binance_funding"
API = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOLS = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ether"}
PER_DAY = 3  # funding every 8 hours


def collect(as_of: date, http=None) -> list[dict]:
    http = http or http_client()
    end = int(datetime.combine(as_of + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    for symbol, name in SYMBOLS.items():
        response = http.get(API, params={"symbol": symbol, "limit": 1000, "endTime": end})
        response.raise_for_status()
        rates = pd.Series([float(r["fundingRate"]) for r in response.json()], dtype=float)
        if len(rates) < 90:
            continue
        three_days = rates.rolling(3 * PER_DAY).mean().dropna() * PER_DAY * 365  # annualised
        ctx = context(three_days, lookback=len(three_days))
        rows.append(evidence(
            as_of, "crypto", SOURCE, "number",
            f"{name} perpetual funding (Binance), 3-day average {ctx['last']:+.1%} a year (z {ctx['z']:+.1f}, "
            f"{ordinal(ctx['pct'])} percentile of {len(rates) // PER_DAY} days; high positive = crowded longs paying "
            f"shorts, negative = shorts crowded)",
            ctx["last"], "https://www.binance.com/en/futures/funding-history/perpetual/real-time-funding-rate"))
    if not rows:
        raise RuntimeError("no funding-rate history")
    return rows
