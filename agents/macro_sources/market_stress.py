"""Market stress from yfinance: VIX term structure, SKEW, MOVE, copper/gold ratio. No key."""
from datetime import date, timedelta

import pandas as pd

from agents.macro_sources._common import context, describe, evidence

SOURCE = "market_stress"
URL = "https://finance.yahoo.com"


def _download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(symbols, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
    return data["Close"]


def collect(as_of: date, download=_download) -> list[dict]:
    close = download(["^VIX", "^VIX3M", "^SKEW", "^MOVE", "HG=F", "GC=F"], as_of - timedelta(days=400), as_of)
    close = close[close.index <= pd.Timestamp(as_of)]
    rows = []

    def add(factor, name, series, unit="", digits=2):
        series = series.dropna()
        if len(series) > 20:
            ctx = context(series)
            rows.append(evidence(as_of, factor, SOURCE, "number", describe(name, ctx, unit, digits),
                                 ctx["last"], URL))

    add("risk_appetite", "VIX", close["^VIX"])
    add("risk_appetite", "VIX/VIX3M (above 1 = inverted, stress)", close["^VIX"] / close["^VIX3M"])
    add("risk_appetite", "CBOE SKEW (tail-risk demand)", close["^SKEW"], digits=1)
    add("fed_us_rates", "MOVE (US rates volatility)", close["^MOVE"], digits=1)
    add("growth_recession", "Copper/gold ratio x1000", close["HG=F"] / close["GC=F"] * 1000, digits=3)
    if not rows:
        raise RuntimeError("no market stress series available")
    return rows
