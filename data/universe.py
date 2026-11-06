"""The 50-ETF universe defined by Belfius (session of 27 Oct 2026)."""
import pandas as pd

from data.config import UNIVERSE_CSV


def load_universe() -> pd.DataFrame:
    """Columns: id, theme, ticker, isin, name."""
    return pd.read_csv(UNIVERSE_CSV)


def tickers() -> list[str]:
    return load_universe()["ticker"].tolist()
