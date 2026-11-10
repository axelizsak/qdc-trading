"""The 50-ETF universe defined by Belfius (session of 27 Oct 2026)."""
import pandas as pd

from data.config import UNIVERSE_CSV

# Exchange Belfius uses to execute and value each ETF (ETF list, Belfius session p. 9-10).
# Xetra for all ETFs except these.
REFERENCE_EXCHANGE = {
    "BEL 20": "Euronext Brussels",
    "COPPER LVG": "Euronext Milan",
    "PALLADIUM LVG": "Euronext Milan",
}
DEFAULT_EXCHANGE = "Xetra"
YAHOO_SUFFIX = {"Xetra": ".DE", "Euronext Brussels": ".BR", "Euronext Milan": ".MI"}


def load_universe() -> pd.DataFrame:
    """Columns: id, theme, ticker, isin, name."""
    return pd.read_csv(UNIVERSE_CSV)


def tickers() -> list[str]:
    return load_universe()["ticker"].tolist()


def reference_exchange(ticker: str) -> str:
    return REFERENCE_EXCHANGE.get(ticker, DEFAULT_EXCHANGE)


def yahoo_suffix(ticker: str) -> str:
    """Yahoo suffix of the ETF's reference exchange, e.g. '.DE' for Xetra."""
    return YAHOO_SUFFIX[reference_exchange(ticker)]
