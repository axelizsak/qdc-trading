"""The 50-ETF universe defined by Belfius (session of 27 Oct 2026), and its exchanges."""
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

# How each service names the exchange: OpenFIGI exchange code and Yahoo symbol suffix.
# GR and IM checked with OpenFIGI, BB checked with BEL 20.
EXCHANGES = {
    "Xetra":             {"openfigi_code": "GR", "yahoo_suffix": ".DE"},
    "Euronext Brussels": {"openfigi_code": "BB", "yahoo_suffix": ".BR"},
    "Euronext Milan":    {"openfigi_code": "IM", "yahoo_suffix": ".MI"},
}


def load_universe() -> pd.DataFrame:
    """Columns: id, theme, ticker, isin, name."""
    return pd.read_csv(UNIVERSE_CSV)


def tickers() -> list[str]:
    return load_universe()["ticker"].tolist()


def reference_exchange(ticker: str) -> str:
    """Exchange Belfius uses for this ETF, e.g. 'Xetra'."""
    if ticker in REFERENCE_EXCHANGE:
        return REFERENCE_EXCHANGE[ticker]
    return DEFAULT_EXCHANGE


def yahoo_suffix(ticker: str) -> str:
    """Yahoo suffix of the ETF's reference exchange, e.g. '.DE' for Xetra."""
    return EXCHANGES[reference_exchange(ticker)]["yahoo_suffix"]
