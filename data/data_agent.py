"""[1] Data agent. Owner: Person 2. No LLM.

Input:  ISIN list (data/etf_universe.csv)
Output: clean prices in the Prices format (date, ticker, close_eur),
        plus returns, volatility, drawdowns and correlations.

Source: yfinance, EUR listing preferred.
Fallback: another listing of the same ETF + FX conversion, or a manual export
from justETF / Investing.com. The ISIN -> Yahoo symbol mapping is produced by
data/check_etfs_yfinance.py (data/etf_yahoo_map.csv).
"""
from datetime import date

import pandas as pd


def fetch_prices(start: date, end: date) -> pd.DataFrame:
    """Prices for every ETF in the universe, in EUR. Format: Prices."""
    raise NotImplementedError


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns, volatility, drawdowns, correlations."""
    raise NotImplementedError
