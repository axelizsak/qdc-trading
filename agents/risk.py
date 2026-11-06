"""[5] Risk Manager. Owner: Person 3. No LLM.

Input:  target weights (Portfolio format) + current holdings, cash and prices
Output: valid orders (Orders format). It controls concentration and excess
        volatility, rejects extreme positions, and enforces the trading rules:
        sufficient cash and at most MAX_ORDERS_PER_DAY orders.
"""
from datetime import date

import pandas as pd

from common.config import MAX_ORDERS_PER_DAY  # noqa: F401


def run(target: pd.DataFrame, holdings: pd.DataFrame, cash_eur: float,
        prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Orders format, at most MAX_ORDERS_PER_DAY rows."""
    raise NotImplementedError
