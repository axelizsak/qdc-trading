"""[5] Risk Manager. Owner: Person 3. No LLM.

Input:  the orders the human approved (Orders format) + current holdings, cash and prices
Output: the portfolio after those orders, assessed on concentration, volatility, drawdown,
        exposure, leverage and liquidity (Risk report format: one row per metric, with its
        limit and a status ok | warning | breach). The human reads it, then accepts it or sends
        the day back to the Portfolio Manager. The trading rules (at most MAX_ORDERS_PER_DAY
        orders, enough cash, no selling more than held) are checked by the pipeline before this step.
"""
from datetime import date

import pandas as pd


def run(orders: pd.DataFrame, holdings: pd.DataFrame, cash_eur: float,
        prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Risk report format, one row per metric."""
    raise NotImplementedError
