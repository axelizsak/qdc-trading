"""[2] Technical agent. Owner: Person 2. No LLM.

Input:  prices (Prices format)
Output: one signal per ETF (Signals format), from moving averages, momentum,
        trend detection and volatility breakouts.
"""
from datetime import date

import pandas as pd

AGENT = "technical"


def run(prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Signals format, agent == "technical"."""
    raise NotImplementedError
