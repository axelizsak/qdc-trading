"""[2] Technical agent. Owner: Person 2. LLM: interpretation layer only.

Input:  prices (date, ticker, close_eur) + features from the data agent
Steps:
  1. Deterministic indicators: moving averages, momentum, trend, volatility breakouts.
  2. Models from ml_library/, chosen on backtest/ results.
  3. LLM layer that reads 1 + 2 and interprets them.
Output: two signals per ETF in the Signals format
        (ticker, date, agent, score [-1,1], confidence, rationale):
  - agent == "technical_det": deterministic score from indicators + models
  - agent == "technical_llm": LLM-interpreted score
The Portfolio Manager weights the two separately, so the review loop can measure
whether the LLM layer adds value.
"""
from datetime import date

import pandas as pd

AGENT_DET = "technical_det"
AGENT_LLM = "technical_llm"


def run(prices: pd.DataFrame, features: pd.DataFrame, as_of: date, lessons: str = "") -> pd.DataFrame:
    """Signals format, with rows for both AGENT_DET and AGENT_LLM."""
    raise NotImplementedError
