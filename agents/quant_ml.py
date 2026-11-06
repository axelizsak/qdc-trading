"""[4] Quant / ML agent. Owner: Person 3. LLM: model selection only.

Input:  prices + features
Output: best model per ETF from the model library, and its forecast as a signal.

Walk-forward backtest. The agent selects and tunes models but does not write new code.
To add a model (LSTM, mean reversion, regime detection, ...), add it to the
library: it becomes one more candidate the agent tests.
"""
from datetime import date

import pandas as pd

AGENT = "quant_ml"

MODEL_LIBRARY = ["linear_regression", "random_forest", "gradient_boosting", "arima"]


def run(prices: pd.DataFrame, features: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Signals format, agent == "quant_ml"."""
    raise NotImplementedError
