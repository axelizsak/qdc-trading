"""Common walk-forward backtest. Owner: Person 2.

Input:  a model name from ml_library.MODEL_LIBRARY + prices (date, ticker, close_eur)
Output: one row per (model, ticker) with the same METRICS for every model.
"""
import pandas as pd

# Starting set, to agree as a team. Hit rate is also what the review loop uses.
METRICS = ["hit_rate", "rmse", "mean_return_when_long"]


def run(model: str, prices: pd.DataFrame, train_days: int, horizon_days: int) -> pd.DataFrame:
    """Columns: model, ticker, *METRICS."""
    raise NotImplementedError
