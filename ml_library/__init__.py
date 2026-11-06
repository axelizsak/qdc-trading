"""Shared ML model library. Owner: Person 2.

Used by the agents (starting with the Technical agent). It is not an agent itself.
To add a model, add it here and register it in MODEL_LIBRARY. Every model is
evaluated by backtest/ with the same metrics.
"""

# Starting set. Candidates to add: LSTM, mean reversion, regime detection, ...
MODEL_LIBRARY = ["linear_regression", "random_forest", "gradient_boosting", "arima"]
