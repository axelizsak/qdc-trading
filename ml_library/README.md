# ml_library

The shared ML models used by the agents. It is a library, not an agent: the Technical agent calls these models today, and other agents can use them later.

**Starting set:** linear regression, random forest, gradient boosting, ARIMA. These are registered in `MODEL_LIBRARY` in `__init__.py`.

**Rules:**
- Each model forecasts the forward return per ETF from prices and features. Prices come from the data agent (`date, ticker, close_eur`).
- Agents select and tune models from this library. They don't write new model code at runtime.
- To add a model (LSTM, mean reversion, regime detection, ...), register it here. It then becomes one more candidate, scored by `backtest/` with the same metrics as the others.
