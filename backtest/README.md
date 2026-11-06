# backtest

A common walk-forward backtest for every model in `ml_library/`, and later for agent signals too.

Every model is scored with **the same metrics** on **the same splits**, so results are comparable and an agent can pick the best model per ETF on evidence.

- **Walk-forward:** train on data up to day *t*, forecast the next period, move forward, repeat. The model never sees future data.
- **Metrics:** the starting set is in `walk_forward.py` (`METRICS`) and still has to be agreed as a team. Hit rate is in it on purpose, because the review loop uses hit rate to update agent weights.
