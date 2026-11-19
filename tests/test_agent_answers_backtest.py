"""Tests for backtest/agent_answers_backtest.py: no network, no logs, small made-up numbers."""

import numpy as np
import pandas as pd

from backtest.agent_answers_backtest import TOP, sharpe, top_weights


def test_top_weights_takes_the_best_scores_at_equal_weight():
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, -0.2])
    weights = top_weights(scores)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights.iloc[:TOP], 1 / TOP)
    assert np.allclose(weights.iloc[TOP:], 0.0)


def test_top_weights_shares_the_last_places_between_tied_etfs():
    # two ETFs above the cut, four tied at 0 for the three places left
    scores = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    weights = top_weights(scores)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights.iloc[:2], 1 / TOP)
    assert np.allclose(weights.iloc[2:6], 3 / 4 / TOP)
    assert weights.iloc[6] == 0.0


def test_sharpe_is_growth_a_year_over_volatility_a_year():
    period_returns = pd.Series([0.02, -0.01, 0.03, 0.0])
    hold = 12
    periods_a_year = 252 / hold
    growth = (1 + period_returns).prod() ** (periods_a_year / 4) - 1
    volatility = period_returns.std() * np.sqrt(periods_a_year)
    assert np.isclose(sharpe(period_returns, hold), growth / volatility)
