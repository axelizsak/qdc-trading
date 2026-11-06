import pandas as pd
import pytest

from agents.portfolio import combine_scores
from common.schemas import validate_orders, validate_portfolio
from data.mock import mock_prices, mock_signals
from data.universe import load_universe


def test_universe_has_50_unique_etfs():
    u = load_universe()
    assert len(u) == 50
    assert u["isin"].is_unique
    assert u["ticker"].is_unique


def test_mock_data_matches_schemas():
    mock_prices(days=30)
    mock_signals()


def test_combine_scores_is_weighted_sum():
    signals = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "agent": ["technical", "macro_news", "technical", "macro_news"],
        "score": [1.0, -1.0, 0.5, 0.5],
    })
    out = combine_scores(signals, {"technical": 0.75, "macro_news": 0.25})
    assert out["A"] == pytest.approx(0.5)
    assert out["B"] == pytest.approx(0.5)


def test_combine_scores_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError):
        combine_scores(pd.DataFrame(columns=["ticker", "agent", "score"]), {"technical": 0.5})


def test_portfolio_and_orders_validation():
    validate_portfolio(pd.DataFrame({"ticker": ["WORLD", "GOLD"], "target_weight": [0.6, 0.3]}))
    with pytest.raises(ValueError):
        validate_portfolio(pd.DataFrame({"ticker": ["WORLD"], "target_weight": [1.2]}))
    with pytest.raises(ValueError):
        validate_orders(pd.DataFrame({"date": ["2026-12-07"], "side": ["HOLD"], "ticker": ["WORLD"], "units": [1]}))
