import pandas as pd
import pytest

from data.schemas import validate_orders, validate_portfolio
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


def test_portfolio_and_orders_validation():
    validate_portfolio(pd.DataFrame({"ticker": ["WORLD", "GOLD"], "target_weight": [0.6, 0.3]}))
    with pytest.raises(ValueError):
        validate_portfolio(pd.DataFrame({"ticker": ["WORLD"], "target_weight": [1.2]}))
    with pytest.raises(ValueError):
        validate_orders(pd.DataFrame({"date": ["2026-12-07"], "side": ["HOLD"], "ticker": ["WORLD"], "units": [1]}))
