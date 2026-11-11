from datetime import date

import pandas as pd
import pytest

from data import data_agent


@pytest.fixture
def yahoo_map(tmp_path, monkeypatch):
    path = tmp_path / "etf_yahoo_map.csv"
    pd.DataFrame({
        "ticker": ["WORLD", "S&P 500", "GOLD", "QUANTUM"],
        "isin": ["x1", "x2", "x3", "x4"],
        "status": ["OK", "OK", "OK", "INTROUVABLE"],
        "yahoo_symbol": ["IWDA.AS", "CSPX.L", "SGLN.L", None],
        "currency": ["EUR", "USD", "GBp", None],
    }).to_csv(path, index=False)
    monkeypatch.setattr(data_agent, "YAHOO_MAP_CSV", path)
    return path


def fake_download(symbols, start, end):
    """Constant prices, with a holiday on CSPX.L, so conversions are easy to check."""
    idx = pd.bdate_range(start, end)
    data = {
        "IWDA.AS": 100.0,
        "CSPX.L": 550.0,   # USD
        "SGLN.L": 6000.0,  # pence
        "EURUSD=X": 1.10,  # 1 EUR = 1.10 USD
        "EURGBP=X": 0.85,  # 1 EUR = 0.85 GBP
    }
    df = pd.DataFrame({s: data[s] for s in symbols}, index=idx)
    df.loc[idx[2], "CSPX.L"] = float("nan")
    return df


def test_fetch_prices_converts_to_eur_and_uses_belfius_tickers(yahoo_map):
    p = data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), download=fake_download)
    last = p.groupby("ticker")["close_eur"].last()
    assert set(last.index) == {"WORLD", "S&P 500", "GOLD"}
    assert last["WORLD"] == pytest.approx(100.0)
    assert last["S&P 500"] == pytest.approx(550 / 1.10, abs=1e-3)
    assert last["GOLD"] == pytest.approx(60 / 0.85, abs=1e-3)


def test_fetch_prices_fills_holidays_and_skips_unmapped_etfs(yahoo_map):
    p = data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), download=fake_download)
    assert (p.groupby("ticker").size() == 10).all()  # 10 business days, holiday filled
    assert "QUANTUM" not in set(p["ticker"])


def test_missing_map_tells_you_to_run_the_check_script(tmp_path, monkeypatch):
    monkeypatch.setattr(data_agent, "YAHOO_MAP_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="check_etfs_yfinance"):
        data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), download=fake_download)



def test_reference_exchanges_follow_the_belfius_list():
    from data.universe import tickers, yahoo_suffix
    assert yahoo_suffix("WORLD") == ".DE"
    assert yahoo_suffix("BEL 20") == ".BR"
    assert yahoo_suffix("COPPER LVG") == yahoo_suffix("PALLADIUM LVG") == ".MI"
    assert sum(yahoo_suffix(t) != ".DE" for t in tickers()) == 3


def test_pick_best_prefers_reference_exchange_then_eur():
    from data.check_etfs_yfinance import pick_best

    listings = [
        {"symbol": "QQQ3.L", "currency": "USD", "days_1y": 253},
        {"symbol": "QQQ3.MI", "currency": "EUR", "days_1y": 250},
        {"symbol": "QQQ3.DE", "currency": "EUR", "days_1y": 120},
    ]
    best, status = pick_best("NASDAQ 3X", listings)
    assert (best["symbol"], status) == ("QQQ3.DE", "PARTIAL")

    best, status = pick_best("NASDAQ 3X", listings[:2])
    assert (best["symbol"], status) == ("QQQ3.MI", "OTHER EXCHANGE")

    assert pick_best("NASDAQ 3X", []) == (None, "NOT FOUND")
    assert pick_best("BEL 20", [{"symbol": "B20.BR", "currency": "EUR", "days_1y": 250}])[1] == "OK"
