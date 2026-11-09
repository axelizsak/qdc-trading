from datetime import date

import pandas as pd
import pytest

from data import data_agent


@pytest.fixture
def yahoo_map(tmp_path, monkeypatch):
    path = tmp_path / "etf_yahoo_map.csv"
    pd.DataFrame({
        "name": ["WORLD", "S&P 500", "GOLD", "QUANTUM"],
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


def test_fetch_prices_fills_holidays_and_reports_gaps(yahoo_map):
    p = data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), download=fake_download)
    assert (p.groupby("ticker").size() == 10).all()  # 10 business days, holiday filled
    assert "QUANTUM" in data_agent.last_report["missing"]
    assert data_agent.last_report["no_data"] == []


def test_fetch_prices_flags_suspicious_jumps(yahoo_map):
    def jumpy(symbols, start, end):
        df = fake_download(symbols, start, end)
        df.iloc[5:, df.columns.get_loc("IWDA.AS")] = 200.0
        return df

    data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), download=jumpy)
    assert [j[0] for j in data_agent.last_report["jumps"]] == ["WORLD"]


def test_missing_map_tells_you_to_run_the_check_script(tmp_path, monkeypatch):
    monkeypatch.setattr(data_agent, "YAHOO_MAP_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="check_etfs_yfinance"):
        data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), download=fake_download)
