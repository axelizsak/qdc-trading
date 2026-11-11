"""Data agent: step 1 (data/symbols.py) and step 2 (data/data_agent.py). No network: fake sources."""
from datetime import date

import pandas as pd
import pytest

from data import data_agent, symbols
from data.schemas import PRICES

DAYS = pd.bdate_range("2026-11-02", "2026-11-13")   # 10 business days


def fake_ohlc(close: float, missing_day=None) -> pd.DataFrame:
    """Constant bars, optionally with one day without a quote."""
    index = []
    for day in DAYS:
        if day != missing_day:
            index.append(day)
    rows = {"Open": close * 0.99, "High": close * 1.02, "Low": close * 0.98, "Close": close}
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def fake_yahoo(symbol, start, end):
    prices = {"EUNL.DE": 100.0, "BEL.BR": 50.0, "LSK8.DE": 0.4}
    if symbol not in prices:
        return pd.DataFrame(columns=data_agent.OHLC_COLUMNS)
    missing_day = DAYS[2] if symbol == "BEL.BR" else None   # a Brussels holiday
    bars = fake_ohlc(prices[symbol], missing_day)
    keep = []
    for day in bars.index:
        if start <= day.date() <= end:
            keep.append(day)
    return bars.loc[keep]


def fake_eodhd(symbol, start, end):
    assert symbol == "3QQQ.XETRA"
    return fake_ohlc(300.0)


SYMBOLS = pd.DataFrame([
    {"ticker": "WORLD", "yahoo_symbol": "EUNL.DE", "status": "OK"},
    {"ticker": "BEL 20", "yahoo_symbol": "BEL.BR", "status": "OK"},
    {"ticker": "STOXX SHORT", "yahoo_symbol": "LSK8.DE", "status": "OK"},
    {"ticker": "NASDAQ 3X", "yahoo_symbol": None, "status": "EODHD"},
    {"ticker": "QUANTUM", "yahoo_symbol": "QUTM.DE", "status": "NO DATA"},
])


def fetch(**kwargs):
    arguments = {"start": date(2026, 11, 2), "end": date(2026, 11, 13), "yahoo": fake_yahoo,
                 "eodhd": fake_eodhd, "symbols": SYMBOLS, "today": date(2026, 11, 20)}
    arguments.update(kwargs)
    return data_agent.fetch_prices(**arguments)


# ---------- Step 2: fetch_prices ----------

def test_fetch_prices_returns_ohlc_with_belfius_tickers():
    p = fetch()
    assert list(p.columns) == PRICES
    assert set(p["ticker"]) == {"WORLD", "BEL 20", "STOXX SHORT", "NASDAQ 3X"}
    last = p.groupby("ticker").last()
    assert last.loc["WORLD", "close_eur"] == pytest.approx(100.0)
    assert last.loc["WORLD", "high_eur"] == pytest.approx(102.0)


def test_nasdaq_3x_comes_from_eodhd_and_no_data_etfs_are_left_out():
    p = fetch()
    assert p[p["ticker"] == "NASDAQ 3X"]["close_eur"].iloc[-1] == pytest.approx(300.0)
    assert "QUANTUM" not in set(p["ticker"])


def test_a_day_without_quote_stays_absent():
    p = fetch()
    days = p.groupby("ticker").size()
    assert days["WORLD"] == 10
    assert days["BEL 20"] == 9                       # the Brussels holiday is not filled
    assert DAYS[2].date() not in set(p[p["ticker"] == "BEL 20"]["date"])


def test_only_closed_days_are_kept():
    p = fetch(today=date(2026, 11, 10))              # run on 10/11: last close is 09/11
    assert p["date"].max() == date(2026, 11, 9)


def test_missing_symbols_table_tells_you_to_run_step_1(tmp_path, monkeypatch):
    monkeypatch.setattr(symbols, "SYMBOLS_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="python -m data.symbols"):
        data_agent.fetch_prices(date(2026, 11, 2), date(2026, 11, 13), yahoo=fake_yahoo, eodhd=fake_eodhd)


def test_eodhd_without_api_key_leaves_the_etf_out(monkeypatch):
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    bars = data_agent.eodhd_ohlc("3QQQ.XETRA", date(2026, 11, 2), date(2026, 11, 13))
    assert bars.empty


# ---------- Quality checks ----------

def test_check_prices_reports_without_changing_the_data():
    p = fetch()
    before = p.copy()
    p.loc[(p["ticker"] == "WORLD") & (p["date"] == date(2026, 11, 13)), "close_eur"] = 130.0   # +30 %
    reports = data_agent.check_prices(p, SYMBOLS)

    assert list(reports) == ["coverage", "stale", "missing_days", "bad_bars", "big_moves", "left_out"]
    assert len(reports["stale"]) == 0
    assert reports["missing_days"]["date"].tolist() == [DAYS[2].date()]
    assert reports["missing_days"]["missing"].iloc[0] == "BEL 20"
    assert reports["big_moves"]["ticker"].tolist() == ["WORLD"]
    assert reports["left_out"]["ticker"].tolist() == ["QUANTUM"]
    assert len(p) == len(before)


def test_check_prices_finds_impossible_bars_and_stale_etfs():
    p = fetch()
    p.loc[p.index[0], "high_eur"] = 1.0                                        # high below the close
    p = p[~((p["ticker"] == "WORLD") & (p["date"] == date(2026, 11, 13)))]   # WORLD one day late
    reports = data_agent.check_prices(p, SYMBOLS)
    assert len(reports["bad_bars"]) == 1
    assert reports["stale"]["ticker"].tolist() == ["WORLD"]


# ---------- Step 1: symbols ----------

class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_find_yahoo_symbol_adds_the_suffix_of_the_exchange():
    def post(url, json, timeout):
        assert json[0]["exchCode"] == "GR"
        return FakeResponse(200, [{"data": [{"ticker": "EUNL"}]}])

    assert symbols.find_yahoo_symbol("IE00B4L5Y983", "Xetra", post=post) == "EUNL.DE"


def test_find_yahoo_symbol_returns_none_when_openfigi_knows_nothing():
    def post(url, json, timeout):
        return FakeResponse(200, [{"warning": "No identifier found."}])

    assert symbols.find_yahoo_symbol("XX0000000000", "Xetra", post=post) is None


def test_build_symbols_uses_fixes_skips_eodhd_and_sets_statuses():
    universe = pd.DataFrame([
        {"ticker": "WORLD", "isin": "IE00B4L5Y983"},
        {"ticker": "STOXX SHORT", "isin": "FR0010424143"},
        {"ticker": "NASDAQ 3X", "isin": "IE00BLRPRL42"},
        {"ticker": "QUANTUM", "isin": "IE0007Y8Y157"},
        {"ticker": "CHINA A", "isin": "IE00BQT3WG13"},
    ])
    asked_openfigi = []

    def find(isin, exchange):
        asked_openfigi.append(isin)
        return {"IE00B4L5Y983": "EUNL.DE", "IE0007Y8Y157": "QUTM.DE"}.get(isin)

    def check(symbol):
        if symbol == "QUTM.DE":
            return {"days_1y": 0, "currency": None, "last_date": None}
        return {"days_1y": 253, "currency": "EUR", "last_date": date(2026, 11, 10)}

    table = symbols.build_symbols(universe, find=find, check=check, pause_seconds=0)
    status = dict(zip(table["ticker"], table["status"]))
    assert status == {"WORLD": "OK", "STOXX SHORT": "OK", "NASDAQ 3X": "EODHD",
                      "QUANTUM": "NO DATA", "CHINA A": "NOT FOUND"}
    assert table.set_index("ticker").loc["STOXX SHORT", "yahoo_symbol"] == "LSK8.DE"
    assert "FR0010424143" not in asked_openfigi      # manual fix: no OpenFIGI call
    assert "IE00BLRPRL42" not in asked_openfigi      # EODHD: nothing done in step 1
    assert list(table.columns) == symbols.SYMBOLS_COLUMNS


# ---------- Universe ----------

def test_reference_exchanges_follow_the_belfius_list():
    from data.universe import tickers, yahoo_suffix
    assert yahoo_suffix("WORLD") == ".DE"
    assert yahoo_suffix("BEL 20") == ".BR"
    assert yahoo_suffix("COPPER LVG") == yahoo_suffix("PALLADIUM LVG") == ".MI"
    not_xetra = 0
    for ticker in tickers():
        if yahoo_suffix(ticker) != ".DE":
            not_xetra += 1
    assert not_xetra == 3


# ---------- Unit splits ----------

def split_history(jump: bool) -> pd.DataFrame:
    """EURO BANKS around its split of 1 for 80 on 26/10/2026, with or without the jump in the history."""
    days = pd.DatetimeIndex(["2026-10-21", "2026-10-22", "2026-10-26", "2026-10-27"])
    if jump:
        close = [348.0, 349.85, 4.4275, 4.45]
    else:
        close = [4.35, 4.3731, 4.4275, 4.45]
    bars = pd.DataFrame({"Close": close}, index=days)
    bars["Open"] = bars["Close"]
    bars["High"] = bars["Close"]
    bars["Low"] = bars["Close"]
    return bars[data_agent.OHLC_COLUMNS]


def test_split_fix_divides_the_prices_before_the_split():
    fixed = data_agent.apply_split_fix("EURO BANKS", split_history(jump=True))
    assert fixed.loc["2026-10-22", "Close"] == pytest.approx(349.85 / 80)
    assert fixed.loc["2026-10-22", "High"] == pytest.approx(349.85 / 80)
    assert fixed.loc["2026-10-26", "Close"] == pytest.approx(4.4275)        # after the split: unchanged


def test_split_fix_is_not_applied_twice():
    history = split_history(jump=False)                                     # the source already adjusted it
    fixed = data_agent.apply_split_fix("EURO BANKS", history)
    assert fixed["Close"].tolist() == history["Close"].tolist()
    other = data_agent.apply_split_fix("WORLD", split_history(jump=True))     # no split known for WORLD
    assert other.loc["2026-10-22", "Close"] == pytest.approx(349.85)
