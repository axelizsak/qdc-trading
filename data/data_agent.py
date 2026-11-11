"""[1] Data agent, step 2: daily OHLC prices of the ETFs. Owner: Person 2. No LLM.

Input:  data/etf_symbols.csv, written once by step 1 (`python -m data.symbols`)
Output: prices in the Prices format (date, ticker, open_eur, high_eur, low_eur, close_eur),
        plus quality checks (check_prices) and, later, features (compute_features).

Where the prices come from:
  - ETFs with status OK:     Yahoo Finance, on the Belfius reference exchange (yahoo_ohlc)
  - ETFs in EODHD_SYMBOLS:   EODHD, official Xetra prices (eodhd_ohlc). Needs EODHD_API_KEY in .env.
Both return the same table, so everything after the download is shared.

Rules (official Belfius Q&A, and ours):
  - Real prices, not adjusted for dividends: Belfius values the portfolio at real closes
    and does not credit dividends. All in EUR, on the exchange Belfius uses.
  - Closed days only: a row dated today would be an intraday price, not a close.
  - Real prices only: nothing is added, filled or moved. Each ETF keeps its own trading days
    on its own exchange; a day without a quote is simply absent (NA in a wide table).
    Indicators are computed ETF by ETF on each ETF's last available prices.
  - An ETF without a price is left out and reported, never replaced by another exchange's price.

Quick run on real data:  python -m data.data_agent
"""
import os
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import requests

from data.config import DATA_DIR  # importing data.config also reads the .env file
from data.schemas import validate_prices
from data.symbols import EODHD_SYMBOLS, load_symbols

PRICED_STATUSES = ["OK", "EODHD"]
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]

EODHD_URL = "https://eodhd.com/api/eod/"
EODHD_FREE_HISTORY_DAYS = 365   # the free plan only gives the past year

BIG_MOVE_PERCENT = 20           # daily moves above this are reported by check_prices
MISSING_DAYS_LOOKBACK = 365     # check_prices lists missing days over this many days

Downloader = Callable[[str, date, date], pd.DataFrame]


# ---------- Sources: one function per source, same output --------------------------------

def yahoo_ohlc(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily Open, High, Low, Close of one Yahoo symbol, real prices. `end` is included."""
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end + timedelta(days=1),
                      auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=OHLC_COLUMNS)

    # Recent yfinance versions add the symbol as a second column level: keep only the field names
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    ohlc = raw[OHLC_COLUMNS].copy()
    ohlc.index = pd.to_datetime(ohlc.index).tz_localize(None).normalize()
    return ohlc.dropna(subset=["Close"])


def eodhd_ohlc(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily Open, High, Low, Close of one EODHD symbol (official Xetra prices). `end` is included.

    Reads EODHD_API_KEY from the environment (.env). Never prints the request URL: it contains the key.
    """
    api_key = os.getenv("EODHD_API_KEY")
    if api_key is None or api_key == "":
        print(f"  EODHD_API_KEY is missing from .env: {symbol} is left out")
        return pd.DataFrame(columns=OHLC_COLUMNS)

    # The free plan only covers the past year
    earliest = end - timedelta(days=EODHD_FREE_HISTORY_DAYS)
    if start < earliest:
        start = earliest

    params = {"api_token": api_key, "fmt": "json", "from": start.isoformat(), "to": end.isoformat()}
    response = requests.get(EODHD_URL + symbol, params=params, timeout=30)
    if response.status_code != 200:
        print(f"  EODHD error {response.status_code} for {symbol}")
        return pd.DataFrame(columns=OHLC_COLUMNS)

    rows = []
    for bar in response.json():
        rows.append({"date": bar["date"], "Open": bar["open"], "High": bar["high"],
                     "Low": bar["low"], "Close": bar["close"]})
    if len(rows) == 0:
        return pd.DataFrame(columns=OHLC_COLUMNS)

    ohlc = pd.DataFrame(rows)
    ohlc.index = pd.to_datetime(ohlc["date"])
    ohlc = ohlc[OHLC_COLUMNS]
    return ohlc.dropna(subset=["Close"])


# ---------- Step 2: the prices table -----------------------------------------------------

def fetch_prices(start: date, end: date,
                 yahoo: Downloader = yahoo_ohlc,
                 eodhd: Downloader = eodhd_ohlc,
                 symbols: pd.DataFrame | None = None,
                 today: date | None = None) -> pd.DataFrame:
    """Real daily OHLC prices of every priceable ETF, in EUR. Format: Prices.

    `yahoo`, `eodhd`, `symbols` and `today` are injectable so tests never call the network.
    """
    if symbols is None:
        symbols = load_symbols()
    if today is None:
        today = date.today()

    # Closed days only: today's row would be an intraday price, not a close
    last_closed_day = today - timedelta(days=1)
    if end > last_closed_day:
        end = last_closed_day

    frames = []
    for etf in symbols.itertuples():
        if etf.status not in PRICED_STATUSES:
            continue

        if etf.ticker in EODHD_SYMBOLS:
            source = "EODHD"
            source_symbol = EODHD_SYMBOLS[etf.ticker]
            ohlc = eodhd(source_symbol, start, end)
        else:
            source = "Yahoo"
            source_symbol = etf.yahoo_symbol
            ohlc = yahoo(source_symbol, start, end)

        # Keep only the requested period, whatever the source returned (e.g. an intraday row for today)
        ohlc.index = pd.to_datetime(ohlc.index)
        in_period = (ohlc.index.date >= start) & (ohlc.index.date <= end)
        ohlc = ohlc[in_period]

        if ohlc.empty:
            print(f"{etf.ticker:18} {source:6} {source_symbol:11} no data for this period")
            continue

        frames.append(pd.DataFrame({
            "date": ohlc.index.date,
            "ticker": etf.ticker,
            "open_eur": ohlc["Open"].astype(float).round(4).values,
            "high_eur": ohlc["High"].astype(float).round(4).values,
            "low_eur": ohlc["Low"].astype(float).round(4).values,
            "close_eur": ohlc["Close"].astype(float).round(4).values,
        }))

    if len(frames) == 0:
        prices = pd.DataFrame(columns=["date", "ticker", "open_eur", "high_eur", "low_eur", "close_eur"])
    else:
        prices = pd.concat(frames, ignore_index=True)
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return validate_prices(prices)


# ---------- Quality checks: they report, they never change the data ----------------------

def check_prices(prices: pd.DataFrame, symbols: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Six reports on the prices table. Each value is a table; an empty table means nothing to report.

    coverage      first and last date, number of days and last close per ETF
    stale         ETFs without a price on the most recent day of the table
    missing_days  over the last year, days where some ETFs have a price and others do not
    bad_bars      prices <= 0, high below open/close, low above open/close
    big_moves     daily moves above BIG_MOVE_PERCENT, on each ETF's own series
    left_out      ETFs of the universe without any price, with their status
    """
    if symbols is None:
        symbols = load_symbols()
    reports = {}

    # 1. Coverage
    reports["coverage"] = prices.groupby("ticker").agg(
        first=("date", "min"), last=("date", "max"),
        days=("date", "size"), last_close=("close_eur", "last"),
    ).reset_index()

    # 2. Freshness
    latest_day = prices["date"].max()
    stale = []
    for row in reports["coverage"].itertuples():
        if row.last < latest_day:
            stale.append({"ticker": row.ticker, "last_day": row.last, "latest_day": latest_day})
    reports["stale"] = pd.DataFrame(stale, columns=["ticker", "last_day", "latest_day"])

    # 3. Missing days over the last year
    one_year_ago = latest_day - timedelta(days=MISSING_DAYS_LOOKBACK)
    recent = prices[prices["date"] >= one_year_ago]
    recent_tickers = sorted(recent["ticker"].unique())
    missing = []
    for day, group in recent.groupby("date"):
        present = set(group["ticker"])
        absent = []
        for ticker in recent_tickers:
            if ticker not in present:
                absent.append(ticker)
        if len(absent) > 0:
            missing.append({"date": day, "n_missing": len(absent), "n_quoted": len(present),
                            "missing": ", ".join(absent)})
    reports["missing_days"] = pd.DataFrame(missing, columns=["date", "n_missing", "n_quoted", "missing"])

    # 4. Consistency of each bar
    not_positive = (prices[["open_eur", "high_eur", "low_eur", "close_eur"]] <= 0).any(axis=1)
    high_too_low = prices["high_eur"] < prices[["open_eur", "close_eur"]].max(axis=1)
    low_too_high = prices["low_eur"] > prices[["open_eur", "close_eur"]].min(axis=1)
    reports["bad_bars"] = prices[not_positive | high_too_low | low_too_high].reset_index(drop=True)

    # 5. Big daily moves, on each ETF's own series (previous available close)
    big_moves = []
    for ticker, ticker_rows in prices.groupby("ticker"):
        previous_close = None
        for row in ticker_rows.itertuples():
            if previous_close is not None:
                move = (row.close_eur / previous_close - 1) * 100
                if abs(move) > BIG_MOVE_PERCENT:
                    big_moves.append({"ticker": ticker, "date": row.date, "previous_close": previous_close,
                                      "close": row.close_eur, "move_percent": round(move, 2)})
            previous_close = row.close_eur
    reports["big_moves"] = pd.DataFrame(
        big_moves, columns=["ticker", "date", "previous_close", "close", "move_percent"])

    # 6. ETFs left out
    priced = set(prices["ticker"])
    left_out = []
    for etf in symbols.itertuples():
        if etf.ticker not in priced:
            left_out.append({"ticker": etf.ticker, "yahoo_symbol": etf.yahoo_symbol, "status": etf.status})
    reports["left_out"] = pd.DataFrame(left_out, columns=["ticker", "yahoo_symbol", "status"])

    return reports


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns, volatility, drawdowns, correlations."""
    raise NotImplementedError


if __name__ == "__main__":
    end = date.today() - timedelta(days=1)
    p = fetch_prices(date(2020, 1, 1), end)
    out = DATA_DIR / "cache" / f"prices_ohlc_{end.isoformat()}.csv"
    out.parent.mkdir(exist_ok=True)
    p.to_csv(out, index=False)

    reports = check_prices(p)
    print(f"\n{p['ticker'].nunique()} ETFs priced, {len(p)} rows. Written to {out}")
    for name in reports:
        if name == "coverage":
            continue
        report = reports[name]
        print(f"\n{name}: {len(report)} rows")
        if len(report) > 0:
            print(report.head(15).to_string(index=False))
