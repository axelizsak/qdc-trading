"""[1] Data agent, step 2: daily OHLC prices of the ETFs. Owner: Person 2. No LLM.

Input:  data/etf_symbols.csv, written once by step 1 (`python -m data.symbols`)
Output: prices in the Prices format (date, ticker, open_eur, high_eur, low_eur, close_eur, adj_close_eur, volume),
        plus quality checks (check_prices) and indicators (compute_features, in data/features.py).

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
    Only exception: a unit split the source did not apply to the history (SPLIT_FIXES). The prices
    before it are divided by the split ratio and the volumes multiplied by it, so the history is
    continuous; no value really changes.
  - Two closes: close_eur is the real close (valuation, orders, risk: what Belfius uses);
    adj_close_eur is the close adjusted for dividends by the source (indicators and model training:
    no fake drop on ex-dividend days). Equal for accumulating ETFs. NA when the source gives none.
  - Volume is the number of units traded on the Belfius exchange only (an ETF also trades off-exchange),
    so it is partial. NA when the source gives none, 0 when the ETF did not trade that day.
  - An ETF without a price is left out and reported, never replaced by another exchange's price.

Quick run on real data:  python -m data.data_agent
"""
import os
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import requests

from data.config import DATA_DIR  # importing data.config also reads the .env file
from data.schemas import PRICES, validate_prices
from data.symbols import EODHD_SYMBOLS, load_symbols

PRICED_STATUSES = ["OK", "EODHD"]
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
BAR_COLUMNS = OHLC_COLUMNS + ["AdjClose", "Volume"]   # what each source function returns
ADJUSTED_TOLERANCE = 0.001      # check_prices: adjusted close counted as different above 0.1 %

EODHD_URL = "https://eodhd.com/api/eod/"
EODHD_FREE_HISTORY_DAYS = 365   # the free plan only gives the past year

# Unit splits the source did not apply to the history: {Belfius ticker: split}.
# Prices before `first_day` are divided by `ratio`, so the history is continuous (same number of units
# as today). Prices from `first_day` on, the ones Belfius uses, are not changed.
# Applied only if the jump is still in the data (if the source fixes its history, nothing is divided twice).
SPLIT_FIXES = {
    # Split of 1 for 80. Source: Deutsche Boerse adjusted history, checked on two days:
    # 09/09/2026 Yahoo 401.05 / Deutsche Boerse 5.0131 = 80.00; 22/10/2026 Yahoo 349.85 / 4.3731 = 80.00.
    "EURO BANKS": {"first_day": date(2026, 10, 26), "ratio": 80},
}

BIG_MOVE_PERCENT = 20           # daily moves above this are reported by check_prices
MISSING_DAYS_LOOKBACK = 365     # check_prices lists missing days over this many days

Downloader = Callable[[str, date, date], pd.DataFrame]


# ---------- Sources: one function per source, same output --------------------------------

def yahoo_ohlc(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily Open, High, Low, Close, AdjClose and Volume of one Yahoo symbol. `end` is included.

    Open/High/Low/Close are real prices; AdjClose is Yahoo's close adjusted for dividends and splits.
    """
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end + timedelta(days=1),
                      auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    # Recent yfinance versions add the symbol as a second column level: keep only the field names
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    bars = raw[OHLC_COLUMNS].copy()
    if "Adj Close" in raw.columns:
        bars["AdjClose"] = raw["Adj Close"]
    else:
        bars["AdjClose"] = float("nan")     # no adjusted close from the source: left empty
    if "Volume" in raw.columns:
        bars["Volume"] = raw["Volume"]
    else:
        bars["Volume"] = float("nan")       # no volume from the source: left empty, never invented
    bars.index = pd.to_datetime(bars.index).tz_localize(None).normalize()
    return bars.dropna(subset=["Close"])


def eodhd_ohlc(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily Open, High, Low, Close, AdjClose and Volume of one EODHD symbol (official Xetra prices). `end` is included.

    Reads EODHD_API_KEY from the environment (.env). Never prints the request URL: it contains the key.
    """
    api_key = os.getenv("EODHD_API_KEY")
    if api_key is None or api_key == "":
        print(f"  EODHD_API_KEY is missing from .env: {symbol} is left out")
        return pd.DataFrame(columns=BAR_COLUMNS)

    # The free plan only covers the past year
    earliest = end - timedelta(days=EODHD_FREE_HISTORY_DAYS)
    if start < earliest:
        start = earliest

    params = {"api_token": api_key, "fmt": "json", "from": start.isoformat(), "to": end.isoformat()}
    try:
        response = requests.get(EODHD_URL + symbol, params=params, timeout=30)
    except requests.RequestException as exc:
        # The error text holds the request URL, so the key: never print it or let it reach logs/incidents.csv
        print(f"  EODHD request failed for {symbol} ({type(exc).__name__}): {symbol} is left out")
        return pd.DataFrame(columns=BAR_COLUMNS)
    if response.status_code != 200:
        print(f"  EODHD error {response.status_code} for {symbol}")
        return pd.DataFrame(columns=BAR_COLUMNS)

    rows = []
    for bar in response.json():
        rows.append({"date": bar["date"], "Open": bar["open"], "High": bar["high"],
                     "Low": bar["low"], "Close": bar["close"], "AdjClose": bar.get("adjusted_close"),
                     "Volume": bar.get("volume")})
    if len(rows) == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)

    bars = pd.DataFrame(rows)
    bars.index = pd.to_datetime(bars["date"])
    bars = bars[BAR_COLUMNS]
    bars["AdjClose"] = pd.to_numeric(bars["AdjClose"], errors="coerce")   # missing -> NA
    bars["Volume"] = pd.to_numeric(bars["Volume"], errors="coerce")       # missing -> NA
    return bars.dropna(subset=["Close"])


# ---------- Step 2: the prices table -----------------------------------------------------

def apply_split_fix(ticker: str, ohlc: pd.DataFrame) -> pd.DataFrame:
    """Divide the prices before a unit split by its ratio, if the source left the jump in the history.

    The adjusted closes before the split are divided too, if the source left the jump in them as well
    (same check, on AdjClose). The volumes before the split are multiplied by the same ratio: after a split of 1 for 80 there are
    80 times more units, so the old volumes are put in today's units too.
    """
    if ticker not in SPLIT_FIXES:
        return ohlc
    split = SPLIT_FIXES[ticker]

    before = ohlc[ohlc.index.date < split["first_day"]]
    after = ohlc[ohlc.index.date >= split["first_day"]]
    if before.empty or after.empty:
        return ohlc

    # Is the jump still there? Last close before the split vs first close after it.
    jump = float(before["Close"].iloc[-1]) / float(after["Close"].iloc[0])
    if jump < split["ratio"] / 2:
        print(f"  {ticker}: split of 1 for {split['ratio']} already applied by the source, nothing changed")
        return ohlc

    adjusted = ohlc.copy()
    is_before = adjusted.index.date < split["first_day"]
    for column in OHLC_COLUMNS:
        adjusted.loc[is_before, column] = adjusted.loc[is_before, column] / split["ratio"]
    if "AdjClose" in adjusted.columns and adjusted.loc[~is_before, "AdjClose"].notna().any():
        adjusted_jump = float(before["AdjClose"].iloc[-1]) / float(after["AdjClose"].iloc[0])
        if adjusted_jump >= split["ratio"] / 2:
            adjusted.loc[is_before, "AdjClose"] = adjusted.loc[is_before, "AdjClose"] / split["ratio"]
    if "Volume" in adjusted.columns:
        adjusted.loc[is_before, "Volume"] = adjusted.loc[is_before, "Volume"] * split["ratio"]
    print(f"  {ticker}: prices before {split['first_day']} divided by {split['ratio']}, "
          f"volumes multiplied by it (unit split)")
    return adjusted


def fetch_prices(start: date, end: date,
                 yahoo: Downloader = yahoo_ohlc,
                 eodhd: Downloader = eodhd_ohlc,
                 symbols: pd.DataFrame | None = None,
                 today: date | None = None) -> pd.DataFrame:
    """Real daily OHLC prices, adjusted closes and volumes of every priceable ETF, in EUR. Format: Prices.

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

        ohlc = apply_split_fix(etf.ticker, ohlc)

        # Adjusted close and volume: kept as given, NA when the source has none
        if "AdjClose" in ohlc.columns:
            adj_close = pd.to_numeric(ohlc["AdjClose"], errors="coerce").astype(float).round(4).values
        else:
            adj_close = float("nan")
        if "Volume" in ohlc.columns:
            volume = pd.to_numeric(ohlc["Volume"], errors="coerce").astype(float).values
        else:
            volume = float("nan")

        frames.append(pd.DataFrame({
            "date": ohlc.index.date,
            "ticker": etf.ticker,
            "open_eur": ohlc["Open"].astype(float).round(4).values,
            "high_eur": ohlc["High"].astype(float).round(4).values,
            "low_eur": ohlc["Low"].astype(float).round(4).values,
            "close_eur": ohlc["Close"].astype(float).round(4).values,
            "adj_close_eur": adj_close,
            "volume": volume,
        }))

    if len(frames) == 0:
        prices = pd.DataFrame(columns=PRICES)
    else:
        prices = pd.concat(frames, ignore_index=True)
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return validate_prices(prices)


# ---------- Quality checks: they report, they never change the data ----------------------

def check_prices(prices: pd.DataFrame, symbols: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Eight reports on the prices table. Each value is a table; an empty table means nothing to report.

    coverage      first and last date, number of days and last close per ETF
    stale         ETFs without a price on the most recent day of the table
    missing_days  over the last year, days where some ETFs have a price and others do not
    bad_bars      prices <= 0, high below open/close, low above open/close
    big_moves     daily moves above BIG_MOVE_PERCENT, on each ETF's own series
    left_out      ETFs of the universe without any price, with their status
    no_volume     over the last year, per ETF: days with a volume of 0 (no trade: the close is
                  probably the previous one repeated) and days without any volume from the source
    adjusted      per ETF where the adjusted close differs from the close (dividends) or is missing:
                  expected for distributing ETFs, to look at for accumulating ones
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

    # 7. Days without trade (volume 0) or without volume, over the last year
    no_volume = []
    for ticker, ticker_rows in recent.groupby("ticker"):
        volume = pd.to_numeric(ticker_rows["volume"], errors="coerce")
        zero_days = int((volume == 0).sum())
        empty_days = int(volume.isna().sum())
        if zero_days > 0 or empty_days > 0:
            no_volume.append({"ticker": ticker, "days": len(ticker_rows), "zero_volume_days": zero_days,
                              "zero_volume_percent": round(zero_days / len(ticker_rows) * 100, 1),
                              "no_volume_data_days": empty_days})
    reports["no_volume"] = pd.DataFrame(
        no_volume, columns=["ticker", "days", "zero_volume_days", "zero_volume_percent", "no_volume_data_days"])

    # 8. Adjusted close vs close: dividends (distributing ETFs) or a missing adjusted close
    adjusted_rows = []
    for ticker, ticker_rows in prices.groupby("ticker"):
        adjusted = pd.to_numeric(ticker_rows["adj_close_eur"], errors="coerce")
        ratio = adjusted / ticker_rows["close_eur"]
        different = int(((ratio - 1).abs() > ADJUSTED_TOLERANCE).sum())
        missing = int(adjusted.isna().sum())
        if different > 0 or missing > 0:
            adjusted_rows.append({"ticker": ticker, "days": len(ticker_rows), "days_adjusted": different,
                                  "oldest_ratio": round(float(ratio.iloc[0]), 4) if pd.notna(ratio.iloc[0]) else None,
                                  "days_missing": missing})
    reports["adjusted"] = pd.DataFrame(
        adjusted_rows, columns=["ticker", "days", "days_adjusted", "oldest_ratio", "days_missing"])

    return reports


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """The indicators of every ETF, in the shared Features format (long). See data/features.py."""
    from data import features
    return features.compute_features(prices)


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
