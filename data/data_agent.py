"""[1] Data agent. Owner: Person 2. No LLM.

Input:  ISIN list (data/etf_universe.csv)
Output: ETF mapping (ISIN -> Yahoo ticker), clean prices in the Prices format
        (date, ticker, close_eur), plus returns, volatility, drawdowns and correlations.

Source: yfinance, EUR listing preferred.
Fallback: another listing of the same ETF + FX conversion, or a manual export
from justETF / Investing.com. The ISIN -> Yahoo symbol mapping is produced by
data/check_etfs_yfinance.py (data/etf_yahoo_map.csv).

How fetch_prices works:
  1. Read data/etf_yahoo_map.csv (Belfius ticker -> Yahoo symbol + currency).
  2. Download adjusted closes from Yahoo. Adjusted = dividends reinvested, so returns
     are total returns; the latest adjusted close equals the real close.
  3. Convert every listing to EUR (GBp is pence: divide by 100 first).
  4. Put all ETFs on one business-day calendar. A missing day (holiday on one
     exchange) takes the previous close, for at most MAX_FILL_DAYS days.
  5. Rename to the Belfius ticker and return the Prices format.
ETFs that cannot be priced are left out.
"""
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from data.config import DATA_DIR
from data.schemas import validate_prices
from data.universe import tickers

YAHOO_MAP_CSV = DATA_DIR / "etf_yahoo_map.csv"
MAX_FILL_DAYS = 3  # longest gap filled with the previous close (exchange holidays)

# Manual fixes for ETFs the ISIN search maps badly. They override etf_yahoo_map.csv.
# Belfius ticker -> (Yahoo symbol, currency)
MANUAL_SYMBOLS = {
    "EURO STOXX 50": ("CSX5.AS", "EUR"),
    "CHINA A": ("CNYA.AS", "EUR"),
    "COPPER MINERS": ("4COP.DE", "EUR"),
    "OVERNIGHT CASH": ("CSH2.PA", "EUR"),
}

Downloader = Callable[[list[str], date, date], pd.DataFrame]


def load_yahoo_map() -> pd.DataFrame:
    """Columns: ticker, yahoo_symbol, currency. Only ETFs with a symbol."""
    if not YAHOO_MAP_CSV.exists():
        raise FileNotFoundError(
            f"{YAHOO_MAP_CSV.name} is missing: run `python data/check_etfs_yfinance.py` first")
    m = pd.read_csv(YAHOO_MAP_CSV).rename(columns={"name": "ticker"})
    m = m[~m["ticker"].isin(MANUAL_SYMBOLS)]
    manual = pd.DataFrame([(t, sym, ccy) for t, (sym, ccy) in MANUAL_SYMBOLS.items()],
                          columns=["ticker", "yahoo_symbol", "currency"])
    m = pd.concat([m, manual], ignore_index=True).dropna(subset=["yahoo_symbol"])
    m["currency"] = m["currency"].fillna("EUR")
    return m[["ticker", "yahoo_symbol", "currency"]]


def yahoo_download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Adjusted closes, one column per Yahoo symbol, dates as index. `end` is included."""
    import yfinance as yf

    raw = yf.download(symbols, start=start, end=end + timedelta(days=1),
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):  # a single symbol
        close = close.to_frame(symbols[0])
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def _fx_symbol(currency: str) -> str:
    """Yahoo symbol of EUR/<currency>: units of <currency> for 1 EUR."""
    return f"EUR{currency}=X"


def fetch_prices(start: date, end: date, download: Downloader = yahoo_download) -> pd.DataFrame:
    """Prices for every ETF in the universe, in EUR. Format: Prices.

    `download` is injectable so tests never call the network.
    """
    ymap = load_yahoo_map()
    ymap["base_ccy"] = ymap["currency"].replace({"GBp": "GBP", "GBX": "GBP"})
    foreign = sorted(set(ymap["base_ccy"]) - {"EUR"})

    symbols = ymap["yahoo_symbol"].tolist() + [_fx_symbol(c) for c in foreign]
    raw = download(symbols, start, end)

    calendar = pd.bdate_range(start, end)
    raw = raw.reindex(raw.index.union(calendar)).sort_index()

    columns = {}
    for row in ymap.itertuples():
        if row.yahoo_symbol not in raw.columns or raw[row.yahoo_symbol].dropna().empty:
            continue
        px = raw[row.yahoo_symbol].astype(float)
        if row.currency in ("GBp", "GBX"):
            px = px / 100
        if row.base_ccy != "EUR":
            fx = raw.get(_fx_symbol(row.base_ccy))
            if fx is None or fx.dropna().empty:
                continue
            px = px / fx.ffill()  # price in EUR = price in CCY / (CCY per EUR)
        columns[row.ticker] = px

    wide = pd.DataFrame(columns).reindex(calendar)
    wide = wide.ffill(limit=MAX_FILL_DAYS)
    wide = wide.where(wide > 0)

    prices = (wide.rename_axis("date").reset_index()
              .melt(id_vars="date", var_name="ticker", value_name="close_eur")
              .dropna(subset=["close_eur"]))
    prices["date"] = prices["date"].dt.date
    prices["close_eur"] = prices["close_eur"].round(4)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return validate_prices(prices)


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns, volatility, drawdowns, correlations."""
    raise NotImplementedError


if __name__ == "__main__":
    # Quick check on real data: python -m data.data_agent
    today = date.today()
    p = fetch_prices(today - timedelta(days=400), today)
    out = DATA_DIR / "cache" / f"prices_{today.isoformat()}.csv"
    out.parent.mkdir(exist_ok=True)
    p.to_csv(out, index=False)
    summary = p.groupby("ticker").agg(first=("date", "min"), last=("date", "max"),
                                      days=("date", "size"), last_eur=("close_eur", "last"))
    print(summary.to_string())
    print(f"\n{p['ticker'].nunique()} / 50 ETFs priced. Not priced: {sorted(set(tickers()) - set(p['ticker']))}")
    print(f"Written to {out}")
