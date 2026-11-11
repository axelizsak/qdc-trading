"""[1] Data agent, step 1: the Yahoo symbol of each ETF. Owner: Person 2. No LLM.

Run once, and again only when the ETF list, SYMBOL_FIXES or EODHD_SYMBOLS change:

    python -m data.symbols        # writes data/etf_symbols.csv (about 3 minutes)

For each ETF of data/etf_universe.csv:
  0. ETFs in EODHD_SYMBOLS are skipped: Yahoo has no listing for them on their reference
     exchange, so step 2 (data/data_agent.py) downloads their prices from EODHD instead.
  1. Yahoo symbol: the manual fix if there is one, otherwise OpenFIGI
     (ISIN + reference exchange -> ticker of that listing, plus the Yahoo suffix).
  2. Check on Yahoo that the symbol has prices, in EUR.

Status of each ETF in etf_symbols.csv:
  OK         symbol found, prices on Yahoo, in EUR      -> priced from Yahoo in step 2
  EODHD      skipped on purpose (see EODHD_SYMBOLS)     -> priced from EODHD in step 2
  NO DATA    symbol found, but Yahoo has no prices      -> left out, reported
  NOT FOUND  OpenFIGI knows no listing on the exchange  -> left out, reported
  NOT EUR    prices found, but not in EUR               -> left out, reported
"""
import time
from typing import Callable

import pandas as pd
import requests

from data.config import DATA_DIR
from data.universe import EXCHANGES, load_universe, reference_exchange

SYMBOLS_CSV = DATA_DIR / "etf_symbols.csv"
SYMBOLS_COLUMNS = ["ticker", "isin", "exchange", "yahoo_symbol", "source", "status",
                   "currency", "days_1y", "last_date"]

# Manual fixes, used instead of OpenFIGI: {Belfius ticker: Yahoo symbol}.
# OpenFIGI returns the Bloomberg ticker; for these two old Lyxor funds it differs
# from the Xetra trading symbol that Yahoo uses.
SYMBOL_FIXES = {
    "STOXX SHORT": "LSK8.DE",     # Bloomberg BXXP GY   - source: live.deutsche-boerse.com, ISIN FR0010424143
    "EURO STOXX 2X": "LYMZ.DE",   # Bloomberg LYXLVE GY - source: Amundi factsheet, ISIN FR0010468983
}

# ETFs whose listing on the reference exchange is not on Yahoo: {Belfius ticker: EODHD symbol}.
# Step 1 skips them; step 2 downloads their prices from EODHD (free plan, official Deutsche Boerse
# data distributor). 3QQQ.XETRA closes checked identical to Xetra on 11 days (27/10 to 10/11/2026).
EODHD_SYMBOLS = {
    "NASDAQ 3X": "3QQQ.XETRA",
}

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_PAUSE_SECONDS = 3   # without an API key, OpenFIGI allows about 25 requests per minute

NO_CHECK = {"days_1y": 0, "currency": None, "last_date": None}


def find_yahoo_symbol(isin: str, exchange: str, post: Callable = requests.post) -> str | None:
    """ISIN + exchange name -> Yahoo symbol, e.g. ('IE00B4L5Y983', 'Xetra') -> 'EUNL.DE'.

    Returns None when OpenFIGI knows no listing of this ISIN on this exchange.
    `post` is injectable so tests never call the network.
    """
    openfigi_code = EXCHANGES[exchange]["openfigi_code"]
    yahoo_suffix = EXCHANGES[exchange]["yahoo_suffix"]
    query = [{"idType": "ID_ISIN", "idValue": isin, "exchCode": openfigi_code}]

    for attempt in range(1, 4):
        try:
            response = post(OPENFIGI_URL, json=query, timeout=20)
        except requests.exceptions.ConnectionError:
            print(f"  network error for {isin}, attempt {attempt}/3, waiting 10 s")
            time.sleep(10)
            continue

        if response.status_code == 429:
            print(f"  OpenFIGI rate limit, attempt {attempt}/3, waiting 60 s")
            time.sleep(60)
            continue

        if response.status_code != 200:
            raise RuntimeError(f"OpenFIGI error {response.status_code} for {isin}: {response.text[:200]}")

        answer = response.json()[0]
        if "data" not in answer:
            return None
        ticker = answer["data"][0]["ticker"]
        return ticker + yahoo_suffix

    raise RuntimeError(f"OpenFIGI did not answer for {isin} after 3 attempts")


def check_on_yahoo(symbol: str) -> dict:
    """Number of trading days over one year, currency and last date of a Yahoo symbol.

    Yahoo sometimes answers empty by mistake, so an empty answer is asked a second time.
    """
    import yfinance as yf

    history = pd.DataFrame()
    for attempt in range(1, 3):
        try:
            history = yf.Ticker(symbol).history(period="1y", auto_adjust=False)
        except Exception:
            history = pd.DataFrame()
        if len(history) > 0:
            break
        time.sleep(5)

    if len(history) == 0:
        return dict(NO_CHECK)

    currency = None
    try:
        currency = yf.Ticker(symbol).fast_info["currency"]
    except Exception:
        currency = None

    last_date = history.index[-1].date()
    return {"days_1y": len(history), "currency": currency, "last_date": last_date}


def build_symbols(universe: pd.DataFrame | None = None,
                  find: Callable = find_yahoo_symbol,
                  check: Callable = check_on_yahoo,
                  pause_seconds: float = OPENFIGI_PAUSE_SECONDS) -> pd.DataFrame:
    """One row per ETF: Yahoo symbol, where it comes from and its status. Format: SYMBOLS_COLUMNS.

    `find` and `check` are injectable so tests never call the network.
    """
    if universe is None:
        universe = load_universe()

    rows = []
    for etf in universe.itertuples():
        exchange = reference_exchange(etf.ticker)

        # 0. ETFs priced by EODHD: nothing to do here, step 2 takes care of them
        if etf.ticker in EODHD_SYMBOLS:
            rows.append({
                "ticker": etf.ticker, "isin": etf.isin, "exchange": exchange,
                "yahoo_symbol": None, "source": "EODHD", "status": "EODHD",
                "currency": None, "days_1y": 0, "last_date": None,
            })
            print(f"{etf.ticker:18} {exchange:18} {'-':11} EODHD     prices from EODHD in step 2")
            continue

        # 1. Yahoo symbol: manual fix if there is one, otherwise OpenFIGI
        if etf.ticker in SYMBOL_FIXES:
            symbol = SYMBOL_FIXES[etf.ticker]
            source = "manual fix"
        else:
            symbol = find(etf.isin, exchange)
            source = "OpenFIGI"
            time.sleep(pause_seconds)

        # 2. Check on Yahoo
        if symbol is None:
            result = dict(NO_CHECK)
            status = "NOT FOUND"
        else:
            result = check(symbol)
            if result["days_1y"] == 0:
                status = "NO DATA"
            elif result["currency"] != "EUR":
                status = "NOT EUR"
            else:
                status = "OK"

        rows.append({
            "ticker": etf.ticker, "isin": etf.isin, "exchange": exchange,
            "yahoo_symbol": symbol, "source": source, "status": status,
            "currency": result["currency"], "days_1y": result["days_1y"], "last_date": result["last_date"],
        })
        print(f"{etf.ticker:18} {exchange:18} {str(symbol):11} {status:9} "
              f"{result['days_1y']:4} days  {result['currency']}")

    return pd.DataFrame(rows, columns=SYMBOLS_COLUMNS)


def load_symbols() -> pd.DataFrame:
    """The table written by `python -m data.symbols`. Format: SYMBOLS_COLUMNS."""
    if not SYMBOLS_CSV.exists():
        raise FileNotFoundError(f"{SYMBOLS_CSV.name} is missing: run `python -m data.symbols` first")
    return pd.read_csv(SYMBOLS_CSV)


if __name__ == "__main__":
    table = build_symbols()
    table.to_csv(SYMBOLS_CSV, index=False)
    print()
    print(table["status"].value_counts().to_string())
    print(f"\nWritten to {SYMBOLS_CSV}")
