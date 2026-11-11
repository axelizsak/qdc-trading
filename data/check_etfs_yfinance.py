"""
Finds, for each of the 50 QDC ETFs, the Yahoo symbol on the exchange Belfius uses
(Xetra for almost all, Euronext Brussels for BEL 20, Euronext Milan for COPPER LVG
and PALLADIUM LVG). Orders are executed and valued at that exchange's closing price.

Usage:  python data/check_etfs_yfinance.py   (from the repo root)
Output: data/etf_yahoo_map.csv (chosen Yahoo symbol per ETF + status)

How it works, one function per step:
  candidates(ticker, isin)   every Yahoo symbol listing this ETF
  describe(symbol)           currency and history of one listing (None if no data)
  pick_best(ticker, listings) the listing to keep, and its status
  build_yahoo_map()          the three steps above for the 50 ETFs -> one table
  main()                     builds the table and writes data/etf_yahoo_map.csv

Statuses:
  OK              found on the reference exchange, at least 200 days over one year
  PARTIAL         found on the reference exchange, but fewer than 200 days
  OTHER EXCHANGE  not found on the reference exchange: another listing is used
  NOT FOUND       no listing with data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from data.universe import load_universe, reference_exchange, yahoo_suffix  # noqa: E402

YAHOO_MAP_CSV = Path(__file__).resolve().parent / "etf_yahoo_map.csv"
MIN_DAYS = 200  # trading days over the last year for an OK status

# Symbols the ISIN search does not find on the reference exchange.
# They are tried on top of the search results.
EXTRA_CANDIDATES = {
    "EURO STOXX 50": ["SXRT.DE"],
    "CHINA A": ["36BZ.DE"],
    "COPPER MINERS": ["4COP.DE"],
    "OVERNIGHT CASH": ["CSH2.DE", "LYXO.DE"],
}


def candidates(ticker: str, isin: str) -> list[str]:
    """Every Yahoo symbol listing this ETF: Yahoo's ISIN search, plus our manual extras."""
    import yfinance as yf

    try:
        quotes = yf.Search(isin, max_results=10).quotes
    except Exception:
        quotes = []
    found = [q["symbol"] for q in quotes if q.get("symbol")]
    return list(dict.fromkeys(EXTRA_CANDIDATES.get(ticker, []) + found))


def describe(symbol: str) -> dict | None:
    """Currency and one year of history for one listing. None if Yahoo has no data."""
    import yfinance as yf

    try:
        t = yf.Ticker(symbol)
        h = t.history(period="1y", auto_adjust=False)
        currency = t.fast_info.get("currency")
    except Exception:
        return None
    if len(h) == 0:
        return None
    return {"symbol": symbol, "currency": currency, "days_1y": len(h), "last_date": h.index[-1].date()}


def pick_best(ticker: str, listings: list[dict]) -> tuple[dict | None, str]:
    """The listing to keep and its status.

    Preference: the Belfius reference exchange, then a EUR listing, then the longest history.
    """
    suffix = yahoo_suffix(ticker)
    ranked = sorted(listings, key=lambda c: (c["symbol"].endswith(suffix), c["currency"] == "EUR",
                                             c["days_1y"]), reverse=True)
    best = ranked[0] if ranked else None
    if best is None:
        return None, "NOT FOUND"
    if not best["symbol"].endswith(suffix):
        return best, "OTHER EXCHANGE"
    return best, "OK" if best["days_1y"] >= MIN_DAYS else "PARTIAL"


def build_yahoo_map(verbose: bool = True) -> pd.DataFrame:
    """Runs candidates -> describe -> pick_best for the 50 ETFs. Takes a few minutes."""
    rows = []
    for etf in load_universe().itertuples():
        listings = [d for d in (describe(s) for s in candidates(etf.ticker, etf.isin)) if d]
        best, status = pick_best(etf.ticker, listings)
        rows.append({"name": etf.ticker, "isin": etf.isin, "exchange": reference_exchange(etf.ticker),
                     "status": status, "yahoo_symbol": best and best["symbol"],
                     "currency": best and best["currency"], "days_1y": best and best["days_1y"],
                     "last_date": best and best["last_date"],
                     "other_symbols": ", ".join(d["symbol"] for d in listings if d is not best)})
        if verbose:
            print(f"{status:14} {etf.ticker:18} {etf.isin}  -> {best and best['symbol']} "
                  f"({best and best['currency']}, {best and best['days_1y']} days)")
    return pd.DataFrame(rows)


def main() -> None:
    df = build_yahoo_map()
    df.to_csv(YAHOO_MAP_CSV, index=False)
    print("\n", df["status"].value_counts().to_string())
    print("\nTo check:")
    print(df[df["status"] != "OK"][["name", "exchange", "status", "yahoo_symbol", "other_symbols"]]
          .to_string(index=False))
    print(f"\nWritten to {YAHOO_MAP_CSV}")


if __name__ == "__main__":
    main()
