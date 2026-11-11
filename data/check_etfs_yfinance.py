"""
Finds, for each of the 50 QDC ETFs, the Yahoo symbol on the exchange Belfius uses
(Xetra for almost all, Euronext Brussels for BEL 20, Euronext Milan for COPPER LVG
and PALLADIUM LVG). Orders are executed and valued at that exchange's closing price.

Usage:  python data/check_etfs_yfinance.py   (from the repo root)
Output: data/etf_yahoo_map.csv (chosen Yahoo symbol per ETF + status)

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
import yfinance as yf  # noqa: E402

from data.universe import load_universe, reference_exchange, yahoo_suffix  # noqa: E402

# Symbols the ISIN search does not find on the reference exchange.
# They are tried on top of the search results.
EXTRA_CANDIDATES = {
    "EURO STOXX 50": ["SXRT.DE"],
    "CHINA A": ["36BZ.DE"],
    "COPPER MINERS": ["4COP.DE"],
    "OVERNIGHT CASH": ["CSH2.DE", "LYXO.DE"],
}


def candidates(ticker: str, isin: str) -> list[str]:
    try:
        quotes = yf.Search(isin, max_results=10).quotes
    except Exception:
        quotes = []
    found = [q["symbol"] for q in quotes if q.get("symbol")]
    return list(dict.fromkeys(EXTRA_CANDIDATES.get(ticker, []) + found))


def describe(symbol: str) -> dict | None:
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="1y", auto_adjust=False)
        currency = t.fast_info.get("currency")
    except Exception:
        return None
    if len(h) == 0:
        return None
    return {"symbol": symbol, "currency": currency, "days_1y": len(h), "last_date": h.index[-1].date()}


rows = []
for etf in load_universe().itertuples():
    suffix = yahoo_suffix(etf.ticker)
    cands = [c for c in (describe(s) for s in candidates(etf.ticker, etf.isin)) if c]
    # Preference: reference exchange, then EUR listing, then the longest history.
    cands.sort(key=lambda c: (c["symbol"].endswith(suffix), c["currency"] == "EUR", c["days_1y"]), reverse=True)
    best = cands[0] if cands else None
    if best is None:
        status = "NOT FOUND"
    elif not best["symbol"].endswith(suffix):
        status = "OTHER EXCHANGE"
    else:
        status = "OK" if best["days_1y"] >= 200 else "PARTIAL"
    rows.append({"name": etf.ticker, "isin": etf.isin, "exchange": reference_exchange(etf.ticker),
                 "status": status, "yahoo_symbol": best and best["symbol"],
                 "currency": best and best["currency"], "days_1y": best and best["days_1y"],
                 "last_date": best and best["last_date"],
                 "other_symbols": ", ".join(c["symbol"] for c in cands[1:])})
    print(f"{status:14} {etf.ticker:18} {etf.isin}  -> {best and best['symbol']} "
          f"({best and best['currency']}, {best and best['days_1y']} days)")

df = pd.DataFrame(rows)
out = Path(__file__).resolve().parent / "etf_yahoo_map.csv"
df.to_csv(out, index=False)
print("\n", df["status"].value_counts().to_string())
print("\nTo check:")
print(df[df["status"] != "OK"][["name", "exchange", "status", "yahoo_symbol", "other_symbols"]].to_string(index=False))
print(f"\nWritten to {out}")
