"""[1] Data agent, ex-dividend dates of the distributing ETFs. Owner: Person 2 (first version by Axel). No LLM.

Why: dividends are not credited in the competition, so a distributing ETF loses its dividend on the ex-date
with nothing in return. The Risk Manager warns about an ETF that goes ex-dividend before the window ends.

Yahoo publishes the dividends already paid, not the dates to come. A distributing ETF pays on a regular
schedule, so the next ex-date is estimated from the payment made a year earlier: same day, moved to a
weekday, same amount. An expected payment that is due but not seen yet stays expected, dated today, for
LATE_DAYS days. Every row says what it rests on (`basis`), so nobody reads an estimate as an announcement.

Output: the Ex-dividends format (ticker, ex_date, amount_eur, basis), for the ex-dates from `start` to `end`.

Quick run on real data:  python -m data.dividends
"""
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from data.schemas import EX_DIVIDENDS, validate_ex_dividends
from data.symbols import load_symbols
from data.universe import load_universe

LATE_DAYS = 15  # a payment expected this long ago and still not seen is dropped: the schedule changed
SAME_PAYMENT_DAYS = 25  # an actual payment this close to an expected one is that payment
Downloader = Callable[[str], pd.Series]


def yahoo_dividends(symbol: str) -> pd.Series:
    """Every dividend Yahoo has for `symbol`: amount per unit in the listing's currency, indexed by ex-date."""
    import yfinance as yf
    paid = yf.Ticker(symbol).dividends
    paid.index = pd.to_datetime(paid.index).tz_localize(None).normalize()
    return paid.astype(float)


def _weekday(day: date) -> date:
    return day + timedelta(days={5: 2, 6: 1}.get(day.weekday(), 0))


def _a_year_later(day: date) -> date:
    try:
        return _weekday(day.replace(year=day.year + 1))
    except ValueError:  # 29 February
        return _weekday(day + timedelta(days=365))


def expected_ex_dividends(start: date, end: date, download: Downloader = yahoo_dividends,
                          symbols: pd.DataFrame | None = None, today: date | None = None) -> pd.DataFrame:
    """Ex-dates from `start` to `end` for every distributing ETF: the ones already paid, and the ones expected
    from last year's schedule. `download`, `symbols` and `today` are injectable so tests never call the network.
    An ETF whose dividends cannot be read is left out and reported: the Risk Manager then has no date for it."""
    today = date.today() if today is None else today
    symbols = load_symbols() if symbols is None else symbols
    universe = load_universe()
    paying = universe.loc[universe["distributing"].astype(bool), "ticker"]
    yahoo = symbols.set_index("ticker")["yahoo_symbol"]
    rows = []
    for ticker in paying:
        symbol = yahoo.get(ticker)
        if not isinstance(symbol, str) or not symbol:
            print(f"  {ticker}: no Yahoo symbol, no ex-dividend date")
            continue
        try:
            paid = download(symbol)
        except Exception as exc:
            print(f"  {ticker}: dividends not read ({type(exc).__name__}), no ex-dividend date")
            continue
        paid = paid[paid > 0]
        actual = [(d.date(), float(a)) for d, a in paid.items()]
        for day, amount in actual:
            if start <= day <= end:
                rows.append({"ticker": ticker, "ex_date": day.isoformat(), "amount_eur": amount, "basis": "paid"})
        for day, amount in actual:
            due = _a_year_later(day)
            if any(abs((seen - due).days) <= SAME_PAYMENT_DAYS for seen, _ in actual):
                continue  # this year's payment is already in the history
            if due < today:
                if (today - due).days > LATE_DAYS:
                    continue
                note, due = f"expected around {due:%d %b}, not seen yet (paid on {day:%d %b %Y} a year earlier)", today
            else:
                note = f"expected: paid on {day:%d %b %Y} a year earlier"
            if start <= due <= end:
                rows.append({"ticker": ticker, "ex_date": due.isoformat(), "amount_eur": amount, "basis": note})
    out = pd.DataFrame(rows, columns=EX_DIVIDENDS).drop_duplicates(["ticker", "ex_date"])
    return validate_ex_dividends(out.sort_values(["ex_date", "ticker"]).reset_index(drop=True))


if __name__ == "__main__":
    from data.config import TRADING_DAYS
    table = expected_ex_dividends(date.today(), TRADING_DAYS[-1] + timedelta(days=2))
    print(table.to_string(index=False) if len(table) else "No ex-dividend date expected in the window.")
