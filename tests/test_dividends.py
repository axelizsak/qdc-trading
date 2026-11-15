"""Expected ex-dividend dates (data/dividends.py), with made-up dividend histories: no network."""
from datetime import date

import pandas as pd

from data import dividends
from data.schemas import validate_ex_dividends

SYMBOLS = pd.DataFrame({"ticker": ["BEL 20", "NASDAQ INCOME", "EUR GOVT BOND", "CHINA LC"],
                        "yahoo_symbol": ["BEL.BR", "JEQP.DE", "EUNH.DE", "IQQC.DE"]})
PAID = {
    "BEL.BR": {"2024-12-10": 0.58, "2025-12-09": 0.85},                       # once a year, early December
    "JEQP.DE": {"2025-11-13": 0.22, "2025-12-11": 0.23, "2026-11-12": 0.24},  # every month
    "EUNH.DE": {"2026-01-15": 1.42, "2026-07-16": 1.41},                      # January and July
}


def download(symbol: str) -> pd.Series:
    if symbol not in PAID:
        raise RuntimeError("Yahoo down")
    return pd.Series(PAID[symbol].values(), index=pd.to_datetime(list(PAID[symbol])), dtype=float)


def expected(start, end, today):
    return dividends.expected_ex_dividends(start, end, download, SYMBOLS, today).set_index("ticker")


def test_next_ex_dates_come_from_last_years_schedule():
    table = expected(date(2026, 12, 7), date(2026, 12, 23), today=date(2026, 12, 7))
    validate_ex_dividends(table.reset_index())
    assert table.loc["BEL 20", ["ex_date", "amount_eur"]].tolist() == ["2026-12-09", 0.85]
    assert table.loc["BEL 20", "basis"] == "expected: paid on 09 Dec 2025 a year earlier"
    assert table.loc["NASDAQ INCOME", "ex_date"] == "2026-12-11"  # the monthly payment of 11 Dec 2025
    assert "EUR GOVT BOND" not in table.index  # it pays in January and July
    assert "CHINA LC" not in table.index       # its dividends could not be read: left out, never guessed


def test_a_payment_already_made_is_not_expected_again():
    table = expected(date(2026, 11, 9), date(2026, 11, 30), today=date(2026, 11, 16))
    assert table.loc["NASDAQ INCOME", ["ex_date", "basis"]].tolist() == ["2026-11-12", "paid"]  # not 13 Nov again


def test_a_payment_that_is_due_but_not_seen_stays_expected_for_a_while():
    late = expected(date(2026, 12, 7), date(2026, 12, 23), today=date(2026, 12, 14))
    assert late.loc["BEL 20", "ex_date"] == "2026-12-14"  # due on the 9th, not in the history yet: still to come
    assert "not seen yet" in late.loc["BEL 20", "basis"]
    gone = expected(date(2026, 12, 7), date(2027, 1, 31), today=date(2027, 1, 20))
    assert "BEL 20" not in gone.index  # six weeks late: the schedule changed, the estimate is dropped


def test_an_expected_date_falls_on_a_weekday():
    assert dividends._a_year_later(date(2025, 12, 12)) == date(2026, 12, 14)  # Saturday 12 Dec 2026 -> Monday
    assert dividends._a_year_later(date(2024, 2, 29)) == date(2025, 2, 28)
