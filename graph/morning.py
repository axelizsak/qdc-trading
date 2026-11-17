"""The morning run: everything that is the same for every customer, computed once, before anyone opens the app.

1. Prices of the 50 ETFs (data agent).
2. The day's Macro/News run, unless it is already in the logs: it is the slow part, at the free tier's pace.
3. The five model portfolios (fixed rules) and the sheet of each ETF.
The results go to logs/. The customer app only reads them, so a customer never waits for an agent.

Run:  python -m graph.morning            (today)
      python -m graph.morning --date 2026-12-07
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents import etf_sheets, macro_news, model_portfolios
from data import config as settings
from data import data_agent
from data.schemas import MACRO_VIEWS, validate_etf_sheets, validate_macro_views, validate_profile_portfolios

HISTORY_DAYS = 1150  # calendar days of prices: three years of sessions for the 3-year return
PORTFOLIOS, SHEETS = "profile_portfolios.csv", "etf_sheets.csv"


def macro_views(logs: Path, as_of: date) -> pd.DataFrame:
    """The latest macro views up to `as_of` (one day of them), empty when there are none yet."""
    path = logs / "macro_views.csv"
    if not path.exists():
        return pd.DataFrame(columns=MACRO_VIEWS)
    views = validate_macro_views(pd.read_csv(path))
    views = views[views["date"].astype(str) <= as_of.isoformat()]
    return views[views["date"] == views["date"].max()] if len(views) else views


def save(logs: Path, name: str, rows: pd.DataFrame, validate) -> None:
    """Writes the day's rows to a log file, in place of the same day's rows if the run is done again."""
    path = logs / name
    if path.exists():
        kept = pd.read_csv(path)
        rows = pd.concat([kept[kept["date"].astype(str) != str(rows["date"].iloc[0])], rows.astype({"date": str})])
    logs.mkdir(parents=True, exist_ok=True)
    validate(rows).to_csv(path, index=False)


def run(as_of: date, logs: Path | None = None, fetch=data_agent.fetch_prices, news=macro_news.run) -> dict:
    """Does the morning's work for `as_of` and returns what it wrote. `fetch` and `news` are injectable (tests)."""
    logs = Path(logs) if logs else settings.LOGS_DIR
    prices = fetch(as_of - timedelta(days=HISTORY_DAYS), as_of)
    note = ""
    views = macro_views(logs, as_of)
    if views.empty or str(views["date"].iloc[0]) != as_of.isoformat():
        lessons = (logs / "lessons.md").read_text() if (logs / "lessons.md").exists() else ""
        try:
            news(as_of, lessons, logs, prices=prices)
        except Exception as exc:  # no key, quota used, no network: the sheets show the last views there are
            note = f"Macro/News did not run ({type(exc).__name__}: {exc}): the sheets show the latest views in the logs"
        views = macro_views(logs, as_of)
    portfolios = model_portfolios.build(as_of)
    sheets = etf_sheets.build(prices, views, macro_news.load_exposure(), as_of)
    save(logs, PORTFOLIOS, portfolios, validate_profile_portfolios)
    save(logs, SHEETS, sheets, validate_etf_sheets)
    return {"portfolios": portfolios, "sheets": sheets, "note": note,
            "views_of": str(views["date"].iloc[0]) if len(views) else None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute what every customer shares, once, in the morning.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD, default today")
    parser.add_argument("--logs-dir", type=Path, default=settings.LOGS_DIR)
    args = parser.parse_args(argv)
    out = run(args.date, args.logs_dir)
    print(f"Morning run of {args.date}: {out['portfolios']['profile'].nunique()} model portfolios, "
          f"{out['sheets']['ticker'].nunique()} ETF sheets, macro views of {out['views_of'] or 'no day yet'}.")
    if out["note"]:
        print(out["note"])


if __name__ == "__main__":
    main()
