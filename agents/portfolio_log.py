"""The Portfolio Manager's recommendation log. Owner: Person 3. No LLM.

Every recommendation (one per run of the PM, so one per round of a day) is appended to <logs>/recommendations.csv
with an id, a timestamp, the reference price and its date for each order, the weights before and after, the order's
impact and timing. Corrections are new rows, never overwrites: this is part of the audit trail (recommendations,
then the human's decisions in orders.csv and reviews.csv, then fills).
TODO for the shared PR: promote RECOMMENDATIONS to data/schemas.py with a validate_ function and a mock.
"""
from pathlib import Path

import pandas as pd

RECOMMENDATIONS = ["recommendation_id", "created_at", "date", "ticker", "side", "units", "price_eur", "price_date",
                   "value_eur", "weight_now", "weight_target", "impact", "timing", "why"]


def record_recommendation(logs_dir: Path | str, recommendation_id: str, created_at: str, as_of,
                          detail: pd.DataFrame, price_dates: pd.Series, reasons: dict[str, str]) -> None:
    """Append one row per proposed order (a single row with side NONE when no order is proposed)."""
    rows = [{"recommendation_id": recommendation_id, "created_at": created_at, "date": str(as_of),
             "ticker": r.ticker, "side": r.side, "units": r.units, "price_eur": r.price,
             "price_date": price_dates.get(r.ticker, ""), "value_eur": r.value, "weight_now": r.now,
             "weight_target": r.to, "impact": r.gain, "timing": r.timing, "why": reasons.get(r.ticker, "")}
            for r in detail.itertuples()]
    if not rows:
        rows = [{"recommendation_id": recommendation_id, "created_at": created_at, "date": str(as_of), "side": "NONE"}]
    new = pd.DataFrame(rows, columns=RECOMMENDATIONS)
    path = Path(logs_dir) / "recommendations.csv"
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
