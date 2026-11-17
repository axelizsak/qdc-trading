"""The executive summary as a PowerPoint deck: built from the summary's Markdown, nothing else."""
import io
from pathlib import Path

import pytest
from pptx import Presentation

from app.deck import to_pptx

SUMMARY = """# Executive summary · Mon 16 Nov 2026

## Orders to submit

| Side | ETF | Units | Fill price | Value |
|---|---|---:|---:|---:|
| BUY | STAPLES | 36.5644 | €54.78 | €2,003 |
| BUY | EUROPE 600 | 5.2399 | €328.25 | €1,720 |

Cash: €10,000 now, €6,277 after the orders.

## Why: the Portfolio Manager's memo

### Portfolio Manager · Mon 16 Nov 2026 (practice day: not a trading date, next trading date 07 Dec)

Recommendation REC-1

I stay close to the baseline. These two have the highest final scores. I keep a cash cushion against a pullback.

Portfolio value EUR 10,000. Market view (score of WORLD): +0.01, so the code portfolios invest 85%. The LLM's portfolio counts for 60% of the final blend.

#### Agent weights

| Agent | Start | Today | Sessions judged | Mean result a session | Right |
|---|---:|---:|---:|---:|---:|
| macro_etf | 0.25 | 0.25 | 3 | +0.16% | 67% |

#### Portfolios (share of the portfolio value)

| ETF | Now | Baseline | Optimizer | LLM | Final |
|---|---:|---:|---:|---:|---:|
| STAPLES | 0.0% | 15.3% | 21.6% | 19.0% | 20.0% |
| EUROPE 600 | 0.0% | 16.0% | 19.0% | 16.0% | 17.2% |

#### Assumptions

- An order fills at the last close before its day.

## Risk board · 2026-11-16

0 breach(es), 1 warning(s), 21 check(s) inside their reference levels.

| Area | Status | In one line |
|---|---|---|
| Rules | OK | 5 orders for the day. |
| Dividends | Warning | BEL 20 goes ex-dividend on 9 Dec. |
"""


def slides(markdown: str) -> int:
    return len(Presentation(io.BytesIO(to_pptx(markdown))).slides)


def test_a_summary_becomes_a_deck():
    assert to_pptx(SUMMARY)[:2] == b"PK"  # a .pptx is a zip file
    assert slides(SUMMARY) >= 5


def test_a_missing_section_only_drops_its_slide():
    without_risk = SUMMARY.split("## Risk board")[0]
    assert 0 < slides(without_risk) < slides(SUMMARY)


def test_text_that_is_not_a_summary_gives_no_deck():
    with pytest.raises(ValueError):
        to_pptx("nothing here")


def test_the_saved_summaries_all_make_a_deck():
    for path in sorted((Path(__file__).resolve().parent.parent / "logs" / "summary").glob("*.md")):
        assert slides(path.read_text(encoding="utf-8")) >= 5, path.name
