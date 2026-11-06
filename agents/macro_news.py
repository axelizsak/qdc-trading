"""[3] Macro / News agent. Owner: Axel. LLM: yes.

Input:  web search (Claude API web search tool) + lessons journal
Output: macro summary, plus a signal per theme (rates, geopolitics, commodities)
        mapped onto the ETFs via the `theme` column of data/etf_universe.csv.
"""
from datetime import date

import pandas as pd

AGENT = "macro_news"


def run(as_of: date, lessons: str = "") -> tuple[str, pd.DataFrame]:
    """Returns (macro summary, signals in the Signals format)."""
    raise NotImplementedError
