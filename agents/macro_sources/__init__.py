"""Collectors for the Macro/News agent: one module per source, each returns Evidence rows.

Only free sources that need no key live here; data that would need a key (FRED, AGSI, ...) is
looked up by the LLM's web search instead (agents/macro_llm.py). The code collects and puts numbers
in context; the LLM only interprets them. A collector that fails is skipped and its failure is
reported. The agent never writes a new collector itself: new sources found by exploration are
proposed, and a person decides.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from agents.macro_sources import cftc_cot, funding_rates, market_stress, open_meteo, polymarket, stablecoins
from data.schemas import EVIDENCE, validate_evidence

COLLECTORS = {
    "market_stress": market_stress.collect,
    "polymarket": polymarket.collect,
    "open_meteo": open_meteo.collect,
    # v2 (2026-11-08), see logs/source_changes.csv
    "cftc_cot": cftc_cot.collect,
    "stablecoins": stablecoins.collect,
    "funding_rates": funding_rates.collect,
}


def collect_all(as_of: date, collectors: dict | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run every collector in parallel. Returns (evidence, {collector: error} for the ones that failed)."""
    collectors = COLLECTORS if collectors is None else collectors
    rows, failures = [], {}

    def run(item):
        name, collect = item
        try:
            return name, validate_evidence(pd.DataFrame(collect(as_of), columns=EVIDENCE)), None
        except Exception as exc:
            return name, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=6) as pool:
        for name, df, error in pool.map(run, collectors.items()):
            if error:
                failures[name] = error
            else:
                rows.append(df)
    evidence = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=EVIDENCE)
    return evidence, failures
