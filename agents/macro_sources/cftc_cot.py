"""Speculative positioning from the CFTC Commitments of Traders report (public API, no key, weekly).

Net non-commercial position as a share of open interest, set against its own 3-year history.
Extreme positioning often precedes reversals: the LLM interprets, the code only measures.
"""
from datetime import date, timedelta

import pandas as pd

from agents.macro_sources._common import context, evidence, http_client, ordinal

SOURCE = "cftc_cot"
API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"  # legacy report, futures only
MARKETS = {  # CFTC contract code: (label, factor)
    "088691": ("gold (COMEX)", "gold_silver"),
    "085692": ("copper (COMEX)", "copper_metals"),
    "067651": ("WTI crude oil (NYMEX)", "oil"),
    "13874A": ("E-mini S&P 500", "risk_appetite"),
    "209742": ("Nasdaq-100 mini", "tech_ai_semis"),
    "043602": ("10-year Treasury note", "fed_us_rates"),
    "099741": ("euro (Euro FX)", "eurusd"),
    "133741": ("bitcoin (CME)", "crypto"),
}
WEEKS = 156


def collect(as_of: date, http=None) -> list[dict]:
    http = http or http_client()
    codes = ",".join(f"'{c}'" for c in MARKETS)
    start = (as_of - timedelta(weeks=WEEKS + 2)).isoformat()
    response = http.get(API, params={
        "$select": "cftc_contract_market_code,report_date_as_yyyy_mm_dd,open_interest_all,"
                   "noncomm_positions_long_all,noncomm_positions_short_all",
        "$where": f"cftc_contract_market_code in ({codes}) AND report_date_as_yyyy_mm_dd >= '{start}' "
                  f"AND report_date_as_yyyy_mm_dd <= '{as_of.isoformat()}T23:59:59'",
        "$order": "report_date_as_yyyy_mm_dd", "$limit": 5000,
    })
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    if df.empty:
        raise RuntimeError("no CFTC data")
    for col in ["open_interest_all", "noncomm_positions_long_all", "noncomm_positions_short_all"]:
        df[col] = pd.to_numeric(df[col])
    df["net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    df["net_share"] = df["net"] / df["open_interest_all"]
    rows = []
    for code, (label, factor) in MARKETS.items():
        market = df[df["cftc_contract_market_code"] == code].sort_values("report_date_as_yyyy_mm_dd")
        if len(market) < 20:
            continue
        ctx = context(market["net_share"], lookback=WEEKS)
        net, week = int(market["net"].iloc[-1]), str(market["report_date_as_yyyy_mm_dd"].iloc[-1])[:10]
        change = int(market["net"].iloc[-1] - market["net"].iloc[-2])
        rows.append(evidence(
            as_of, factor, SOURCE, "number",
            f"CFTC positioning, week to {week}: speculators net {'long' if net >= 0 else 'short'} {abs(net):,} "
            f"{label} contracts, {ctx['last']:+.1%} of open interest (z {ctx['z']:+.1f}, {ordinal(ctx['pct'])} "
            f"percentile of {ctx['n']} weeks; weekly change {change:+,} contracts)",
            ctx["last"], "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"))
    if not rows:
        raise RuntimeError("no CFTC market with enough history")
    return rows
