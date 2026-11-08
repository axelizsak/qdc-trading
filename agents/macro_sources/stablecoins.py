"""Total stablecoin supply from DefiLlama (public API, no key): new money waiting to enter crypto."""
from datetime import date, datetime, timezone

import pandas as pd

from agents.macro_sources._common import context, evidence, http_client, ordinal

SOURCE = "defillama_stablecoins"
API = "https://stablecoins.llama.fi/stablecoincharts/all"


def collect(as_of: date, http=None) -> list[dict]:
    http = http or http_client()
    response = http.get(API)
    response.raise_for_status()
    points = {datetime.fromtimestamp(int(p["date"]), tz=timezone.utc).date(): sum(p["totalCirculatingUSD"].values())
              for p in response.json() if p.get("totalCirculatingUSD")}
    supply = pd.Series(points).sort_index()
    supply = supply[supply.index <= as_of]
    if len(supply) < 400:
        raise RuntimeError("not enough stablecoin history")
    change_30 = supply.pct_change(30).dropna()
    ctx = context(change_30, lookback=365)
    week = supply.iloc[-1] / supply.iloc[-8] - 1
    return [evidence(
        as_of, "crypto", SOURCE, "number",
        f"Stablecoin supply ${supply.iloc[-1] / 1e9:,.1f}bn on {supply.index[-1]}: {week:+.2%} over 7 days, "
        f"{ctx['last']:+.2%} over 30 days (30-day growth at the {ordinal(ctx['pct'])} percentile of the last year, "
        f"z {ctx['z']:+.1f}; rising supply = more cash ready to buy crypto)",
        ctx["last"], "https://defillama.com/stablecoins")]
