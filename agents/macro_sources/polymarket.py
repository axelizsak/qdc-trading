"""Prediction-market probabilities from Polymarket (public Gamma API, no key)."""
import json
from datetime import date, datetime, timezone

from agents.macro_sources._common import evidence, http_client

SOURCE = "polymarket"
API = "https://gamma-api.polymarket.com/public-search"
# search query -> factor (no liquid markets found for the ECB, EUR/USD or copper: Claude's searches cover them)
QUERIES = {
    "fed rate cut": "fed_us_rates",
    "fed hike": "fed_us_rates",
    "recession": "growth_recession",
    "crude oil": "oil",
    "saudi oil": "oil",
    "gold": "gold_silver",
    "ai bubble": "tech_ai_semis",
    "ceasefire": "geopolitics_defence",
    "invade": "geopolitics_defence",
    "military clash": "geopolitics_defence",
    "bitcoin price": "crypto",
    "ethereum": "crypto",
}
MARKETS_PER_QUERY = 3
MIN_VOLUME_USD = 50_000
MIN_PROBABILITY, MAX_PROBABILITY = 0.01, 0.99  # near-certain markets carry no information
MAX_DAYS_TO_CLOSE = 180  # longer-dated markets say little about the next few weeks


def _open_markets(event: dict, now: datetime) -> list[dict]:
    markets = []
    for m in event.get("markets") or []:
        end = m.get("endDate")
        if m.get("closed") or not end or not m.get("outcomePrices"):
            continue
        days_left = (datetime.fromisoformat(end.replace("Z", "+00:00")) - now).days
        if not 0 <= days_left <= MAX_DAYS_TO_CLOSE:
            continue
        if float(m.get("volume") or 0) < MIN_VOLUME_USD:
            continue
        if not MIN_PROBABILITY < float(json.loads(m["outcomePrices"])[0]) < MAX_PROBABILITY:
            continue
        markets.append(m)
    return markets


def collect(as_of: date, http=None) -> list[dict]:
    http = http or http_client()
    now = datetime.now(timezone.utc)
    rows, seen = [], set()
    for query, factor in QUERIES.items():
        response = http.get(API, params={"q": query, "limit_per_type": 10, "events_status": "active"})
        response.raise_for_status()
        markets = [m for e in response.json().get("events") or [] for m in _open_markets(e, now)]
        markets.sort(key=lambda m: float(m.get("volume") or 0), reverse=True)
        for m in [m for m in markets if m["id"] not in seen][:MARKETS_PER_QUERY]:
            seen.add(m["id"])
            yes = float(json.loads(m["outcomePrices"])[0])
            week = m.get("oneWeekPriceChange")
            change = f", {float(week) * 100:+.0f} pts over a week" if week is not None else ""
            rows.append(evidence(
                as_of, factor, SOURCE, "number",
                f"Polymarket: '{m['question']}' Yes = {yes:.0%}{change} "
                f"(volume ${float(m.get('volume') or 0):,.0f}, closes {m['endDate'][:10]})",
                yes, f"https://polymarket.com/market/{m.get('slug', '')}"))
    if not rows:
        raise RuntimeError("no open Polymarket market matched the queries")
    return rows
