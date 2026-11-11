"""ETF analysts of the Macro/News agent: one Sonnet "hedge-fund quant" per theme of the universe.

Each analyst searches the news for its ETFs and gives every ETF a direction in [-1, 1], a confidence,
a rationale, a counter-argument and its sources. Leveraged and inverse ETFs on an index reuse the view
on their base ETF (e.g. NASDAQ 3X <- NASDAQ); on a commodity, the analyst judges the commodity itself.
The code applies the leverage in both cases. Views are logged in logs/etf_views.csv and blended into
the macro_news score (agents/macro_news.py).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from agents.macro_llm import UNTRUSTED, Usage, ask, extract_json
from data import config as settings
from data.schemas import ETF_VIEWS, LIST_SEP, validate_etf_views
from data.universe import load_universe

# Leveraged and inverse ETFs (leverage comes from data/macro_exposure.csv).
SAME_AS = {  # on an index already covered by another analyst: reuse that ETF's view
    "STOXX SHORT": "EURO STOXX 50",
    "EURO STOXX 2X": "EURO STOXX 50",
    "NASDAQ 3X": "NASDAQ",
    "S&P 500 SHORT": "S&P 500",
}
UNDERLYING = {  # on a commodity: the analyst judges the commodity itself
    "COPPER LVG": "the copper price",
    "PALLADIUM LVG": "the palladium price",
}

# What moves each theme's ETFs: the analyst's coverage brief.
THEME_BRIEFS = {
    "Global equity (benchmark)": "global earnings revisions; US mega-cap tech (about 70% of the index is US); Fed path "
                                 "and real yields; EUR/USD; risk sentiment (VIX).",
    "US equity": "S&P 500 earnings and guidance; mega-cap tech and AI spending; Fed path; US data (CPI, payrolls, "
                 "retail sales); EUR/USD.",
    "European equity": "euro-area PMIs and growth; ECB path; energy prices; China demand for exporters; banks and "
                       "luxury; for IBEX 35, Spanish banks and rates.",
    "Home market": "Belgian large caps (AB InBev, KBC, UCB, argenx, Ageas, Solvay...): their news and results; ECB "
                   "path; euro-area growth.",
    "Size & style factors": "small caps: rate cuts and domestic growth; minimum volatility: defensive rotation; "
                            "dividend leaders: bond yields and value rotation.",
    "Income strategies": "Nasdaq-100 direction and implied volatility: a covered-call fund earns more premium when "
                         "volatility is high but gives up the upside of a rally.",
    "Equity / Tech": "AI capex and hyperscaler spending; Nvidia, TSMC and ASML results and guidance; chip export "
                     "controls; rates for growth stocks; biotech: FDA decisions, M&A and rates; cybersecurity: breaches "
                     "and earnings; robotics: industrial orders.",
    "Frontier tech": "speculative risk appetite and retail flows; space: launch and contract news (NASA, defence); "
                     "quantum: funding rounds, technical milestones, hype cycles; rates.",
    "Financials": "ECB path and the slope of the yield curve; Italian and French spreads; credit quality and "
                  "provisions; bank earnings, buybacks and M&A.",
    "Health care": "big pharma results and pipelines (obesity drugs); US drug-pricing policy; defensive rotation; "
                   "EUR/USD.",
    "Emerging markets": "the dollar and EM flows; China stimulus and data; commodity prices; India: oil and foreign "
                        "flows; Korea: chip exports; Brazil: commodities, Selic and politics; China A: policy support.",
    "Energy & commodities": "oil: OPEC+ decisions, US inventories (EIA), Middle East and Russia, demand data; refining "
                            "margins; producers move more than the oil price.",
    "Energy transition": "uranium: spot and term prices, Sprott trust buying, reactor and AI-power news; clean energy "
                         "and hydrogen: rates, subsidies and policy, project cancellations, Chinese competition.",
    "Industrial & strategic metals": "copper: LME and COMEX prices, Chinese demand and stocks, mine disruptions, "
                                     "tariffs; rare earths: Chinese export controls and defence demand.",
    "Precious metals": "real yields and the dollar; central-bank buying; gold ETF flows; CFTC positioning; "
                       "geopolitics; miners move more than the metal and also depend on costs.",
    "Fixed income / Bonds": "Fed and ECB rate expectations; inflation prints; term premium and bond supply; credit "
                            "spreads for high yield; the US Treasury fund also moves with EUR/USD.",
    "Money market (cash)": "the ECB deposit rate only: near-riskless carry, so its score should stay small.",
    "Defensive sector": "defence: budgets, NATO, conflicts and ceasefires, contract awards; staples: risk-off "
                        "rotation, input costs, consumer demand.",
    "Digital assets": "spot bitcoin ETF flows; funding rates and leverage; stablecoin supply; regulation; risk "
                      "appetite; crypto equities move more than bitcoin.",
    "Leveraged & inverse": "copper: LME and COMEX prices, Chinese demand and stocks, supply; palladium: car-catalyst "
                           "demand, Russian supply, substitution by platinum.",
}

SYSTEM = """You are a senior quant researcher at a top hedge fund, covering {theme} ETFs. You are known for calibrated, evidence-based calls, and for saying "no edge" when there is none.

Today is {today}. Your training data ends well before today: prices, rates, data releases and news may all have changed since then, so search for current facts before you judge, even when you feel sure.

Your desk runs a EUR portfolio of European-listed ETFs and trades on 7, 10, 14, 17 and 21 December 2026, so your horizon is the next few trading days. For a EUR investor, an ETF that holds dollar assets also moves with EUR/USD.

What drives your ETFs: {brief}

How to work:
1. Search first, then judge. Start with one broad search on this week's news for your theme, then target the two or three drivers that matter most right now. Make each search count: combine related terms in one query (for example "gold price real yields dollar ETF flows this week"), and cover ETFs that share drivers with the same searches. You have about {searches} searches; if they run out, judge from what you found.
2. For each ETF, weigh what is new against what is already priced in, and look for scheduled catalysts inside the horizon (with their date), positioning and flows.
3. Score each ETF:
   - direction in [-1, 1], the move you expect over the next few trading days: 0.1 to 0.3 is a slight tilt, 0.3 to 0.6 a clear view backed by a specific catalyst or data point, above 0.6 only when several independent and recent facts point the same way (same scale for negative views).
   - confidence in [0, 1], how much evidence supports the call: 0.2 thin or mixed, 0.5 solid, 0.8 strong and recent.
   On most days most ETFs deserve a small score. With no relevant news, give 0 with a low confidence: that is a valid answer.
4. Write the rationale your portfolio manager wants: 2 to 4 sentences of dated facts with numbers, ending with the next catalyst and its date when there is one. Then the strongest counter-argument, and the URLs you relied on: at least one for every ETF, the page you relied on most for it, even when you conclude there is no edge.

Never invent a number or a date. Report analysts' forecasts as attributed opinions, not as facts. {untrusted}

Answer with only a JSON object in a ```json block, with one entry per ETF you were given:
{{"etfs": [{{"ticker": "<ticker exactly as given>", "direction": <number>, "confidence": <number>, "rationale": "...", "counter_argument": "...", "sources": ["<url>"]}}]}}"""
GENERIC_BRIEF = "holdings, sector and macro drivers, flows, positioning and upcoming catalysts."


def system_prompt(theme: str, as_of: date) -> str:
    return SYSTEM.format(theme=theme, today=as_of.isoformat(), brief=THEME_BRIEFS.get(theme, GENERIC_BRIEF),
                         searches=settings.MACRO_ANALYST_SEARCHES, untrusted=UNTRUSTED)


def _etf_lines(etfs: pd.DataFrame) -> str:
    lines = []
    for r in etfs.itertuples():
        line = f'- "{r.ticker}": {r.name}'
        if r.ticker in UNDERLYING:
            line += f" · judge {UNDERLYING[r.ticker]} itself: the code applies the leverage"
        lines.append(line)
    return "\n".join(lines)


def _urls(sources) -> list[str]:
    """Source URLs, given as strings or as {"url": ...} objects."""
    urls = [s.get("url") if isinstance(s, dict) else s for s in sources or []]
    return [u.strip() for u in urls if isinstance(u, str) and u.strip().startswith("http")]


def analyse_theme(client, usage: Usage, as_of: date, theme: str, etfs: pd.DataFrame,
                  attempts: int = 2) -> tuple[list[dict], dict[str, str]]:
    """One analyst. ETFs its answer leaves without a usable view are asked again, once.

    Returns (views, {ticker: why it still has no view}).
    """
    rows, remaining, dropped = [], etfs, {}
    for _ in range(attempts):
        text = ask(client, usage, settings.LLM_REASONING, system_prompt(theme, as_of),
                   f"Your ETFs:\n{_etf_lines(remaining)}",
                   tools=[{"type": settings.WEB_SEARCH_TOOL_REASONING, "name": "web_search",
                           "max_uses": settings.MACRO_ANALYST_SEARCHES}],
                   output_config={"effort": "medium"})
        new, dropped = _parse(text, as_of, set(remaining["ticker"]))
        rows += new
        remaining = remaining[remaining["ticker"].isin(dropped)]
        if remaining.empty:
            return rows, {}
    return rows, dropped


def _match(answered, expected: set[str]) -> str | None:
    """The expected ticker an answer refers to: exact, or the start of a longer string such as
    'S&P 500 · iShares Core S&P 500 UCITS ETF' (the longest match wins: GOLD MINERS, not GOLD)."""
    if not isinstance(answered, str):
        return None
    text = answered.strip().strip('"').upper()
    hits = [e for e in expected if text == e.upper() or text.startswith(e.upper() + " ")
            or text.startswith(e.upper() + '"')]
    return max(hits, key=len) if hits else None


def _parse(text: str, as_of: date, expected: set[str]) -> tuple[list[dict], dict[str, str]]:
    """Usable views in an answer, and why each expected ETF without one was dropped."""
    try:
        answer = extract_json(text)
    except ValueError:
        return [], {t: "no JSON in the answer" for t in expected}
    rows, dropped = [], {}
    for view in answer.get("etfs", []) if isinstance(answer, dict) else []:
        ticker = _match(view.get("ticker"), expected)
        if ticker is None or ticker in {r["ticker"] for r in rows}:
            continue
        urls = _urls(view.get("sources"))
        if not urls:
            dropped[ticker] = "no source URL"
            continue
        rows.append({"date": as_of.isoformat(), "ticker": ticker,
                     "direction": float(np.clip(view.get("direction", 0), -1, 1)),
                     "confidence": float(np.clip(view.get("confidence", 0), 0, 1)),
                     "rationale": view.get("rationale", ""), "counter_argument": view.get("counter_argument", ""),
                     "sources": LIST_SEP.join(dict.fromkeys(urls))})
    for ticker in expected - {r["ticker"] for r in rows}:
        dropped.setdefault(ticker, "left out of the answer")
    return rows, dropped


def analyse_all(client, usage: Usage, as_of: date) -> tuple[pd.DataFrame, dict[str, str]]:
    """One analyst per theme, in parallel. Returns (ETF views, {analyst: what went wrong})."""
    universe = load_universe()
    rows, failures = [], {}
    covered = universe[~universe["ticker"].isin(SAME_AS)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {theme: pool.submit(analyse_theme, client, usage, as_of, theme, etfs)
                for theme, etfs in covered.groupby("theme", sort=False)}
        for theme, job in jobs.items():
            try:
                views, missing = job.result()
            except Exception as exc:
                failures[f"analyst {theme}"] = f"{type(exc).__name__}: {exc}"
                continue
            rows += views
            if missing:
                failures[f"analyst {theme}"] = "no view for " + "; ".join(
                    f"{ticker} ({why})" for ticker, why in sorted(missing.items()))
    by_ticker = {r["ticker"]: r for r in rows}
    for ticker, base in SAME_AS.items():
        if base in by_ticker:
            rows.append({**by_ticker[base], "ticker": ticker,
                         "rationale": f"Same underlying as {base}: {by_ticker[base]['rationale']}"})
    return validate_etf_views(pd.DataFrame(rows, columns=ETF_VIEWS)), failures


def blend(signals: pd.DataFrame, views: pd.DataFrame, exposure: pd.DataFrame,
          share: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """macro_news score = (1 - share) x factor score + share x analyst score, per ETF.

    The analyst score is the view times the ETF's leverage (from the exposure table), clipped to [-1, 1].
    An ETF without an analyst view keeps its factor score. Returns (blended signals, comparison table).
    """
    share = settings.MACRO_ANALYST_SHARE if share is None else share
    leverage = exposure.set_index("ticker")["multiplier"].astype(float)
    by_ticker = views.set_index("ticker")
    out, table = signals.copy(), []
    for i, row in out.iterrows():
        factor_score = float(row["score"])
        analyst = None
        if row["ticker"] in by_ticker.index:
            view = by_ticker.loc[row["ticker"]]
            analyst = float(np.clip(leverage[row["ticker"]] * view["direction"], -1, 1))
            out.at[i, "score"] = round((1 - share) * factor_score + share * analyst, 4)
            out.at[i, "confidence"] = round((1 - share) * float(row["confidence"]) + share * view["confidence"], 4)
            out.at[i, "rationale"] = f"{row['rationale']} | analyst {analyst:+.2f}: {str(view['rationale'])[:160]}"
        table.append({"ticker": row["ticker"], "factors": factor_score, "analyst": analyst,
                      "final": float(out.at[i, "score"])})
    return out, pd.DataFrame(table)
