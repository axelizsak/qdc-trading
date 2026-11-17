"""[3] Macro / News agent. Owner: Axel. LLM: yes.

The code collects and calculates, the LLM judges:
1. Collectors (agents/macro_sources/) bring numbers, each set against its own history.
2. The fast model runs targeted web searches: news per factor, asset-manager notes, central-bank tone, calendar.
3. It also explores freely for signals from new sources; those enter the registry "on trial".
4. The reasoning model reads all evidence and logs/lessons.md and gives one view per factor (agents/macro_llm.py).
5. Code turns the views into one score per ETF with data/macro_exposure.csv: the macro_market score.
   No LLM in this step.
6. In parallel, 20 ETF analysts (one "hedge-fund quant" per theme, agents/macro_analysts.py)
   search the news and score their ETFs: the macro_etf score. Before judging, each one reads what the
   code calculated from prices (agents/macro_tape.py): what the ETF already did, and how the analysts'
   own past calls on it turned out.

Two scores, not one blend: the Portfolio Manager weights them and learns which one predicts better.

Writes to logs/: evidence.csv, macro_views.csv, etf_views.csv, source_registry.csv; the daily run also writes
macro/<date>.md and the day's macro_market and macro_etf rows of signals.csv (the pipeline does the same).

Daily run (every business day):
    python -m agents.macro_news                  # today
    python -m agents.macro_news --date 2026-12-07
    python -m agents.macro_news --force          # run a date again (otherwise read back from logs/)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from agents import llm, macro_analysts, macro_llm, macro_tape
from agents.macro_sources import collect_all
from data import config as settings
from data.config import MACRO_FACTORS
from data.schemas import (
    ETF_VIEWS, EVIDENCE, LIST_SEP, MACRO_VIEWS, SIGNALS, SOURCE_REGISTRY, split_list, validate_evidence,
    validate_macro_exposure, validate_macro_views, validate_signals, validate_source_registry,
)
from data.universe import load_universe

AGENT_MARKET = "macro_market"  # the day's macro views through the exposure table
AGENT_ETF = "macro_etf"  # the ETF analysts' own call
AGENTS = [AGENT_MARKET, AGENT_ETF]


def run(as_of: date, lessons: str = "", logs_dir: Path | None = None, client=None,
        collectors: dict | None = None, prices: pd.DataFrame | None = None) -> tuple[str, pd.DataFrame]:
    """Returns (the day's report in Markdown, signals in the Signals format: macro_market and macro_etf rows).

    `prices` (Prices format, from the data agent) lets the analysts read each ETF's tape and their own
    graded calls. Without prices they judge from the news alone.
    """
    logs = Path(logs_dir) if logs_dir else settings.LOGS_DIR
    client = client or llm.client()
    usage = macro_llm.Usage()
    registry = load_registry(logs)
    on_trial = set(registry.loc[registry["status"] == "testing", "source"])
    dropped = set(registry.loc[registry["status"] == "dropped", "source"])
    exposure = load_exposure()
    table = macro_tape.closes(prices, as_of)
    grades = macro_tape.grade_calls(past_etf_views(logs, as_of), table, exposure.set_index("ticker")["multiplier"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        numbers = pool.submit(collect_all, as_of, collectors)
        analysts = pool.submit(macro_analysts.analyse_all, client, usage, as_of, table, grades)
        searches = pool.submit(macro_llm.search_all, client, usage, as_of)
        exploring = pool.submit(macro_llm.explore, client, usage, as_of, registry["source"].tolist(), _etf_list())
        ev_numbers, failures = numbers.result()
        ev_search, search_failures = searches.result()
        failures.update(search_failures)
        try:
            ev_explore = exploring.result()
        except Exception as exc:
            ev_explore, failures["exploration"] = [], f"{type(exc).__name__}: {exc}"
        etf_views, analyst_failures = analysts.result()
        failures.update(analyst_failures)

    frames = [f for f in (ev_numbers, pd.DataFrame(ev_search, columns=EVIDENCE),
                          pd.DataFrame(ev_explore, columns=EVIDENCE)) if len(f)]
    if not frames:
        raise RuntimeError("no evidence collected: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))
    ev = validate_evidence(pd.concat(frames, ignore_index=True))
    ev = ev[~ev["source"].isin(dropped)].reset_index(drop=True)

    views = macro_llm.reason(client, usage, as_of, ev, lessons, on_trial)
    market = score_etfs(views, as_of, exposure)
    etf = macro_analysts.etf_signals(etf_views, exposure, as_of, AGENT_ETF)
    signals = validate_signals(pd.concat([market, etf], ignore_index=True))

    _save(logs / "evidence.csv", ev, as_of)
    _save(logs / "macro_views.csv", views, as_of)
    if len(etf_views):
        _save(logs / "etf_views.csv", etf_views, as_of)
    save_registry(logs, register_sources(registry, ev, as_of))
    return render(as_of, views, ev, failures, usage, etf_views, signals, grades, table), signals


def past_etf_views(logs: Path, as_of: date) -> pd.DataFrame | None:
    """The analysts' views logged before `as_of`, or None."""
    path = logs / "etf_views.csv"
    if not path.exists():
        return None
    views = pd.read_csv(path)
    return views[views["date"].astype(str) < as_of.isoformat()]


# --- Scores: views x exposure table, no LLM --------------------------------------------------

def load_exposure() -> pd.DataFrame:
    return validate_macro_exposure(pd.read_csv(settings.MACRO_EXPOSURE_CSV))


def score_etfs(views: pd.DataFrame, as_of: date, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """The macro_market score = multiplier x Σ exposure × direction / max(Σ|exposure|, 1), clipped to [-1, 1].

    Dividing by at least 1 keeps an ETF with little macro exposure (e.g. cash) near 0.
    confidence = the views' confidence, weighted by each factor's |exposure|.
    """
    exposure = load_exposure() if exposure is None else exposure
    table = exposure.set_index("ticker")
    e = table[MACRO_FACTORS].astype(float)
    by_factor = views.set_index("factor")
    direction = by_factor["direction"].reindex(MACRO_FACTORS).fillna(0.0)
    confidence = by_factor["confidence"].reindex(MACRO_FACTORS).fillna(0.0)
    weight = e.abs().sum(axis=1)
    contribution = e.mul(direction, axis=1)
    score = (table["multiplier"].astype(float) * contribution.sum(axis=1) / weight.clip(lower=1)).clip(-1, 1)
    conf = (e.abs().mul(confidence, axis=1).sum(axis=1) / weight.replace(0, np.nan)).fillna(0.0)
    rows = []
    for ticker in table.index:
        top = contribution.loc[ticker].abs().sort_values(ascending=False).head(3)
        drivers = ", ".join(f"{f} {direction[f]:+.2f} x exposure {e.loc[ticker, f]:+.1f}"
                            for f in top.index if top[f] > 0) or "no macro view on its factors"
        lever = table.loc[ticker, "multiplier"]
        rows.append({"ticker": ticker, "date": as_of.isoformat(), "agent": AGENT_MARKET,
                     "score": round(float(score[ticker]), 4), "confidence": round(float(conf[ticker]), 4),
                     "rationale": f"macro: {drivers}" + (f"; leverage {lever:+g}" if lever != 1 else "")})
    return validate_signals(pd.DataFrame(rows, columns=SIGNALS))


# --- Source registry and hit rates -------------------------------------------------------------

def load_registry(logs: Path) -> pd.DataFrame:
    path = logs / "source_registry.csv"
    if path.exists():
        return validate_source_registry(pd.read_csv(path))
    return pd.DataFrame(columns=SOURCE_REGISTRY)


def save_registry(logs: Path, registry: pd.DataFrame) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    validate_source_registry(registry).to_csv(logs / "source_registry.csv", index=False)


def register_sources(registry: pd.DataFrame, ev: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Add new sources (exploration ones "on trial") and keep each source's factor list up to date."""
    registry = registry.copy()
    for source, rows in ev.groupby("source"):
        factors = set(rows["factor"])
        known = registry["source"] == source
        if known.any():
            current = set(split_list(registry.loc[known, "factors"].iloc[0]))
            registry.loc[known, "factors"] = LIST_SEP.join(sorted(current | factors))
            continue
        kind = rows["kind"].iloc[0]
        new = pd.DataFrame([{"source": source, "kind": kind, "factors": LIST_SEP.join(sorted(factors)),
                             "status": "testing" if kind == "exploration" else "active",
                             "first_seen": as_of.isoformat(), "n_uses": 0, "hit_rate": np.nan}])
        registry = new if registry.empty else pd.concat([registry, new], ignore_index=True)
    return registry


def grade_views(logs: Path, table: pd.DataFrame, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every past macro view that can be judged: date, factor, direction, move, hit, sources.

    `table`: closes before the day of the review (agents/macro_tape.closes). A view of day T is judged on the
    move of the ETFs exposed to its factor (exposure-weighted, leverage included) from the last close before T
    to HORIZON_SESSIONS sessions later. It is right when it has the sign of that move. A view smaller than
    macro_tape.MIN_CALL is no call."""
    columns = ["date", "factor", "direction", "move", "hit", "sources"]
    path = logs / "macro_views.csv"
    if not path.exists() or table.empty:
        return pd.DataFrame(columns=columns)
    views = pd.read_csv(path)
    exposure = load_exposure() if exposure is None else exposure
    effective = exposure.set_index("ticker")[MACRO_FACTORS].mul(exposure.set_index("ticker")["multiplier"], axis=0)
    rows = []
    for day, day_views in views.groupby(views["date"].astype(str).str[:10]):
        everything = pd.DataFrame({"date": day, "ticker": effective.index, "direction": 1.0})
        realised = macro_tape.grade_calls(everything, table, pd.Series(dtype=float)).set_index("ticker")["ret"]
        if realised.empty:
            continue  # no close after that day yet
        for v in day_views.itertuples():
            exposed = effective[v.factor].reindex(realised.index).dropna()
            if abs(v.direction) < macro_tape.MIN_CALL or exposed.abs().sum() == 0:
                continue
            move = float((exposed * realised[exposed.index]).sum() / exposed.abs().sum())
            rows.append({"date": day, "factor": v.factor, "direction": float(v.direction), "move": move,
                         "hit": bool(np.sign(move) == np.sign(v.direction)), "sources": v.sources})
    return pd.DataFrame(rows, columns=columns)


def review_sources(logs_dir: Path | None, table: pd.DataFrame, exposure: pd.DataFrame | None = None) -> dict:
    """Grade every source on the views that cited it, and update logs/source_registry.csv. Called by the review loop.

    Stateless: uses and hit rates are recomputed from all the views that can be judged, so running it twice
    changes nothing. A source on trial (found by exploration) that is wrong too often is dropped, and one
    that is right often enough is proposed as a regular source. A regular source is never dropped by code:
    the weak ones are listed for a person to decide.
    Returns {graded: the judged views, dropped, weak, propose_active}."""
    logs = Path(logs_dir) if logs_dir else settings.LOGS_DIR
    graded = grade_views(logs, table, exposure)
    registry = load_registry(logs)
    if registry.empty:
        return {"graded": graded, "dropped": [], "weak": [], "propose_active": []}
    registry = registry.set_index("source")
    uses, hits = {}, {}
    for v in graded.itertuples():
        for source in split_list(v.sources):
            uses[source] = uses.get(source, 0) + 1
            hits[source] = hits.get(source, 0) + int(v.hit)
    registry["n_uses"] = [uses.get(source, 0) for source in registry.index]
    registry["hit_rate"] = [hits[source] / uses[source] if uses.get(source) else np.nan for source in registry.index]
    mature = registry["n_uses"] >= settings.SOURCE_MIN_USES
    below = mature & (registry["hit_rate"] < settings.SOURCE_DROP_HIT_RATE)
    drop = below & (registry["status"] == "testing")
    registry.loc[drop, "status"] = "dropped"
    weak = below & (registry["status"] == "active")
    propose = mature & (registry["status"] == "testing") & (registry["hit_rate"] >= settings.SOURCE_PROMOTE_HIT_RATE)
    save_registry(logs, registry.reset_index())
    return {"graded": graded, "dropped": sorted(registry.index[drop]), "weak": sorted(registry.index[weak]),
            "propose_active": sorted(registry.index[propose])}


# --- Logs and report ---------------------------------------------------------------------------

def _save(path: Path, df: pd.DataFrame, as_of: date, keep=None) -> None:
    """Write `df` to `path`, replacing the rows of the same date (and matching `keep`, if given)."""
    if path.exists():
        old = pd.read_csv(path)
        same_day = old["date"].astype(str) == as_of.isoformat()
        old = old[~(same_day & keep(old))] if keep is not None else old[~same_day]
        if len(old):
            df = pd.concat([old, df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _etf_list() -> str:
    universe = load_universe()
    return "\n".join(f"- {theme}: {', '.join(group['ticker'])}" for theme, group in universe.groupby("theme", sort=False))


BRIEF_CHARS = 3600  # the Portfolio Manager reads the first 4,000 characters of the report: the brief must fit
BRIEF_CALLS = 6
BRIEF_CATALYSTS = 8


GENERIC_WORDS = {"us", "euro", "area", "eurozone", "the", "of", "for", "and", "at", "in", "on", "to", "a", "after",
                 "before", "plus", "data", "release", "report", "final", "flash", "meeting", "decision", "close",
                 "november", "december", "october", "q1", "q2", "q3", "q4"}


def _words(event: str) -> list[str]:
    return ["".join(c for c in word if c.isalnum()) for word in event.lower().split()]


def same_event(a: str, b: str) -> bool:
    """Two analysts' wordings of one event: they start with the same name (Nvidia, ECB, OPEC...) or share two
    words that mean something ('US payrolls' and 'US CPI' share none)."""
    first, second = _words(a), _words(b)
    if first and second and first[0] == second[0] and first[0] not in GENERIC_WORDS:
        return True
    return len((set(first) & set(second)) - GENERIC_WORDS - {""}) >= 2


def catalysts(etf_views: pd.DataFrame | None) -> list[tuple[str, str, list[str]]]:
    """The dated events the analysts found, soonest first: (ISO date, event, ETFs it matters for).
    Analysts word the same event differently: on a given day, wordings of one event are merged (same_event),
    under the wording of the analyst with the strongest call."""
    if etf_views is None or etf_views.empty or "catalyst_date" not in etf_views:
        return []
    found = etf_views[etf_views["catalyst_date"].notna() & etf_views["catalyst"].notna()]
    events: list[tuple[str, str, list[str]]] = []
    for view in found.sort_values("direction", key=abs, ascending=False).itertuples():
        day, text = str(view.catalyst_date)[:10], str(view.catalyst)
        known = next((e for e in events if e[0] == day and same_event(e[1], text)), None)
        if known is None:
            events.append((day, text, [view.ticker]))
        else:
            known[2].append(view.ticker)
    return sorted(events)


def brief(as_of: date, views: pd.DataFrame, etf_views: pd.DataFrame | None, signals: pd.DataFrame | None,
          grades: pd.DataFrame | None) -> list[str]:
    """What matters for the next sessions, first in the report and short: the events ahead, the analysts'
    strongest calls, the macro views in one line. Built by code from the run's own output."""
    lines = ["## Brief for the Portfolio Manager", "",
             f"Horizon: the next {settings.HORIZON_SESSIONS} trading sessions."]
    if grades is not None and len(grades):
        right, total = macro_tape.record(grades)
        if total:
            lines[-1] += f" The ETF analysts' record before today: {right} of {total} calls right ({right / total:.0%})."
    ahead = catalysts(etf_views)
    if ahead:
        lines += ["", "**Events ahead** (found by the ETF analysts in today's searches)", ""]
        for day, event, tickers in ahead[:BRIEF_CATALYSTS]:
            names = ", ".join(tickers[:5]) + (f" +{len(tickers) - 5}" if len(tickers) > 5 else "")
            what = event.split(". ")[0].rstrip(".")  # the event, without the analyst's comment on it
            lines.append(f"- {date.fromisoformat(day):%a %d %b}: {what[:110]} ({names})")
    if etf_views is not None and signals is not None and len(etf_views):
        scores = signals.pivot(index="ticker", columns="agent", values="score").reindex(columns=AGENTS)
        calls = etf_views.set_index("ticker")
        calls = calls[calls["direction"].abs() >= macro_tape.MIN_CALL]
        calls = calls.reindex(calls["direction"].abs().sort_values(ascending=False).index).head(BRIEF_CALLS)
        if len(calls):
            lines += ["", f"**Strongest ETF calls** ({AGENT_ETF} = analyst view x leverage; {AGENT_MARKET} = the macro "
                          "views through the exposure table)", "",
                      f"| ETF | {AGENT_ETF} | {AGENT_MARKET} | Why |", "|---|---|---|---|"]
            for ticker, call in calls.iterrows():
                why = str(call["rationale"]).replace("|", "/")
                why = why[:200] + ("..." if len(why) > 200 else "")
                lines.append(f"| {ticker} | {scores.loc[ticker, AGENT_ETF]:+.2f} | {scores.loc[ticker, AGENT_MARKET]:+.2f} | {why} |")
        else:
            lines += ["", "**Strongest ETF calls**: none today. No analyst found an edge on its ETFs."]
    ranked = views.reindex(views["direction"].abs().sort_values(ascending=False).index)
    lines += ["", "**Macro views** (+1 to -1): " + ", ".join(f"{v.factor} {v.direction:+.2f}" for v in ranked.itertuples())]
    while len("\n".join(lines)) > BRIEF_CHARS and any(line.startswith("| ") and len(line) > 120 for line in lines):
        longest = max((line for line in lines if line.startswith("| ")), key=len)  # shorten the longest "why"
        lines[lines.index(longest)] = longest[:len(longest) - 40].rstrip(" |.") + "... |"
    return lines


def render(as_of: date, views: pd.DataFrame, ev: pd.DataFrame, failures: dict[str, str], usage,
           etf_views: pd.DataFrame | None = None, signals: pd.DataFrame | None = None,
           grades: pd.DataFrame | None = None, table: pd.DataFrame | None = None) -> str:
    lines = [f"# Macro/News, {as_of}", "", f"Cost of this run: {usage.line()}", "",
             *brief(as_of, views, etf_views, signals, grades), "",
             "## Views", "", "| Factor | Direction | Confidence | Sources |", "|---|---|---|---|"]
    lines += [f"| {v.factor} | {v.direction:+.2f} | {v.confidence:.2f} | {v.sources.replace(LIST_SEP, ', ')} |"
              for v in views.itertuples()]
    missing = [f for f in MACRO_FACTORS if f not in set(views["factor"])]
    if missing:
        lines += ["", f"No view (treated as 0): {', '.join(missing)}"]
    for v in views.itertuples():
        lines += ["", f"### {v.factor} ({v.direction:+.2f}, confidence {v.confidence:.2f})", "", v.rationale,
                  "", f"Counter-argument: {v.counter_argument}"]
    if etf_views is not None and signals is not None and len(etf_views):
        scores = signals.pivot(index="ticker", columns="agent", values="score")
        scores = scores.reindex(columns=AGENTS).dropna(subset=[AGENT_ETF])
        tape = "the tape of each ETF and their own graded calls" if table is not None and len(table) else "no prices"
        lines += ["", "## ETF analysts", "",
                  f"{len(etf_views)} ETFs scored by the theme analysts over the next "
                  f"{settings.HORIZON_SESSIONS} sessions. They read {tape} before judging. "
                  f"{AGENT_MARKET} = the views above through the exposure table; {AGENT_ETF} = analyst view x leverage. "
                  "Strongest analyst scores first."]
        if grades is not None and len(grades):
            right, total = macro_tape.record(grades)
            lines += ["", f"Analysts' record before today: {right} of {total} past calls right"
                          + (f" ({right / total:.0%})" if total else "") + f", graded over up to "
                          f"{settings.HORIZON_SESSIONS} sessions ({len(grades) - total} were no call)."]
        lines += ["", f"| ETF | {AGENT_MARKET} | {AGENT_ETF} | Next catalyst | Analyst rationale |", "|---|---|---|---|---|"]
        by_ticker = etf_views.reindex(columns=ETF_VIEWS).set_index("ticker")
        for ticker, r in scores.reindex(scores[AGENT_ETF].abs().sort_values(ascending=False).index).iterrows():
            market = "—" if pd.isna(r[AGENT_MARKET]) else f"{r[AGENT_MARKET]:+.2f}"
            view = by_ticker.loc[ticker]
            ahead = (f"{date.fromisoformat(str(view['catalyst_date'])[:10]):%d %b}: {view['catalyst']}".replace("|", "/")
                     if isinstance(view["catalyst"], str) and view["catalyst"].strip() else "—")
            why = str(view["rationale"]).replace("|", "/")[:240]
            lines.append(f"| {ticker} | {market} | {r[AGENT_ETF]:+.2f} | {ahead} | {why} |")
    found = ev[ev["kind"] == "exploration"]
    lines += ["", "## Exploration findings (on trial)", ""]
    lines += [f"- {r.source} ({r.factor}): {r.summary} {r.url}".rstrip() for r in found.itertuples()] or ["None."]
    lines += ["", "## Sources", "", f"{len(ev)} evidence items from {ev['source'].nunique()} sources "
              "(full list in logs/evidence.csv).", ""]
    for source, rows in ev.groupby("source"):
        urls = list(dict.fromkeys(u for u in rows["url"] if isinstance(u, str) and u))[:3]
        lines.append(f"- {source} ({len(rows)}): {' '.join(urls)}".rstrip())
    if failures:
        lines += ["", "## Failed collectors and searches", ""] + [f"- {k}: {v}" for k, v in failures.items()]
    return "\n".join(lines) + "\n"


def fetch_prices(as_of: date) -> pd.DataFrame | None:
    """Prices for the analysts' tape and record, from the data agent. None when they cannot be fetched:
    the run goes on without them."""
    try:
        from data import data_agent
        return data_agent.fetch_prices(as_of - timedelta(days=90), as_of)
    except Exception as exc:
        print(f"no prices for the analysts ({type(exc).__name__}): they judge from the news alone")
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Daily run of the Macro/News agent.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD, default today")
    parser.add_argument("--force", action="store_true", help="run again even if the date is already in logs/")
    args = parser.parse_args(argv)
    logs = settings.LOGS_DIR
    report = logs / "macro" / f"{args.date.isoformat()}.md"
    if report.exists() and not args.force:
        print(f"{args.date}: already done, see {report} (use --force to run again)")
        return
    lessons = (logs / "lessons.md").read_text(encoding="utf-8") if (logs / "lessons.md").exists() else ""
    summary, signals = run(args.date, lessons, prices=fetch_prices(args.date))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(summary, encoding="utf-8", newline="\n")
    _save(logs / "signals.csv", signals, args.date, keep=lambda old: old["agent"].isin(AGENTS))
    print(summary.split("\n### ")[0])
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
