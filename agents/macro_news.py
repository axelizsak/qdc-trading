"""[3] Macro / News agent. Owner: Axel. LLM: yes.

The code collects and calculates, the LLM judges:
1. Collectors (agents/macro_sources/) bring numbers, each set against its own history.
2. Haiku runs targeted web searches: news per factor, asset-manager notes, central-bank tone, calendar.
3. Sonnet explores freely for signals from new sources; those enter the registry "on trial".
4. Sonnet reads all evidence and logs/lessons.md and gives one view per factor (agents/macro_llm.py).
5. Code turns the views into one score per ETF with data/macro_exposure.csv. No LLM in this step.
6. In parallel, 20 ETF analysts (one Sonnet "hedge-fund quant" per theme, agents/macro_analysts.py)
   search the news and score their ETFs; code blends them in: score = 50% factors + 50% analysts.

Writes to logs/: evidence.csv, macro_views.csv, etf_views.csv, source_registry.csv; the daily run also writes
macro/<date>.md and the day's macro_news rows of signals.csv (the pipeline does the same).

Daily run (every business day):
    python -m agents.macro_news                  # today
    python -m agents.macro_news --date 2026-12-07
    python -m agents.macro_news --force          # run a date again (otherwise read back from logs/)
"""
import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from agents import macro_analysts, macro_llm
from agents.macro_sources import collect_all
from data import config as settings
from data.config import MACRO_FACTORS
from data.schemas import (
    EVIDENCE, LIST_SEP, MACRO_VIEWS, SIGNALS, SOURCE_REGISTRY, split_list, validate_evidence,
    validate_macro_exposure, validate_macro_views, validate_signals, validate_source_registry,
)
from data.universe import load_universe

AGENT = "macro_news"


def run(as_of: date, lessons: str = "", logs_dir: Path | None = None, client=None,
        collectors: dict | None = None) -> tuple[str, pd.DataFrame]:
    """Returns (the day's report in Markdown, signals in the Signals format)."""
    logs = Path(logs_dir) if logs_dir else settings.LOGS_DIR
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    usage = macro_llm.Usage()
    registry = load_registry(logs)
    on_trial = set(registry.loc[registry["status"] == "testing", "source"])
    dropped = set(registry.loc[registry["status"] == "dropped", "source"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        numbers = pool.submit(collect_all, as_of, collectors)
        analysts = pool.submit(macro_analysts.analyse_all, client, usage, as_of)
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
    exposure = load_exposure()
    signals, comparison = macro_analysts.blend(score_etfs(views, as_of, exposure), etf_views, exposure)

    _save(logs / "evidence.csv", ev, as_of)
    _save(logs / "macro_views.csv", views, as_of)
    if len(etf_views):
        _save(logs / "etf_views.csv", etf_views, as_of)
    save_registry(logs, register_sources(registry, ev, as_of))
    return render(as_of, views, ev, failures, usage, etf_views, comparison), signals


# --- Scores: views x exposure table, no LLM --------------------------------------------------

def load_exposure() -> pd.DataFrame:
    return validate_macro_exposure(pd.read_csv(settings.MACRO_EXPOSURE_CSV))


def score_etfs(views: pd.DataFrame, as_of: date, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """score = multiplier x Σ exposure × direction / max(Σ|exposure|, 1), clipped to [-1, 1].

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
        rows.append({"ticker": ticker, "date": as_of.isoformat(), "agent": AGENT,
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


def update_source_hit_rates(as_of: date, returns: Mapping[str, float], logs_dir: Path | None = None,
                            exposure: pd.DataFrame | None = None) -> dict:
    """Grade the macro views of `as_of` against the ETF returns that followed. Called by the review loop.

    returns: realised return per ticker over the period the views were for.
    A view is right when its direction has the sign of the factor's realised move, measured as the
    exposure-weighted return of the ETFs exposed to that factor. Every source a view cites gets one more
    use and a new hit rate. A source wrong too often is dropped; a source on trial that is right often
    enough is proposed as a regular source (a person decides, the status is not changed automatically).
    """
    logs = Path(logs_dir) if logs_dir else settings.LOGS_DIR
    views = pd.read_csv(logs / "macro_views.csv")
    views = validate_macro_views(views[views["date"].astype(str) == as_of.isoformat()])
    registry = load_registry(logs).set_index("source")
    exposure = load_exposure() if exposure is None else exposure
    table = exposure.set_index("ticker")
    realised = pd.Series(returns, dtype=float).dropna()
    effective = table[MACRO_FACTORS].mul(table["multiplier"], axis=0).reindex(realised.index).dropna()
    graded = 0
    for v in views.itertuples():
        exposed = effective[v.factor]
        if v.direction == 0 or exposed.abs().sum() == 0:
            continue
        move = float((exposed * realised[exposed.index]).sum() / exposed.abs().sum())
        hit = float(np.sign(move) == np.sign(v.direction))
        graded += 1
        for source in split_list(v.sources):
            if source not in registry.index:
                continue
            n = int(registry.at[source, "n_uses"])
            previous = registry.at[source, "hit_rate"]
            registry.at[source, "hit_rate"] = hit if n == 0 or pd.isna(previous) else (previous * n + hit) / (n + 1)
            registry.at[source, "n_uses"] = n + 1
    mature = registry["n_uses"] >= settings.SOURCE_MIN_USES
    drop = mature & (registry["hit_rate"] < settings.SOURCE_DROP_HIT_RATE) & (registry["status"] != "dropped")
    registry.loc[drop, "status"] = "dropped"
    propose = mature & (registry["status"] == "testing") & (registry["hit_rate"] >= settings.SOURCE_PROMOTE_HIT_RATE)
    save_registry(logs, registry.reset_index())
    return {"graded_views": graded, "dropped": sorted(registry.index[drop]),
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


def render(as_of: date, views: pd.DataFrame, ev: pd.DataFrame, failures: dict[str, str], usage,
           etf_views: pd.DataFrame | None = None, comparison: pd.DataFrame | None = None) -> str:
    lines = [f"# Macro/News, {as_of}", "", f"Cost of this run: {usage.line()}", "",
             "## Views", "", "| Factor | Direction | Confidence | Sources |", "|---|---|---|---|"]
    lines += [f"| {v.factor} | {v.direction:+.2f} | {v.confidence:.2f} | {v.sources.replace(LIST_SEP, ', ')} |"
              for v in views.itertuples()]
    missing = [f for f in MACRO_FACTORS if f not in set(views["factor"])]
    if missing:
        lines += ["", f"No view (treated as 0): {', '.join(missing)}"]
    for v in views.itertuples():
        lines += ["", f"### {v.factor} ({v.direction:+.2f}, confidence {v.confidence:.2f})", "", v.rationale,
                  "", f"Counter-argument: {v.counter_argument}"]
    if etf_views is not None and comparison is not None and len(etf_views):
        lines += ["", "## ETF analysts", "",
                  f"{len(etf_views)} ETFs scored by the theme analysts. Final macro_news score = "
                  f"{1 - settings.MACRO_ANALYST_SHARE:.0%} factor views + {settings.MACRO_ANALYST_SHARE:.0%} analyst "
                  "(analyst view x leverage). Strongest final scores first.", "",
                  "| ETF | Factors | Analyst | Final | Analyst rationale |", "|---|---|---|---|---|"]
        rationale = etf_views.set_index("ticker")["rationale"]
        for r in comparison.reindex(comparison["final"].abs().sort_values(ascending=False).index).itertuples():
            analyst = "—" if pd.isna(r.analyst) else f"{r.analyst:+.2f}"
            why = str(rationale.get(r.ticker, "")).replace("|", "/")[:140]
            lines.append(f"| {r.ticker} | {r.factors:+.2f} | {analyst} | {r.final:+.2f} | {why} |")
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
    lessons = (logs / "lessons.md").read_text() if (logs / "lessons.md").exists() else ""
    summary, signals = run(args.date, lessons)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(summary)
    _save(logs / "signals.csv", signals, args.date, keep=lambda old: old["agent"] == AGENT)
    print(summary.split("\n### ")[0])
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
