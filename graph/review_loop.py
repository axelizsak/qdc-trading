"""Review loop: what the system learns from its own past, before each day's decisions. Owner: Axel.
(It calls each agent's own grading code: Person 3 owns the grading of the Portfolio and Risk Managers.)

It runs first thing in a day, right after the data step, and on its own:
    python -m graph.review_loop                    # today, on logs/
    python -m graph.review_loop --date 2026-12-10

Stateless: every number is recomputed from the logs and the prices. Running it twice changes nothing, and
a change to an agent's settings shows in the next review. No LLM: it reports facts with their numbers.

1. Grade   every agent's past scores against what the ETFs did over the next HORIZON_SESSIONS sessions.
           The technical rule score is also recomputed for the last REVIEW_BACKFILL_DAYS days from prices
           (what the rule would have said each morning), so it is judged on months, not on a few days.
2. Weigh   the agent weights that evidence supports -> logs/agent_weights.csv, the evidence of model
           re-assessment. The Portfolio Manager learns its weights from the same evidence, the same way.
3. Check   the macro sources: one on trial that is wrong too often is dropped -> logs/source_registry.csv.
4. Write   logs/lessons.md, newest review first. Macro/News, technical_llm and the Portfolio Manager read it
           before they decide.
5. Record  the re-assessment and the lessons in the logbook (the audit trail).

To change how the system learns: REVIEW_BACKFILL_DAYS and HORIZON_SESSIONS in data/config.py, the learning
speed and the weight bounds in agents/portfolio_config.py (LEARN_*, AGENT_WEIGHT_*).
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents import logbook, macro_news, macro_tape, portfolio_backfill, portfolio_learning, risk_review
from data import config as settings
from data.schemas import AGENT_WEIGHTS, SIGNALS, validate_agent_weights
from data.universe import load_universe

JOURNAL = "lessons.md"
HEADER = ("# Lessons journal\n\nWritten by the review loop (graph/review_loop.py) from the logs and the prices: "
          "facts with their numbers, no opinion. Newest review first. The agents read it before they decide.\n")
REVIEWS_KEPT = 5  # older reviews stay in git and in the logbook
MIN_CALLS = 5  # calls needed before a hit rate is quoted
MISSES = 3


def starting_weights(agents: list[str]) -> dict[str, float]:
    """Weights before any evidence: each family (technical, macro) counts the same, split equally between
    its agents in `agents`. A family without an agent gives its share to the other."""
    families = {name: [a for a in members if a in agents] for name, members in settings.AGENT_FAMILIES.items()}
    families = {name: members for name, members in families.items() if members}
    return {a: 1 / len(families) / len(members) for members in families.values() for a in members}


def technical_history(prices: pd.DataFrame, features: pd.DataFrame | None, as_of: date,
                      days: int | None = None) -> pd.DataFrame:
    """The technical rule score recomputed for the last `days` quote days before `as_of`, each from the prices
    known that morning (agents/portfolio_backfill.py). Signals format. It is evidence for the learning only:
    logs/signals.csv stays the record of what the agents really said."""
    days = settings.REVIEW_BACKFILL_DAYS if days is None else days
    return portfolio_backfill.backfill_technical(prices, portfolio_backfill.quote_days(prices, days, as_of), features)


def evidence(logs: Path, as_of: date, backfill: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every score that can be judged on `as_of`: the logged ones (without the neutral rows of a failed agent)
    and the recomputed technical ones, dated before `as_of`. A logged score wins over a recomputed one."""
    frames = [f for f in (backfill, portfolio_learning.load_history(logs)) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=SIGNALS)
    history = pd.concat(frames, ignore_index=True)
    history["date"] = history["date"].astype(str).str[:10]
    history = history[history["date"] < as_of.isoformat()]
    return history.drop_duplicates(["ticker", "date", "agent"], keep="last").reset_index(drop=True)


def grade_agents(history: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Every past score with what its ETF did next: agent + the columns of macro_tape.GRADES."""
    frames = []
    for agent, rows in history.groupby("agent"):
        calls = rows.rename(columns={"score": "direction"})[["date", "ticker", "direction"]]
        frames.append(macro_tape.grade_calls(calls, table, pd.Series(dtype=float)).assign(agent=agent))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["agent", *macro_tape.GRADES])


def _share(frame: pd.DataFrame) -> str:
    """'55% (n=120)' of the judged calls of `frame` that were right, or 'too few calls'."""
    judged = frame[frame["hit"].notna()]
    if len(judged) < MIN_CALLS:
        return f"too few calls (n={len(judged)})"
    return f"{judged['hit'].astype(bool).mean():.0%} (n={len(judged)})"


def _ranked(table: pd.DataFrame, most: int = 8) -> str:
    """'A 6/6, B 5/7, ... , Z 0/5': the rows of a table sorted best first (columns right, calls). When there
    are more than `most`, the best and the worst halves are kept."""
    rows = [f"{name} {int(r.right)}/{int(r.calls)}" for name, r in table.iterrows()]
    if len(rows) > most:
        rows = rows[:most // 2] + ["..."] + rows[-most // 2:]
    return ", ".join(rows)


def _day(iso: str) -> str:
    return f"{date.fromisoformat(str(iso)[:10]):%d %b}"


def write_review(as_of: date, stats: pd.DataFrame, grades: pd.DataFrame, analysts: pd.DataFrame,
                 sources: dict, trust: float, track: dict, risk_text: str, backfill_days: int) -> list[str]:
    """The review of the day as Markdown lines: facts with their numbers, the most useful first."""
    lines = [f"## Review of {as_of.isoformat()}", "",
             f"Every score is judged on the {settings.HORIZON_SESSIONS} sessions that followed it."
             + (f" The technical rule score is also recomputed for the last {backfill_days} quote days." if backfill_days else "")]

    lines += ["", "**Agent weights** (starting weight -> weight the evidence supports)", ""]
    for r in stats.itertuples():
        proof = ("no evidence yet" if not r.days else f"{r.days} days judged, rank correlation {r.ic:+.2f}"
                 + ("" if pd.isna(r.hit_rate) else f", hit rate {r.hit_rate:.0%}"))
        lines.append(f"- {r.agent}: {r.prior:.0%} -> {r.weight:.0%} ({proof})")
    if stats.empty:
        lines.append("- No score can be judged yet: every agent keeps its starting weight.")

    calls = grades[grades["hit"].notna()]
    if len(calls):
        lines += ["", f"**Where each agent is right** (calls with a score of at least {macro_tape.MIN_CALL})", ""]
        for agent, mine in calls.groupby("agent"):
            lines.append(f"- {agent}: calls up right {_share(mine[mine['direction'] > 0])}, "
                         f"calls down right {_share(mine[mine['direction'] < 0])}")
        last = calls["date"].max()
        worst = calls[(calls["date"] == last) & ~calls["hit"].astype(bool)]
        worst = worst.reindex((worst["direction"].abs() * worst["ret"].abs()).sort_values(ascending=False).index).head(MISSES)
        if len(worst):
            lines += ["", f"**Largest misses of the last judged day** ({_day(last)})", ""]
            lines += [f"- {m.agent} {m.direction:+.2f} on {m.ticker}: {m.ret:+.1%} over {m.sessions} "
                      f"session{'s' if m.sessions > 1 else ''}" for m in worst.itertuples()]

    judged = analysts[analysts["hit"].notna()] if len(analysts) else analysts
    if len(judged):
        themes = load_universe().set_index("ticker")["theme"]
        by_theme = judged.assign(theme=judged["ticker"].map(themes)).groupby("theme")["hit"].agg(
            right=lambda h: int(h.astype(bool).sum()), calls="size")
        by_theme = by_theme[by_theme["calls"] >= MIN_CALLS].assign(rate=lambda t: t["right"] / t["calls"])
        if len(by_theme):
            lines += ["", "**ETF analysts by theme** (calls right / calls made, best first)", "",
                      "- " + _ranked(by_theme.sort_values("rate", ascending=False))]

    views = sources.get("graded", pd.DataFrame())
    if len(views):
        by_factor = views.groupby("factor")["hit"].agg(right="sum", calls="size")
        by_factor = by_factor[by_factor["calls"] >= 3].assign(rate=lambda t: t["right"] / t["calls"])
        if len(by_factor):
            lines += ["", "**Macro views by factor** (views right / views judged, best first)", "",
                      "- " + _ranked(by_factor.sort_values("rate", ascending=False))]
    notes = [f"dropped (on trial, wrong too often): {', '.join(sources['dropped'])}" if sources.get("dropped") else "",
             f"weak regular sources, for a person to decide: {', '.join(sources['weak'])}" if sources.get("weak") else "",
             f"on trial and right often enough to become regular: {', '.join(sources['propose_active'])}"
             if sources.get("propose_active") else ""]
    if any(notes):
        lines += ["", "**Macro sources**", ""] + [f"- {n[0].upper()}{n[1:]}" for n in notes if n]

    if track.get("days"):
        lines += ["", f"**Portfolio Manager**: over {track['days']} day(s) its LLM portfolio returned {track['llm']:+.2%} "
                      f"on average against {track[track['anchor']]:+.2%} for the code's {track['anchor']} portfolio, "
                      f"so the LLM counts for {trust:.0%} of the final blend."]
    if risk_text:
        lines += ["", f"**Risk Manager**: {risk_text}"]
    return lines


def save_journal(logs: Path, as_of: date, review_lines: list[str]) -> str:
    """Put the review first in <logs>/lessons.md (replacing one of the same date) and keep the REVIEWS_KEPT newest.
    Returns the whole journal."""
    path = logs / JOURNAL
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    sections = ["## Review of " + part for part in old.split("## Review of ")[1:]]
    title = review_lines[0]
    sections = [s for s in sections if not s.startswith(title)]
    journal = HEADER + "\n" + "\n".join(["\n".join(review_lines) + "\n", *[s.rstrip("\n") + "\n" for s in sections]][:REVIEWS_KEPT])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(journal, encoding="utf-8", newline="\n")
    return journal


def save_weights(logs: Path, as_of: date, stats: pd.DataFrame) -> None:
    """One set of weights per review date in <logs>/agent_weights.csv (a review run again replaces its set)."""
    if stats.empty:
        return
    new = validate_agent_weights(pd.DataFrame({"date": as_of.isoformat(), "agent": stats["agent"],
                                               "weight": stats["weight"], "hit_rate": stats["hit_rate"]},
                                              columns=AGENT_WEIGHTS))
    path = logs / "agent_weights.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"].astype(str) != as_of.isoformat()], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def review(as_of: date, prices: pd.DataFrame, logs: Path | str | None = None, features: pd.DataFrame | None = None,
           backfill: bool = True) -> dict:
    """Run the review of `as_of` on <logs> and return what it found:
    weights, stats (one row per agent), lessons (the review as text), journal (the whole lessons.md),
    backfill (the recomputed technical scores, for the Portfolio Manager's own learning), sources.
    `backfill=False` skips the recomputed technical scores (a mock technical agent has no rule to replay)."""
    logs = Path(logs) if logs else settings.LOGS_DIR
    table = macro_tape.closes(prices, as_of)
    past = technical_history(prices, features, as_of) if backfill and len(prices) else pd.DataFrame(columns=SIGNALS)
    history = evidence(logs, as_of, past)
    agents = [a for a in settings.SIGNAL_AGENTS if a in set(history["agent"])]
    history = history[history["agent"].isin(agents)]
    if agents:
        weights, stats = portfolio_learning.learn_weights(history, prices, starting_weights(agents), as_of)
    else:
        weights, stats = {}, pd.DataFrame(columns=["agent", "days", "ic", "hit_rate", "prior", "weight"])
    grades = grade_agents(history, table)
    exposure = macro_news.load_exposure()
    analysts = macro_tape.grade_calls(macro_news.past_etf_views(logs, as_of), table, exposure.set_index("ticker")["multiplier"])
    sources = macro_news.review_sources(logs, table, exposure)
    trust, track = portfolio_learning.learn_trust(portfolio_learning.load_records(logs), prices, as_of)
    risk_text = risk_review.grading_summary(logs, prices, as_of)

    lines = write_review(as_of, stats, grades, analysts, sources, trust, track, risk_text,
                         past["date"].nunique() if len(past) else 0)
    save_weights(logs, as_of, stats)
    journal = save_journal(logs, as_of, lines)
    if len(stats):
        moved = "; ".join(f"{r.agent} {r.prior:.0%} -> {r.weight:.0%}" for r in stats.itertuples())
        logbook.record_learning(logs, as_of, "reassessment", "Agent weights re-assessed", moved, source="review loop",
                                data={r.agent: {"prior": float(r.prior), "weight": float(r.weight), "days": int(r.days)}
                                      for r in stats.itertuples()})
        logbook.record_learning(logs, as_of, "lesson", "Review of the day written to lessons.md",
                                f"{len(lines)} lines; " + moved, source="review loop")
    return {"weights": weights, "stats": stats, "lessons": "\n".join(lines) + "\n", "journal": journal,
            "backfill": past, "sources": sources, "grades": grades}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Review what the agents got right and wrong, and re-weight them.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD, default today")
    parser.add_argument("--logs-dir", type=Path, default=settings.LOGS_DIR)
    args = parser.parse_args(argv)
    from data import data_agent
    prices = data_agent.fetch_prices(args.date - timedelta(days=400), args.date)
    result = review(args.date, prices, args.logs_dir, data_agent.compute_features(prices))
    print(result["lessons"])
    print(f"Written: {args.logs_dir / JOURNAL}, {args.logs_dir / 'agent_weights.csv'}")


if __name__ == "__main__":
    main()
