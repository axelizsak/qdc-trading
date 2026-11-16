"""Performance monitoring and learning: what the system learns from its own past, every morning. Owner: Axel.
(It calls each agent's own grading code: Person 3 owns the grading of the Risk Manager.)

It runs first thing in a day, right after the data step, and on its own:
    python -m graph.review_loop                    # today, on logs/
    python -m graph.review_loop --date 2026-12-10
    python -m graph.review_loop --rebuild          # replay the technical answers since 2022 again (after a rule change)

No LLM: it reports facts with their numbers.

1. Record  the technical agent's answers of every session not recorded yet, with what each ETF then did
           (logs/technical_answers.csv, from LEARNING_START). Computed once, then one session a day.
2. Learn   every weight of the system with one rule (agents/learning.py): after each session, a weight is
           multiplied by e^(LEARNING_SPEED x what following it earned). Three sets: the technical agent's three
           answers (since 2022), the four agents (since their first score), Claude's portfolio against the code's.
           -> logs/weights.csv, the evidence of model re-assessment. Every number is recomputed from the logs.
3. Check   the macro sources: one on trial that is wrong too often is dropped -> logs/source_registry.csv.
4. Write   logs/lessons.md, newest review first. Macro/News, technical_llm and the Portfolio Manager read it.
5. Record  the re-assessment and the lessons in the logbook (the audit trail).

To change how the system learns: LEARNING_SPEED, WEIGHT_MIN, WEIGHT_MAX and LEARNING_START in data/config.py; the
starting weights in AGENT_FAMILIES (data/config.py), SIGNAL_START (agents/strategy_config.py) and TRUST_START
(agents/portfolio_config.py).
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents import learning, logbook, macro_news, macro_tape, portfolio_learning, risk_review, technical
from agents import portfolio_config as pm_cfg
from data import config as settings
from data.schemas import SIGNALS, TECHNICAL_ANSWERS, WEIGHTS, validate_technical_answers, validate_weights
from data.universe import load_universe

JOURNAL = "lessons.md"
ANSWERS = "technical_answers.csv"
WEIGHTS_FILE = "weights.csv"
HEADER = ("# Lessons journal\n\nWritten by the review loop (graph/review_loop.py) from the logs and the prices: "
          "facts with their numbers, no opinion. Newest review first. The agents read it before they decide.\n")
REVIEWS_KEPT = 5  # older reviews stay in git and in the logbook
MIN_CALLS = 5  # calls needed before a hit rate is quoted
MISSES = 3
PORTFOLIOS = {"llm": "llm", pm_cfg.ANCHOR: "code"}  # pm_portfolios.csv source -> expert


# --- 1. The technical answers since LEARNING_START ---------------------------------------------------------------

def load_answers(path: Path) -> pd.DataFrame:
    """The recorded answers ('n/a' is an answer, not a missing value: only empty cells are)."""
    if not path.exists():
        return pd.DataFrame(columns=TECHNICAL_ANSWERS)
    return validate_technical_answers(pd.read_csv(path, keep_default_na=False, na_values=[""]))


def update_answers(logs: Path, prices: pd.DataFrame, features: pd.DataFrame, as_of: date, fetch=None) -> pd.DataFrame:
    """<logs>/technical_answers.csv brought up to the session before `as_of`. Only sessions after the last one
    recorded are computed. When nothing is recorded yet, `fetch(start, end) -> (prices, features)` brings the prices
    since LEARNING_START (the day's prices cover about a year); without it, the sessions the prices allow."""
    path = logs / ANSWERS
    stored = load_answers(path)
    last = str(stored["date"].max()) if len(stored) else None
    if last is None and fetch is not None:
        prices, features = fetch(settings.LEARNING_START - timedelta(days=technical.PRICE_DAYS + 30), as_of)
    quote_days = sorted(set(pd.to_datetime(prices["date"]).dt.date))
    days = [d for d in quote_days if settings.LEARNING_START <= d < as_of and (last is None or d.isoformat() > last)]
    if not days:
        return stored
    new = learning.technical_answers(prices, features, days, as_of)
    out = validate_technical_answers(pd.concat([stored, new], ignore_index=True) if len(stored) else new)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# --- 2. The three sets of weights -----------------------------------------------------------------------------------

def technical_weights(answers: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, dict[str, float]]:
    """The technical agent's three answers, learned session by session since LEARNING_START."""
    results = learning.answer_results(answers) if len(answers) else pd.DataFrame(columns=learning.RESULTS)
    return learning.daily_weights(results, learning.shares(technical.starting_weights()), as_of,
                                  known=settings.TECHNICAL_SIGNALS)


def agent_calls(logs: Path, answers: pd.DataFrame, technical_frame: pd.DataFrame) -> pd.DataFrame:
    """Every score the four agents gave (Signals format), from their first logged score. A technical_det score
    missing from the logs (a day the pipeline did not run) is replayed from the recorded answers with that day's
    technical weights: what the agent would have said that morning."""
    logged = portfolio_learning.load_history(logs)
    logged = logged[logged["agent"].isin(settings.SIGNAL_AGENTS)].assign(date=lambda s: s["date"].astype(str).str[:10])
    if logged.empty:
        return pd.DataFrame(columns=SIGNALS)
    days = answers[answers["date"].astype(str) >= logged["date"].min()]
    days = days[~days["date"].astype(str).isin(set(logged.loc[logged["agent"] == technical.AGENT_DET, "date"]))]
    replayed = []
    for day, rows in days.groupby(days["date"].astype(str)):
        weights = learning.weights_before(technical_frame, date.fromisoformat(day))
        replayed.append(pd.DataFrame({"ticker": rows["ticker"], "date": day, "agent": technical.AGENT_DET,
                                      "score": technical.blend(technical.signal_scores(rows), weights).values,
                                      "confidence": (rows["confirmations"] / 2).values, "rationale": "replayed"}))
    return pd.concat([logged[SIGNALS], *replayed], ignore_index=True)


def agent_weights(calls: pd.DataFrame, prices: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, dict[str, float]]:
    """The four agents, learned session by session from their first score."""
    results = (learning.call_results(calls.rename(columns={"agent": "name"}), prices, as_of) if len(calls)
               else pd.DataFrame(columns=learning.RESULTS))
    return learning.daily_weights(results, learning.starting_weights, as_of, known=sorted(set(calls["agent"])))


def portfolio_weights(logs: Path, prices: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, dict[str, float]]:
    """Claude's portfolio against the code's (ANCHOR), learned session by session from the first one logged."""
    records = portfolio_learning.load_records(logs)
    records = records[records["source"].isin(PORTFOLIOS)].assign(name=lambda r: r["source"].map(PORTFOLIOS))
    results = (learning.portfolio_results(records, prices, as_of) if len(records)
               else pd.DataFrame(columns=learning.RESULTS))
    start = {"llm": pm_cfg.TRUST_START, "code": 1 - pm_cfg.TRUST_START}
    return learning.daily_weights(results, learning.shares(start), as_of, known=list(start))


# --- 3 and 4. The lessons -----------------------------------------------------------------------------------------

def grade_agents(history: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Every past score with what its ETF did over the next HORIZON_SESSIONS: agent + the columns of macro_tape.GRADES."""
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


WEIGHT_TITLES = {"technical": "The technical agent's three answers", "agents": "The four agents",
                 "portfolio": "Claude's portfolio (llm) against the code's (code)"}


def weight_lines(board: pd.DataFrame) -> list[str]:
    """The weights, level by level: start -> today, with the record behind each one."""
    lines = ["**Weights** (each session, a weight is multiplied by e^(" f"{settings.LEARNING_SPEED:g}"
             " x what following it earned in that session); start -> today", ""]
    for level, rows in board.groupby("level", sort=False):
        parts = []
        for r in rows.itertuples():
            record = ("no session judged yet" if not r.sessions else
                      f"{r.mean_result:+.3%} a session over {int(r.sessions)}, right {r.right_share:.0%}")
            parts.append(f"{r.name} {r.start:.0%} -> {r.weight:.0%} ({record})")
        lines.append(f"- {WEIGHT_TITLES[level]}: " + "; ".join(parts))
    return lines


def write_review(as_of: date, board: pd.DataFrame, grades: pd.DataFrame, analysts: pd.DataFrame,
                 sources: dict, risk_text: str) -> list[str]:
    """The review of the day as Markdown lines: facts with their numbers, the most useful first."""
    lines = [f"## Review of {as_of.isoformat()}", ""] + weight_lines(board)
    calls = grades[grades["hit"].notna()]
    if len(calls):
        lines += ["", f"**Where each agent is right** (each score judged on the {settings.HORIZON_SESSIONS} sessions "
                      f"that followed it; calls with a score of at least {macro_tape.MIN_CALL})", ""]
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


# --- The review ---------------------------------------------------------------------------------------------------

def review(as_of: date, prices: pd.DataFrame, logs: Path | str | None = None, features: pd.DataFrame | None = None,
           replay_technical: bool = True, fetch=None) -> dict:
    """Run the review of `as_of` on <logs> and return what the day needs:
    technical_weights (today's weights of the three answers), agent_multipliers (learning.weights_now turns them into
    today's weights over the agents that score), trust (how much Claude's portfolio counts), board (the scoreboard:
    level, name, start, weight, sessions, mean_result, right_share), weights (logs/weights.csv), lessons (the review
    as text), journal (the whole lessons.md), sources, grades.
    `replay_technical=False` skips the technical answers (a mock technical agent has no rules to replay)."""
    logs = Path(logs) if logs else settings.LOGS_DIR
    table = macro_tape.closes(prices, as_of)
    answers = (update_answers(logs, prices, features, as_of, fetch) if replay_technical and len(prices)
               else pd.DataFrame(columns=TECHNICAL_ANSWERS))
    tech_frame, _ = technical_weights(answers, as_of)
    calls = agent_calls(logs, answers, tech_frame)
    agent_frame, multipliers = agent_weights(calls, prices, as_of)
    trust_frame, _ = portfolio_weights(logs, prices, as_of)

    starts = {"technical": technical.starting_weights(), "agents": learning.starting_weights(sorted(set(calls["agent"]))),
              "portfolio": {"llm": pm_cfg.TRUST_START, "code": 1 - pm_cfg.TRUST_START}}
    frames = {"technical": tech_frame, "agents": agent_frame, "portfolio": trust_frame}
    weights = validate_weights(pd.concat([f.assign(level=level) for level, f in frames.items() if len(f)],
                                         ignore_index=True).reindex(columns=WEIGHTS))
    board = pd.concat([learning.scoreboard(f, starts[level]).assign(level=level) for level, f in frames.items() if len(f)],
                      ignore_index=True)
    logs.mkdir(parents=True, exist_ok=True)
    weights.to_csv(logs / WEIGHTS_FILE, index=False)

    grades = grade_agents(calls, table)
    exposure = macro_news.load_exposure()
    analysts = macro_tape.grade_calls(macro_news.past_etf_views(logs, as_of), table, exposure.set_index("ticker")["multiplier"])
    sources = macro_news.review_sources(logs, table, exposure)
    risk_text = risk_review.grading_summary(logs, prices, as_of)
    lines = write_review(as_of, board, grades, analysts, sources, risk_text)
    journal = save_journal(logs, as_of, lines)

    today = weights[weights["date"] == as_of.isoformat()]
    now = {level: dict(zip(rows["name"], rows["weight"].astype(float))) for level, rows in today.groupby("level")}
    moved = "; ".join(f"{r.level} {r.name} {r.start:.0%} -> {r.weight:.0%}" for r in board.itertuples())
    logbook.record_learning(logs, as_of, "reassessment", "Weights re-assessed", moved, source="review loop",
                            data={f"{r.level}/{r.name}": {"start": float(r.start), "weight": float(r.weight),
                                                          "sessions": int(r.sessions)} for r in board.itertuples()})
    logbook.record_learning(logs, as_of, "lesson", "Review of the day written to lessons.md",
                            f"{len(lines)} lines; " + moved, source="review loop")
    return {"technical_weights": now.get("technical") or technical.starting_weights(),
            "agent_multipliers": multipliers, "trust": now.get("portfolio", {}).get("llm", pm_cfg.TRUST_START),
            "board": board, "weights": weights, "lessons": "\n".join(lines) + "\n", "journal": journal,
            "sources": sources, "grades": grades}


def long_history(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real prices and indicators from `start` to `end` (data agent): for the first replay of the technical answers."""
    from data import data_agent
    prices = data_agent.fetch_prices(start, end)
    return prices, data_agent.compute_features(prices)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Review what the agents got right and wrong, and re-weight them.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD, default today")
    parser.add_argument("--logs-dir", type=Path, default=settings.LOGS_DIR)
    parser.add_argument("--rebuild", action="store_true", help="replay the technical answers since LEARNING_START again")
    args = parser.parse_args(argv)
    if args.rebuild:
        (args.logs_dir / ANSWERS).unlink(missing_ok=True)
    prices, features = long_history(args.date - timedelta(days=400), args.date)
    result = review(args.date, prices, args.logs_dir, features, fetch=long_history)
    print(result["lessons"])
    print(f"Written: {args.logs_dir / JOURNAL}, {args.logs_dir / WEIGHTS_FILE}, {args.logs_dir / ANSWERS}")


if __name__ == "__main__":
    main()
