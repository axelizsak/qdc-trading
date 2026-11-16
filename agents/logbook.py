"""The Logbook: the audit trail of every day. Owner: Person 3. No LLM.

Every agent and every human step already writes its own file in <logs>. The Logbook joins them into one append-only
ledger, <logs>/logbook.csv (LOGBOOK format), with one row per event, tells each day as a page, <logs>/logbook/<date>.md,
and keeps an overview of the whole competition in <logs>/logbook/INDEX.md.

Events of a day, in the order the page tells them
  signals            one row per signal agent: how many ETFs, the strongest and weakest scores              (signals.csv)
  macro_report       the Macro/News report of the day, with the cost line and a fingerprint                 (macro/<date>.md)
  agent_weights      the agent weights learned for the day and their hit rates                              (agent_weights.csv)
  target_allocation  the Portfolio Manager's target weights and memo, one per round                         (record_round)
  recommendation     every order the Portfolio Manager proposed, round by round, with its reasons           (recommendations.csv)
  risk_report        the Risk Manager's count of ok, warning and breach, and one risk_flag per warning;
                     one set per round when recorded at the gates, else the last round                      (record_round, risk_reports.csv)
  human_review       every human answer at the three gates, with the feedback                               (reviews.csv)
  order_decision     each proposed order next to the human decision and the final order                     (orders.csv)
  human_vs_ai        the day in one line: orders kept, changed, rejected, added, sent back, value traded    (orders.csv, reviews.csv)
  fill               each executed order at its fill price (code calculates it, see below)                  (orders.csv + prices)
  valuation          holdings, cash, value and return since the start, against the benchmark                (orders.csv + prices)
  summary            the approved executive summary, as a fingerprint                                       (summary/<date>.md)
  lesson             a lesson the learning agent wrote for the agents to read                               (record_learning)
  reassessment       a change the learning agent made (a weight, a trust level, a setting) and why          (record_learning)
  incident           every step that failed and was replaced by its fallback                                (incidents.csv)

Three ways in
  record_round(...)     at each gate, with what the pipeline holds in memory: the PM's target and memo and the risk report of
                        that round. The files only keep the last round, so this is the only way to archive the earlier ones.
  record_day(...)       after the order log: everything the files tell about the day. Events from the gates stay as they are.
  record_learning(...)  the learning agent's lessons and re-assessments.

Rules of the ledger
  - Append only. A row is never changed. When a day is run again and an event is no longer true, a `superseded` row
    that names it is appended, and the new event follows. Nothing is deleted, so the history of corrections is on record.
  - Hash chain. Each row holds the hash of the row before it, so changing an old row breaks every hash after it.
    `verify` finds the first broken row. The page and the index print the hash of the last row: commit them and git pins it.
  - Fills are computed, not read from the broker: the last close before the order day on the ETF's own exchange (the
    Belfius rule), from the price table passed in. Sells and buys of a day both count. Days before the first trading
    day are practice and form a separate book, as in graph/pipeline.py.
  - The `data` column holds each event's numbers as JSON, so the index is built from the ledger alone.
Usage:
    python -m agents.logbook --date 2026-12-07            # record that day (needs prices: fetched like the pipeline does)
    python -m agents.logbook --verify                     # check the hash chain
Called from graph/pipeline.py (see the handoff): record_round after each of the PM and risk steps,
record_day after the order log.
"""
import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as settings
from data.schemas import LOGBOOK, validate_logbook

LEDGER = "logbook.csv"
GENESIS = "0" * 64
HASHED = [c for c in LOGBOOK if c != "hash"]
SECTIONS = [("Signals", ["signals"]), ("Macro/News", ["macro_report"]), ("Agent weights", ["agent_weights"]),
            ("Portfolio Manager", ["target_allocation", "recommendation"]),
            ("Risk Manager", ["risk_report", "risk_flag"]), ("Human reviews", ["human_review"]),
            ("Orders: AI and human", ["order_decision", "human_vs_ai"]), ("Fills", ["fill"]),
            ("Portfolio", ["valuation"]), ("Executive summary", ["summary"]),
            ("Learning agent", ["lesson", "reassessment"]), ("Incidents", ["incident"])]
PAGE_ORDER = [kind for _, kinds in SECTIONS for kind in kinds]
SECTION_OF = {kind: i for i, (_, kinds) in enumerate(SECTIONS) for kind in kinds}
LEARNING_KINDS = {"lesson", "reassessment"}
STATE_KINDS = {"target_allocation", "risk_report", "risk_flag"}  # recorded at the gates
STATE, LEARNING = "state:", "learning:"  # source prefixes of events the end-of-day pass must leave alone
GATES = {"orders": 0, "risk": 1, "summary": 2}
TOP = 3  # strongest and weakest signals named per agent
POSITIONS = 8  # positions named in a target allocation
MEMO_CHARS = 2000


# --- Small helpers ---------------------------------------------------------------------------------------------

def _text(value) -> str:
    """A cell as clean one-line text: empty for missing values, no line breaks, whole numbers without .0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return " ".join(str(value).split())


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8") if path.exists() else pd.DataFrame()


def _on(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame["date"].astype(str).str[:10] == day]


def _num(value, fmt: str) -> str:
    return "" if value is None or pd.isna(value) else format(float(value), fmt)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _json(data: dict | None) -> str:
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in (data or {}).items()}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), default=float) if clean else ""


def _data(row) -> dict:
    return json.loads(row["data"]) if row.get("data") else {}


def _event(day: str, kind: str, actor: str, title: str, detail: str = "", source: str = "", ticker="",
           round_="", gate="", data: dict | None = None) -> dict:
    row = {"date": day, "kind": kind, "actor": actor, "ticker": _text(ticker), "round": _text(round_),
           "gate": _text(gate), "title": _text(title), "detail": _text(detail), "data": _json(data), "source": source}
    key = json.dumps([row[k] for k in ("date", "kind", "actor", "ticker", "round", "gate", "title", "detail", "source")])
    row["event_id"] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return row


# --- The day's events, one function per source -------------------------------------------------------------------

def signal_events(logs: Path, day: str) -> list[dict]:
    rows = _on(_read(logs / "signals.csv"), day)
    out = []
    order = [a for a in settings.SIGNAL_AGENTS if a in set(rows.get("agent", []))]
    order += sorted(set(rows.get("agent", [])) - set(order))
    for agent in order:
        part = rows[rows["agent"] == agent].sort_values("score", ascending=False)
        pick = lambda frame: ", ".join(f"{r.ticker} {r.score:+.2f}" for r in frame.itertuples())  # noqa: E731
        detail = (f"mean score {part['score'].mean():+.2f}, mean confidence {part['confidence'].mean():.2f}; "
                  f"strongest: {pick(part.head(TOP))}; weakest: {pick(part.tail(TOP).iloc[::-1])}")
        out.append(_event(day, "signals", agent, f"{agent}: {len(part)} ETFs scored", detail, "signals.csv"))
    return out


def macro_events(logs: Path, day: str) -> list[dict]:
    report = logs / "macro" / f"{day}.md"
    if not report.exists():
        return []
    cost = next((line.strip() for line in report.read_text(encoding="utf-8").splitlines()
                 if line.startswith("Cost of this run")), "")
    views = len(_on(_read(logs / "macro_views.csv"), day))
    detail = f"{views} macro views; {cost}; fingerprint {_digest(report)}".replace("; ;", ";")
    return [_event(day, "macro_report", "macro_news", "Macro/News report written", detail, f"macro/{day}.md")]


def weight_events(logs: Path, day: str) -> list[dict]:
    rows = _on(_read(logs / "agent_weights.csv"), day)
    if rows.empty:
        return []
    parts = [f"{r.agent} {r.weight:.0%}" + (f" (hit rate {r.hit_rate:.0%})" if pd.notna(r.hit_rate) else "")
             for r in rows.itertuples()]
    return [_event(day, "agent_weights", "portfolio_manager", "Agent weights for the day", "; ".join(parts),
                   "agent_weights.csv")]


def recommendation_events(logs: Path, day: str) -> list[dict]:
    rows = _on(_read(logs / "recommendations.csv"), day)
    if rows.empty:
        return []
    ids = list(dict.fromkeys(rows.sort_values("created_at", kind="stable")["recommendation_id"]))
    out = []
    for row in rows.itertuples():
        round_ = ids.index(row.recommendation_id) + 1
        if row.side == "NONE":
            out.append(_event(day, "recommendation", "portfolio_manager", "No order proposed",
                              f"recommendation {row.recommendation_id}", "recommendations.csv", round_=round_))
            continue
        detail = (f"at {_num(row.price_eur, '.2f')} EUR (close of {_text(row.price_date)}), value "
                  f"{_num(row.value_eur, ',.0f')} EUR; weight {_num(row.weight_now, '.1%')} to {_num(row.weight_target, '.1%')}; "
                  f"impact {_num(row.impact, '+.4f')}; timing {_num(row.timing, '+.2f')}; why: {_text(row.why)}")
        out.append(_event(day, "recommendation", "portfolio_manager",
                          f"{row.side} {_text(row.units)} {row.ticker}", detail, "recommendations.csv",
                          ticker=row.ticker, round_=round_))
    return out


def risk_events(rows: pd.DataFrame, day: str, round_="", source: str = "risk_reports.csv") -> list[dict]:
    """The risk report as events: its count of ok, warning and breach, and one event per metric that is not ok."""
    if rows is None or rows.empty:
        return []
    counts = rows["status"].value_counts()
    title = f"Risk report: {counts.get('ok', 0)} ok, {counts.get('warning', 0)} warning, {counts.get('breach', 0)} breach"
    data = {"ok": int(counts.get("ok", 0)), "warning": int(counts.get("warning", 0)), "breach": int(counts.get("breach", 0))}
    out = [_event(day, "risk_report", "risk_manager", title, f"{len(rows)} metrics", source, round_=round_, data=data)]
    for row in rows[rows["status"] != "ok"].itertuples():
        limit = f" against {_num(row.limit, 'g')}" if pd.notna(row.limit) else ""
        out.append(_event(day, "risk_flag", "risk_manager", f"{str(row.status).upper()}: {row.metric}",
                          f"value {_num(row.value, 'g')}{limit}; {_text(row.comment)}", source, round_=round_,
                          data={"status": str(row.status)}))
    return out


def review_events(logs: Path, day: str) -> list[dict]:
    rows = _on(_read(logs / "reviews.csv"), day)
    rows = rows.assign(_gate=rows["gate"].map(GATES)).sort_values(["round", "_gate"], kind="stable") if len(rows) else rows
    return [_event(day, "human_review", "human", f"{r.gate} review: {r.decision}", _text(r.feedback), "reviews.csv",
                   round_=r.round, gate=r.gate) for r in rows.itertuples()]


def order_events(logs: Path, day: str) -> list[dict]:
    rows = _on(_read(logs / "orders.csv"), day)
    out = []
    for r in rows.itertuples():
        ai = f"AI proposed {r.ai_side} {_text(r.ai_units)}" if pd.notna(r.ai_side) else "no AI order"
        final = f"{r.side} {_text(r.units)}" if pd.notna(r.side) else "no order"
        reason = f"; reason: {_text(r.reason)}" if _text(r.reason) else ""
        out.append(_event(day, "order_decision", "human", f"{r.ticker}: {r.decision}", f"{ai}; final: {final}{reason}",
                          "orders.csv", ticker=r.ticker))
    return out


def incident_events(logs: Path, day: str) -> list[dict]:
    return [_event(day, "incident", "pipeline", f"{r.step} failed", _text(r.error), "incidents.csv")
            for r in _on(_read(logs / "incidents.csv"), day).itertuples()]


def summary_events(logs: Path, day: str) -> list[dict]:
    path = logs / "summary" / f"{day}.md"
    if not path.exists():
        return []
    n = len(path.read_text(encoding="utf-8"))
    return [_event(day, "summary", "human", "Executive summary approved", f"{n} characters; fingerprint {_digest(path)}",
                   f"summary/{day}.md")]


# --- The book: fills and value, computed by code ----------------------------------------------------------------------

def _closes(prices: pd.DataFrame | None) -> pd.DataFrame:
    if prices is None or not len(prices):
        return pd.DataFrame()
    day = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    return prices.assign(_day=day).pivot(index="_day", columns="ticker", values="close_eur").sort_index().ffill()


def _book_name(day: str) -> str:
    return "competition" if day >= settings.TRADING_DAYS[0].isoformat() else "practice"


def _book_rows(logs: Path, day: str) -> pd.DataFrame:
    """Executed orders of the book `day` belongs to, up to and including `day`."""
    log = _read(logs / "orders.csv")
    if log.empty:
        return log
    start = settings.TRADING_DAYS[0].isoformat()
    dates = log["date"].astype(str).str[:10]
    same = (dates >= start) if day >= start else (dates < start)
    done = log[same & (dates <= day) & (log["decision"] != "rejected") & log["units"].notna()].copy()
    done["date"] = done["date"].astype(str).str[:10]
    return done


def fill_price(closes: pd.DataFrame, ticker: str, day: str) -> tuple[float, str] | None:
    """(price, date of that price): the last close before `day`, or None when there is none."""
    if ticker not in closes.columns:
        return None
    before = closes.loc[settings.fills_from(closes.index, day), ticker].dropna()
    return (float(before.iloc[-1]), str(before.index[-1])) if len(before) else None


def book_events(logs: Path, day: str, closes: pd.DataFrame) -> list[dict]:
    done = _book_rows(logs, day)
    cash, units, first_fill, out = float(settings.START_CAPITAL_EUR), {}, None, []
    for r in done.itertuples():
        sign = 1.0 if r.side == "BUY" else -1.0
        found = fill_price(closes, r.ticker, r.date)
        if found is None:
            if r.date == day:
                out.append(_event(day, "fill", "code", f"{r.side} {_text(r.units)} {r.ticker}: no fill price",
                                  "no close before the order day in the price table; not valued", "orders.csv",
                                  ticker=r.ticker))
            continue
        price, price_date = found
        first_fill = price_date if first_fill is None else min(first_fill, price_date)
        cash -= sign * float(r.units) * price
        units[r.ticker] = units.get(r.ticker, 0.0) + sign * float(r.units)
        if r.date == day:
            out.append(_event(day, "fill", "code", f"{r.side} {_text(r.units)} {r.ticker} @ {price:.2f}",
                              f"computed: close of {price_date} ({settings.fill_rule()}), value "
                              f"{float(r.units) * price:,.2f} EUR", "orders.csv", ticker=r.ticker,
                              data={"price": price, "units": float(r.units), "side": r.side}))
    on_day = closes.loc[closes.index <= day] if len(closes) else closes
    held, missing, value = [], [], 0.0
    for ticker, n in sorted(units.items()):
        if n <= 1e-9:
            continue
        last = on_day[ticker].dropna() if ticker in on_day.columns else pd.Series(dtype=float)
        if last.empty:
            missing.append(ticker)
            continue
        value += n * float(last.iloc[-1])
        held.append(f"{ticker} {n:g} ({n * float(last.iloc[-1]):,.0f} EUR)")
    total = cash + value
    start = float(settings.START_CAPITAL_EUR)
    parts = [f"cash {cash:,.2f} EUR ({cash / total:.0%})" if total > 0 else f"cash {cash:,.2f} EUR",
             "holdings: " + (", ".join(held) if held else "none")]
    if missing:
        parts.append("not valued (no price): " + ", ".join(missing))
    bench, bench_ret = settings.BENCHMARK, None
    if first_fill and bench in on_day.columns and on_day[bench].dropna().size:
        base = closes.loc[first_fill, bench] if first_fill in closes.index else np.nan
        if pd.notna(base):
            bench_ret = float(on_day[bench].dropna().iloc[-1]) / float(base) - 1
            parts.append(f"{bench} {bench_ret:+.2%} since {first_fill}")
    when = f"the close of {on_day.index[-1]}" if len(on_day) else "no price table"
    data = {"value": total, "cash": cash, "ret": total / start - 1, "bench_ret": bench_ret, "book": _book_name(day)}
    out.append(_event(day, "valuation", "code", f"Portfolio value {total:,.2f} EUR ({total / start - 1:+.2%} since the start)",
                      "; ".join(parts) + f"; valued at {when}", "orders.csv", data=data))
    return out


def human_events(logs: Path, day: str, closes: pd.DataFrame) -> list[dict]:
    """The day in one line: what the human kept, changed, rejected and added, and how often the day was sent back."""
    rows, reviews = _on(_read(logs / "orders.csv"), day), _on(_read(logs / "reviews.csv"), day)
    if rows.empty and reviews.empty:
        return []
    count = rows["decision"].value_counts() if len(rows) else pd.Series(dtype=int)
    approved, modified, rejected, added = (int(count.get(k, 0)) for k in ("approved", "modified", "rejected", "added"))
    proposed = approved + modified + rejected
    sent_back = int((reviews["decision"] == "revised").sum()) if len(reviews) else 0
    stopped = reviews[reviews["decision"] == "stopped"] if len(reviews) else reviews
    ai_value = done_value = 0.0
    priced = len(closes) > 0
    for r in rows.itertuples():
        for side, units, is_ai in ((r.ai_side, r.ai_units, True), (r.side, r.units, False)):
            if pd.isna(units) or pd.isna(side) or (not is_ai and r.decision == "rejected"):
                continue
            found = fill_price(closes, r.ticker, str(r.date)[:10])
            if found is None:
                priced = False
                continue
            if is_ai:
                ai_value += float(units) * found[0]
            else:
                done_value += float(units) * found[0]
    if len(stopped):
        title = f"Human against AI: no trade today (stopped at the {stopped.iloc[0]['gate']} review)"
    elif proposed:
        title = f"Human against AI: {approved} of {proposed} proposed orders kept as proposed" + (f", {added} added" if added else "")
    else:
        title = "Human against AI: no order was proposed" + (f", {added} added" if added else "")
    parts = [f"approved {approved}, modified {modified}, rejected {rejected}, added {added}",
             f"the day was sent back {sent_back} time(s) to the Portfolio Manager"]
    if priced and (ai_value or done_value):
        parts.append(f"gross value: AI proposed {ai_value:,.0f} EUR, executed {done_value:,.0f} EUR")
    data = {"proposed": proposed, "approved": approved, "modified": modified, "rejected": rejected, "added": added,
            "sent_back": sent_back, "stopped": bool(len(stopped)),
            "ai_value": ai_value if priced else None, "executed_value": done_value if priced else None}
    return [_event(day, "human_vs_ai", "human", title, "; ".join(parts), "orders.csv, reviews.csv", data=data)]


def build_events(logs_dir: Path | str, as_of, prices: pd.DataFrame | None = None, with_risk: bool = True) -> list[dict]:
    """Every event of `as_of` as the source files tell it now, in the order of the page.
    `with_risk` False leaves out the risk report of the files (the gates have recorded every round already)."""
    logs, day = Path(logs_dir), str(as_of)[:10]
    closes = _closes(prices)
    risk = risk_events(_on(_read(logs / "risk_reports.csv"), day), day) if with_risk else []
    events = (signal_events(logs, day) + macro_events(logs, day) + weight_events(logs, day)
              + recommendation_events(logs, day) + risk + review_events(logs, day)
              + order_events(logs, day) + human_events(logs, day, closes) + book_events(logs, day, closes)
              + summary_events(logs, day) + incident_events(logs, day))
    return sorted(events, key=lambda e: (SECTION_OF[e["kind"]], int(e["round"] or 0), PAGE_ORDER.index(e["kind"])))


# --- The ledger ----------------------------------------------------------------------------------------------------

def load(logs_dir: Path | str) -> pd.DataFrame:
    """The whole ledger as text (no type guessing, so the hashes can be recomputed)."""
    path = Path(logs_dir) / LEDGER
    if not path.exists():
        return pd.DataFrame(columns=LOGBOOK, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def _hash(row: dict) -> str:
    return hashlib.sha256(json.dumps([_text(row[c]) for c in HASHED]).encode("utf-8")).hexdigest()


def live_ids(ledger: pd.DataFrame, day: str) -> dict[str, dict]:
    """Event id -> its latest row, for the events of `day` that have not been superseded since."""
    last, struck = {}, {}
    for r in ledger[ledger["date"] == day].to_dict("records"):
        if r["kind"] == "superseded":
            struck[r["detail"]] = int(r["seq"])
        else:
            last[r["event_id"]] = r
    return {i: r for i, r in last.items() if int(r["seq"]) > struck.get(i, 0)}


def _commit(logs: Path, day: str, events: list[dict], strike_if, now: datetime | None) -> dict:
    """Append the events that are not live yet; strike the live ones that are not in `events` and that `strike_if` names."""
    ledger = load(logs)
    current = {e["event_id"] for e in events}
    live = live_ids(ledger, day)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = [{"date": day, "kind": "superseded", "actor": "logbook", "ticker": r["ticker"], "round": r["round"],
            "gate": r["gate"], "title": f"Superseded: {r['title']}", "detail": i, "source": LEDGER,
            "event_id": f"struck-{i}"} for i, r in live.items() if i not in current and strike_if(r)]
    n_struck = len(new)
    new += [e for e in events if e["event_id"] not in live]
    prev = ledger["hash"].iloc[-1] if len(ledger) else GENESIS
    seq = len(ledger)
    for row in new:
        seq += 1
        for column in LOGBOOK:
            row.setdefault(column, "")
        row.update(seq=str(seq), recorded_at=stamp, prev_hash=prev)
        row["hash"] = prev = _hash(row)
    if new:
        path = logs / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(new, columns=LOGBOOK).to_csv(path, index=False, header=not path.exists(), mode="a",
                                                  lineterminator="\n", encoding="utf-8")
    write_page(logs, day)
    write_index(logs)
    return {"added": len(new) - n_struck, "superseded": n_struck}


def _protected(row: dict) -> bool:
    return row["source"].startswith((STATE, LEARNING))


def record_day(logs_dir: Path | str, as_of, prices: pd.DataFrame | None = None, now: datetime | None = None) -> dict:
    """Append what is new for `as_of` from the files, strike what is no longer true, and write the day's page.
    Events recorded at the gates and by the learning agent are left as they are.
    Returns {"added": n, "superseded": n}. Running it twice in a row adds nothing."""
    logs, day = Path(logs_dir), str(as_of)[:10]
    gates_recorded = any(r["kind"] == "risk_report" and r["source"].startswith(STATE)
                         for r in live_ids(load(logs), day).values())
    events = build_events(logs, day, prices, with_risk=not gates_recorded)
    return _commit(logs, day, events, lambda row: not _protected(row), now)


def record_round(logs_dir: Path | str, as_of, round_: int, *, target: pd.DataFrame | None = None, memo: str = "",
                 proposed: pd.DataFrame | None = None, risk_report: pd.DataFrame | None = None,
                 now: datetime | None = None) -> dict:
    """Archive one round of the day as it is, from what the pipeline holds in memory: the Portfolio Manager's target
    (PORTFOLIO format) and memo, and the risk report of that round (RISK_REPORT format). Call it after the Portfolio
    Manager (target, memo, proposed) and again after the Risk Manager (risk_report); each call adds what it is given.
    A round recorded again (a rerun of the day) replaces its earlier events."""
    logs, day, source = Path(logs_dir), str(as_of)[:10], f"{STATE} round {round_}"
    events = []
    if target is not None:
        weights = target.set_index("ticker")["target_weight"].astype(float)
        weights = weights[weights > 0].sort_values(ascending=False)
        named = ", ".join(f"{t} {w:.1%}" for t, w in weights.head(POSITIONS).items())
        more = f" and {len(weights) - POSITIONS} more" if len(weights) > POSITIONS else ""
        orders = f"; {len(proposed)} order(s) proposed" if proposed is not None else ""
        detail = (f"{named or 'no positions'}{more}; cash {1 - weights.sum():.1%}{orders}"
                  + (f"; memo: {_text(memo)[:MEMO_CHARS]}" if _text(memo) else ""))
        events.append(_event(day, "target_allocation", "portfolio_manager", "Target allocation", detail,
                             source, round_=round_, data={"positions": len(weights), "invested": float(weights.sum())}))
    if risk_report is not None:
        events += risk_events(risk_report, day, round_, source)
    kinds = ({"target_allocation"} if target is not None else set()) | ({"risk_report", "risk_flag"} if risk_report is not None else set())
    # a round recorded again replaces itself; a risk report taken from the files (the day was recorded first) gives way
    mine = lambda row: ((row["source"] == source and row["kind"] in kinds)  # noqa: E731
                        or (risk_report is not None and row["source"] == "risk_reports.csv"
                            and row["kind"] in {"risk_report", "risk_flag"}))
    return _commit(logs, day, events, mine, now)


def record_learning(logs_dir: Path | str, as_of, kind: str, title: str, detail: str = "", data: dict | None = None,
                    source: str = "", now: datetime | None = None) -> dict:
    """Archive what the learning agent did on `as_of`: a `lesson` it wrote or a `reassessment` it made (a weight, a trust
    level, a setting: say what it was, what it is now and which evidence moved it). `data` carries the numbers."""
    if kind not in LEARNING_KINDS:
        raise ValueError(f"kind must be one of {sorted(LEARNING_KINDS)}")
    day = str(as_of)[:10]
    label = source.removeprefix(LEARNING).strip()
    event = _event(day, kind, "learning_agent", title, detail, f"{LEARNING} {label}".strip(), data=data)
    return _commit(Path(logs_dir), day, [event], lambda row: False, now)


def verify(logs_dir: Path | str) -> list[str]:
    """Problems found in the ledger (empty list = intact): a wrong sequence, a broken link or an altered row."""
    ledger = load(logs_dir)
    problems, prev = [], GENESIS
    for i, row in enumerate(ledger.to_dict("records"), start=1):
        if row["seq"] != str(i):
            problems.append(f"row {i}: seq is {row['seq']}")
        if row["prev_hash"] != prev:
            problems.append(f"row {i}: the link to the row before is broken")
        if row["hash"] != _hash(row):
            problems.append(f"row {i}: the content was changed")
        prev = row["hash"]
    if len(ledger):
        try:
            validate_logbook(ledger)
        except ValueError as exc:
            problems.append(str(exc))
    return problems


# --- The day as a page, the competition as an index -------------------------------------------------------------------

def _head(ledger: pd.DataFrame) -> str:
    return (f"Ledger: {len(ledger)} rows, last hash `{ledger['hash'].iloc[-1][:16]}`. Commit it: git then pins the chain."
            if len(ledger) else "Ledger: empty.")


def render_day(ledger: pd.DataFrame, day: str) -> str:
    live = sorted(live_ids(ledger, day).values(), key=lambda r: int(r["seq"]))
    lines = [f"# Logbook, {day}", ""]
    if not live:
        return "\n".join(lines + ["Nothing recorded for this day.", ""])
    stamps = sorted(r["recorded_at"] for r in live)
    lines += [f"{len(live)} event{'s' if len(live) != 1 else ''}, recorded {stamps[0]} to {stamps[-1]}. "
              f"Source of truth: `logbook.csv` (rows {live[0]['seq']} to {live[-1]['seq']}).", ""]
    for name, kinds in SECTIONS:
        part = sorted((r for r in live if r["kind"] in kinds),
                      key=lambda r: (int(r["round"] or 0), kinds.index(r["kind"]), int(r["seq"])))
        if not part:
            continue
        lines += [f"## {name}", ""]
        for r in part:
            where = f" [round {r['round']}]" if r["round"] else ""
            detail = f": {r['detail']}" if r["detail"] else ""
            lines.append(f"- **{r['title']}**{where}{detail}")
        lines.append("")
    struck = ledger[(ledger["date"] == day) & (ledger["kind"] == "superseded")]
    if len(struck):
        lines += ["## Corrections", "", f"{len(struck)} earlier event(s) were superseded when the day was run again; "
                  "they stay in `logbook.csv`.", ""]
    return "\n".join(lines + [_head(ledger), ""])


def write_page(logs: Path, day: str) -> Path:
    path = logs / "logbook" / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_day(load(logs), day).encode("utf-8"))  # bytes: the same file on every system
    return path


def _cell(value, fmt: str = "") -> str:
    return "-" if value is None or value == "" else (format(value, fmt) if fmt else str(value))


def render_index(ledger: pd.DataFrame, problems: list[str] | None = None) -> str:
    """One line per day: what the human did with the AI's orders, the risk warnings, and the value against the benchmark."""
    bench = settings.BENCHMARK
    lines = ["# Logbook index", "",
             f"Value is the portfolio at the last close up to the day, in EUR; return is since the start of the book "
             f"(practice and competition are separate books); {bench} is the benchmark over the same period. "
             "Kept = AI orders executed as proposed.", "",
             f"| Day | Book | AI orders | Kept | Changed | Rejected | Added | Sent back | Risk warnings | Value | Return | {bench} | Lessons |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for day in sorted(set(ledger["date"])):
        live = sorted(live_ids(ledger, day).values(), key=lambda r: int(r["seq"]))

        def last(kind):
            rows = [r for r in live if r["kind"] == kind]
            return _data(rows[-1]) if rows else {}
        human, value = last("human_vs_ai"), last("valuation")
        reports = [int(r["round"] or 0) for r in live if r["kind"] == "risk_report"]
        flags = ([r for r in live if r["kind"] == "risk_flag" and int(r["round"] or 0) == max(reports)] if reports else None)
        lessons = sum(1 for r in live if r["kind"] in LEARNING_KINDS)
        ret, mark = value.get("ret"), value.get("bench_ret")
        lines.append("| " + " | ".join([
            day, _cell(value.get("book")), _cell(human.get("proposed")), _cell(human.get("approved")),
            _cell(human.get("modified")), _cell(human.get("rejected")), _cell(human.get("added")),
            _cell(human.get("sent_back")), "-" if flags is None else str(len(flags)),
            _cell(value.get("value"), ",.2f"), "-" if ret is None else f"{ret:+.2%}",
            "-" if mark is None else f"{mark:+.2%}", str(lessons) if lessons else "-"]) + " |")
    status = "intact" if not problems else "BROKEN: " + "; ".join(problems[:3])
    return "\n".join(lines + ["", f"Hash chain: {status}. {_head(ledger)}", ""])


def write_index(logs: Path) -> Path:
    path = logs / "logbook" / "INDEX.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = load(logs)
    path.write_bytes(render_index(ledger, verify(logs)).encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, help="record this day")
    parser.add_argument("--verify", action="store_true", help="check the hash chain")
    parser.add_argument("--logs-dir", type=Path, default=settings.LOGS_DIR)
    args = parser.parse_args(argv)
    if args.verify or not args.date:
        problems = verify(args.logs_dir)
        print("Logbook intact." if not problems else "\n".join(problems))
        return
    from data import data_agent
    prices = data_agent.fetch_prices(args.date - timedelta(days=400), args.date)
    result = record_day(args.logs_dir, args.date, prices)
    print(f"{args.date}: {result['added']} event(s) added, {result['superseded']} superseded. "
          f"Page: {args.logs_dir / 'logbook' / f'{args.date}.md'}")


if __name__ == "__main__":
    main()
