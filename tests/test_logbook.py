"""The Logbook: events from every source, fills and value by code, an append-only hash-chained ledger."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agents import logbook, portfolio_log
from data import mock
from data.config import SIGNAL_AGENTS
from data.schemas import LOGBOOK, ORDER_LOG, RISK_REPORT, validate_logbook

DAY = "2026-12-07"
NOW = datetime(2026, 12, 7, 8, 0, 0, tzinfo=timezone.utc)


def price_table() -> pd.DataFrame:
    """Flat prices, except: WORLD 100 -> 110 on the 7th, GOLD 50 -> 55, DEFENSE 20 -> 18 (the closes of the 4th are the fills)."""
    rows = []
    for day in pd.bdate_range("2026-11-16", "2026-12-08"):
        for ticker, base, on_seventh in (("WORLD", 100.0, 110.0), ("GOLD", 50.0, 55.0), ("DEFENSE", 20.0, 18.0),
                                         ("BITCOIN", 10.0, 10.0)):
            close = on_seventh if day.date().isoformat() >= DAY else base
            rows.append({"date": day.date().isoformat(), "ticker": ticker, "close_eur": close})
    return pd.DataFrame(rows)


def write(logs, name, frame):
    logs.mkdir(parents=True, exist_ok=True)
    frame.to_csv(logs / name, index=False, lineterminator="\n")


@pytest.fixture
def logs(tmp_path):
    folder = tmp_path / "logs"
    write(folder, "signals.csv", mock.mock_signals(DAY))
    write(folder, "weights.csv", mock.mock_weights((DAY, "2026-12-10")))
    write(folder, "reviews.csv", mock.mock_reviews(DAY))
    risk = mock.mock_risk_report(DAY)
    write(folder, "risk_reports.csv", risk[RISK_REPORT])
    write(folder, "orders.csv", mock.mock_order_log(DAY))
    write(folder, "incidents.csv", mock.mock_incidents(DAY))
    columns = portfolio_log.RECOMMENDATIONS
    rows = [
        ["rec-b", "2026-12-07T08:05:00", DAY, "GOLD", "BUY", 8, 50.0, "2026-12-04", 400.0, 0.0, 0.04, 0.0012, 0.1, "trend"],
        ["rec-a", "2026-12-07T08:01:00", DAY, "WORLD", "BUY", 10, 100.0, "2026-12-04", 1000.0, 0.0, 0.10, 0.0030, 0.0, "core"],
        ["rec-b", "2026-12-07T08:05:00", DAY, "WORLD", "BUY", 10, 100.0, "2026-12-04", 1000.0, 0.0, 0.10, 0.0030, 0.0, "core"],
    ]
    write(folder, "recommendations.csv", pd.DataFrame(rows, columns=columns))
    (folder / "macro").mkdir()
    (folder / "macro" / f"{DAY}.md").write_bytes(b"# Macro/News\n\nCost of this run: $3.727 (52 Claude calls)\n")
    (folder / "summary").mkdir()
    (folder / "summary" / f"{DAY}.md").write_bytes(b"Buy the world, a little gold.\n")
    return folder


def kinds(events):
    return [e["kind"] for e in events]


def test_every_source_becomes_events(logs):
    events = logbook.build_events(logs, DAY, price_table())
    found = set(kinds(events))
    assert found == {"signals", "macro_report", "agent_weights", "recommendation", "risk_report", "risk_flag",
                     "human_review", "order_decision", "human_vs_ai", "fill", "valuation", "summary", "incident"}
    assert kinds(events).count("signals") == len(SIGNAL_AGENTS) and kinds(events).count("human_review") == 5
    assert kinds(events).count("order_decision") == 4
    flags = [e for e in events if e["kind"] == "risk_flag"]
    assert len(flags) == 1 and flags[0]["title"] == "WARNING: leverage"


def test_events_follow_the_page_order(logs):
    ranks = [logbook.SECTION_OF[k] for k in kinds(logbook.build_events(logs, DAY, price_table()))]
    assert ranks == sorted(ranks)


def test_recommendation_rounds_follow_the_creation_time(logs):
    events = [e for e in logbook.build_events(logs, DAY, price_table()) if e["kind"] == "recommendation"]
    rounds = {(e["ticker"], e["round"]) for e in events}
    assert rounds == {("WORLD", "1"), ("WORLD", "2"), ("GOLD", "2")}


def test_reviews_are_listed_by_round_then_gate(logs):
    events = [e for e in logbook.build_events(logs, DAY, price_table()) if e["kind"] == "human_review"]
    assert [(e["round"], e["gate"]) for e in events] == [("1", "orders"), ("1", "risk"), ("2", "orders"),
                                                         ("2", "risk"), ("2", "summary")]
    assert "too much in leveraged ETFs" in events[1]["detail"]


def test_fills_are_the_last_close_before_the_order_day(logs):
    events = logbook.build_events(logs, DAY, price_table())
    fills = {e["ticker"]: e["title"] for e in events if e["kind"] == "fill"}
    assert fills == {"WORLD": "BUY 10 WORLD @ 100.00", "GOLD": "BUY 4 GOLD @ 50.00", "DEFENSE": "BUY 5 DEFENSE @ 20.00"}
    assert "close of 2026-12-04" in next(e for e in events if e["ticker"] == "WORLD" and e["kind"] == "fill")["detail"]


def test_valuation_cash_value_and_benchmark(logs):
    value = next(e for e in logbook.build_events(logs, DAY, price_table()) if e["kind"] == "valuation")
    # cash 10000 - 10 x 100 - 4 x 50 - 5 x 20 = 8700; at the close of the 7th 1100 + 220 + 90 = 1410
    assert value["title"] == "Portfolio value 10,110.00 EUR (+1.10% since the start)"
    assert "cash 8,700.00 EUR (86%)" in value["detail"]
    assert "WORLD 10 (1,100 EUR)" in value["detail"] and "GOLD 4 (220 EUR)" in value["detail"]
    assert "WORLD +10.00% since 2026-12-04" in value["detail"]
    assert value["detail"].endswith("valued at the close of 2026-12-07")


def test_a_rejected_order_is_not_filled(logs):
    events = logbook.build_events(logs, DAY, price_table())
    assert not [e for e in events if e["kind"] == "fill" and e["ticker"] == "BITCOIN"]
    rejected = next(e for e in events if e["kind"] == "order_decision" and e["ticker"] == "BITCOIN")
    assert "AI proposed BUY 2; final: no order" in rejected["detail"] and "too volatile" in rejected["detail"]


def test_sells_fund_buys_and_earlier_days_count(logs):
    first = pd.DataFrame([["2026-12-07", "WORLD", "BUY", 10, "approved", "BUY", 10, ""]], columns=ORDER_LOG)
    second = pd.DataFrame([["2026-12-10", "WORLD", "SELL", 4, "approved", "SELL", 4, ""]], columns=ORDER_LOG)
    write(logs, "orders.csv", pd.concat([first, second], ignore_index=True))
    prices = price_table()
    later = prices[prices["ticker"] == "WORLD"].iloc[:1].assign(date="2026-12-09", close_eur=120.0)
    later2 = later.assign(date="2026-12-10", close_eur=130.0)
    prices = pd.concat([prices, later, later2], ignore_index=True)
    events = logbook.build_events(logs, "2026-12-10", prices)
    sell = next(e for e in events if e["kind"] == "fill")
    assert sell["title"] == "SELL 4 WORLD @ 120.00"                    # the close of the 9th, the day before
    value = next(e for e in events if e["kind"] == "valuation")
    # cash 10000 - 10 x 100 + 4 x 120 = 9480; 6 WORLD at 130 = 780
    assert value["title"].startswith("Portfolio value 10,260.00 EUR")


def test_practice_days_form_their_own_book(logs):
    practice = pd.DataFrame([["2026-11-20", "WORLD", "BUY", 5, "approved", "BUY", 5, ""]], columns=ORDER_LOG)
    write(logs, "orders.csv", pd.concat([practice, mock.mock_order_log(DAY)], ignore_index=True))
    competition = next(e for e in logbook.build_events(logs, DAY, price_table()) if e["kind"] == "valuation")
    assert "cash 8,700.00 EUR" in competition["detail"] and "WORLD 10 " in competition["detail"]
    trial = next(e for e in logbook.build_events(logs, "2026-11-20", price_table()) if e["kind"] == "valuation")
    assert trial["title"] == "Portfolio value 10,000.00 EUR (+0.00% since the start)"
    assert "cash 9,500.00 EUR" in trial["detail"]


def test_no_price_is_said_and_never_guessed(logs):
    events = logbook.build_events(logs, DAY, None)
    fills = [e for e in events if e["kind"] == "fill"]
    assert fills and all("no fill price" in e["title"] for e in fills)
    value = next(e for e in events if e["kind"] == "valuation")
    assert value["title"].startswith("Portfolio value 10,000.00 EUR")


def test_an_empty_day_still_gets_a_valuation(tmp_path):
    events = logbook.build_events(tmp_path / "nothing", DAY, price_table())
    assert kinds(events) == ["valuation"]
    assert "cash 10,000.00 EUR" in events[0]["detail"] and "holdings: none" in events[0]["detail"]


def test_recording_twice_adds_nothing(logs):
    first = logbook.record_day(logs, DAY, price_table(), NOW)
    assert first["added"] > 10 and first["superseded"] == 0
    before = (logs / "logbook.csv").read_bytes()
    assert logbook.record_day(logs, DAY, price_table(), NOW) == {"added": 0, "superseded": 0}
    assert (logs / "logbook.csv").read_bytes() == before


def test_ledger_is_valid_clean_text_with_unix_line_endings(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    raw = (logs / "logbook.csv").read_bytes()
    assert b"\r" not in raw and b"nan" not in raw.lower().replace(b"finance", b"")
    ledger = logbook.load(logs)
    assert list(ledger.columns) == LOGBOOK and validate_logbook(ledger) is ledger
    assert ledger["recorded_at"].eq("2026-12-07T08:00:00Z").all()


def test_a_rerun_with_other_orders_strikes_the_old_events(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    kept = mock.mock_order_log(DAY).iloc[:1]                         # only the WORLD order is left
    write(logs, "orders.csv", kept)
    result = logbook.record_day(logs, DAY, price_table(), NOW)
    assert result["superseded"] >= 6 and result["added"] >= 1
    ledger = logbook.load(logs)
    live = logbook.live_ids(ledger, DAY).values()
    titles = [r["title"] for r in live]
    assert "BUY 10 WORLD @ 100.00" in titles and "BUY 4 GOLD @ 50.00" not in titles
    assert (ledger["kind"] == "superseded").sum() == result["superseded"]
    assert "Superseded: BUY 4 GOLD @ 50.00" in set(ledger["title"])    # the old row itself is still there
    assert logbook.verify(logs) == []


def test_an_event_that_comes_back_is_live_again(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    original = (logs / "orders.csv").read_bytes()
    write(logs, "orders.csv", mock.mock_order_log(DAY).iloc[:1])
    logbook.record_day(logs, DAY, price_table(), NOW)
    (logs / "orders.csv").write_bytes(original)
    logbook.record_day(logs, DAY, price_table(), NOW)
    live = [r["title"] for r in logbook.live_ids(logbook.load(logs), DAY).values()]
    assert "BUY 4 GOLD @ 50.00" in live and logbook.verify(logs) == []


def test_the_hash_chain_detects_a_changed_row(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    assert logbook.verify(logs) == []
    path = logs / "logbook.csv"
    path.write_bytes(path.read_bytes().replace(b"BUY 4 GOLD @ 50.00", b"BUY 4 GOLD @ 40.00", 1))
    problems = logbook.verify(logs)
    assert problems and "content was changed" in problems[0]


def test_the_hash_chain_detects_a_deleted_row(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    path = logs / "logbook.csv"
    lines = path.read_bytes().split(b"\n")
    del lines[3]
    path.write_bytes(b"\n".join(lines))
    assert any("seq" in p or "link" in p for p in logbook.verify(logs))


def test_the_next_day_continues_the_chain(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    first_rows = len(logbook.load(logs))
    logbook.record_day(logs, "2026-12-10", price_table(), NOW)
    ledger = logbook.load(logs)
    assert len(ledger) > first_rows and ledger["seq"].tolist() == [str(i) for i in range(1, len(ledger) + 1)]
    assert logbook.verify(logs) == []
    assert ledger.loc[first_rows, "prev_hash"] == ledger.loc[first_rows - 1, "hash"]


def test_the_page_tells_the_day(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    raw = (logs / "logbook" / f"{DAY}.md").read_bytes()
    assert b"\r" not in raw
    page = raw.decode("utf-8")
    assert page.startswith("# Logbook, 2026-12-07")
    for heading in ("## Signals", "## Portfolio Manager", "## Risk Manager", "## Human reviews", "## Fills", "## Portfolio"):
        assert heading in page
    assert page.index("## Portfolio Manager") < page.index("## Risk Manager") < page.index("## Fills")
    assert "- **BUY 10 WORLD @ 100.00**" in page and "too much in leveraged ETFs" in page


def test_the_page_names_corrections_and_hides_struck_events(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    write(logs, "orders.csv", mock.mock_order_log(DAY).iloc[:1])
    logbook.record_day(logs, DAY, price_table(), NOW)
    page = (logs / "logbook" / f"{DAY}.md").read_text(encoding="utf-8")
    assert "BUY 4 GOLD @ 50.00" not in page and "## Corrections" in page


def test_recording_one_day_leaves_the_other_days_alone(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    before = len(logbook.load(logs))
    assert logbook.record_day(logs, "2026-12-14", price_table(), NOW)["superseded"] == 0
    assert (logbook.load(logs)["date"].iloc[:before] == DAY).all()


def test_text_cleaning():
    assert logbook._text(np.nan) == "" and logbook._text(3.0) == "3" and logbook._text(0.25) == "0.25"
    assert logbook._text("two\nlines  here") == "two lines here"
    assert logbook._text(None) == ""


def test_the_mock_is_a_valid_logbook():
    assert list(mock.mock_logbook().columns) == LOGBOOK


def test_validate_logbook_rejects_gaps_and_unknown_kinds():
    good = mock.mock_logbook()
    with pytest.raises(ValueError, match="seq"):
        validate_logbook(good.drop(index=1).reset_index(drop=True))
    with pytest.raises(ValueError, match="unknown kind"):
        validate_logbook(good.assign(kind="gossip"))


# --- Recording at the gates ---------------------------------------------------------------------------------------

def target_table():
    return pd.DataFrame({"ticker": ["WORLD", "GOLD"], "target_weight": [0.10, 0.04]})


def risk_table(status="warning"):
    return pd.DataFrame([[DAY, "volatility", 0.14, 0.20, "ok", "fine"],
                         [DAY, "leverage", 0.12, 0.10, status, "too much leverage"]], columns=RISK_REPORT)


def live_titles(logs, day=DAY):
    return [r["title"] for r in logbook.live_ids(logbook.load(logs), day).values()]


def json_of(event):
    return json.loads(event["data"])


def test_each_round_keeps_its_target_memo_and_risk_report(tmp_path):
    logs = tmp_path / "logs"
    proposed = pd.DataFrame({"date": [DAY], "side": ["BUY"], "ticker": ["WORLD"], "units": [10.0]})
    logbook.record_round(logs, DAY, 1, target=target_table(), memo="Round one memo.\nSecond line.", proposed=proposed, now=NOW)
    logbook.record_round(logs, DAY, 1, risk_report=risk_table(), now=NOW)
    logbook.record_round(logs, DAY, 2, target=target_table().assign(target_weight=[0.05, 0.02]),
                         risk_report=risk_table("ok"), now=NOW)
    ledger = logbook.load(logs)
    live = list(logbook.live_ids(ledger, DAY).values())
    assert [r["round"] for r in live if r["kind"] == "target_allocation"] == ["1", "2"]
    first = next(r for r in live if r["kind"] == "target_allocation" and r["round"] == "1")
    assert "WORLD 10.0%, GOLD 4.0%; cash 86.0%; 1 order(s) proposed; memo: Round one memo. Second line." in first["detail"]
    reports = {r["round"]: r["title"] for r in live if r["kind"] == "risk_report"}
    assert reports == {"1": "Risk report: 1 ok, 1 warning, 0 breach", "2": "Risk report: 2 ok, 0 warning, 0 breach"}
    assert [r["round"] for r in live if r["kind"] == "risk_flag"] == ["1"]     # round 1's warning is on record
    assert logbook.verify(logs) == []


def test_recording_a_round_twice_adds_nothing_and_a_rerun_replaces_it(tmp_path):
    logs = tmp_path / "logs"
    logbook.record_round(logs, DAY, 1, risk_report=risk_table(), now=NOW)
    assert logbook.record_round(logs, DAY, 1, risk_report=risk_table(), now=NOW) == {"added": 0, "superseded": 0}
    logbook.record_round(logs, DAY, 2, risk_report=risk_table("ok"), now=NOW)
    again = logbook.record_round(logs, DAY, 1, risk_report=risk_table("breach"), now=NOW)   # the day was run again
    assert again["superseded"] == 2 and again["added"] == 2
    titles = live_titles(logs)
    assert "BREACH: leverage" in titles and "WARNING: leverage" not in titles
    assert titles.count("Risk report: 2 ok, 0 warning, 0 breach") == 1               # round 2 was not touched


def test_the_end_of_day_pass_keeps_the_gate_events_and_does_not_duplicate_the_risk_report(logs):
    logbook.record_round(logs, DAY, 1, target=target_table(), memo="m", risk_report=risk_table(), now=NOW)
    logbook.record_round(logs, DAY, 2, risk_report=risk_table("ok"), now=NOW)
    logbook.record_day(logs, DAY, price_table(), NOW)
    live = list(logbook.live_ids(logbook.load(logs), DAY).values())
    assert [r["round"] for r in live if r["kind"] == "risk_report"] == ["1", "2"]    # not a third one from the file
    assert any(r["kind"] == "target_allocation" for r in live) and any(r["kind"] == "fill" for r in live)
    assert logbook.record_day(logs, DAY, price_table(), NOW) == {"added": 0, "superseded": 0}
    page = (logs / "logbook" / f"{DAY}.md").read_text(encoding="utf-8")
    assert page.index("Target allocation") < page.index("BUY 10 WORLD") < page.index("Risk report: 1 ok, 1 warning")


# --- Human against AI -------------------------------------------------------------------------------------------------

def test_human_against_ai_counts_and_value(logs):
    event = next(e for e in logbook.build_events(logs, DAY, price_table()) if e["kind"] == "human_vs_ai")
    assert event["title"] == "Human against AI: 1 of 3 proposed orders kept as proposed, 1 added"
    assert "approved 1, modified 1, rejected 1, added 1; the day was sent back 1 time(s)" in event["detail"]
    # AI: 10 WORLD x 100 + 8 GOLD x 50 + 2 BITCOIN x 10 = 1,420; executed: 1,000 + 4 x 50 + 5 DEFENSE x 20 = 1,300
    assert "AI proposed 1,420 EUR, executed 1,300 EUR" in event["detail"]
    assert json_of(event) == {"added": 1, "ai_value": 1420.0, "approved": 1, "executed_value": 1300.0, "modified": 1,
                              "proposed": 3, "rejected": 1, "sent_back": 1, "stopped": False}


def test_a_stopped_day_is_named_and_traded_nothing(logs):
    rejected = mock.mock_order_log(DAY).iloc[:2].assign(decision="rejected", side=np.nan, units=np.nan,
                                                       reason="no trade today (risk review): Fed tonight")
    write(logs, "orders.csv", rejected)
    write(logs, "reviews.csv", pd.DataFrame([[DAY, 1, "orders", "approved", np.nan],
                                              [DAY, 1, "risk", "stopped", "Fed tonight, no trade"]],
                                             columns=["date", "round", "gate", "decision", "feedback"]))
    event = next(e for e in logbook.build_events(logs, DAY, price_table()) if e["kind"] == "human_vs_ai")
    assert event["title"] == "Human against AI: no trade today (stopped at the risk review)"
    assert json_of(event)["stopped"] is True and json_of(event)["executed_value"] == 0.0


def test_human_against_ai_without_prices_has_counts_but_no_values(logs):
    event = next(e for e in logbook.build_events(logs, DAY, None) if e["kind"] == "human_vs_ai")
    assert json_of(event)["proposed"] == 3 and json_of(event)["ai_value"] is None
    assert "gross value" not in event["detail"]


# --- The learning agent's events ----------------------------------------------------------------------------------------

def test_the_learning_agents_events_are_recorded_once_and_survive_the_day(logs):
    result = logbook.record_learning(logs, DAY, "lesson", "Macro/News ran ahead of the market",
                                     "hit rate 38% over 3 days", data={"hit_rate": 0.38}, source="weights", now=NOW)
    assert result == {"added": 1, "superseded": 0}
    again = logbook.record_learning(logs, DAY, "lesson", "Macro/News ran ahead of the market",
                                    "hit rate 38% over 3 days", data={"hit_rate": 0.38}, source="weights", now=NOW)
    assert again["added"] == 0
    logbook.record_learning(logs, DAY, "reassessment", "macro_news weight 33% to 25%", "evidence: hit rate 38%", now=NOW)
    logbook.record_day(logs, DAY, price_table(), NOW)
    logbook.record_day(logs, DAY, price_table(), NOW)
    ledger = logbook.load(logs)
    rows = ledger[ledger["kind"].isin(["lesson", "reassessment"])]
    assert len(rows) == 2 and set(rows["actor"]) == {"learning_agent"} and (ledger["kind"] == "superseded").sum() == 0
    assert rows.iloc[0]["source"] == "learning: weights"
    page = (logs / "logbook" / f"{DAY}.md").read_text(encoding="utf-8")
    assert "## Learning agent" in page and "- **macro_news weight 33% to 25%**" in page
    assert logbook.verify(logs) == []


def test_the_learning_agent_can_only_log_lessons_and_reassessments(tmp_path):
    with pytest.raises(ValueError, match="kind must be"):
        logbook.record_learning(tmp_path, DAY, "gossip", "x")


# --- The index ------------------------------------------------------------------------------------------------------------

def test_the_index_has_one_line_per_day_from_the_ledger_alone(logs):
    logbook.record_round(logs, DAY, 1, risk_report=risk_table(), now=NOW)
    logbook.record_learning(logs, DAY, "lesson", "a lesson", now=NOW)
    logbook.record_day(logs, DAY, price_table(), NOW)
    text = (logs / "logbook" / "INDEX.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith(f"| {DAY} |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    #        day  book         AI  kept chg rej add back warn value      return   WORLD    lessons
    assert cells == [DAY, "competition", "3", "1", "1", "1", "1", "1", "1", "10,110.00", "+1.10%", "+10.00%", "1"]
    assert "Hash chain: intact." in text and "last hash" in text
    assert b"\r" not in (logs / "logbook" / "INDEX.md").read_bytes()


def test_the_index_marks_a_broken_chain_and_lists_practice_and_competition_days(logs):
    practice = pd.DataFrame([["2026-11-20", "WORLD", "BUY", 5, "approved", "BUY", 5, ""]], columns=ORDER_LOG)
    write(logs, "orders.csv", pd.concat([practice, mock.mock_order_log(DAY)], ignore_index=True))
    logbook.record_day(logs, "2026-11-20", price_table(), NOW)
    logbook.record_day(logs, DAY, price_table(), NOW)
    text = (logs / "logbook" / "INDEX.md").read_text(encoding="utf-8")
    days = [line.split("|")[1].strip() for line in text.splitlines() if line.startswith("| 2026")]
    assert days == ["2026-11-20", DAY] and "| practice |" in text and "| competition |" in text
    path = logs / "logbook.csv"
    path.write_bytes(path.read_bytes().replace(b"BUY 5 WORLD", b"BUY 6 WORLD", 1))
    logbook.write_index(logs)
    assert "BROKEN" in (logs / "logbook" / "INDEX.md").read_text(encoding="utf-8")


def test_validate_logbook_rejects_data_that_is_not_json():
    with pytest.raises(ValueError, match="JSON"):
        validate_logbook(mock.mock_logbook().assign(data="not json"))
    validate_logbook(mock.mock_logbook().assign(data='{"a":1}'))


def test_gates_recorded_after_the_end_of_day_pass_replace_the_risk_report_of_the_files(logs):
    logbook.record_day(logs, DAY, price_table(), NOW)
    assert [r["round"] for r in logbook.live_ids(logbook.load(logs), DAY).values() if r["kind"] == "risk_report"] == [""]
    logbook.record_round(logs, DAY, 1, risk_report=risk_table(), now=NOW)
    live = list(logbook.live_ids(logbook.load(logs), DAY).values())
    assert [r["round"] for r in live if r["kind"] == "risk_report"] == ["1"]
    assert logbook.record_day(logs, DAY, price_table(), NOW) == {"added": 0, "superseded": 0}


def test_the_command_line_checks_the_chain_without_prices(logs, capsys):
    logbook.record_day(logs, DAY, price_table(), NOW)
    logbook.main(["--verify", "--logs-dir", str(logs)])
    assert capsys.readouterr().out.strip() == "Logbook intact."
    path = logs / "logbook.csv"
    path.write_bytes(path.read_bytes().replace(b"BUY 4 GOLD @ 50.00", b"BUY 4 GOLD @ 40.00", 1))
    logbook.main(["--logs-dir", str(logs)])                       # no date: it only verifies
    assert "content was changed" in capsys.readouterr().out
