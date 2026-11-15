"""Macro/News agent without network: fake Claude client, fake HTTP, fake market data."""
import json
import re
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from agents import macro_analysts, macro_llm, macro_news, macro_tape
from agents.macro_sources import _common, cftc_cot, collect_all, funding_rates, market_stress, polymarket, stablecoins
from data import config as settings
from data.config import MACRO_FACTORS
from data.schemas import validate_macro_exposure, validate_signals, validate_source_registry
from data.universe import load_universe, tickers

DAY = date(2026, 12, 7)


# --- Fakes -----------------------------------------------------------------------------------

def _response(text, stop="end_turn", searches=1):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0,
                            server_tool_use=SimpleNamespace(web_search_requests=searches, web_fetch_requests=0))
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop, usage=usage)


class FakeClaude:
    """Answers like Claude would: facts for searches, findings for exploration, views for reasoning."""

    def __init__(self, direction=0.5):
        self.calls = []
        self.direction = direction
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls.append(request)
        tool_types = {t["type"] for t in request.get("tools", [])}
        if "quant researcher" in request.get("system", ""):
            tickers = re.findall(r'^- "(.+?)":', request["messages"][0]["content"], flags=re.M)
            etfs = [{"ticker": t, "direction": self.direction, "confidence": 0.5, "rationale": f"{t} news",
                     "counter_argument": "c", "sources": ["https://example.com/etf"],
                     "next_catalyst": {"date": "2026-12-09", "event": "Fed rate decision"}} for t in tickers]
            return _response("```json\n" + json.dumps({"etfs": etfs}) + "\n```")
        if "format" in request.get("output_config", {}):
            views = [{"factor": f, "direction": self.direction, "confidence": 0.6, "rationale": "r",
                      "counter_argument": "c", "sources": ["web_news", "not_a_source"]} for f in MACRO_FACTORS]
            return _response(json.dumps({"views": views}), searches=0)
        if settings.WEB_FETCH_TOOL_REASONING in tool_types:
            finding = {"name": "Baltic Dry Index", "what": "shipping rates", "url": "https://example.com/bdi",
                       "fact": "up 12% this week", "value": 12.0, "why": "global trade", "factor": "growth_recession"}
            return _response("Found one.\n```json\n" + json.dumps({"findings": [finding]}) + "\n```")
        fact = {"factor": "oil", "date": "2026-12-05", "summary": "Brent up 3%.", "url": "https://example.com/oil"}
        return _response("```json\n" + json.dumps({"facts": [fact]}) + "\n```")


def _number_collector(as_of):
    return [_common.evidence(as_of, "risk_appetite", "market_stress", "number", "VIX = 25 (z +2.0)", 25.0)]


def _broken_collector(as_of):
    raise RuntimeError("site down")


def _http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- End to end ------------------------------------------------------------------------------

def test_run_gives_two_scores_per_etf_logs_and_a_report(tmp_path):
    client = FakeClaude()
    summary, signals = macro_news.run(DAY, "lesson: be careful", tmp_path, client,
                                      {"stress": _number_collector, "broken": _broken_collector})

    validate_signals(signals)
    assert len(signals) == 100 and set(signals["agent"]) == {"macro_market", "macro_etf"}
    assert signals.groupby("agent")["ticker"].nunique().to_dict() == {"macro_etf": 50, "macro_market": 50}
    assert "Cost of this run: $" in summary
    assert "broken: RuntimeError: site down" in summary
    assert "explore_baltic_dry_index" in summary

    registry = validate_source_registry(pd.read_csv(tmp_path / "source_registry.csv")).set_index("source")
    assert registry.loc["explore_baltic_dry_index", "status"] == "testing"
    assert registry.loc["web_news", "status"] == "active"
    views = pd.read_csv(tmp_path / "macro_views.csv")
    assert len(views) == 15 and not views["sources"].str.contains("not_a_source").any()
    assert len(pd.read_csv(tmp_path / "evidence.csv")) > 3

    searches = [c for c in client.calls if c["model"] == settings.LLM_FAST]
    assert len(searches) == len(macro_llm.search_topics())
    assert all(c["tools"][0]["type"] == "web_search_20250305" for c in searches)
    sonnet = [c for c in client.calls if c["model"] == settings.LLM_REASONING]
    assert all(c["fallbacks"] == "default" for c in sonnet)


def test_on_trial_sources_are_flagged_and_dropped_ones_ignored(tmp_path):
    macro_news.run(DAY, "", tmp_path, FakeClaude(), {"stress": _number_collector})
    registry = pd.read_csv(tmp_path / "source_registry.csv")
    registry.loc[registry["source"] == "market_stress", "status"] = "dropped"
    registry.to_csv(tmp_path / "source_registry.csv", index=False)

    client = FakeClaude()
    macro_news.run(date(2026, 12, 8), "", tmp_path, client, {"stress": _number_collector})

    prompt = next(c for c in client.calls if "format" in c.get("output_config", {}))["messages"][0]["content"]
    assert "market_stress" not in prompt
    assert "explore_baltic_dry_index (on trial)" in prompt


def test_daily_cli_reads_back_a_date_already_done(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    (tmp_path / "macro").mkdir()
    (tmp_path / "macro" / "2026-12-07.md").write_text("done")
    macro_news.main(["--date", "2026-12-07"])
    assert "already done" in capsys.readouterr().out


def test_pause_turn_is_resumed():
    replies = iter([_response("searching...", stop="pause_turn"), _response('```json\n{"facts": []}\n```')])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: next(replies)))
    usage = macro_llm.Usage()
    text = macro_llm.ask(client, usage, settings.LLM_FAST, "system", "prompt")
    assert macro_llm.extract_json(text) == {"facts": []}
    assert usage.calls == 2 and usage.web_searches == 2
    assert usage.cost_usd == pytest.approx(2 * (1000 * 0.10 + 200 * 0.50) / 1e6 + 2 * settings.WEB_SEARCH_USD)


# --- Scores ----------------------------------------------------------------------------------

def _views(**directions):
    return pd.DataFrame([{"factor": f, "direction": d, "confidence": 0.8} for f, d in directions.items()])


def test_scores_follow_exposure_leverage_and_stay_in_range():
    scores = macro_news.score_etfs(_views(oil=1.0, risk_appetite=-1.0), DAY).set_index("ticker")["score"]
    assert scores["OIL & GAS E&P"] > 0 > scores["NASDAQ"]
    assert scores["NASDAQ 3X"] == pytest.approx(max(-1, 3 * scores["NASDAQ"]), abs=1e-3)
    assert scores["S&P 500 SHORT"] == pytest.approx(-scores["S&P 500"], abs=1e-3)
    assert scores.between(-1, 1).all()
    assert scores["OVERNIGHT CASH"] == 0


def test_cash_stays_near_zero_on_a_strong_rates_view():
    scores = macro_news.score_etfs(_views(ecb_rates=1.0), DAY).set_index("ticker")["score"]
    assert abs(scores["OVERNIGHT CASH"]) <= 0.1 < scores["EURO BANKS"]


def test_exposure_table_covers_the_universe():
    exposure = validate_macro_exposure(pd.read_csv(settings.MACRO_EXPOSURE_CSV))
    assert exposure["ticker"].tolist() == tickers()


# --- Hit rates -------------------------------------------------------------------------------

def test_hit_rates_update_and_wrong_sources_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SOURCE_MIN_USES", 1)
    pd.DataFrame([
        {"source": "good", "kind": "number", "factors": "oil", "status": "testing", "first_seen": "2026-12-01",
         "n_uses": 0, "hit_rate": np.nan},
        {"source": "bad", "kind": "text", "factors": "oil", "status": "active", "first_seen": "2026-12-01",
         "n_uses": 0, "hit_rate": np.nan},
    ]).to_csv(tmp_path / "source_registry.csv", index=False)
    pd.DataFrame([
        {"date": "2026-12-07", "factor": "oil", "direction": 0.8, "confidence": 0.7, "rationale": "r",
         "counter_argument": "c", "sources": "good"},
        {"date": "2026-12-07", "factor": "risk_appetite", "direction": 0.5, "confidence": 0.5, "rationale": "r",
         "counter_argument": "c", "sources": "bad"},
    ]).to_csv(tmp_path / "macro_views.csv", index=False)
    returns = {t: -0.02 for t in tickers()}
    returns.update({"OIL & GAS E&P": 0.05, "WORLD ENERGY": 0.04})

    result = macro_news.update_source_hit_rates(DAY, returns, tmp_path)

    registry = pd.read_csv(tmp_path / "source_registry.csv").set_index("source")
    assert result["graded_views"] == 2
    assert registry.loc["good", "hit_rate"] == 1 and registry.loc["good", "n_uses"] == 1
    assert result["propose_active"] == ["good"] and registry.loc["good", "status"] == "testing"
    assert result["dropped"] == ["bad"] and registry.loc["bad", "status"] == "dropped"


# --- Collectors ------------------------------------------------------------------------------

def test_failing_collector_is_skipped_and_reported():
    ev, failures = collect_all(DAY, {"ok": _number_collector, "broken": _broken_collector})
    assert len(ev) == 1 and failures == {"broken": "RuntimeError: site down"}


def test_market_stress_puts_numbers_in_context():
    idx = pd.bdate_range(end="2026-12-07", periods=260)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({s: 20 + rng.normal(0, 1, len(idx)) for s in ["^VIX", "^VIX3M", "^SKEW", "^MOVE", "HG=F", "GC=F"]},
                         index=idx)
    rows = market_stress.collect(DAY, download=lambda *args: close)
    assert len(rows) == 5 and all("percentile" in r["summary"] for r in rows)


def test_polymarket_keeps_open_informative_markets():
    soon = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    market = {"id": "1", "question": "Fed cut in December?", "outcomePrices": '["0.3", "0.7"]', "closed": False,
              "endDate": soon, "volume": "1000000", "oneWeekPriceChange": 0.05, "slug": "fed-cut"}
    certain = {**market, "id": "2", "outcomePrices": '["0.999", "0.001"]'}
    closed = {**market, "id": "3", "closed": True}
    far = {**market, "id": "4", "endDate": "2099-12-10T00:00:00Z"}
    no_prices = {k: v for k, v in {**market, "id": "5"}.items() if k != "outcomePrices"}
    payload = {"events": [{"markets": [market, certain, closed, far, no_prices]}]}
    rows = polymarket.collect(DAY, http=_http(lambda request: httpx.Response(200, json=payload)))
    assert [r["value"] for r in rows] == [0.3]
    assert "+5 pts over a week" in rows[0]["summary"]



# --- v2 sources ------------------------------------------------------------------------------

def test_ordinal():
    assert [_common.ordinal(n) for n in (1, 2, 3, 11, 12, 22, 73, 100)] == \
        ["1st", "2nd", "3rd", "11th", "12th", "22nd", "73rd", "100th"]


def test_cftc_cot_measures_net_positioning():
    weeks = pd.date_range(end="2026-12-01", periods=40, freq="W-TUE")
    payload = [{"cftc_contract_market_code": "088691", "report_date_as_yyyy_mm_dd": f"{d:%Y-%m-%d}T00:00:00.000",
                "open_interest_all": "100000", "noncomm_positions_long_all": str(50000 + i * 100),
                "noncomm_positions_short_all": "10000"} for i, d in enumerate(weeks)]
    rows = cftc_cot.collect(DAY, http=_http(lambda request: httpx.Response(200, json=payload)))
    assert len(rows) == 1 and rows[0]["factor"] == "gold_silver"
    assert rows[0]["value"] == pytest.approx(0.439)
    assert "net long 43,900 gold" in rows[0]["summary"] and "weekly change +100" in rows[0]["summary"]


def test_stablecoin_supply_growth():
    start = int(pd.Timestamp("2025-06-01", tz="UTC").timestamp())
    payload = [{"date": str(start + i * 86400), "totalCirculatingUSD": {"peggedUSD": 2e11 * (1.001 ** i)}}
               for i in range(500)]
    rows = stablecoins.collect(DAY, http=_http(lambda request: httpx.Response(200, json=payload)))
    assert rows[0]["factor"] == "crypto" and rows[0]["value"] == pytest.approx(1.001 ** 30 - 1)


def test_funding_rates_are_annualised():
    payload = [{"fundingRate": "0.0001"} for _ in range(120)]
    rows = funding_rates.collect(DAY, http=_http(lambda request: httpx.Response(200, json=payload)))
    assert len(rows) == 2 and rows[0]["value"] == pytest.approx(0.0001 * 3 * 365)


def test_x_posts_are_searched_on_x_only(tmp_path):
    client = FakeClaude()
    macro_news.run(DAY, "", tmp_path, client, {"stress": _number_collector})
    restricted = [c for c in client.calls if c.get("tools") and c["tools"][0].get("allowed_domains")]
    assert len(restricted) == 1 and restricted[0]["tools"][0]["allowed_domains"] == ["x.com"]
    assert "posts" in restricted[0]["messages"][0]["content"]


# --- ETF analysts ----------------------------------------------------------------------------

def test_run_scores_the_theme_analysts_separately(tmp_path):
    client = FakeClaude(direction=0.5)
    summary, signals = macro_news.run(DAY, "", tmp_path, client, {"stress": _number_collector})

    analyst_calls = [c for c in client.calls if "quant researcher" in c.get("system", "")]
    assert len(analyst_calls) == 20  # one per theme of the universe
    assert all(c["tools"][0]["type"] == settings.WEB_SEARCH_TOOL_REASONING and c["fallbacks"] == "default"
               for c in analyst_calls)
    assert any("judge the copper price itself" in c["messages"][0]["content"] for c in analyst_calls)
    assert not any("NASDAQ 3X" in c["messages"][0]["content"] for c in analyst_calls)  # reuses NASDAQ's view
    views = pd.read_csv(tmp_path / "etf_views.csv").set_index("ticker")
    assert len(views) == 50
    assert views.loc["NASDAQ 3X", "direction"] == views.loc["NASDAQ", "direction"]
    assert views.loc["NASDAQ 3X", "rationale"].startswith("Same underlying as NASDAQ")
    assert "## ETF analysts" in summary and "| ETF | macro_market | macro_etf |" in summary
    validate_signals(signals)
    by_agent = {agent: rows.set_index("ticker") for agent, rows in signals.groupby("agent")}
    assert by_agent["macro_etf"]["rationale"].str.startswith("analyst +0.50").all()
    assert by_agent["macro_market"]["rationale"].str.startswith("macro: ").all()
    assert by_agent["macro_etf"].loc["WORLD", "score"] == 0.5  # the analyst's own call, not a blend
    assert "no prices" in summary  # no price table given: the analysts judged from the news alone
    assert by_agent["macro_etf"].loc["WORLD", "rationale"].startswith("analyst +0.50; next: 09 Dec Fed rate decision: ")
    assert views.loc["WORLD", "catalyst_date"] == "2026-12-09" and views.loc["NASDAQ 3X", "catalyst"] == "Fed rate decision"


def test_report_opens_with_a_brief_the_portfolio_manager_can_read_in_full(tmp_path):
    summary, _ = macro_news.run(DAY, "", tmp_path, FakeClaude(direction=0.5), {"stress": _number_collector})
    start, end = summary.index("## Brief for the Portfolio Manager"), summary.index("## Views")
    assert summary.index("Cost of this run") < start < end < 4000  # the Portfolio Manager reads 4,000 characters
    brief = summary[start:end]
    assert "- Wed 09 Dec: Fed rate decision (" in brief and brief.count("Fed rate decision") == 1  # one event, many ETFs
    assert brief.count("\n| ") == 1 + macro_news.BRIEF_CALLS  # the table header, then the strongest calls
    assert "**Macro views** (+1 to -1): ecb_rates +0.50" in brief


def test_a_catalyst_needs_an_event_and_a_date_inside_the_window():
    assert macro_analysts._catalyst({"date": "2026-12-09", "event": " Fed  decision "}, DAY) == ("2026-12-09", "Fed decision")
    for bad in ({"date": "2026-12-01", "event": "already past"}, {"date": "2027-02-01", "event": "too far"},
                {"date": "next week", "event": "no date"}, {"date": "2026-12-09", "event": " "}, None, "Fed"):
        assert macro_analysts._catalyst(bad, DAY) == (None, None)


def test_analyst_score_applies_leverage_and_skips_etfs_without_a_view():
    exposure = macro_news.load_exposure()
    views = pd.DataFrame({"date": "2026-12-07", "ticker": ["NASDAQ 3X", "STOXX SHORT", "WORLD"],
                          "direction": [0.5, 0.5, 0.2], "confidence": [0.8, 0.8, 0.3], "rationale": "r",
                          "counter_argument": "c", "sources": "https://example.com"})
    signals = macro_analysts.etf_signals(views, exposure, DAY, "macro_etf").set_index("ticker")
    assert signals["score"].to_dict() == {"NASDAQ 3X": 1.0, "STOXX SHORT": -1.0, "WORLD": 0.2}  # 3 x 0.5 clipped, -2 x 0.5
    assert signals.loc["WORLD", "confidence"] == 0.3 and set(signals["agent"]) == {"macro_etf"}
    assert "GOLD" not in signals.index  # no view, no score: the Portfolio Manager uses the other agents


# --- What the analysts read before they judge: the tape and their own record ---------------------

def _closes(**series) -> pd.DataFrame:
    """Prices of a few ETFs on consecutive days ending 2026-12-06 (the day before DAY)."""
    rows = [{"date": (pd.Timestamp("2026-12-06") - pd.Timedelta(days=len(values) - 1 - i)).date(), "ticker": t,
             "close_eur": v} for t, values in series.items() for i, v in enumerate(values)]
    return pd.DataFrame(rows)


def test_tape_shows_what_the_price_already_did_and_never_the_day_itself():
    prices = _closes(WORLD=[100.0] * 19 + [100.0, 110.0, 99.0])
    today = pd.DataFrame([{"date": DAY, "ticker": "WORLD", "close_eur": 500.0}])  # not known in the morning
    table = macro_tape.closes(pd.concat([prices, today]), DAY)
    assert table.index.max() == "2026-12-06"
    line = macro_tape.tape_lines(table)["WORLD"]
    assert line.startswith("tape: last close 06 Dec; 1 session -10.0%, 5 sessions -1.0%, 20 sessions -1.0%; volatility")
    assert macro_tape.tape_lines(macro_tape.closes(None, DAY)) == {}


def test_past_calls_are_graded_on_the_sessions_that_followed():
    prices = _closes(WORLD=[100.0, 100.0, 102.0, 104.0, 108.0, 110.0], **{"STOXX SHORT": [50.0, 50.0, 49.0, 48.0, 47.0, 46.0]})
    table = macro_tape.closes(prices, DAY)
    views = pd.DataFrame({"date": ["2026-12-03", "2026-12-03", "2026-12-06", "2026-12-07"],
                          "ticker": ["WORLD", "STOXX SHORT", "WORLD", "WORLD"],
                          "direction": [-0.3, 0.4, 0.05, 0.9]})
    leverage = pd.Series({"WORLD": 1.0, "STOXX SHORT": -2.0})
    grades = macro_tape.grade_calls(views, table, leverage, sessions=3).set_index(["date", "ticker"])

    # called on 3 Dec knowing the 2 Dec close (100): three sessions later the close is 108
    assert grades.loc[("2026-12-03", "WORLD"), "ret"] == pytest.approx(0.08)
    assert grades.loc[("2026-12-03", "WORLD"), "hit"] is False  # said down, went up
    # a view of +0.4 on the index is a view of -0.8 on its -2x ETF, which fell: right
    assert grades.loc[("2026-12-03", "STOXX SHORT"), "hit"] is True
    # called on 6 Dec: only one session is known so far, and 0.05 is no call
    assert grades.loc[("2026-12-06", "WORLD"), "sessions"] == 1 and grades.loc[("2026-12-06", "WORLD"), "hit"] is None
    assert ("2026-12-07", "WORLD") not in grades.index  # today's call cannot be graded yet

    grades = grades.reset_index()
    assert macro_tape.record(grades) == (1, 2)
    assert macro_tape.record_line(grades, ["WORLD"]) == "Your record on these ETFs so far: 0 of 1 calls right (0%)."
    lines = macro_tape.scorecard_lines(grades)
    assert lines["WORLD"] == ("your calls: 06 Dec +0.05 -> +1.9% over 1 session (no call); "
                              "03 Dec -0.30 -> +8.0% over 3 sessions (wrong)")


def test_analysts_read_the_tape_and_their_record(tmp_path):
    prices = _closes(WORLD=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0], GOLD=[50.0, 50.0, 50.0, 49.0, 48.0, 47.0])
    pd.DataFrame({"date": "2026-12-03", "ticker": ["WORLD", "GOLD"], "direction": [0.4, 0.4], "confidence": 0.5,
                  "rationale": "r", "counter_argument": "c", "sources": "https://example.com"}
                 ).to_csv(tmp_path / "etf_views.csv", index=False)
    client = FakeClaude()
    summary, _ = macro_news.run(DAY, "", tmp_path, client, {"stress": _number_collector}, prices=prices)

    asked = {c["system"].split(", covering ")[1].split(" ETFs")[0]: c["messages"][0]["content"]
             for c in client.calls if "quant researcher" in c.get("system", "")}
    world = asked["Global equity (benchmark)"]
    assert "  tape: last close 06 Dec; 1 session +1.0%, 5 sessions +5.0%" in world
    assert "  your calls: 03 Dec +0.40 -> +3.0% over 3 sessions (right)" in world
    assert "Your record on these ETFs so far: 1 of 1 calls right (100%)." in world
    assert "03 Dec +0.40 -> -4.0% over 3 sessions (wrong)" in asked["Precious metals"]  # 50 on 2 Dec, 48 on 5 Dec
    assert "No call of yours on these ETFs can be graded yet." in asked["Financials"]
    assert "next 3 trading sessions" in next(c["system"] for c in client.calls if "quant researcher" in c.get("system", ""))
    assert "Analysts' record before today: 1 of 2 past calls right (50%)" in summary
    assert len(pd.read_csv(tmp_path / "etf_views.csv")) == 2 + 50  # the old views stay, today's are added


def test_a_failing_analyst_is_reported_and_the_run_goes_on(tmp_path):
    client = FakeClaude()
    original = client._create

    def flaky(**request):
        if "Precious metals" in request.get("system", ""):
            raise RuntimeError("search timeout")
        return original(**request)

    client.messages.create = client.beta.messages.create = flaky
    summary, signals = macro_news.run(DAY, "", tmp_path, client, {"stress": _number_collector})
    assert "analyst Precious metals: RuntimeError: search timeout" in summary
    assert len(pd.read_csv(tmp_path / "etf_views.csv")) == 50 - 3  # GOLD, GOLD MINERS, SILVER MINERS


def test_an_empty_analyst_answer_is_asked_again():
    replies = iter([_response('```json\n{"etfs": []}\n```'), _response(
        '```json\n{"etfs": [{"ticker": "GOLD", "direction": 0.3, "confidence": 0.5, "rationale": "r", '
        '"counter_argument": "c", "sources": [{"url": "https://example.com/gold"}]}]}\n```')])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: next(replies))))
    universe = load_universe()
    rows, missing = macro_analysts.analyse_theme(client, macro_llm.Usage(), DAY, "Precious metals",
                                                 universe[universe["ticker"] == "GOLD"])
    assert [r["ticker"] for r in rows] == ["GOLD"] and rows[0]["sources"] == "https://example.com/gold"
    assert missing == {}


def test_every_theme_has_a_coverage_brief_in_its_prompt():
    for theme in load_universe()["theme"].unique():
        assert theme in macro_analysts.THEME_BRIEFS
        prompt = macro_analysts.system_prompt(theme, DAY)
        assert macro_analysts.THEME_BRIEFS[theme] in prompt and "2026-12-07" in prompt and "{" not in prompt.split("```json")[0]


def test_etfs_missing_from_an_answer_are_asked_again_and_reported():
    view = {"direction": 0.2, "confidence": 0.4, "rationale": "r", "counter_argument": "c"}
    first = {"etfs": [{**view, "ticker": "GOLD", "sources": ["https://example.com/gold"]},
                      {**view, "ticker": "GOLD MINERS", "sources": []}]}  # SILVER MINERS left out
    second = {"etfs": [{**view, "ticker": "GOLD MINERS", "sources": ["https://example.com/miners"]}]}
    replies, prompts = iter([first, second]), []

    def create(**request):
        prompts.append(request["messages"][0]["content"])
        return _response("```json\n" + json.dumps(next(replies)) + "\n```")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    etfs = load_universe().query("theme == 'Precious metals'")
    rows, missing = macro_analysts.analyse_theme(client, macro_llm.Usage(), DAY, "Precious metals", etfs)

    assert sorted(r["ticker"] for r in rows) == ["GOLD", "GOLD MINERS"]
    assert '"GOLD MINERS"' in prompts[1] and '"SILVER MINERS"' in prompts[1] and '- "GOLD":' not in prompts[1]
    assert missing == {"SILVER MINERS": "left out of the answer"}


def test_an_answer_without_json_is_reported():
    rows, dropped = macro_analysts._parse("Sorry, I found nothing.", DAY, {"GOLD"})
    assert rows == [] and dropped == {"GOLD": "no JSON in the answer"}


def test_a_ticker_copied_with_its_name_is_still_matched():
    expected = {"S&P 500", "GOLD", "GOLD MINERS"}
    assert macro_analysts._match("S&P 500 · iShares Core S&P 500 UCITS ETF USD (Acc)", expected) == "S&P 500"
    assert macro_analysts._match("gold miners", expected) == "GOLD MINERS"
    assert macro_analysts._match("GOLD MINERS · VanEck", expected) == "GOLD MINERS"
    assert macro_analysts._match("SILVER", expected) is None
