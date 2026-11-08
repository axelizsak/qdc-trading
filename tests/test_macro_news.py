"""Macro/News agent without network: fake Claude client, fake HTTP, fake market data."""
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from agents import macro_llm, macro_news
from agents.macro_sources import _common, collect_all, market_stress, polymarket
from data import config as settings
from data.config import MACRO_FACTORS
from data.schemas import validate_macro_exposure, validate_signals, validate_source_registry
from data.universe import tickers

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

def test_run_gives_50_signals_logs_and_a_report(tmp_path):
    client = FakeClaude()
    summary, signals = macro_news.run(DAY, "lesson: be careful", tmp_path, client,
                                      {"stress": _number_collector, "broken": _broken_collector})

    validate_signals(signals)
    assert len(signals) == 50 and set(signals["agent"]) == {"macro_news"}
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

