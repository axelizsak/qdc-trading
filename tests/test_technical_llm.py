"""The LLM layer of the technical agent (agents/technical_llm.py), with a fake Claude: no network."""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from agents import technical, technical_llm
from data import config as settings
from data.features import compute_features
from data.schemas import validate_signals
from tests.test_technical import made_up_prices

DAY = date(2026, 12, 7)


class FakeClaude:
    """Answers with one score per ETF of the prompt, or with whatever `answer` returns."""

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls.append(request)
        tickers = [line.split('"')[1] for line in request["messages"][0]["content"].splitlines() if line.startswith('- "')]
        etfs = self.answer(tickers) if self.answer else [
            {"ticker": t, "score": 0.4, "confidence": 0.6, "rationale": f"{t}: buy setup but RSI 2 at 95"} for t in tickers]
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, server_tool_use=None)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"etfs": etfs}))],
                               stop_reason="end_turn", usage=usage)


@pytest.fixture(scope="module")
def sheet():
    prices = made_up_prices()
    return technical.build_sheet(prices, compute_features(prices), pd.to_datetime(prices["date"]).max().date())


def test_every_etf_of_the_sheet_gets_an_llm_score(sheet):
    client = FakeClaude()
    signals = technical_llm.interpret(sheet, DAY, "lesson: buy setups on stretched ETFs gave back", client)

    validate_signals(signals)
    assert set(signals["agent"]) == {"technical_llm"} and sorted(signals["ticker"]) == sorted(sheet["ticker"])
    assert (signals["score"] == 0.4).all() and signals["rationale"].str.contains("RSI 2 at 95").all()
    request = client.calls[0]
    assert request["model"] == settings.LLM_REASONING and "tools" not in request  # it reads no news
    assert request["output_config"]["format"]["type"] == "json_schema"
    prompt = request["messages"][0]["content"]
    assert "lesson: buy setups on stretched ETFs gave back" in prompt
    assert f"next {settings.HORIZON_SESSIONS} trading sessions" in request["system"]


def test_the_llm_reads_the_codes_answers_not_raw_prices(sheet):
    lines = technical_llm.sheet_lines(sheet).splitlines()
    assert len(lines) == len(sheet)
    world = next(line for line in lines if line.startswith('- "WORLD"'))
    assert "rule signal buy (+1.0)" in world and "trend short up, medium up, long up -> setup buy" in world
    short = next(line for line in lines if line.startswith('- "S&P 500 SHORT"'))
    assert "leverage -1, read on S&P 500" in short and "mean reversion: not applicable" in short
    assert "close_eur" not in "\n".join(lines)


def test_bad_answers_are_dropped_and_scores_stay_in_range(sheet):
    def answer(tickers):
        return [{"ticker": tickers[0], "score": 7, "confidence": -1, "rationale": "x"},      # clipped
                {"ticker": tickers[0], "score": -1, "confidence": 1, "rationale": "twice"},  # answered twice: first kept
                {"ticker": "NOT AN ETF", "score": 1, "confidence": 1, "rationale": "x"},
                {"ticker": tickers[1], "score": "high", "confidence": 1, "rationale": "x"},   # not a number
                {"ticker": tickers[2], "score": 0.25, "confidence": 0.5, "rationale": ""}]

    signals = technical_llm.interpret(sheet, DAY, "", FakeClaude(answer)).set_index("ticker")
    tickers = sheet["ticker"].tolist()
    assert list(signals.index) == [tickers[0], tickers[2]]
    assert signals.loc[tickers[0], ["score", "confidence"]].tolist() == [1.0, 0.0]
    assert signals.loc[tickers[2], "rationale"] == "no rationale given"


def test_without_a_key_or_when_the_call_fails_there_is_no_llm_score(sheet, capsys):
    assert technical_llm.interpret(sheet, DAY).empty  # no key in tests
    assert "no ANTHROPIC_API_KEY" in capsys.readouterr().out

    def down(tickers):
        raise RuntimeError("API down")

    assert technical_llm.interpret(sheet, DAY, "", FakeClaude(down)).empty
    assert technical_llm.interpret(sheet.iloc[0:0], DAY, "", FakeClaude()).empty


def test_run_gives_both_technical_scores_when_the_llm_answers():
    prices = made_up_prices()
    features = compute_features(prices)
    as_of = pd.to_datetime(prices["date"]).max().date()
    both = technical.run(prices, features, as_of, "", client=FakeClaude())
    assert both.groupby("agent")["ticker"].nunique().to_dict() == {"technical_det": 42, "technical_llm": 42}
    assert set(technical.run(prices, features, as_of, use_llm=False)["agent"]) == {"technical_det"}
    assert set(technical.run(prices, features, as_of)["agent"]) == {"technical_det"}  # no key: the rule score alone
