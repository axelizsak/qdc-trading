"""The onboarding coach: the code keeps the sheet, decides when it is complete and finds the contradictions."""
import json
from datetime import date
from types import SimpleNamespace

import pytest

from agents import profiler
from data.profiles import PROFILES

KNOWN = {"First name": "Thomas", "Age": "29", "On the savings account": "EUR 14,500"}
FULL = {"goal": "a home deposit", "horizon_years": 4, "amount_eur": 6000, "wealth_range": "10k_50k", "monthly_eur": 150,
        "buffer_months": 6, "loss_reaction": "wait", "max_loss_pct": 10, "knowledge_1": "right", "knowledge_2": "wrong",
        "experience": "funds", "preferences": "none"}


def reply(message="Next question?", choices=(), profile=None, why="", **answers):
    return {"message": message, "choices": list(choices), "answers": {field: answers.get(field) for field in profiler.SHEET},
            "profile": profile, "why": why}


class FakeClaude:
    """Answers with the replies it was given, in order, and keeps every request."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls.append(request)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0, server_tool_use=None)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(self.replies.pop(0)))],
                               stop_reason="end_turn", usage=usage)


def test_the_sheet_is_what_claude_is_asked_to_fill():
    answers = profiler.TURN_SCHEMA["properties"]["answers"]
    assert answers["required"] == list(profiler.SHEET) == list(FULL)
    assert all({"type": "null"} in field["anyOf"] for field in answers["properties"].values())  # empty until settled
    assert profiler.TURN_SCHEMA["properties"]["profile"]["anyOf"][0]["enum"] == list(PROFILES)


def test_clean_keeps_only_valid_answers():
    dirty = {"goal": "  a   home  ", "horizon_years": -2, "amount_eur": 6000, "monthly_eur": True, "wealth_range": "rich",
             "loss_reaction": "wait", "max_loss_pct": 10.0, "preferences": "", "profile": "late_starter", "extra": 1}
    assert profiler.clean(dirty) == {"goal": "a home", "amount_eur": 6000, "loss_reaction": "wait", "max_loss_pct": 10}
    assert profiler.clean(None) == {}
    assert len(profiler.clean({"goal": "x" * 500})["goal"]) == profiler.TEXT_CHARS
    assert profiler.clean(FULL) == FULL and profiler.missing(FULL) == []
    assert profiler.missing({"goal": "a home"})[:2] == ["horizon_years", "amount_eur"]


def test_checks_find_the_contradictions():
    assert profiler.checks(FULL) == [] and profiler.checks({}) == []
    unaware = profiler.checks({**FULL, "knowledge_1": "wrong", "max_loss_pct": 20})
    assert len(unaware) == 2 and "knowledge" in unaware[0] and "5 years" in unaware[1]  # horizon 4 with a 20% loss
    assert "knowledge" in profiler.checks({**FULL, "knowledge_1": "wrong", "loss_reaction": "add"})[0]
    assert profiler.checks({**FULL, "horizon_years": 12, "max_loss_pct": 30}) == []  # long horizon: a high loss is fine
    assert "buffer" in profiler.checks({**FULL, "buffer_months": 1})[0]
    assert "wealth range" in profiler.checks({**FULL, "amount_eur": 60_000})[0]
    assert profiler.checks({**FULL, "amount_eur": 600_000, "wealth_range": "over_250k"}) == []


def test_a_turn_adds_the_answers_to_the_sheet():
    claude = FakeClaude(reply("And how much now?", ["EUR 2,000", "EUR 6,000"], goal="a home deposit", horizon_years=4))
    chat = [{"role": "coach", "text": "What is it for?"}, {"role": "customer", "text": "A home in 4 years " + "x" * 900}]
    out = profiler.turn(KNOWN, chat, {"experience": "funds"}, client=claude, today=date(2026, 12, 1))
    assert out["sheet"] == {"experience": "funds", "goal": "a home deposit", "horizon_years": 4}
    assert out["message"] == "And how much now?" and out["choices"] == ["EUR 2,000", "EUR 6,000"]
    assert out["profile"] is None and out["why"] == "" and "amount_eur" in out["missing"] and out["cost_usd"] > 0
    request = claude.calls[0]
    assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]
    assert request["messages"][0]["content"] == profiler.OPENING
    assert len(request["messages"][2]["content"]) == profiler.MESSAGE_CHARS  # a long message is cut
    assert request["output_config"]["format"]["schema"] is profiler.TURN_SCHEMA
    system = request["system"]
    assert "- On the savings account: EUR 14,500" in system and "- experience: funds" in system
    assert "Still missing: goal, horizon_years" in system and "2026-12-01" in system and profiler.COACH in system
    assert "contradict each other" not in system.split("The sheet so far")[1]  # nothing to raise


def test_the_code_not_claude_decides_when_the_sheet_is_complete():
    early = reply("All set!", profile="goal_builder", why="You save for a home.", goal="a home deposit")
    claude = FakeClaude(early, reply("How many years until you need it?"))
    out = profiler.turn(KNOWN, [{"role": "customer", "text": "a home"}], {}, client=claude)
    assert len(claude.calls) == 2 and "Still missing: horizon_years" in claude.calls[1]["system"]  # asked once more
    assert out["profile"] is None and out["message"] == "How many years until you need it?"
    stubborn = FakeClaude(early, early)
    assert profiler.turn(KNOWN, [], {}, client=stubborn)["profile"] is None


def test_the_profile_comes_with_the_last_answer():
    almost = {k: v for k, v in FULL.items() if k != "preferences"}
    last = reply("That is all I need.", ["Great"], profile="goal_builder", why="You  save for a home.", preferences="none")
    out = profiler.turn(KNOWN, [{"role": "customer", "text": "no wish"}], almost, client=FakeClaude(last))
    assert out["profile"] == "goal_builder" and out["why"] == "You save for a home." and out["sheet"] == FULL
    assert out["missing"] == [] and out["choices"] == [] and out["flags"] == []
    forgetful = FakeClaude(reply("Thanks!", preferences="none"), last)
    assert profiler.turn(KNOWN, [], almost, client=forgetful)["profile"] == "goal_builder"
    assert "nothing: set the profile in this turn" in forgetful.calls[1]["system"]
    with pytest.raises(RuntimeError, match="no profile"):
        profiler.turn(KNOWN, [], FULL, client=FakeClaude(reply("Thanks!"), reply("Thanks again!")))


def test_contradictions_go_back_to_claude():
    claude = FakeClaude(reply("Is one month of buffer right?", ["Yes", "No"]))
    out = profiler.turn(KNOWN, [], {"buffer_months": 1}, client=claude)
    assert "Thin buffer" in claude.calls[0]["system"] and "Raise each one once" in claude.calls[0]["system"]
    assert out["flags"] == ["Thin buffer: less than 3 months of spending stays in savings"]


def test_a_turn_needs_a_key_and_a_chat_of_reasonable_length(monkeypatch):
    monkeypatch.setattr(profiler.settings, "ANTHROPIC_API_KEY", None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        profiler.turn(KNOWN, [])
    with pytest.raises(ValueError, match="longer"):
        profiler.turn(KNOWN, [{"role": "customer", "text": "hi"}] * (profiler.MAX_MESSAGES + 1), client=FakeClaude())
