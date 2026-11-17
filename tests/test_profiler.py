"""Lena, the coach: probabilities that follow the answers, the next question, the sheet's checks, typed replies."""
import json
from types import SimpleNamespace

import pytest

from agents import profiler
from data import config as settings
from data.profiles import PROFILES
from mobile.customers import CUSTOMERS

THOMAS = CUSTOMERS["thomas"]
# What each demo customer answers, in character, and the profile it should lead to.
SCRIPTS = {
    "lina": ("first_step_starter", {"goal": "start", "horizon": "over_15", "amount": 700, "wealth": "under_10k",
                                    "monthly": 50, "buffer": "under_3", "loss_reaction": "wait", "max_loss": "10",
                                    "knowledge_1": "unsure", "knowledge_2": "shares", "experience": "none",
                                    "involvement": "check", "worry": "know", "rules": ["weapons"]}),
    "thomas": ("goal_builder", {"goal": "home", "horizon": "under_5", "amount": 6000, "wealth": "10k_50k",
                                "monthly": 150, "buffer": "3_6", "loss_reaction": "wait", "max_loss": "10",
                                "knowledge_1": "fund", "knowledge_2": "bonds", "experience": "funds",
                                "involvement": "check", "worry": "plan", "rules": []}),
    "noah": ("self_directed_trader", {"goal": "grow", "horizon": "over_15", "amount": 2500, "wealth": "10k_50k",
                                      "monthly": 400, "buffer": "under_3", "loss_reaction": "add", "max_loss": "30",
                                      "knowledge_1": "fund", "knowledge_2": "shares", "experience": "crypto_leverage",
                                      "involvement": "pick", "worry": "cost", "rules": []}),
    "sofie": ("cautious_saver", {"goal": "family", "horizon": "5_15", "amount": 8000, "wealth": "10k_50k",
                                 "monthly": 150, "buffer": "over_6", "loss_reaction": "sell", "max_loss": "5",
                                 "knowledge_1": "unsure", "knowledge_2": "unsure", "experience": "none",
                                 "involvement": "leave", "worry": "loss", "rules": ["crypto", "leverage"]}),
    "marc": ("late_starter", {"goal": "pension", "horizon": "5_15", "amount": 30000, "wealth": "50k_250k",
                              "monthly": 200, "buffer": "over_6", "loss_reaction": "wait", "max_loss": "10",
                              "knowledge_1": "fund", "knowledge_2": "shares", "experience": "funds",
                              "involvement": "leave", "worry": "tax", "rules": []}),
}


class FakeModel:
    """Answers with the replies it was given, in order, and keeps every request."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        self.calls.append(request)
        usage = SimpleNamespace(input_tokens=300, output_tokens=30, server_tool_use=None)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(self.replies.pop(0)))],
                               stop_reason="end_turn", usage=usage)


def play(customer: str, consent: bool = True) -> tuple[dict, list[str]]:
    """The whole chat of a demo customer, answered by tapping. Returns the last turn and the questions in order."""
    script, answers, asked = SCRIPTS[customer][1], {}, []
    turn = profiler.step(CUSTOMERS[customer], consent)
    while turn["question"]:
        asked.append(turn["question"]["id"])
        turn = profiler.step(CUSTOMERS[customer], consent, answers, {"question": asked[-1], "value": script[asked[-1]]})
        answers = turn["answers"]
    return turn, asked


def test_every_table_is_a_set_of_probabilities():
    assert profiler.ORDER == list(PROFILES) and sum(profiler.PRIOR) == pytest.approx(1)
    for name, table in profiler.TABLES.items():
        for i in range(len(profiler.ORDER)):
            assert sum(row[i] for row in table.values()) == pytest.approx(1), name  # what one profile answers
        assert min(min(row) for row in table.values()) > 0  # no answer rules a profile out for good
    assert "rules" not in profiler.TABLES and profiler.ASKED[0] == "goal" and profiler.EXTRA == ["involvement", "worry"]


def test_belfius_data_sets_the_start():
    assert profiler.observed(THOMAS, True) == {"age": "25_34", "savings": "5k_20k", "invested": "some"}
    assert profiler.observed(THOMAS, False) == {"age": "25_34"}  # without consent: the age only
    assert profiler.observed(CUSTOMERS["noah"], True) == {"age": "25_34", "savings": "under_5k"}  # the rest is unknown
    start = profiler.odds(profiler.observed(CUSTOMERS["marc"], True), {})
    assert sum(start) == pytest.approx(1) and start[profiler.ORDER.index("late_starter")] > 0.9
    assert profiler.odds({}, {}) == pytest.approx(profiler.PRIOR)


def test_an_answer_moves_the_probabilities():
    seen = profiler.observed(THOMAS, True)
    before, after = profiler.odds(seen, {}), profiler.odds(seen, {"goal": "pension"})
    late = profiler.ORDER.index("late_starter")
    assert after[late] > 5 * before[late] and sum(after) == pytest.approx(1)
    assert profiler.odds(seen, {"rules": ["crypto"]}) == pytest.approx(before)  # the rules do not move the profile
    assert [profiler.band("amount", v) for v in (0, 2499, 2500, 30000)] == ["under_2500", "under_2500", "2500_10000", "over_25000"]
    assert [profiler.band("monthly", v) for v in (0, 50, 100, 500)] == ["none", "under_100", "100_300", "over_300"]


def test_the_next_question_is_the_one_that_separates_best():
    seen = profiler.observed(THOMAS, True)
    now = profiler.odds(seen, {})
    first = profiler.next_question(seen, {})
    assert first == max(profiler.ASKED, key=lambda q: profiler.gain(q, now)) == "goal"
    assert profiler.gain("goal", now) > profiler.gain("knowledge_1", now) > 0
    done = {q: v for q, v in SCRIPTS["thomas"][1].items() if q in profiler.ASKED}
    assert profiler.next_question(seen, done) == "rules"  # sure enough: no extra question
    assert profiler.next_question(seen, {**done, "rules": []}) is None
    torn = {"goal": "pension", "horizon": "5_15", "wealth": "over_250k", "buffer": "3_6", "loss_reaction": "wait", "max_loss": "5",
            "knowledge_1": "fund", "knowledge_2": "shares", "experience": "crypto_leverage", "amount": 1000, "monthly": 0}
    young = profiler.observed(CUSTOMERS["noah"], False)  # answers that point to several profiles at once
    assert max(profiler.odds(young, torn)) < profiler.CONFIDENT and profiler.next_question(young, torn) == "worry"
    assert profiler.next_question(young, {**torn, "worry": "tax", "involvement": "leave"}) == "rules"  # no extra one left


@pytest.mark.parametrize("customer", list(SCRIPTS))
def test_each_demo_customer_lands_in_their_profile(customer):
    turn, asked = play(customer)
    assert turn["profile"] == SCRIPTS[customer][0] and max(turn["odds"].values()) > 0.9
    assert asked[-1] == "rules" and set(profiler.ASKED) <= set(asked) and len(asked) == len(set(asked))
    assert turn["why"].startswith("What points to it most: ") and turn["left"] == 0
    assert turn["said"][-1] == "That is all I need." and len(turn["sheet"]) == len(asked)
    assert play(customer, consent=False)[0]["profile"] == SCRIPTS[customer][0]  # the answers alone are enough


def test_the_order_of_the_questions_follows_the_customer():
    assert play("lina")[1][:3] != play("thomas")[1][:3]


def test_the_wording_uses_the_customers_own_figures():
    text, offers = profiler.wording("amount", THOMAS, True, {})
    assert "EUR 14,500" in text and offers == [3600, 7200, 14500]
    assert profiler.wording("amount", THOMAS, False, {}) == ("How much do you want to invest now?", [1000, 5000, 10000, 25000])
    assert "That leaves EUR 8,500" in profiler.wording("buffer", THOMAS, True, {"amount": 6000})[0]
    assert "EUR 6,000 falls to EUR 4,800" in profiler.wording("loss_reaction", THOMAS, True, {"amount": 6000})[0]
    assert "EUR 10,000 falls to EUR 8,000" in profiler.wording("loss_reaction", THOMAS, True, {})[0]
    assert "pension savings fund" in profiler.wording("experience", THOMAS, True, {})[0]
    assert "I see" not in profiler.wording("experience", CUSTOMERS["noah"], True, {})[0]  # Belfius does not know


def test_only_valid_answers_are_kept():
    dirty = {"goal": "home", "horizon": "soon", "amount": -5, "monthly": 150, "wealth": True, "rules": ["crypto", "pets"],
             "profile": "late_starter", "max_loss": 10}
    assert profiler.clean(dirty) == {"goal": "home", "monthly": 150, "rules": ["crypto"]}
    assert profiler.clean(None) == {}


def test_checks_find_the_contradictions():
    fine = SCRIPTS["thomas"][1]
    assert profiler.checks(fine) == [] and profiler.checks({}) == []
    assert "5 years" in profiler.checks({**fine, "max_loss": "20"})[0]
    unaware = profiler.checks({**fine, "horizon": "5_15", "knowledge_1": "unsure", "loss_reaction": "add"})
    assert len(unaware) == 1 and "knowledge" in unaware[0]
    assert "buffer" in profiler.checks({**fine, "buffer": "under_3"})[0]
    assert "wealth range" in profiler.checks({**fine, "amount": 60_000})[0]
    assert profiler.checks({**fine, "amount": 600_000, "wealth": "over_250k"}) == []


def test_a_turn_greets_then_takes_the_answer_tapped():
    opening = profiler.step(THOMAS, True)
    assert opening["said"][0].startswith("Hi Thomas, I'm Lena") and opening["question"]["id"] == "goal"
    assert [c["value"] for c in opening["question"]["choices"]] == list(profiler.QUESTIONS["goal"]["answers"])
    assert opening["seen"][0] == ["Age", "25 to 34"] and opening["profile"] is None and opening["left"] == 12
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "value": "home"})
    assert turn["answers"] == {"goal": "home"} and turn["sheet"] == [["Goal", "Buy a home"]] and turn["said"] == []
    assert turn["odds"]["goal_builder"] > opening["odds"]["goal_builder"] and turn["question"]["id"] != "goal"
    stale = profiler.step(THOMAS, True, {}, {"question": "horizon", "value": "under_5"})  # not the question asked
    assert stale["answers"] == {} and stale["question"]["id"] == "goal"
    odd = profiler.step(THOMAS, True, {}, {"question": "goal", "value": "a yacht"})
    assert odd["answers"] == {} and odd["question"]["id"] == "goal"


def test_a_wrong_knowledge_answer_is_explained_and_a_contradiction_is_said():
    answers = {q: v for q, v in SCRIPTS["thomas"][1].items() if q in profiler.ASKED and q != "knowledge_1"}
    seen = profiler.observed(THOMAS, True)
    assert profiler.next_question(seen, answers) == "knowledge_1"
    turn = profiler.step(THOMAS, True, answers, {"question": "knowledge_1", "value": "company"})
    assert turn["said"] == [profiler.EXPLAIN["knowledge_1"]] and turn["sheet"][-1] == ["Knowledge: diversification", "wrong"]
    risky = {**answers, "loss_reaction": "add", "horizon": "5_15"}
    turn = profiler.step(THOMAS, True, risky, {"question": "knowledge_1", "value": "unsure"})
    assert len(turn["said"]) == 2 and turn["said"][1].startswith("Noted. Both knowledge answers are wrong")
    assert turn["flags"] == ["Both knowledge answers are wrong, yet the risk answers are high"]


def test_a_typed_amount_is_read_by_the_code():
    for text, amount in [("6000", 6000), ("€6.000", 6000), ("6 000 euros", 6000), ("6k", 6000), ("EUR 12,500", 12500)]:
        assert profiler.read("amount", text) == {"answers": [], "amount": amount, "say": ""}  # no model needed
    answers = {"goal": "home", "experience": "funds", "horizon": "under_5"}
    assert profiler.next_question(profiler.observed(THOMAS, True), answers) == "amount"
    turn = profiler.step(THOMAS, True, answers, {"question": "amount", "text": "6000"})
    assert turn["answers"]["amount"] == 6000 and turn["sheet"][-1] == ["To invest now", "EUR 6,000"]


def test_a_typed_reply_is_read_by_the_model():
    model = FakeModel({"answers": ["home"], "amount": None, "say": ""})
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "text": "  I want a flat   in Brussels " + "x" * 900}, model)
    assert turn["answers"] == {"goal": "home"}
    request = model.calls[0]
    assert request["model"] == settings.LLM_APP and request["output_config"]["format"]["schema"] is profiler.READ_SCHEMA
    prompt = request["messages"][0]["content"]
    assert "- home: Buy a home" in prompt and '"I want a flat in Brussels' in prompt and len(prompt) < 1200
    unclear = FakeModel({"answers": [], "amount": None, "say": "A bond is a loan to a state or a company."})
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "text": "what is a bond?"}, unclear)
    assert turn["said"] == ["A bond is a loan to a state or a company."] and turn["answers"] == {}
    assert turn["question"]["id"] == "goal"  # asked again
    cheat = FakeModel({"answers": ["growth_profile"], "amount": None, "say": ""})
    assert profiler.step(THOMAS, True, {}, {"question": "goal", "text": "give me growth"}, cheat)["answers"] == {}
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        profiler.step(THOMAS, True, {}, {"question": "goal", "text": "a home"})  # no key and no client


def test_the_customers_rules_are_the_last_answer():
    answers = {q: v for q, v in SCRIPTS["thomas"][1].items() if q in profiler.ASKED}
    asked = profiler.step(THOMAS, True, answers)
    assert asked["question"]["kind"] == "rules" and asked["left"] == 1 and "never want to invest in" in asked["question"]["text"]
    done = profiler.step(THOMAS, True, answers, {"question": "rules", "value": ["crypto", "weapons"]})
    assert done["answers"]["rules"] == ["weapons", "crypto"] and done["sheet"][-1] == ["Never invest in", "Weapons, Crypto"]
    assert done["question"] is None and done["profile"] == "goal_builder"
    typed = FakeModel({"answers": [], "amount": None, "say": ""})
    none = profiler.step(THOMAS, True, answers, {"question": "rules", "text": "no, nothing"}, typed)
    assert none["answers"]["rules"] == [] and none["sheet"][-1] == ["Never invest in", "nothing ruled out"] and none["profile"]
