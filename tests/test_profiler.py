"""Lena, the coach: the archetype's probabilities, the next question, the risk level by rules, the checks, typed replies."""
import json
from types import SimpleNamespace

import pytest

from agents import profiler
from data import config as settings
from data.lessons import LESSONS
from data.profiles import ARCHETYPES, STRATEGIES
from mobile.customers import CUSTOMERS

THOMAS = CUSTOMERS["thomas"]
# What each demo customer answers, in character, the archetype it should lead to and the strategy the rules give.
# Noah keeps 3 to 6 months of buffer: a thinner one would hold him at Balanced.
SCRIPTS = {
    "lina": ("first_step_starter", "balanced", {"goal": "start", "horizon": "over_15", "amount": 700, "wealth": "under_10k",
                                    "monthly": 50, "buffer": "under_3", "loss_reaction": "wait", "max_loss": "10",
                                    "knowledge_1": "unsure", "knowledge_2": "shares", "experience": "none",
                                    "involvement": "check", "worry": "know", "rules": ["weapons"]}),
    "thomas": ("goal_builder", "defensive", {"goal": "home", "horizon": "under_5", "amount": 6000, "wealth": "10k_50k",
                                "monthly": 150, "buffer": "3_6", "loss_reaction": "wait", "max_loss": "10",
                                "knowledge_1": "fund", "knowledge_2": "bonds", "experience": "funds",
                                "involvement": "check", "worry": "plan", "rules": []}),
    "noah": ("self_directed_trader", "opportunity", {"goal": "grow", "horizon": "over_15", "amount": 2500, "wealth": "10k_50k",
                                      "monthly": 400, "buffer": "3_6", "loss_reaction": "add", "max_loss": "30",
                                      "knowledge_1": "fund", "knowledge_2": "shares", "experience": "crypto_leverage",
                                      "involvement": "pick", "worry": "cost", "rules": []}),
    "sofie": ("cautious_saver", "defensive", {"goal": "family", "horizon": "5_15", "amount": 8000, "wealth": "10k_50k",
                                 "monthly": 150, "buffer": "over_6", "loss_reaction": "sell", "max_loss": "5",
                                 "knowledge_1": "unsure", "knowledge_2": "unsure", "experience": "none",
                                 "involvement": "leave", "worry": "loss", "rules": ["crypto", "leverage"]}),
    "marc": ("late_starter", "balanced", {"goal": "pension", "horizon": "5_15", "amount": 30000, "wealth": "50k_250k",
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
    script, answers, asked = SCRIPTS[customer][2], {}, []
    turn = profiler.step(CUSTOMERS[customer], consent)
    while turn["question"]:
        asked.append(turn["question"]["id"])
        turn = profiler.step(CUSTOMERS[customer], consent, answers, {"question": asked[-1], "value": script[asked[-1]]})
        answers = turn["answers"]
    return turn, asked


def test_every_table_is_a_set_of_probabilities():
    assert profiler.ORDER == list(ARCHETYPES) and sum(profiler.PRIOR) == pytest.approx(1)
    for name, table in profiler.TABLES.items():
        for i in range(len(profiler.ORDER)):
            assert sum(row[i] for row in table.values()) == pytest.approx(1), name  # what one profile answers
        assert min(min(row) for row in table.values()) > 0  # no answer rules a profile out for good
    assert "rules" not in profiler.TABLES and profiler.ALWAYS[0] == "goal"
    assert set(profiler.ALWAYS) | set(profiler.OTHERS) == set(profiler.TABLES) - set(profiler.OBSERVED)


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
    assert profiler.required(seen) == profiler.ALWAYS  # Belfius shows his savings: no wealth question
    assert profiler.required(profiler.observed(THOMAS, False)) == [*profiler.ALWAYS, "wealth"]
    assert profiler.next_question(seen, {}) == max(profiler.ALWAYS, key=lambda q: profiler.gain(q, now)) == "goal"
    assert profiler.gain("goal", now) > profiler.gain("knowledge_1", now) > 0
    done = {q: v for q, v in SCRIPTS["thomas"][2].items() if q in profiler.ALWAYS}
    assert profiler.next_question(seen, done) == "rules"  # sure enough: nothing more is asked
    assert profiler.next_question(seen, {**done, "rules": []}) is None
    torn = {"goal": "pension", "horizon": "5_15", "amount": 1000, "loss_reaction": "wait", "knowledge_1": "fund"}
    young = profiler.observed(CUSTOMERS["noah"], True)  # answers that point to several profiles at once
    extra = profiler.next_question(young, torn)
    assert max(profiler.odds(young, torn)) < profiler.CONFIDENT and extra in profiler.OTHERS  # one more, the best one
    assert profiler.next_question(young, {**torn, extra: next(iter(profiler.QUESTIONS[extra]["answers"]))}) == "rules"  # no room left


@pytest.mark.parametrize("customer", list(SCRIPTS))
def test_each_demo_customer_lands_in_their_archetype_and_strategy(customer):
    turn, asked = play(customer)
    assert turn["archetype"] == SCRIPTS[customer][0] and max(turn["odds"].values()) > 0.8
    assert turn["strategy"] == SCRIPTS[customer][1] and turn["strategy_why"]
    assert asked[-1] == "rules" and set(profiler.ALWAYS) <= set(asked) and len(asked) == len(set(asked))
    assert len(asked) <= profiler.MAX_QUESTIONS  # a short chat, whoever the customer is
    assert turn["why"].startswith("What points to it most: ") and turn["left"] == 0
    assert turn["said"][-1] == "That is all I need." and len(turn["sheet"]) == len(asked)
    alone, questions = play(customer, consent=False)
    assert alone["archetype"] == SCRIPTS[customer][0] and "wealth" in questions  # the answers alone are enough
    assert alone["strategy"] == SCRIPTS[customer][1]
    assert len(questions) == profiler.MAX_QUESTIONS


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


def test_the_risk_level_follows_fixed_rules_on_the_answers():
    bold = {"horizon": "over_15", "loss_reaction": "add", "knowledge_1": "fund"}
    assert profiler.strategy(bold) == ("opportunity", ["You would invest more after a fall",
                                                      "You answered the knowledge question right",
                                                      "You do not need the money within 5 years"])
    assert profiler.strategy({**bold, "horizon": "under_5"}) == ("defensive", ["You need the money within 5 years"])
    assert profiler.strategy({**bold, "loss_reaction": "sell", "max_loss": "5"})[1] == [
        "You would sell after a fall of a fifth", "A 5% loss over a year would cost you sleep"]
    assert profiler.strategy({**bold, "loss_reaction": "wait"})[0] == "balanced"
    assert profiler.strategy({**bold, "knowledge_1": "unsure"}) == ("balanced", ["The knowledge question was missed"])
    assert profiler.strategy({**bold, "knowledge_1": "unsure", "knowledge_2": "shares"})[0] == "opportunity"  # one is enough
    assert profiler.strategy({**bold, "buffer": "under_3"})[0] == "balanced"  # a thin buffer holds it back
    assert profiler.strategy({**bold, "max_loss": "30"})[0] == "opportunity"
    assert set(STRATEGIES) == {"defensive", "balanced", "opportunity"}


def test_the_archetype_never_sets_the_risk_level():
    """Two customers with the same suitability answers get the same strategy, whatever their archetype."""
    lina, marc = play("lina")[0], play("marc")[0]
    assert lina["archetype"] != marc["archetype"]
    same = {q: SCRIPTS["lina"][2][q] for q in ("horizon", "loss_reaction", "knowledge_1")}
    assert profiler.strategy(same) == profiler.strategy({**same, "goal": "pension", "amount": 30000})


def test_the_check_on_complex_products_needs_every_answer_right():
    right = {key: spec["right"] for key, spec in profiler.QUIZ.items()}
    assert profiler.quiz(right) == {"passed": True, "right": 3, "notes": [
        profiler.QUIZ[key]["answers"][value][1] for key, value in right.items()]}
    one_wrong = profiler.quiz({**right, "leverage_day": "ten"})
    assert not one_wrong["passed"] and one_wrong["right"] == 2 and one_wrong["notes"][0].startswith("No:")
    assert profiler.quiz({}) == {"passed": False, "right": 0, "notes": ["No answer."] * 3}
    assert profiler.quiz("all right")["passed"] is False  # not a dict: nothing is right
    for spec in profiler.QUIZ.values():
        assert spec["right"] in spec["answers"] and all(why for _, why in spec["answers"].values())


def test_every_archetype_starts_with_three_real_lessons():
    for spec in ARCHETYPES.values():
        assert len(spec["lessons"]) == 3 and set(spec["lessons"]) <= set(LESSONS)
    firsts = [key for spec in ARCHETYPES.values() for key in spec["lessons"]]
    assert sorted(firsts) == sorted(LESSONS)  # each lesson is someone's first, none twice
    for key, lesson in LESSONS.items():
        assert lesson["title"] and lesson["text"] and lesson["question"].strip(), key
        assert lesson["right"] in lesson["answers"] and len(lesson["answers"]) == 3, key
        assert lesson["answers"][lesson["right"]][1].startswith("Right"), key
        assert all(why.startswith("No") for answer, (_, why) in lesson["answers"].items() if answer != lesson["right"]), key


def test_a_lesson_is_marked_with_the_reason():
    assert profiler.lesson("etf", "slice") == {"right": True, "why": LESSONS["etf"]["answers"]["slice"][1],
                                               "right_answer": "A small slice of about 1,400 companies"}
    wrong = profiler.lesson("etf", "loan")
    assert wrong["right"] is False and wrong["why"].startswith("No:") and wrong["right_answer"].startswith("A small slice")
    assert profiler.lesson("etf", "all of them") == {"right": False, "why": "No answer.",
                                                     "right_answer": "A small slice of about 1,400 companies"}
    assert profiler.lesson("etf", None)["right"] is False


def test_checks_find_the_contradictions():
    fine = SCRIPTS["thomas"][2]
    assert profiler.checks(fine) == [] and profiler.checks({}) == []
    assert "5 years" in profiler.checks({**fine, "max_loss": "20"})[0]
    assert "5 years" in profiler.checks({"horizon": "under_5", "loss_reaction": "add"})[0]  # with the short chat's answers
    unaware = profiler.checks({**fine, "horizon": "5_15", "knowledge_1": "unsure", "loss_reaction": "add"})
    assert len(unaware) == 1 and "knowledge" in unaware[0]
    assert profiler.checks({"knowledge_1": "company", "loss_reaction": "add"}) == unaware  # one knowledge question is enough
    assert profiler.checks({"knowledge_1": "fund", "knowledge_2": "bonds", "loss_reaction": "add"}) == []
    assert "buffer" in profiler.checks({**fine, "buffer": "under_3"})[0]
    assert "wealth range" in profiler.checks({**fine, "amount": 60_000})[0]
    assert profiler.checks({**fine, "amount": 600_000, "wealth": "over_250k"}) == []


def test_a_turn_greets_then_takes_the_answer_tapped():
    opening = profiler.step(THOMAS, True)
    assert opening["said"][0].startswith("Hi Thomas, I'm Lena") and opening["question"]["id"] == "goal"
    assert [c["value"] for c in opening["question"]["choices"]] == list(profiler.QUESTIONS["goal"]["answers"])
    assert opening["seen"][0] == ["Age", "25 to 34"] and opening["archetype"] is None and opening["left"] == 7
    assert opening["strategy"] is None and opening["strategy_why"] == []
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "value": "home"})
    assert turn["answers"] == {"goal": "home"} and turn["sheet"] == [["Goal", "A home or a big purchase"]] and turn["said"] == []
    assert turn["odds"]["goal_builder"] > opening["odds"]["goal_builder"] and turn["question"]["id"] != "goal"
    stale = profiler.step(THOMAS, True, {}, {"question": "horizon", "value": "under_5"})  # not the question asked
    assert stale["answers"] == {} and stale["question"]["id"] == "goal"
    odd = profiler.step(THOMAS, True, {}, {"question": "goal", "value": "a yacht"})
    assert odd["answers"] == {} and odd["question"]["id"] == "goal"


def test_a_wrong_knowledge_answer_is_explained_and_a_contradiction_is_said():
    answers = {q: v for q, v in SCRIPTS["thomas"][2].items() if q in profiler.ALWAYS and q != "knowledge_1"}
    seen = profiler.observed(THOMAS, True)
    assert profiler.next_question(seen, answers) == "knowledge_1"
    turn = profiler.step(THOMAS, True, answers, {"question": "knowledge_1", "value": "company"})
    assert turn["said"] == [profiler.EXPLAIN["knowledge_1"]] and turn["sheet"][-1] == ["Knowledge: diversification", "wrong"]
    risky = {**answers, "loss_reaction": "add", "horizon": "5_15"}
    turn = profiler.step(THOMAS, True, risky, {"question": "knowledge_1", "value": "unsure"})
    assert len(turn["said"]) == 2 and turn["said"][1].startswith("Noted. No knowledge question answered right")
    assert turn["flags"] == ["No knowledge question answered right, yet the risk answers are high"]


def test_a_typed_amount_is_read_by_the_code():
    for text, amount in [("6000", 6000), ("€6.000", 6000), ("6 000 euros", 6000), ("6k", 6000), ("EUR 12,500", 12500)]:
        assert profiler.read("amount", text) == {"answers": [], "amount": amount, "say": ""}  # no model needed
    answers = {"goal": "home", "horizon": "under_5", "loss_reaction": "wait", "knowledge_1": "fund"}
    assert profiler.next_question(profiler.observed(THOMAS, True), answers) == "amount"
    turn = profiler.step(THOMAS, True, answers, {"question": "amount", "text": "6000"})
    assert turn["answers"]["amount"] == 6000 and turn["sheet"][-1] == ["To invest now", "EUR 6,000"]


def test_the_code_reads_plain_typed_replies_at_once():
    """No model call: these come back as fast as a tap."""
    read = profiler.understood
    assert read("rules", "No, nothing") == {"answers": [], "amount": None, "say": ""}
    assert read("rules", "no fossil fuels please")["answers"] == ["fossil"]
    assert read("rules", "Crypto and weapons")["answers"] == ["weapons", "crypto"]
    assert read("rules", "pas de pétrole ni d'armes")["answers"] == ["weapons", "fossil"]
    assert read("rules", "car") is None and read("rules", "crypto is fine") is None  # left to the model
    assert read("goal", "a home or a big purchase!")["answers"] == ["home"] and read("goal", "for my kids") is None
    assert read("horizon", "in 8 years")["answers"] == ["5_15"] and read("horizon", "20 ans")["answers"] == ["over_15"]
    assert read("horizon", "3 years")["answers"] == ["under_5"]
    assert read("max_loss", "10%")["answers"] == ["10"] and read("max_loss", "40 %")["answers"] == ["30"]
    assert read("amount", "6k")["amount"] == 6000 and read("knowledge_1", "fund")["answers"] == ["fund"]
    assert profiler.read("rules", "no", client="never called")["answers"] == []  # the model is not asked


def test_the_code_names_the_language_of_a_reply_for_the_model():
    assert [profiler.language(t) for t in ("I don't know yet", "hmm", "what's an ETF?", "Buy a car")] == ["English"] * 4
    assert [profiler.language(t) for t in ("je sais pas encore", "pour ma retraite", "non", "de rien")] == ["French"] * 4
    assert [profiler.language(t) for t in ("ik weet het niet", "nee")] == ["Dutch"] * 2
    model = FakeModel({"answers": [], "amount": None, "say": "Pas de souci."})
    profiler.read("goal", "je sais pas encore", model)
    assert model.calls[0]["messages"][0]["content"].endswith("Write `say` in French.")


def test_a_reply_nobody_could_read_still_gets_a_word():
    silent = FakeModel({"answers": [], "amount": None, "say": ""})
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "text": "hmm"}, silent)
    assert turn["said"] == ["I did not catch that. Tap one of the answers below."] and turn["question"]["id"] == "goal"
    answers = {q: v for q, v in SCRIPTS["thomas"][2].items() if q in profiler.ALWAYS}
    outside = FakeModel({"answers": [], "amount": None, "say": "Cars sit inside broad ETFs, so I cannot leave them out."})
    turn = profiler.step(THOMAS, True, answers, {"question": "rules", "text": "car"}, outside)
    assert turn["said"] == ["Cars sit inside broad ETFs, so I cannot leave them out."] and "rules" not in turn["answers"]
    assert "cannot be left out on its own" in outside.calls[0]["messages"][0]["content"]  # the question's note
    assert "never by their key" in outside.calls[0]["system"]


def test_a_typed_reply_is_read_by_the_model():
    model = FakeModel({"answers": ["home"], "amount": None, "say": ""})
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "text": "  I want a flat   in Brussels " + "x" * 900}, model)
    assert turn["answers"] == {"goal": "home"}
    request = model.calls[0]
    assert request["model"] == settings.LLM_APP and request["output_config"]["format"]["schema"] is profiler.READ_SCHEMA
    prompt = request["messages"][0]["content"]
    assert "- home: A home or a big purchase" in prompt and '"I want a flat in Brussels' in prompt and len(prompt) < 1200
    unclear = FakeModel({"answers": [], "amount": None, "say": "A bond is a loan to a state or a company."})
    turn = profiler.step(THOMAS, True, {}, {"question": "goal", "text": "what is a bond?"}, unclear)
    assert turn["said"] == ["A bond is a loan to a state or a company."] and turn["answers"] == {}
    assert turn["question"]["id"] == "goal"  # asked again
    cheat = FakeModel({"answers": ["growth_profile"], "amount": None, "say": ""})
    assert profiler.step(THOMAS, True, {}, {"question": "goal", "text": "give me growth"}, cheat)["answers"] == {}
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        profiler.step(THOMAS, True, {}, {"question": "goal", "text": "a home"})  # no key and no client


def test_the_customers_rules_are_the_last_answer():
    answers = {q: v for q, v in SCRIPTS["thomas"][2].items() if q in profiler.ALWAYS}
    asked = profiler.step(THOMAS, True, answers)
    assert asked["question"]["kind"] == "rules" and asked["left"] == 2 and "never want to invest in" in asked["question"]["text"]
    done = profiler.step(THOMAS, True, answers, {"question": "rules", "value": ["crypto", "weapons"]})
    assert done["answers"]["rules"] == ["weapons", "crypto"] and done["sheet"][-1] == ["Never invest in", "Weapons, Crypto"]
    assert done["question"] is None and done["archetype"] == "goal_builder" and done["strategy"] == "defensive"
    typed = FakeModel({"answers": [], "amount": None, "say": ""})
    none = profiler.step(THOMAS, True, answers, {"question": "rules", "text": "no, nothing"}, typed)
    assert none["answers"]["rules"] == [] and none["sheet"][-1] == ["Never invest in", "nothing ruled out"] and none["archetype"]


def test_the_learning_streak_counts_days_in_a_row_with_a_lesson():
    week = ["2026-11-14", "2026-11-16", "2026-11-17", "2026-11-18"]
    assert profiler.streak(week, "2026-11-18") == {"days": 3, "today": True,
                                                   "week": [False, False, True, False, True, True, True]}
    assert profiler.streak(week, "2026-11-19")["days"] == 3  # not done yet today: the streak still stands
    assert profiler.streak(week, "2026-11-20") == {"days": 0, "today": False, "week": [True, False, True, True, True, False, False]}
    assert profiler.streak([], "2026-11-20")["days"] == 0
