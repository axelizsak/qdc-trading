"""The customer app: what the coach may know, the plan of a profile, and the routes the phone calls."""
import re
from datetime import date

import pytest
from starlette.testclient import TestClient

from agents import profiler
from data import config as settings
from data.profiles import PROFILES
from mobile import server
from mobile.customers import ACCOUNT, BASICS, CUSTOMERS, known
from tests.test_profiler import SCRIPTS, FakeModel

DAY = date(2026, 12, 7)


def app(tmp_path, *replies):
    return TestClient(server.create_app(client=FakeModel(*replies), logs=tmp_path))


def chat(phone, customer: str, answers: dict, reply: dict | None = None) -> dict:
    return phone.post("/api/chat", json={"customer": customer, "consent": True, "answers": answers, "reply": reply}).json()


def test_the_coach_knows_only_what_the_customer_shares():
    shared, kept = known("thomas", True), known("thomas", False)
    assert list(shared) == list(kept) == [*BASICS.values(), *ACCOUNT.values()]
    assert shared["On the savings account"] == "EUR 14,500" and shared["Age"] == "29"
    assert all(kept[label] == "not shared" for label in ACCOUNT.values()) and kept["First name"] == "Thomas"
    assert known("noah", True)["Net income a month"] == "not known"  # new customer: Belfius does not have it
    assert all(set(c) == {*BASICS, *ACCOUNT, "city"} for c in CUSTOMERS.values())


@pytest.mark.parametrize("profile", list(PROFILES))
def test_a_plan_splits_the_whole_amount(profile):
    plan = server.plan(profile, 10_000, DAY)
    lines = plan["lines"]
    assert sum(line["eur"] for line in lines) == pytest.approx(10_000) and plan["illustration"] is server.ILLUSTRATION
    core = sum(line["weight"] for line in lines if line["sleeve"] == "core")
    assert core == pytest.approx(PROFILES[profile]["core"]) and plan["left_out"] == []
    assert [line["sleeve"] for line in lines] == sorted(line["sleeve"] for line in lines)  # the core first
    assert {line["asset"] for line in lines} <= {"Shares", *server.ASSETS.values()}
    assert all(line["name"] and line["isin"] and "(" not in line["theme"] for line in lines)


def test_the_customers_rules_leave_etfs_out_of_the_plan():
    assert set(server.RULED_OUT) == set(profiler.QUESTIONS["rules"]["answers"])
    free, ruled = server.plan("self_directed_trader", 10_000, DAY), server.plan("self_directed_trader", 10_000, DAY, ["weapons", "crypto"])
    assert {"BITCOIN", "DEFENSE"} <= {line["ticker"] for line in free["lines"]}
    assert not {"BITCOIN", "DEFENSE"} & {line["ticker"] for line in ruled["lines"]}
    assert sorted((out["ticker"], out["rule"]) for out in ruled["left_out"]) == [("BITCOIN", "Crypto"), ("DEFENSE", "Weapons")]
    assert sum(line["eur"] for line in ruled["lines"]) == pytest.approx(10_000)  # their share went to the largest core ETF
    assert ruled["lines"][0]["weight"] == pytest.approx(free["lines"][0]["weight"] + 0.10)
    assert server.plan("goal_builder", 10_000, DAY, ["leverage"])["left_out"] == []  # nothing of the kind in this plan


def test_example_plans_stay_out_of_the_committed_logs():
    assert server.ILLUSTRATION and server.default_logs_dir() == settings.LOGS_DIR / "mock"


def test_start_gives_the_phone_what_it_shows(tmp_path):
    start = app(tmp_path).get("/api/start").json()
    assert [c["id"] for c in start["customers"]] == list(CUSTOMERS) and "savings_eur" not in start["customers"][0]
    assert start["profiles"] == PROFILES and start["coach"] == "Bel" and start["account_data"] == list(ACCOUNT.values())
    assert start["questions"] == len(profiler.ASKED) + 1


def test_from_the_chat_to_the_approval(tmp_path):
    phone, script = app(tmp_path), SCRIPTS["thomas"][1] | {"rules": ["crypto"]}
    assert phone.get("/api/state", params={"customer": "thomas"}).json()["state"] is None
    assert phone.get("/api/plan", params={"customer": "thomas"}).status_code == 404  # no profile yet
    turn = chat(phone, "thomas", {})
    assert turn["said"][0].startswith("Hi Thomas") and turn["question"]["id"] == "goal"
    while turn["question"]:
        assert not (tmp_path / server.LOG_NAME).exists()  # nothing is written before a profile
        question = turn["question"]["id"]
        turn = chat(phone, "thomas", turn["answers"], {"question": question, "value": script[question]})
    assert turn["profile"] == "goal_builder"
    state = phone.get("/api/state", params={"customer": "thomas"}).json()["state"]
    assert state["profile"] == "goal_builder" and state["answers"] == turn["answers"] and state["decision"] is None
    assert state["sheet"] == turn["sheet"] and state["why"] == turn["why"]
    plan = phone.get("/api/plan", params={"customer": "thomas"}).json()
    assert plan["amount"] == 6000 and sum(line["eur"] for line in plan["lines"]) == pytest.approx(6000)
    assert phone.post("/api/decision", json={"customer": "thomas", "decision": "sold"}).status_code == 400
    approved = phone.post("/api/decision", json={"customer": "thomas", "decision": "approved"}).json()["state"]
    assert approved["decision"] == "approved" and approved["decided_at"]
    log = server.read_log(tmp_path)
    assert list(log["event"]) == ["profiled", "approved"] and set(log["profile"]) == {"goal_builder"}
    assert phone.get("/api/state", params={"customer": "lina"}).json()["state"] is None  # each customer their own
    assert phone.post("/api/reset", json={"customer": "thomas"}).json() == {"state": None}
    assert phone.get("/api/state", params={"customer": "thomas"}).json()["state"] is None
    assert list(server.read_log(tmp_path)["event"]) == ["profiled", "approved", "reset"]  # nothing is erased


def test_a_typed_answer_goes_through_the_model(tmp_path):
    phone = app(tmp_path, {"answers": ["pension"], "amount": None, "say": ""})
    turn = chat(phone, "marc", {}, {"question": "goal", "text": "for when I stop working"})
    assert turn["answers"] == {"goal": "pension"}


def test_what_goes_wrong_is_said(tmp_path):
    phone = app(tmp_path)  # a model with no reply left: reading a typed answer fails
    failed = phone.post("/api/chat", json={"customer": "thomas", "answers": {}, "reply": {"question": "goal", "text": "a home"}})
    assert failed.status_code == 502 and failed.json()["error"].startswith("Bel cannot read typed answers")
    assert phone.post("/api/chat", json={"customer": "nobody", "answers": {}}).status_code == 404
    assert phone.get("/api/state", params={"customer": "nobody"}).status_code == 404
    assert phone.post("/api/decision", json={"customer": "thomas", "decision": "approved"}).status_code == 400
    assert phone.post("/api/reset", json={"customer": "nobody"}).status_code == 404


def test_the_page_calls_routes_that_exist(tmp_path):
    phone = app(tmp_path)
    page = phone.get("/")
    assert page.status_code == 200 and 'src="app.js"' in page.text and 'href="app.css"' in page.text
    script = phone.get("/app.js").text
    assert phone.get("/app.css").status_code == 200
    routes = {route.path for route in phone.app.routes}
    called = set(re.findall(r'"(/api/\w+)|`(/api/\w+)', script))
    assert called and {a or b for a, b in called} <= routes
