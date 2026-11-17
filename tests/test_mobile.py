"""The customer app: what the coach may know, the plan of a profile, and the routes the phone calls."""
import re
from datetime import date

import pytest
from starlette.testclient import TestClient

from data import config as settings
from data.profiles import PROFILES
from mobile import server
from mobile.customers import ACCOUNT, BASICS, CUSTOMERS, known
from tests.test_profiler import FULL, FakeClaude, reply

DAY = date(2026, 12, 7)
LAST = reply("That is all I need.", profile="goal_builder", why="You save for a home.", preferences="none")
ALMOST = {k: v for k, v in FULL.items() if k != "preferences"}


def app(tmp_path, *replies):
    return TestClient(server.create_app(client=FakeClaude(*replies), logs=tmp_path))


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
    assert core == pytest.approx(PROFILES[profile]["core"])
    assert [line["sleeve"] for line in lines] == sorted(line["sleeve"] for line in lines)  # the core first
    assert {line["asset"] for line in lines} <= {"Shares", *server.ASSETS.values()}
    assert all(line["name"] and line["isin"] and "(" not in line["theme"] for line in lines)


def test_example_plans_stay_out_of_the_committed_logs():
    assert server.ILLUSTRATION and server.default_logs_dir() == settings.LOGS_DIR / "mock"


def test_start_gives_the_phone_what_it_shows(tmp_path):
    start = app(tmp_path).get("/api/start").json()
    assert [c["id"] for c in start["customers"]] == list(CUSTOMERS) and "savings_eur" not in start["customers"][0]
    assert start["profiles"] == PROFILES and start["coach"] == "Bel" and start["account_data"] == list(ACCOUNT.values())
    assert list(start["sheet"]) == list(FULL)


def test_from_the_chat_to_the_approval(tmp_path):
    phone = app(tmp_path, reply("What is it for?"), LAST)
    assert phone.get("/api/state", params={"customer": "thomas"}).json()["state"] is None
    assert phone.get("/api/plan", params={"customer": "thomas"}).status_code == 404  # no profile yet
    opening = phone.post("/api/chat", json={"customer": "thomas", "consent": True, "messages": []}).json()
    assert opening["message"] == "What is it for?" and opening["profile"] is None
    assert not (tmp_path / server.LOG_NAME).exists()  # nothing is written before a profile
    done = phone.post("/api/chat", json={"customer": "thomas", "consent": True, "sheet": ALMOST,
                                         "messages": [{"role": "customer", "text": "no wish"}]}).json()
    assert done["profile"] == "goal_builder"
    state = phone.get("/api/state", params={"customer": "thomas"}).json()["state"]
    assert state["profile"] == "goal_builder" and state["sheet"] == FULL and state["decision"] is None
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


def test_without_consent_the_coach_gets_no_account_data(tmp_path):
    claude = FakeClaude(reply("What is it for?"))
    TestClient(server.create_app(client=claude, logs=tmp_path)).post(
        "/api/chat", json={"customer": "thomas", "consent": False, "messages": []})
    assert "EUR 14,500" not in claude.calls[0]["system"] and "- Net income a month: not shared" in claude.calls[0]["system"]


def test_what_goes_wrong_is_said(tmp_path):
    phone = app(tmp_path)  # a coach with no reply left: the call fails
    failed = phone.post("/api/chat", json={"customer": "thomas", "messages": []})
    assert failed.status_code == 502 and failed.json()["error"].startswith("Bel cannot answer")
    assert phone.post("/api/chat", json={"customer": "nobody", "messages": []}).status_code == 404
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
