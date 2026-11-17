"""The customer app: what the coach may know, the plan of a profile, and the routes the phone calls."""
import re
from types import SimpleNamespace

import pandas as pd
import pytest
from starlette.testclient import TestClient

from agents import profiler
from data import config as settings
from data import mock
from data.profiles import PROFILES, SLEEVES
from graph import morning
from mobile import server
from mobile.customers import ACCOUNT, BASICS, CUSTOMERS, known
from tests.test_profiler import SCRIPTS, FakeModel



def app(tmp_path, *replies):
    return TestClient(server.create_app(client=FakeModel(*replies), logs=tmp_path, shared=tmp_path))


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
    plan = server.plan(profile, 10_000)
    lines = plan["lines"]
    assert sum(line["eur"] for line in lines) == pytest.approx(10_000) and plan["left_out"] == []
    order = [list(SLEEVES).index(line["sleeve"]) for line in lines]
    assert order == sorted(order) and lines[0]["ticker"] == "WORLD"  # sleeve by sleeve, the largest core ETF first
    for sleeve, share in PROFILES[profile]["sleeves"].items():
        assert sum(line["weight"] for line in lines if line["sleeve"] == sleeve) == pytest.approx(share)
    assert all(line["name"] and line["isin"] and "(" not in line["theme"] for line in lines)


def test_the_customers_rules_leave_etfs_out_of_the_plan():
    assert set(server.RULED_OUT) == set(profiler.QUESTIONS["rules"]["answers"])
    free, ruled = server.plan("self_directed_trader", 10_000), server.plan("self_directed_trader", 10_000, ["weapons", "crypto"])
    assert {"BITCOIN", "DEFENSE"} <= {line["ticker"] for line in free["lines"]}
    assert not {"BITCOIN", "DEFENSE"} & {line["ticker"] for line in ruled["lines"]}
    assert sorted((out["ticker"], out["rule"]) for out in ruled["left_out"]) == [("BITCOIN", "Crypto"), ("DEFENSE", "Weapons")]
    assert sum(line["eur"] for line in ruled["lines"]) == pytest.approx(10_000)  # their share went to the largest core ETF
    assert ruled["lines"][0]["weight"] == pytest.approx(free["lines"][0]["weight"] + 0.12)
    assert server.plan("goal_builder", 10_000, ["leverage"])["left_out"] == []  # nothing of the kind in this plan


def test_demo_customers_stay_out_of_the_committed_logs():
    assert server.customer_logs_dir() == settings.LOGS_DIR / "mock"


def test_an_etf_sheet_comes_from_the_morning_run(tmp_path):
    phone = app(tmp_path, "unused")
    missing = phone.get("/api/etf", params={"ticker": "WORLD"})
    assert missing.status_code == 404 and "morning run" in missing.json()["error"]
    pd.concat([mock.mock_etf_sheets("2026-12-04"), mock.mock_etf_sheets("2026-12-07")]).to_csv(tmp_path / morning.SHEETS, index=False)
    sheet = phone.get("/api/etf", params={"ticker": "WORLD"}).json()
    assert sheet["as_of"] == "2026-12-07" and sheet["name"].startswith("iShares Core MSCI World") and "rows" not in sheet
    assert [line["block"] for line in sheet["lines"]].count("trend") == 3 and sheet["lines"][0] == {
        "block": "returns", "name": "1 month", "value": 0.012, "state": "", "detail": ""}
    assert phone.get("/api/etf", params={"ticker": "NOT AN ETF"}).status_code == 404
    assert phone.post("/api/why", json={"ticker": "WORLD", "block": "trend", "name": "sideways"}).status_code == 404


def test_why_asks_the_model_about_one_line(tmp_path):
    mock.mock_etf_sheets().to_csv(tmp_path / morning.SHEETS, index=False)
    model = FakeModel()
    model.messages.create = lambda **request: (model.calls.append(request), SimpleNamespace(
        content=[SimpleNamespace(type="text", text="It is just under its average.")], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=200, output_tokens=20, server_tool_use=None)))[1]
    phone = TestClient(server.create_app(client=model, logs=tmp_path, shared=tmp_path))
    reply = phone.post("/api/why", json={"ticker": "WORLD", "block": "trend", "name": "short"})
    assert reply.json() == {"answer": "It is just under its average."}
    assert "line 'short' of this sheet read neutral" in model.calls[0]["messages"][0]["content"]
    down = TestClient(server.create_app(client=FakeModel(), logs=tmp_path, shared=tmp_path))  # a model with no reply
    failed = down.post("/api/why", json={"ticker": "WORLD", "block": "trend", "name": "short"})
    assert failed.status_code == 502 and failed.json()["error"] == "Lena cannot explain right now."


def test_start_gives_the_phone_what_it_shows(tmp_path):
    start = app(tmp_path).get("/api/start").json()
    assert [c["id"] for c in start["customers"]] == list(CUSTOMERS) and "savings_eur" not in start["customers"][0]
    assert start["profiles"] == PROFILES and start["coach"] == "Lena" and start["account_data"] == list(ACCOUNT.values())
    assert start["questions"] == len(profiler.ASKED) + 1 and start["sleeves"] == SLEEVES


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
    assert failed.status_code == 502 and failed.json()["error"].startswith("Lena cannot read typed answers")
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
