"""The customer app: what the coach may know, the reference portfolios, the ETF sheets, the customer's own orders, and
the routes the phone calls."""
import re
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from starlette.testclient import TestClient

from agents import model_portfolios
from data import config as settings
from data import mock
from data.lessons import LESSONS
from data.profiles import ARCHETYPES, SLEEVES, STRATEGIES, TRADED
from graph import morning
from mobile import server, wallet
from mobile.customers import ACCOUNT, BASICS, CUSTOMERS, known
from tests.test_profiler import SCRIPTS, FakeModel


def app(tmp_path, *replies):
    return TestClient(server.create_app(client=FakeModel(*replies), logs=tmp_path, shared=tmp_path))


def chat(phone, customer: str, answers: dict, reply: dict | None = None) -> dict:
    return phone.post("/api/chat", json={"customer": customer, "consent": True, "answers": answers, "reply": reply}).json()


def place(phone, customer: str, script: dict) -> dict:
    """The whole chat of a customer, answered by tapping. Returns the last turn."""
    turn = chat(phone, customer, {})
    while turn["question"]:
        question = turn["question"]["id"]
        turn = chat(phone, customer, turn["answers"], {"question": question, "value": script[question]})
    return turn


def morning_run(folder) -> None:
    """What the morning run leaves for the app, from the mocks: sheets, chart points, analysts' views."""
    pd.concat([mock.mock_etf_sheets("2026-12-04"), mock.mock_etf_sheets("2026-12-07")]).to_csv(folder / morning.SHEETS, index=False)
    mock.mock_chart_history("2026-12-07").to_csv(folder / morning.HISTORY, index=False)
    mock.mock_etf_views("2026-12-07").to_csv(folder / server.VIEWS, index=False)


def test_the_coach_knows_only_what_the_customer_shares():
    shared, kept = known("thomas", True), known("thomas", False)
    assert list(shared) == list(kept) == [*BASICS.values(), *ACCOUNT.values()]
    assert shared["On the savings account"] == "EUR 14,500" and shared["Age"] == "29"
    assert all(kept[label] == "not shared" for label in ACCOUNT.values()) and kept["First name"] == "Thomas"
    assert known("noah", True)["Net income a month"] == "not known"  # new customer: Belfius does not have it
    assert all(set(c) == {*BASICS, *ACCOUNT, "city"} for c in CUSTOMERS.values())


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_a_reference_portfolio_splits_the_whole_amount(strategy):
    ref = server.reference(strategy, 10_000)
    lines = ref["lines"]
    assert sum(line["eur"] for line in lines) == pytest.approx(10_000) and ref["left_out"] == []
    order = [list(SLEEVES).index(line["sleeve"]) for line in lines]
    assert order == sorted(order) and lines[0]["ticker"] == "WORLD"  # sleeve by sleeve, the largest core ETF first
    for sleeve, share in STRATEGIES[strategy]["sleeves"].items():
        assert sum(line["weight"] for line in lines if line["sleeve"] == sleeve) == pytest.approx(share)
    assert sum(s["weight"] for s in ref["sleeves"]) == pytest.approx(1) and all(s["weight"] > 0 for s in ref["sleeves"])
    assert all(line["name"] and line["isin"] and line["family"] and "(" not in line["theme"] for line in lines)
    assert ref["history"] == [] and all(line["returns"] == {} for line in lines)  # no morning run given


def test_the_customers_rules_leave_etfs_out_of_the_reference():
    free = server.reference("opportunity", 10_000)
    ruled = server.reference("opportunity", 10_000, ["weapons", "crypto"])
    assert {"BITCOIN", "DEFENSE"} <= {line["ticker"] for line in free["lines"]}
    assert not {"BITCOIN", "DEFENSE"} & {line["ticker"] for line in ruled["lines"]}
    assert sorted((out["ticker"], out["rule"]) for out in ruled["left_out"]) == [("BITCOIN", "Crypto"), ("DEFENSE", "Weapons")]
    assert sum(line["eur"] for line in ruled["lines"]) == pytest.approx(10_000)  # their share went to the largest core ETF
    assert ruled["lines"][0]["weight"] == pytest.approx(free["lines"][0]["weight"] + 0.12)
    assert server.reference("balanced", 10_000, ["leverage"])["left_out"] == []  # nothing of the kind in it


def test_demo_customers_stay_out_of_the_committed_logs():
    assert server.customer_logs_dir() == settings.LOGS_DIR / "mock"


def test_an_etf_sheet_comes_from_the_morning_run_with_its_chart_news_and_reading(tmp_path):
    phone = app(tmp_path, "unused")
    missing = phone.get("/api/etf", params={"ticker": "WORLD"})
    assert missing.status_code == 404 and "morning run" in missing.json()["error"]
    morning_run(tmp_path)
    sheet = phone.get("/api/etf", params={"ticker": "WORLD"}).json()
    assert sheet["as_of"] == "2026-12-07" and sheet["name"].startswith("iShares Core MSCI World") and "rows" not in sheet
    assert [line["block"] for line in sheet["lines"]].count("trend") == 3 and sheet["lines"][0] == {
        "block": "returns", "name": "1 month", "label": "1 month", "value": 0.012, "state": "", "detail": ""}
    assert [line["label"] for line in sheet["lines"] if line["block"] == "macro"] == ["Growth", "Fed / US rates"]
    assert len(sheet["history"]) > 50 and sheet["history"] == sorted(sheet["history"], key=lambda p: p["date"])
    assert sheet["news"]["direction"] == pytest.approx(0.27) and sheet["news"]["counter"]
    assert all(url.startswith("http") for url in sheet["news"]["sources"])
    assert [r["topic"] for r in sheet["reading"]] == ["Trend", "Macro", "News"]
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
    assert start["strategies"] == STRATEGIES and start["archetypes"] == ARCHETYPES and start["traded"] == TRADED == "opportunity"
    model_portfolios.choose(tmp_path, date(2026, 1, 5), "balanced", "the cockpit chose another")
    assert app(tmp_path).get("/api/start").json()["traded"] == "balanced"
    assert start["coach"] == "Lena" and start["account_data"] == list(ACCOUNT.values()) and start["sleeves"] == SLEEVES
    assert len(start["quiz"]) == 3 and all("right" not in q and len(q["choices"]) == 3 for q in start["quiz"])  # no answer key
    assert set(start["lessons"]) == set(LESSONS) and all(
        set(lesson) == {"title", "text", "question", "choices"} for lesson in start["lessons"].values())  # no answer key


def test_from_the_chat_to_the_customers_own_orders(tmp_path):
    phone, script = app(tmp_path), SCRIPTS["thomas"][2] | {"rules": ["crypto"]}
    morning_run(tmp_path)
    assert phone.get("/api/state", params={"customer": "thomas"}).json()["state"] is None
    for route in ("/api/reference", "/api/etfs", "/api/portfolio"):
        assert phone.get(route, params={"customer": "thomas"}).status_code == 404  # no risk level yet
    turn = chat(phone, "thomas", {})
    assert turn["said"][0].startswith("Hi Thomas") and turn["question"]["id"] == "goal"
    while turn["question"]:
        assert not (tmp_path / server.LOG_NAME).exists()  # nothing is written before the result
        question = turn["question"]["id"]
        turn = chat(phone, "thomas", turn["answers"], {"question": question, "value": script[question]})
    assert (turn["archetype"], turn["strategy"]) == ("goal_builder", "defensive")
    state = phone.get("/api/state", params={"customer": "thomas"}).json()["state"]
    assert state["archetype"] == "goal_builder" and state["strategy"] == "defensive" and state["answers"] == turn["answers"]
    assert state["why"] == turn["why"] and state["strategy_why"] == turn["strategy_why"] and state["sheet"] == turn["sheet"]
    assert state["orders"] == [] and state["unlocked"] is False and state["lessons"] == {}

    ref = phone.get("/api/reference", params={"customer": "thomas"}).json()
    assert ref["strategy"] == "defensive" and ref["amount"] == 6000 and sum(l["eur"] for l in ref["lines"]) == pytest.approx(6000)
    assert all(line["lock"] == "" for line in ref["lines"]) and ref["history"]  # the defensive reference holds nothing complex
    etfs = {etf["ticker"]: etf for etf in phone.get("/api/etfs", params={"customer": "thomas"}).json()["etfs"]}
    assert len(etfs) == 50 and etfs["BITCOIN"]["lock"] == "You ruled out crypto"
    assert etfs["NASDAQ 3X"]["lock"] == "Not part of the Defensive risk level" and etfs["WORLD"]["lock"] == ""
    assert etfs["WORLD"]["returns"] == {"1 month": 0.012, "1 year": 0.094}

    bad = phone.post("/api/order", json={"customer": "thomas", "orders": [{"ticker": "NASDAQ 3X", "eur": 100}]})
    assert bad.status_code == 400 and "Defensive" in bad.json()["error"]
    assert phone.post("/api/order", json={"customer": "thomas", "orders": []}).status_code == 400
    assert phone.post("/api/order", json={"customer": "thomas", "orders": [{"ticker": "WORLD", "eur": 0}]}).status_code == 400
    assert not (tmp_path / server.LOG_NAME).read_text().count(",order,")  # nothing was placed
    placed = phone.post("/api/order", json={"customer": "thomas", "orders": [
        {"ticker": "WORLD", "eur": 1000}, {"ticker": "GOLD", "eur": 250.555}]}).json()
    orders = placed["state"]["orders"]
    assert [(o["ticker"], o["eur"]) for o in orders] == [("WORLD", 1000), ("GOLD", 250.56)]
    world_price = server.last_prices(tmp_path)["WORLD"]["price"]
    assert orders[0]["units"] == pytest.approx(1000 / world_price, abs=1e-6) and orders[0]["as_of"] == "2026-12-04"
    held = placed["portfolio"]
    assert held["invested"] == pytest.approx(1250.56) and held["value"] == pytest.approx(1250.56, abs=0.01)
    assert {line["ticker"] for line in held["lines"]} == {"WORLD", "GOLD"} and held["history"] and held["health"]
    assert phone.get("/api/portfolio", params={"customer": "thomas"}).json() == held

    log = server.read_log(tmp_path)
    assert list(log["event"]) == ["profiled", "order", "order"] and set(log["strategy"]) == {"defensive"}
    assert phone.get("/api/state", params={"customer": "lina"}).json()["state"] is None  # each customer their own
    assert phone.post("/api/reset", json={"customer": "thomas"}).json() == {"state": None}
    assert phone.get("/api/state", params={"customer": "thomas"}).json()["state"] is None
    assert list(server.read_log(tmp_path)["event"]) == ["profiled", "order", "order", "reset"]  # nothing is erased


def test_the_check_opens_complex_products_at_the_opportunity_level_only(tmp_path):
    phone = app(tmp_path)
    morning_run(tmp_path)
    assert place(phone, "noah", SCRIPTS["noah"][2])["strategy"] == "opportunity"
    locked = {l["ticker"]: l["lock"] for l in phone.get("/api/reference", params={"customer": "noah"}).json()["lines"]}
    assert locked["BITCOIN"] == "Pass the 3-question check to unlock" and locked["WORLD"] == ""
    assert phone.post("/api/order", json={"customer": "noah", "orders": [{"ticker": "BITCOIN", "eur": 100}]}).status_code == 400
    wrong = phone.post("/api/quiz", json={"customer": "noah", "answers": {"leverage_day": "ten"}}).json()
    assert wrong["passed"] is False and wrong["state"]["unlocked"] is False
    right = {"leverage_day": "thirty", "leverage_hold": "reset", "crypto_fall": "yes"}
    passed = phone.post("/api/quiz", json={"customer": "noah", "answers": right}).json()
    assert passed["passed"] and passed["state"]["unlocked"] is True
    assert phone.post("/api/order", json={"customer": "noah", "orders": [{"ticker": "BITCOIN", "eur": 100}]}).status_code == 200
    assert list(server.read_log(tmp_path)["event"]) == ["profiled", "unlocked", "order"]
    place(phone, "sofie", SCRIPTS["sofie"][2])
    closed = phone.post("/api/quiz", json={"customer": "sofie", "answers": right})
    assert closed.status_code == 400 and "Defensive" in closed.json()["error"]


def test_a_lesson_is_marked_and_remembered(tmp_path):
    phone = app(tmp_path)
    assert phone.post("/api/lesson", json={"customer": "lina", "lesson": "etf", "answer": "slice"}).status_code == 404
    place(phone, "lina", SCRIPTS["lina"][2])
    assert phone.post("/api/lesson", json={"customer": "lina", "lesson": "nope", "answer": "x"}).status_code == 404
    wrong = phone.post("/api/lesson", json={"customer": "lina", "lesson": "etf", "answer": "loan"}).json()
    assert wrong["right"] is False and wrong["right_answer"].startswith("A small slice") and wrong["state"]["lessons"] == {"etf": False}
    right = phone.post("/api/lesson", json={"customer": "lina", "lesson": "etf", "answer": "slice"}).json()
    assert right["right"] is True and right["state"]["lessons"] == {"etf": True}
    again = phone.post("/api/lesson", json={"customer": "lina", "lesson": "etf", "answer": "loan"}).json()
    assert again["state"]["lessons"] == {"etf": True}  # once right, the lesson stays done
    assert phone.get("/api/state", params={"customer": "lina"}).json()["state"]["lessons"] == {"etf": True}
    assert list(server.read_log(tmp_path)["event"]) == ["profiled", "lesson", "lesson", "lesson"]


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
    assert phone.post("/api/order", json={"customer": "thomas", "orders": [{"ticker": "WORLD", "eur": 10}]}).status_code == 404
    assert phone.post("/api/quiz", json={"customer": "thomas", "answers": {}}).status_code == 404
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
    assert set(wallet.RULED_OUT) == set(__import__("agents.profiler").profiler.QUESTIONS["rules"]["answers"])


def test_the_reference_follows_its_versions_and_the_customer_gets_the_orders_to_follow_it(tmp_path):
    phone = app(tmp_path)
    morning_run(tmp_path)
    place(phone, "thomas", SCRIPTS["thomas"][2])  # Defensive
    ref = phone.get("/api/reference", params={"customer": "thomas"}).json()
    assert ref["version"] is None and ref["changes"] == [] and ref["history"]  # the starting mix, drawn from its ETFs
    phone.post("/api/order", json={"customer": "thomas", "orders": [{"ticker": l["ticker"], "eur": l["eur"]} for l in ref["lines"]]})
    held = phone.get("/api/portfolio", params={"customer": "thomas"}).json()
    assert held["proposal"] == {"version": None, "changes": [], "new": False, "orders": []}  # bought exactly the reference

    families = {t: etf["family"] for t, etf in server.universe().items()}  # the team signs a day off: a new version
    engine = pd.Series({"WORLD": 0.20, "EUR GOVT BOND": 0.50, "GOLD": 0.20, "STAPLES": 0.10})
    assert model_portfolios.publish(tmp_path, date.today(), {"defensive": engine}, families)
    ref = phone.get("/api/reference", params={"customer": "thomas"}).json()
    assert ref["version"] == date.today().isoformat() and ref["changes"][0]["ticker"] == "EUR GOVT BOND"
    proposal = phone.get("/api/portfolio", params={"customer": "thomas"}).json()["proposal"]
    assert proposal["new"] and proposal["orders"] and proposal["orders"][0]["side"] == "SELL"
    sides = [o["side"] for o in proposal["orders"]]
    assert sides == sorted(sides, key=lambda side: side != "SELL")  # sales first: they fund the buys

    too_much = phone.post("/api/order", json={"customer": "thomas", "orders": [{"ticker": "WORLD", "eur": 99_999, "side": "SELL"}]})
    assert too_much.status_code == 400 and "You hold" in too_much.json()["error"]
    follow = [{"ticker": o["ticker"], "eur": o["eur"], "side": o["side"]} for o in proposal["orders"]]
    after = phone.post("/api/order", json={"customer": "thomas", "orders": follow}).json()["portfolio"]
    assert after["proposal"]["orders"] == []  # back on the reference
    assert {l["ticker"] for l in after["lines"]} == {"WORLD", "EUR GOVT BOND", "GOLD", "STAPLES"}
