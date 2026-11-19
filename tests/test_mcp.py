"""ReBel in Claude (mobile/mcp_server.py): the app's data through the MCP connector, read-only, with no model."""
import asyncio
import json

import pytest
from mcp import Client

from mobile import mcp_server, server
from tests.test_mobile import app, morning_run, place
from tests.test_profiler import SCRIPTS

TOOLS = {"list_customers", "get_profile", "get_portfolio", "get_reference", "list_etfs", "get_etf", "prepare_order"}


class Failed(Exception):
    """A tool said what goes wrong."""


async def session(rebel, *steps):
    async with Client(rebel) as client:
        return [await step(client) for step in steps]


def call(rebel, tool: str, **arguments):
    """One call as Claude makes it, in-process (no network): the JSON the tool returns, or Failed with its words."""
    result, = asyncio.run(session(rebel, lambda client: client.call_tool(tool, arguments)))
    text = result.content[0].text
    if result.is_error:
        raise Failed(text)
    return json.loads(text)


def test_the_connector_offers_its_tools_read_only():
    listed, = asyncio.run(session(mcp_server.create_server(), lambda client: client.list_tools()))
    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.annotations.read_only_hint and not tool.annotations.destructive_hint and tool.description
    assert "confirms the order in the ReBel app" in tools["prepare_order"].description


def test_a_customer_seen_through_the_connector_is_the_one_of_the_app(tmp_path):
    phone, rebel = app(tmp_path), mcp_server.create_server(tmp_path, tmp_path)
    morning_run(tmp_path)
    place(phone, "thomas", SCRIPTS["thomas"][2])  # Defensive, EUR 6,000
    ref = phone.get("/api/reference", params={"customer": "thomas"}).json()
    phone.post("/api/order", json={"customer": "thomas", "orders": [{"ticker": l["ticker"], "eur": l["eur"]} for l in ref["lines"]]})

    people = {person["id"]: person for person in call(rebel, "list_customers")["customers"]}
    assert people["thomas"]["placed"] and people["thomas"]["strategy"] == "defensive"
    assert people["thomas"]["orders"] == len(ref["lines"]) and not people["noah"]["placed"]
    profile = call(rebel, "get_profile", customer=" Thomas ")
    assert profile["archetype"]["key"] == "goal_builder" and profile["strategy"]["label"] == "Defensive"
    assert profile["strategy"]["why"] and profile["answers"] and profile["rules"] == []

    held = call(rebel, "get_portfolio", customer="thomas")
    shown = phone.get("/api/portfolio", params={"customer": "thomas"}).json()
    assert held["value"] == shown["value"] and held["lines"] == shown["lines"] and held["health"] == shown["health"]
    assert "history" not in held and len(held["placed_orders"]) == len(ref["lines"]) and "never an instruction" in held["note"]
    assert call(rebel, "get_portfolio", customer="thomas", with_chart=True)["history"]

    mine = call(rebel, "get_reference", customer="thomas")
    assert mine["lines"] == ref["lines"] and mine["amount"] == 6000 and mine["label"] == "Defensive" and "history" not in mine
    sample = call(rebel, "get_reference", strategy="Balanced", with_chart=True)
    assert sample["amount"] == 10_000 and sum(l["eur"] for l in sample["lines"]) == pytest.approx(10_000)
    assert 0 < len(sample["history"]) <= mcp_server.CHART_POINTS and "lock" not in sample["lines"][0]

    etfs = {etf["ticker"]: etf for etf in call(rebel, "list_etfs", customer="thomas")["etfs"]}
    assert len(etfs) == 50 and etfs["NASDAQ 3X"]["lock"] == "Not part of the Defensive risk level"
    assert etfs["WORLD"]["lock"] == "" and etfs["WORLD"]["returns"] == {"1 month": 0.012, "1 year": 0.094}
    assert "lock" not in call(rebel, "list_etfs")["etfs"][0]


def test_an_etf_sheet_comes_without_its_table_and_its_chart_on_request(tmp_path):
    rebel = mcp_server.create_server(tmp_path, tmp_path)
    assert call(rebel, "get_etf", ticker="WORLD")["as_of"] is None  # before the first morning run
    morning_run(tmp_path)
    sheet = call(rebel, "get_etf", ticker="world")
    assert sheet["ticker"] == "WORLD" and sheet["as_of"] == "2026-12-07" and sheet["lines"] and sheet["reading"]
    assert "rows" not in sheet and "history" not in sheet and "never an instruction" in sheet["note"]
    drawn = call(rebel, "get_etf", ticker="WORLD", with_chart=True)
    assert drawn["history"][-1] == drawn["last_close"] == server.points(server.chart_points(tmp_path), "WORLD")[-1]
    with pytest.raises(Failed, match="Unknown ETF 'NOPE'"):
        call(rebel, "get_etf", ticker="NOPE")


def test_an_order_is_only_checked_and_never_recorded(tmp_path):
    phone, rebel = app(tmp_path), mcp_server.create_server(tmp_path, tmp_path)
    morning_run(tmp_path)
    place(phone, "noah", SCRIPTS["noah"][2])  # Opportunity, complex products not opened yet
    log = (tmp_path / server.LOG_NAME).read_bytes()

    good = call(rebel, "prepare_order", customer="noah", ticker="WORLD", eur=250)
    price = server.last_prices(tmp_path)["WORLD"]["price"]
    assert good["ok"] and good["problem"] == "" and good["next"] == mcp_server.CONFIRM
    assert good["draft"]["side"] == "BUY" and good["draft"]["units"] == pytest.approx(250 / price, rel=1e-5)
    closed = call(rebel, "prepare_order", customer="noah", ticker="BITCOIN", eur=100)
    assert not closed["ok"] and closed["problem"] == "Pass the 3-question check to unlock" and closed["draft"]["units"] is None
    sale = call(rebel, "prepare_order", customer="noah", ticker="WORLD", eur=100, side="sell")
    assert sale["problem"] == "You hold EUR 0.00 of it"
    assert call(rebel, "prepare_order", customer="noah", ticker="WORLD", eur=0.5)["problem"].startswith("The amount")
    assert call(rebel, "prepare_order", customer="noah", ticker="WORLD", eur=50, side="HOLD")["problem"] == "An order is a buy or a sell"

    assert (tmp_path / server.LOG_NAME).read_bytes() == log  # nothing was written
    assert server.customer_state(tmp_path, "noah")["orders"] == []


def test_what_goes_wrong_is_said(tmp_path):
    rebel = mcp_server.create_server(tmp_path, tmp_path)
    with pytest.raises(Failed, match="Unknown customer 'nobody': one of lina, thomas"):
        call(rebel, "get_profile", customer="nobody")
    assert call(rebel, "get_profile", customer="sofie")["placed"] is False
    for tool, arguments in [("get_portfolio", {}), ("get_reference", {}), ("list_etfs", {}),
                            ("prepare_order", {"ticker": "WORLD", "eur": 100})]:
        with pytest.raises(Failed, match="Sofie has no risk level yet"):
            call(rebel, tool, customer="sofie", **arguments)
    with pytest.raises(Failed, match="Give a customer"):
        call(rebel, "get_reference")
    with pytest.raises(Failed, match="Unknown strategy 'yolo'"):
        call(rebel, "get_reference", strategy="yolo")
    assert not (tmp_path / server.LOG_NAME).exists()


def test_a_long_chart_keeps_its_last_point():
    long = [{"date": str(i), "value": float(i)} for i in range(1000)]
    short = mcp_server.chart(long)
    assert len(short) == mcp_server.CHART_POINTS and short[0] == long[0] and short[-1] == long[-1]
    assert mcp_server.chart(long[:10]) == long[:10]
