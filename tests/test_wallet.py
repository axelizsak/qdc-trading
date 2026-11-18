"""What a customer may buy and what they hold (mobile/wallet.py): locks, orders, the comparison with the reference."""
import pytest

from mobile import wallet

WORLD = {"ticker": "WORLD", "family": "broad_equity", "name": "MSCI World", "theme": "Global equity"}
SEMIS = {"ticker": "SEMIS", "family": "tech_growth", "name": "Semiconductors", "theme": "Equity / Tech"}
BONDS = {"ticker": "GLOBAL AGG BOND", "family": "bonds", "name": "Global bonds", "theme": "Fixed income / Bonds"}
BITCOIN = {"ticker": "BITCOIN", "family": "crypto", "name": "Bitcoin", "theme": "Digital assets"}
NASDAQ_3X = {"ticker": "NASDAQ 3X", "family": "leveraged_inverse", "name": "Nasdaq 3x", "theme": "Leveraged & inverse"}
DEFENSE = {"ticker": "DEFENSE", "family": "cyclical", "name": "Defense", "theme": "Defensive sector"}
ETFS = {etf["ticker"]: etf for etf in (WORLD, SEMIS, BONDS, BITCOIN, NASDAQ_3X, DEFENSE)}
REFERENCE = [{"ticker": "WORLD", "sleeve": "core_equity", "weight": 0.6}, {"ticker": "GLOBAL AGG BOND", "sleeve": "bonds", "weight": 0.3},
             {"ticker": "SEMIS", "sleeve": "satellites", "weight": 0.1}]


def test_complex_products_open_only_at_a_level_that_allows_them_and_after_the_check():
    assert wallet.lock(BITCOIN, "balanced", True) == "Not part of the Balanced risk level"
    assert wallet.lock(NASDAQ_3X, "opportunity", False) == "Pass the 3-question check to unlock"
    assert wallet.lock(BITCOIN, "opportunity", True) == ""
    assert wallet.lock(WORLD, "defensive", False) == "" and wallet.lock(SEMIS, "defensive", False) == ""


def test_the_customers_own_rules_come_first():
    assert set(wallet.RULED_OUT) == {"weapons", "crypto", "leverage", "fossil"}
    assert wallet.lock(BITCOIN, "opportunity", True, ["crypto"]) == "You ruled out crypto"
    assert wallet.lock(DEFENSE, "opportunity", True, ["weapons"]) == "You ruled out weapons"
    assert wallet.ruled_out_by(WORLD, ["weapons", "crypto", "leverage", "fossil"]) is None
    lines = wallet.decorate([WORLD, BITCOIN], "opportunity", False)
    assert [line["lock"] for line in lines] == ["", "Pass the 3-question check to unlock"] and lines[0]["name"] == "MSCI World"


def test_an_order_needs_a_known_etf_an_amount_and_no_lock():
    assert wallet.check_order(WORLD, 250, "balanced", False) == ""
    assert wallet.check_order(None, 250, "balanced", False) == "Unknown ETF"
    for amount in (0, -5, 0.5, "100", True, None, float("nan")):
        assert wallet.check_order(WORLD, amount, "balanced", False).startswith("The amount must be at least")
    assert wallet.check_order(BITCOIN, 250, "opportunity", False) == "Pass the 3-question check to unlock"


def test_holdings_are_valued_at_the_last_close_and_compared_with_the_reference():
    orders = [{"ticker": "WORLD", "eur": 500, "units": 5}, {"ticker": "WORLD", "eur": 300, "units": 2.5},
              {"ticker": "SEMIS", "eur": 200, "units": None}]  # bought before any price was known
    out = wallet.holdings(orders, {"WORLD": 110.0, "SEMIS": 14.0}, ETFS, REFERENCE)
    world, semis = out["lines"]
    assert (world["ticker"], world["invested"], world["value"]) == ("WORLD", 800, 825.0)
    assert semis["value"] == 200 and semis["gain"] == 0  # no units: worth what was paid
    assert out["value"] == 1025 and out["invested"] == 1000 and out["gain"] == pytest.approx(0.025)
    assert sum(line["weight"] for line in out["lines"]) == pytest.approx(1)
    health = {row["sleeve"]: row for row in out["health"]}
    assert health["core_equity"]["state"] == "gap" and health["bonds"]["state"] == "gap"  # 80% vs 60%, 0% vs 30%
    assert health["satellites"]["state"] == "watch" and "cash" not in health  # 20% vs 10%; cash: neither held nor in it
    assert out["ideas"][0].startswith("Core equity: 80% of what you hold") and "Bonds: about 30%" in out["ideas"][1]
    assert [s["key"] for s in out["sleeves"]] == ["core_equity", "satellites"]


def test_an_etf_outside_the_reference_takes_its_familys_sleeve():
    assert wallet.sleeve_of(BONDS, []) == "bonds" and wallet.sleeve_of(BITCOIN, REFERENCE) == "satellites"
    assert wallet.sleeve_of(SEMIS, REFERENCE) == "satellites" and wallet.sleeve_of(WORLD, []) == "core_equity"
    empty = wallet.holdings([], {}, ETFS, REFERENCE)
    assert empty["lines"] == [] and empty["value"] == 0 and empty["health"] == [] and empty["gain"] == 0


def test_the_history_of_a_mix_starts_when_every_etf_has_a_price():
    histories = {"WORLD": [{"date": "2026-01-02", "value": 100}, {"date": "2026-01-09", "value": 110}, {"date": "2026-01-16", "value": 120}],
                 "SEMIS": [{"date": "2026-01-09", "value": 10}, {"date": "2026-01-16", "value": 12}]}
    orders = [{"ticker": "WORLD", "eur": 240, "units": 2}, {"ticker": "SEMIS", "eur": 120, "units": 10},
              {"ticker": "BITCOIN", "eur": 50, "units": None}]
    assert wallet.mix_history(orders, histories) == [{"date": "2026-01-09", "value": 370.0}, {"date": "2026-01-16", "value": 410.0}]
    assert wallet.mix_history([{"ticker": "BITCOIN", "eur": 50, "units": None}], histories) == []


def test_a_sale_takes_its_share_off_what_is_held():
    orders = [{"ticker": "WORLD", "eur": 1000, "units": 10}, {"ticker": "WORLD", "side": "SELL", "eur": 440, "units": 4}]
    out = wallet.holdings(orders, {"WORLD": 110.0}, ETFS, REFERENCE)
    assert out["lines"][0]["value"] == pytest.approx(660) and out["lines"][0]["invested"] == pytest.approx(600)
    gone = wallet.holdings(orders + [{"ticker": "WORLD", "side": "SELL", "eur": 660, "units": 6}], {"WORLD": 110.0}, ETFS, REFERENCE)
    assert gone["lines"] == []
    assert wallet.check_order(WORLD, 500, "balanced", False, side="SELL", held_eur=660) == ""
    assert wallet.check_order(WORLD, 700, "balanced", False, side="SELL", held_eur=660) == "You hold EUR 660.00 of it"
    assert wallet.check_order(BITCOIN, 50, "balanced", False, side="SELL", held_eur=100) == ""  # selling is always open
    assert wallet.check_order(WORLD, 50, "balanced", False, side="HOLD") == "An order is a buy or a sell"


def test_the_orders_that_follow_the_reference_sell_first_and_skip_what_is_locked():
    held = [{"ticker": "WORLD", "value": 1000.0}, {"ticker": "URANIUM", "value": 200.0}]
    reference = [{"ticker": "WORLD", "weight": 0.6, "lock": ""}, {"ticker": "SEMIS", "weight": 0.3, "lock": ""},
                 {"ticker": "BITCOIN", "weight": 0.1, "lock": "Pass the 3-question check to unlock"}]
    orders = wallet.rebalance(held, reference)
    assert [(o["side"], o["ticker"], o["eur"]) for o in orders] == [("SELL", "WORLD", 280.0), ("SELL", "URANIUM", 200.0),
                                                                    ("BUY", "SEMIS", 360.0)]
    assert orders[0]["now"] == pytest.approx(0.8333, abs=1e-4) and orders[0]["target"] == 0.6
    close = [{"ticker": "WORLD", "value": 605.0}, {"ticker": "SEMIS", "value": 395.0}]
    assert wallet.rebalance(close, [{"ticker": "WORLD", "weight": 0.6}, {"ticker": "SEMIS", "weight": 0.4}]) == []  # under 1%
    assert wallet.rebalance([], reference) == []


def test_what_changed_and_the_chart_of_a_mix():
    assert wallet.changes({"A": 0.5, "B": 0.5, "D": 0.1}, {"A": 0.3, "C": 0.7, "D": 0.104}) == [  # D: too small a move
        {"ticker": "C", "before": 0.0, "after": 0.7}, {"ticker": "B", "before": 0.5, "after": 0.0},
        {"ticker": "A", "before": 0.5, "after": 0.3}]
    histories = {"A": [{"date": "d1", "value": 10}, {"date": "d2", "value": 11}],
                 "B": [{"date": "d2", "value": 5}, {"date": "d3", "value": 6}]}
    assert wallet.mix_curve({"A": 0.5, "B": 0.5}, histories) == [
        {"date": "d1", "value": 1.0}, {"date": "d2", "value": 1.05}, {"date": "d3", "value": 1.155}]
