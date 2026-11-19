"""The strategy the team trades: its limits on the Portfolio Manager, the choice logged in the cockpit, the pipeline."""
import json
from datetime import date

import pandas as pd
import pytest

from agents import model_portfolios, portfolio
from data.profiles import COMPLEX_FAMILIES, EQUITY_SLEEVES, STRATEGIES, TRADED, sleeve_of
from data.universe import load_universe
from graph import pipeline
from tests.test_portfolio import prices, propose, signals  # noqa: F401 (the fixtures of the Portfolio Manager's tests)

UNIVERSE = load_universe().set_index("ticker")
EVERYTHING = set(UNIVERSE.index)
DAY = date(2026, 12, 7)


def share(weights: pd.Series, test) -> float:
    return float(sum(w for t, w in weights.items() if test(t)))


def complex_share(weights):
    return share(weights, lambda t: UNIVERSE.loc[t, "family"] in COMPLEX_FAMILIES)


def themes_share(weights):
    return share(weights, lambda t: sleeve_of(t, UNIVERSE.loc[t, "family"]) == "satellites")


def equity_share(weights):
    return share(weights, lambda t: sleeve_of(t, UNIVERSE.loc[t, "family"]) in EQUITY_SLEEVES)


def test_every_reference_portfolio_is_within_its_own_limits():
    for key, spec in STRATEGIES.items():
        held = pd.Series({t: w for t, (_, w) in model_portfolios.weights(key).items()})
        assert equity_share(held) <= spec["limits"]["equity_max"] and themes_share(held) <= spec["limits"]["themes_max"]
        assert complex_share(held) <= spec["limits"]["complex_max"]
    caps = [STRATEGIES[k]["limits"]["equity_max"] for k in ("defensive", "balanced", "opportunity")]
    assert caps == sorted(caps)  # the higher the risk level, the more room for shares


def test_the_opportunity_limits_cut_a_large_bet_on_an_inverse_etf():
    """The case that started it: 13% in STOXX SHORT, above Opportunity's 10% for complex products."""
    target = pd.Series({"STOXX SHORT": 0.13, "WORLD": 0.30, "HEALTH CARE": 0.20})
    out, notes = portfolio.within_strategy(target, "opportunity", UNIVERSE, EVERYTHING)
    assert out["STOXX SHORT"] == pytest.approx(0.10, abs=1e-4) and out["WORLD"] == pytest.approx(0.30)
    assert notes == ["Complex products (STOXX SHORT) cut from 13% to 10%, the Opportunity limit: kept in cash."]


def test_the_defensive_limits_move_the_excess_to_the_core_then_to_bonds():
    target = pd.Series({"SEMIS": 0.30, "NASDAQ": 0.20, "WORLD": 0.25, "BITCOIN": 0.05, "STOXX SHORT": 0.10})
    out, notes = portfolio.within_strategy(target, "defensive", UNIVERSE, EVERYTHING)
    limits = STRATEGIES["defensive"]["limits"]
    assert complex_share(out) == 0 and "BITCOIN" not in out and "STOXX SHORT" not in out  # cut, kept in cash
    assert themes_share(out) <= limits["themes_max"] + 1e-4 and equity_share(out) <= limits["equity_max"] + 1e-4
    assert out[["GLOBAL AGG BOND", "EUR GOVT BOND"]].sum() == pytest.approx(0.30, abs=1e-3)  # what the shares lost
    assert out.sum() == pytest.approx(0.75, abs=1e-3)  # only the complex products went to cash
    assert (out <= 0.30 + 1e-9).all() and len(notes) == 3 and "moved to WORLD, EUROPE 600, EMMKT" in notes[1]
    assert "moved to GLOBAL AGG BOND, EUR GOVT BOND" in notes[2]


def test_a_cut_goes_to_cash_when_its_etfs_cannot_be_bought():
    target = pd.Series({"SEMIS": 0.60})
    out, notes = portfolio.within_strategy(target, "balanced", UNIVERSE, EVERYTHING - {"WORLD", "EUROPE 600", "EMMKT"})
    assert out.to_dict() == {"SEMIS": 0.25} and notes[0].endswith("kept in cash.")
    assert portfolio.within_strategy(target, None, UNIVERSE, EVERYTHING) == (target, [])  # no strategy: as it is


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_every_portfolio_of_the_day_stays_within_the_traded_strategy(prices, signals, strategy, tmp_path):
    target, orders, memo = propose(prices, signals, strategy=strategy, logs_dir=tmp_path)
    final = target.set_index("ticker")["target_weight"]
    limits = STRATEGIES[strategy]["limits"]
    assert complex_share(final) <= limits["complex_max"] + 1e-4 and themes_share(final) <= limits["themes_max"] + 1e-4
    assert equity_share(final) <= limits["equity_max"] + 1e-4
    logged = pd.read_csv(tmp_path / "pm_portfolios.csv")  # the code's portfolios too: the learning compares them
    for _, weights in logged.groupby("source"):
        assert equity_share(weights.set_index("ticker")["weight"]) <= limits["equity_max"] + 1e-3
    assert f"**Strategy.** The team trades the {STRATEGIES[strategy]['label']} strategy" in memo
    assert memo.startswith("## Portfolio Manager")


def test_the_choice_is_logged_and_dated(tmp_path):
    assert model_portfolios.traded(tmp_path, DAY) == TRADED == "opportunity"  # before any choice
    model_portfolios.choose(tmp_path, date(2026, 12, 10), "balanced", "  Fed and ECB   next week ")
    model_portfolios.choose(tmp_path, date(2026, 12, 14), "defensive", "too much risk")
    assert model_portfolios.traded(tmp_path, DAY) == "opportunity"  # a choice applies from its day on
    assert model_portfolios.traded(tmp_path, date(2026, 12, 10)) == "balanced"
    assert model_portfolios.traded(tmp_path, date(2026, 12, 21)) == "defensive"
    model_portfolios.choose(tmp_path, date(2026, 12, 10), "opportunity", "changed our mind")  # same day: replaced
    made = model_portfolios.choices(tmp_path)
    assert made["date"].tolist() == ["2026-12-10", "2026-12-14"] and made["strategy"].iloc[0] == "opportunity"
    assert model_portfolios.choices(tmp_path / "none").empty
    with pytest.raises(ValueError):
        model_portfolios.choose(tmp_path, DAY, "yolo", "no such strategy")


def test_the_pipeline_trades_the_strategy_chosen_for_the_day(tmp_path, monkeypatch):
    logs, seen = tmp_path / "logs", []

    def manager(*args, **extra):
        seen.append(extra.get("strategy"))
        return pipeline._mock_portfolio(*args, **extra)

    monkeypatch.setitem(pipeline.MOCK, "portfolio", manager)
    model_portfolios.choose(logs, DAY, "balanced", "a calmer start")
    pipeline.run_day(DAY, logs)
    assert seen == ["balanced"]
    assert json.loads(pipeline.proposal_path(logs, DAY, 1).read_text())["strategy"] == "balanced"
    model_portfolios.choose(logs, DAY, "defensive", "too late for this day")
    (logs / "checkpoints.sqlite").unlink()
    pipeline.run_day(DAY, logs)
    assert seen == ["balanced"]  # the day was run: it keeps its strategy, nothing is asked again


FAMILIES = dict(zip(UNIVERSE.index, UNIVERSE["family"]))


def test_a_reference_starts_from_its_fixed_mix_then_follows_the_published_versions(tmp_path):
    start, version = model_portfolios.current(tmp_path, "balanced")
    assert version is None and start == model_portfolios.weights("balanced")
    near = pd.Series({t: w for t, (_, w) in start.items()}) * 0.98  # 2% of noise: not worth bothering customers
    assert model_portfolios.publish(tmp_path, DAY, {"balanced": near}, FAMILIES) == {}
    engine = pd.Series({"SEMIS": 0.20, "WORLD": 0.30, "GLOBAL AGG BOND": 0.30, "GOLD": 0.10, "ROBOTICS": 0.015})
    published = model_portfolios.publish(tmp_path, DAY, {"balanced": engine}, FAMILIES)
    assert list(published) == ["balanced"] and published["balanced"] >= model_portfolios.PUBLISH_TURNOVER
    now, version = model_portfolios.current(tmp_path, "balanced")
    assert version == DAY.isoformat() and "ROBOTICS" not in now  # dust is shared out in its sleeve
    assert now["OVERNIGHT CASH"] == ("cash", pytest.approx(0.05)) and now["WORLD"] == ("core_equity", pytest.approx(0.425))
    assert now["SEMIS"][1] > start["SEMIS"][1] > now["HEALTH CARE"][1]  # the engine likes SEMIS: it gains in its sleeve
    assert sum(w for _, w in now.values()) == pytest.approx(1)
    assert model_portfolios.previous(tmp_path, "balanced", version) == start
    assert model_portfolios.current(tmp_path, "balanced", date(2026, 12, 4)) == (start, None)  # before the version
    assert model_portfolios.current(tmp_path, "defensive")[1] is None  # each strategy its own versions
    again = model_portfolios.publish(tmp_path, DAY, {"balanced": engine}, FAMILIES)  # the same day run again
    assert again == published and len(model_portfolios.versions(tmp_path)) == len(now)


def test_a_signed_off_day_moves_the_references_and_a_stopped_one_does_not(tmp_path):
    logs = tmp_path / "logs"
    pipeline.approve_day(DAY, logs)
    found = model_portfolios.versions(logs)
    assert len(found) and set(found["date"]) == {DAY.isoformat()}  # those the engine moved by 10% or more
    for strategy, rows in found.groupby("strategy"):
        wanted = {s: w for s, w in STRATEGIES[strategy]["sleeves"].items() if w > 0}
        assert rows.groupby("sleeve")["target_weight"].sum().to_dict() == pytest.approx(wanted, abs=1e-5)
    other = date(2026, 12, 10)
    pending = pipeline.run_day(other, logs)
    pipeline.resume_day(other, {"decision": "stopped", "feedback": "Fed tonight"}, logs)
    assert set(model_portfolios.versions(logs)["date"]) == {DAY.isoformat()}  # no trade, no new version


def sleeves(reference: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for sleeve, w in reference.values():
        out[sleeve] = round(out.get(sleeve, 0.0) + w, 6)
    return out


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_a_reference_keeps_the_share_of_each_sleeve_whatever_the_engine_does(strategy):
    """The risk level is the table of the deck (data/profiles.py): the engine only moves the ETFs inside a sleeve."""
    engine = pd.Series({"STOXX SHORT": 0.30, "BITCOIN": 0.20, "SEMIS": 0.30, "EUROPE 600": 0.15, "OVERNIGHT CASH": 0.05})
    out = model_portfolios.follow(engine, strategy, FAMILIES)
    wanted = {s: w for s, w in STRATEGIES[strategy]["sleeves"].items() if w > 0}
    assert sleeves(out) == pytest.approx(wanted, abs=1e-5) and all(w >= model_portfolios.MIN_LINE for _, w in out.values())
    held = pd.Series({t: w for t, (_, w) in out.items()})
    assert complex_share(held) <= STRATEGIES[strategy]["limits"]["complex_max"] + 1e-6
    assert out["GOLD"] == ("gold", wanted["gold"])  # the engine holds no gold: the fixed mix's split stays
    assert out["EUROPE 600"][1] > model_portfolios.weights(strategy)["EUROPE 600"][1]  # it gains inside core equity
