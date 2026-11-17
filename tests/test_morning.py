"""The morning run: the fixed model portfolios, the ETF sheets, and what is written for the customer app."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import etf_sheets, macro_news, model_portfolios
from data import mock
from data.profiles import PROFILES, SLEEVE_ETFS, SLEEVES
from data.schemas import validate_etf_sheets, validate_profile_portfolios
from data.universe import tickers
from graph import morning
from tests.test_profiler import FakeModel

DAY = date(2026, 12, 7)


def made_up_prices(days: int = 800) -> pd.DataFrame:
    """Three ETFs: one rising steadily, one falling, one flat. Prices format, the columns the sheets read."""
    dates = pd.bdate_range(end=DAY, periods=days).date
    paths = {"WORLD": 100 * 1.0005 ** np.arange(days), "GOLD": 100 * 0.9995 ** np.arange(days), "OVERNIGHT CASH": np.full(days, 100.0)}
    return pd.DataFrame([(d, ticker, path[i]) for ticker, path in paths.items() for i, d in enumerate(dates)],
                        columns=["date", "ticker", "adj_close_eur"])


def test_every_profile_has_a_whole_portfolio_of_known_etfs():
    known = set(tickers())
    for profile, spec in PROFILES.items():
        assert sum(spec["sleeves"].values()) == pytest.approx(1) and set(spec["sleeves"]) == set(SLEEVES)
        assert 1 <= spec["level"] <= 5 and set(spec["themes"]) <= known
        held = model_portfolios.weights(profile)
        assert sum(weight for _, weight in held.values()) == pytest.approx(1) and set(held) <= known
        for sleeve, share in spec["sleeves"].items():
            assert sum(w for s, w in held.values() if s == sleeve) == pytest.approx(share)
    assert all(sum(parts.values()) == pytest.approx(1) for parts in SLEEVE_ETFS.values())
    assert "BITCOIN" not in model_portfolios.weights("first_step_starter")  # its rule: no crypto, no leverage
    safer = [PROFILES[p]["sleeves"]["bonds"] + PROFILES[p]["sleeves"]["cash"] for p in ("self_directed_trader", "goal_builder", "cautious_saver")]
    assert safer == sorted(safer)  # the lower the risk level, the more bonds and cash


def test_the_model_portfolios_of_a_day_are_in_the_shared_format():
    portfolios = model_portfolios.build(DAY)
    assert set(portfolios["profile"]) == set(PROFILES) and set(portfolios["date"]) == {DAY}
    validate_profile_portfolios(portfolios)
    with pytest.raises(ValueError, match="unknown sleeve"):
        validate_profile_portfolios(portfolios.assign(sleeve="core"))


def test_a_track_record_measures_the_fixed_portfolio():
    record = model_portfolios.track_record(made_up_prices(), "cautious_saver")  # only WORLD, GOLD and cash have prices here
    assert record["years"] == pytest.approx(799 / 252, abs=0.1) and record["worst_fall"] <= 0
    assert 0 < record["return"] < 0.05 and record["volatility"] < 0.01  # a little of the rising ETF, the rest flat


def test_an_etf_sheet_shows_returns_trends_and_macro():
    views = mock.mock_macro_views(DAY.isoformat())
    sheets = etf_sheets.build(made_up_prices(), views, macro_news.load_exposure(), DAY)
    validate_etf_sheets(sheets)
    world = sheets[sheets["ticker"] == "WORLD"].set_index(["block", "name"])
    assert world.loc[("returns", "1 month"), "value"] == pytest.approx(1.0005 ** 21 - 1)
    assert world.loc[("returns", "3 years"), "value"] == pytest.approx(1.0005 ** 756 - 1)
    assert set(world.loc["trend", "state"]) == {"up"} and set(sheets[sheets["ticker"] == "GOLD"].set_index("block").loc["trend", "state"]) == {"down"}
    assert world.loc[("trend", "short"), "detail"].startswith("Price above its 20-day average: yes (+")
    assert "1-month return positive: yes" in world.loc[("trend", "short"), "detail"]
    macro = world.loc["macro"]
    assert len(macro) == etf_sheets.MACRO_LINES and set(macro["state"]) <= {"supportive", "watch", "headwind"}
    assert all(detail.startswith("Today's view: ") for detail in macro["detail"])
    bounce = pd.Series([100.0] * 4 + list(np.linspace(100, 88, 20)) + [97.0])  # back above its average, still down on the month
    assert [(name, state) for _, name, _, state, _ in etf_sheets._trends(bounce)] == [("short", "neutral")]
    short = etf_sheets.build(made_up_prices(days=30), views, macro_news.load_exposure(), DAY)
    assert set(short[short["ticker"] == "WORLD"]["name"]) >= {"1 month", "short"} and "3 years" not in set(short["name"])


def test_a_line_is_explained_from_the_sheet_only():
    sheet = mock.mock_etf_sheets()
    model = FakeModel("The price is just under its 20-day average.")
    model.messages.create = lambda **request: (model.calls.append(request), __import__("types").SimpleNamespace(
        content=[__import__("types").SimpleNamespace(type="text", text="The price is  just under its average.")],
        stop_reason="end_turn", usage=__import__("types").SimpleNamespace(input_tokens=200, output_tokens=20, server_tool_use=None)))[1]
    answer = etf_sheets.explain(sheet[sheet["ticker"] == "WORLD"], "Why is the short trend neutral?", model)
    assert answer == "The price is just under its average."
    request = model.calls[0]
    assert "trend | short | -0.004 | neutral" in request["messages"][0]["content"] and "Never say to buy" in request["system"]


def test_the_morning_run_writes_what_the_app_reads(tmp_path):
    asked = []
    mock.mock_macro_views("2026-12-04").to_csv(tmp_path / "macro_views.csv", index=False)  # last Friday's views

    def news(as_of, lessons, logs, prices=None):
        asked.append(as_of)
        both = pd.concat([pd.read_csv(logs / "macro_views.csv"), mock.mock_macro_views(as_of.isoformat())])
        both.to_csv(logs / "macro_views.csv", index=False)

    out = morning.run(DAY, tmp_path, fetch=lambda start, end: made_up_prices(), news=news)
    assert asked == [DAY] and out["views_of"] == DAY.isoformat() and out["note"] == ""
    portfolios, sheets = pd.read_csv(tmp_path / morning.PORTFOLIOS), pd.read_csv(tmp_path / morning.SHEETS, keep_default_na=False)
    assert set(portfolios["profile"]) == set(PROFILES) and set(sheets["ticker"]) == {"WORLD", "GOLD", "OVERNIGHT CASH"}
    morning.run(DAY, tmp_path, fetch=lambda start, end: made_up_prices(), news=news)
    assert asked == [DAY]  # the day's views are in the logs: Macro/News is not run twice
    assert len(pd.read_csv(tmp_path / morning.PORTFOLIOS)) == len(portfolios)  # a second run replaces the day, it adds nothing


def test_the_morning_run_goes_on_when_the_news_cannot_run(tmp_path):
    mock.mock_macro_views("2026-12-04").to_csv(tmp_path / "macro_views.csv", index=False)

    def no_news(*args, **kwargs):
        raise RuntimeError("no GEMINI_API_KEY")

    out = morning.run(DAY, tmp_path, fetch=lambda start, end: made_up_prices(), news=no_news)
    assert "Macro/News did not run" in out["note"] and out["views_of"] == "2026-12-04"  # the latest views there are
    assert "macro" in set(out["sheets"]["block"])
    empty = morning.run(DAY, tmp_path / "new", fetch=lambda start, end: made_up_prices(), news=no_news)
    assert empty["views_of"] is None and "macro" not in set(empty["sheets"]["block"])  # no views at all: no macro lines
