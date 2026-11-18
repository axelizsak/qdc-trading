"""The morning run: the fixed reference portfolios, the ETF sheets, the charts' points, what the customer app reads."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import etf_sheets, macro_news, model_portfolios
from data import mock
from data.profiles import COMPLEX_FAMILIES, SLEEVE_ETFS, SLEEVES, STRATEGIES, TRADED
from data.schemas import validate_chart_history, validate_etf_sheets, validate_strategy_portfolios
from data.universe import load_universe, tickers
from graph import morning
from tests.test_profiler import FakeModel

DAY = date(2026, 12, 7)


def made_up_prices(days: int = 800) -> pd.DataFrame:
    """Three ETFs: one rising steadily, one falling, one flat. Prices format, the columns the sheets read."""
    dates = pd.bdate_range(end=DAY, periods=days).date
    paths = {"WORLD": 100 * 1.0005 ** np.arange(days), "GOLD": 100 * 0.9995 ** np.arange(days), "OVERNIGHT CASH": np.full(days, 100.0)}
    return pd.DataFrame([(d, ticker, path[i]) for ticker, path in paths.items() for i, d in enumerate(dates)],
                        columns=["date", "ticker", "adj_close_eur"])


def test_every_strategy_has_a_whole_portfolio_of_known_etfs():
    known = set(tickers())
    family = load_universe().set_index("ticker")["family"]
    assert [spec["level"] for spec in STRATEGIES.values()] == [1, 2, 3] and TRADED == "opportunity"
    for strategy, spec in STRATEGIES.items():
        assert sum(spec["sleeves"].values()) == pytest.approx(1) and set(spec["sleeves"]) == set(SLEEVES)
        assert set(spec["themes"]) <= known
        held = model_portfolios.weights(strategy)
        assert sum(weight for _, weight in held.values()) == pytest.approx(1) and set(held) <= known
        for sleeve, share in spec["sleeves"].items():
            assert sum(w for s, w in held.values() if s == sleeve) == pytest.approx(share)
        if not spec["complex"]:  # a level that does not allow complex products holds none
            assert not any(family[ticker] in COMPLEX_FAMILIES for ticker in held)
    assert all(sum(parts.values()) == pytest.approx(1) for parts in SLEEVE_ETFS.values())
    assert "BITCOIN" in model_portfolios.weights("opportunity")
    safer = [STRATEGIES[s]["sleeves"]["bonds"] + STRATEGIES[s]["sleeves"]["cash"] for s in ("opportunity", "balanced", "defensive")]
    assert safer == sorted(safer)  # the lower the risk level, the more bonds and cash


def test_the_reference_portfolios_of_a_day_are_in_the_shared_format():
    portfolios = model_portfolios.build(DAY)
    assert set(portfolios["strategy"]) == set(STRATEGIES) and set(portfolios["date"]) == {DAY}
    validate_strategy_portfolios(portfolios)
    with pytest.raises(ValueError, match="unknown sleeve"):
        validate_strategy_portfolios(portfolios.assign(sleeve="core"))


def test_a_track_record_measures_the_fixed_portfolio():
    record = model_portfolios.track_record(made_up_prices(), "defensive")  # only WORLD, GOLD and cash have prices here
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


def test_the_reading_says_in_words_what_the_sheet_holds():
    lines = [{"block": "trend", "name": "short", "state": "down"}, {"block": "trend", "name": "medium", "state": "down"},
             {"block": "trend", "name": "long", "state": "up"}, {"block": "macro", "name": "growth recession", "state": "supportive"},
             {"block": "macro", "name": "risk appetite", "state": "headwind"}, {"block": "macro", "name": "oil", "state": "watch"}]
    trend, macro, news = etf_sheets.reading(lines, {"direction": -0.3, "confidence": 0.55, "rationale": "..."})
    assert trend == {"topic": "Trend", "tone": "bad", "text": "Rising over 12 months, falling over 1 and 3 months."}
    assert macro == {"topic": "Macro", "tone": "wait", "text": "Tailwind: Growth. Headwind: Risk appetite."}
    assert news["tone"] == "bad" and news["text"] == "Our news analyst leans negative for the next few days (55% sure)."
    calm = etf_sheets.reading([{"block": "macro", "name": "oil", "state": "watch"}], {"direction": 0.05, "confidence": 0.2})
    assert calm[0]["text"] == "No strong macro force either way today." and calm[1]["text"].startswith("Our news analyst sees no clear edge")
    assert etf_sheets.reading([]) == []
    assert [etf_sheets.label(line) for line in lines[:4]] == ["1 month", "3 months", "12 months", "Growth"]
    assert etf_sheets.label({"block": "returns", "name": "volatility"}) == "Volatility"


def test_the_charts_keep_recent_days_and_one_point_a_week_before():
    days = pd.bdate_range(end=DAY, periods=400)
    points = morning.sampled(pd.Series(np.arange(400.0), index=days.date))
    assert list(points.iloc[-morning.DAILY_POINTS:]) == list(np.arange(330.0, 400.0))  # every recent session
    older = pd.to_datetime(pd.Series(points.index[:-morning.DAILY_POINTS]))
    assert older.dt.to_period("W").is_unique and len(older) == len(set(pd.to_datetime(days[:330]).to_period("W")))
    assert points.index.is_monotonic_increasing
    history = morning.chart_history(made_up_prices(), DAY)
    validate_chart_history(history)
    assert set(history[history["kind"] == "strategy"]["name"]) == set(STRATEGIES)
    assert set(history[history["kind"] == "etf"]["name"]) == {"WORLD", "GOLD", "OVERNIGHT CASH"}


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
    assert set(portfolios["strategy"]) == set(STRATEGIES) and set(sheets["ticker"]) == {"WORLD", "GOLD", "OVERNIGHT CASH"}
    history = validate_chart_history(pd.read_csv(tmp_path / morning.HISTORY))
    assert history["date"].max() <= DAY.isoformat() and len(history) == len(out["history"])
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
