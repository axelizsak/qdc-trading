"""The figures and numbers of the midterm report (backtest/report_figures.py), on made-up prices and logs."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agents import model_portfolios
from backtest import report_figures as rf
from data.profiles import STRATEGIES
from data.universe import load_universe

UNIVERSE = load_universe()
DAYS = pd.bdate_range("2021-11-01", "2026-11-18")
LATE = {"EUROPE 600": "2024-02-19", "BITCOIN": "2025-03-24", "NASDAQ 3X": "2025-11-19"}  # priced later, as for real


def made_up_prices() -> pd.DataFrame:
    """A random walk for each of the 50 ETFs, in the Prices format; a few start late."""
    rng = np.random.default_rng(7)
    rows = []
    for k, ticker in enumerate(UNIVERSE["ticker"]):
        days = DAYS[DAYS >= pd.Timestamp(LATE.get(ticker, DAYS[0]))]
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003 + k * 1e-6, 0.01, len(days))))
        rows.append(pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "ticker": ticker, "adj_close_eur": close}))
    prices = pd.concat(rows, ignore_index=True)
    for column in ("open_eur", "high_eur", "low_eur", "close_eur"):
        prices[column] = prices["adj_close_eur"]
    return prices.assign(volume=1000)


@pytest.fixture(scope="module")
def prices():
    return made_up_prices()


def test_the_etfs_come_by_group_then_theme():
    order = rf.ordered(UNIVERSE)
    assert len(order) == 50 and order["ticker"].iloc[0] == "WORLD"
    assert list(dict.fromkeys(order["group"])) == list(rf.GROUPS)  # each group in one block, in order
    assert order.groupby("theme")["group"].nunique().max() == 1


def test_return_and_risk_over_three_years_say_which_etfs_are_short(prices):
    table = rf.return_and_risk(prices, UNIVERSE)
    assert len(table) == 50
    short = set(table.loc[~table["complete"], "ticker"])
    assert short == {"EUROPE 600", "BITCOIN", "NASDAQ 3X"}
    world = table.set_index("ticker").loc["WORLD"]
    closes = rf.closes(prices)["WORLD"]
    window = closes[closes.index >= closes.index.max() - pd.DateOffset(years=3)]
    span = (window.index[-1] - window.index[0]).days / 365.25
    assert world["ret_year"] == pytest.approx((window.iloc[-1] / window.iloc[0]) ** (1 / span) - 1)
    assert world["vol_year"] == pytest.approx(window.pct_change().std() * np.sqrt(252))
    assert table.loc[table["ticker"] == "NASDAQ 3X", "years"].iloc[0] == pytest.approx(1.0, abs=0.02)


def test_the_correlation_is_weekly_and_symmetric(prices):
    tickers = rf.ordered(UNIVERSE)["ticker"].tolist()
    corr = rf.weekly_correlation(prices, tickers)
    assert list(corr.index) == tickers and list(corr.columns) == tickers
    assert np.allclose(np.diag(corr.to_numpy(dtype=float)), 1) and np.allclose(corr, corr.T, equal_nan=True)
    assert not corr.isna().any().any()  # NASDAQ 3X has a year of weeks: enough


def test_a_drawdown_and_the_error_of_a_sharpe_ratio():
    assert rf.drawdown(pd.Series([100, 120, 90, 130])).tolist() == pytest.approx([0, 0, -0.25, 0])
    assert rf.sharpe_error(0.0, 4.0) == pytest.approx(0.5)
    assert rf.sharpe_error(1.0, 5.0) > rf.sharpe_error(0.0, 5.0)


def test_hit_rates_count_only_the_sessions_with_a_call():
    weights = pd.DataFrame({"date": [f"2026-01-{d:02d}" for d in range(1, 9) for _ in (0, 1)],
                            "level": "technical", "name": ["trend", "momentum"] * 8, "weight": 0.5,
                            "result": [0.01, np.nan, -0.01, 0.0, 0.02, 0.01, 0.01, 0.01] * 2})
    hits = rf.hit_rates(weights, window=4)
    assert hits["trend"].iloc[-1] == pytest.approx(0.75)  # +, -, +, + over its last four calls
    assert hits["momentum"].dropna().iloc[-1] == pytest.approx(1.0)  # no call (NaN, 0) does not count
    board = rf.learned(weights).set_index("name")
    assert board.loc["trend", "sessions"] == 8 and board.loc["momentum", "sessions"] == 4


def test_the_replay_of_the_three_answers_uses_the_learning_rule():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2026-01-05", periods=40).strftime("%Y-%m-%d")
    answers = pd.DataFrame([{"date": d, "ticker": t, "suggested_signal": rng.choice(["buy", "watch", "avoid"]),
                             "mr_setup": rng.choice(["buy", "none", "avoid"]), "momentum_13612": rng.normal(),
                             "confirmations": 1, "trend_strength": "absent", "session_return": rng.normal(0, 0.01)}
                            for d in days for t in ("WORLD", "SEMIS", "GOLD", "BITCOIN")])
    frame = rf.replayed(answers, date(2026, 3, 2))
    assert set(frame["name"]) == set(rf.ANSWERS) and set(frame["level"]) == {"technical"}
    first = frame[frame["date"] == frame["date"].min()]
    assert first["weight"].tolist() == pytest.approx([1 / 3] * 3)
    last = frame[frame["date"] == "2026-03-02"]["weight"]
    assert last.sum() == pytest.approx(1) and last.between(0.05 - 1e-9, 0.8 + 1e-9).all()


def test_the_strategies_start_at_one_on_the_last_close_of_2021(prices):
    curves = rf.mix_curves(prices)
    assert curves.index.min() == pd.Timestamp("2021-12-31") and list(curves.columns) == [*STRATEGIES, "world"]
    assert curves.iloc[0].tolist() == pytest.approx([1.0] * 4) and not curves.isna().any().any()
    late = rf.unpriced(prices, "opportunity", rf.SINCE)
    held = model_portfolios.weights("opportunity")
    assert {t for t, _, _ in late} == {t for t in LATE if t in held}
    assert sum(w for _, w, _ in late) == pytest.approx(sum(held[t][1] for t in LATE if t in held))


def test_labels_are_moved_apart():
    assert rf.spread([1.0, 1.01, 5.0], 0.5) == pytest.approx([0.75, 1.25, 5.0], abs=0.26)
    placed = rf.spread([3.0, 3.0, 3.0], 1.0)
    assert sorted(np.diff(sorted(placed))) == pytest.approx([1.0, 1.0])
    assert rf.wrapped("one two three four", 8.5, 0.5)[0] == "one two"
    assert rf.signed(-0.113) == "−11%" and rf.day("2022-01-03") == "3 Jan 2022"


def test_the_whole_report_on_made_up_data(prices, tmp_path):
    results, logs, out = tmp_path / "results", tmp_path / "logs", tmp_path / "out"
    results.mkdir()
    logs.mkdir()
    days = pd.bdate_range("2021-12-31", "2026-11-18")
    walk = 10_000 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.0003, 0.008, (len(days), 3)), axis=0))
    pd.concat([pd.DataFrame({"variant": name, "date": days.strftime("%Y-%m-%d"), "value": walk[:, k],
                             "bench": walk[:, 2], "riskfree": 1.0, "invested": 0.9, "orders": 0, "broken_rules": 0,
                             "failed_days": 0})
               for k, name in enumerate(["current", "learned_three"])]).to_csv(results / "system_backtest_daily.csv",
                                                                                index=False)
    summary = {"what": "the system", "years": 4.9, "sharpe": 1.1, "sharpe_h1": 0.5, "sharpe_h2": 1.6, "ret_year": 0.11,
               "vol_year": 0.1, "beta": 0.5, "orders": 1000, "broken_rules": 0, "bench_sharpe": 0.7,
               "bench_ret_year": 0.1, "bench_vol_year": 0.14}
    pd.DataFrame([{"variant": "current", **summary}, {"variant": "learned_three", **summary, "sharpe": 0.9,
                                                      "sharpe_h1": 0.3}]).to_csv(
        results / "system_backtest_daily_summary.csv", index=False)
    rng = np.random.default_rng(1)
    pd.DataFrame([{"date": d, "ticker": t, "suggested_signal": rng.choice(["buy", "watch", "avoid"]),
                   "mr_setup": rng.choice(["buy", "none"]), "momentum_13612": rng.normal(), "confirmations": 1,
                   "trend_strength": "absent", "session_return": rng.normal(0, 0.01)}
                  for d in pd.bdate_range("2022-01-03", periods=200).strftime("%Y-%m-%d")
                  for t in ("WORLD", "SEMIS", "GOLD")]).to_csv(logs / "technical_answers.csv", index=False)
    pd.DataFrame({"date": ["2026-11-16", "2026-11-18"], "level": "agents", "name": "macro_market",
                  "weight": [0.25, 0.26], "result": [0.01, np.nan]}).to_csv(logs / "weights.csv", index=False)
    (out).mkdir()
    (out / "code_appendix.md").write_text("# Appendix\n\nPython 3.11, **0 automated tests**.\n")
    rf.appendix(out, count=lambda: 1234)
    assert "**1,234 automated tests**" in (out / "code_appendix.md").read_text()
    assert (out / "code_appendix.pdf").read_bytes().startswith(b"%PDF")
    (out / "code_appendix.md").unlink()  # the report itself would count the tests with pytest

    numbers = rf.report(prices, out, logs=logs, results=results)
    assert [p.name for p in numbers["figures"]] == ["1_etf_return_risk.png", "2_etf_correlation.png",
                                                    "3_system_backtest.png", "4_learning.png", "5_strategies.png"]
    assert all(p.stat().st_size > 20_000 for p in numbers["figures"])
    text = (out / "data_exploration.md").read_text()
    assert "nan" not in text.lower().replace("financ", "")
    assert "did better in both halves" not in text  # the second half is a tie here: the page does not say it
    assert "## Key numbers" in text and text.count("![") == 5
    assert len(pd.read_csv(out / "etf_return_risk.csv")) == 50
    assert numbers["appendix"] is None
