"""The figures and numbers of the midterm report's data exploration, written into docs/midterm/. Owner: Axel. No LLM.

    python -m backtest.report_figures                        # fetches the prices (about a minute), then everything
    python -m backtest.report_figures --prices prices.csv    # from prices already saved (Prices format)

Every number is read from what the system already writes, never typed by hand:
  1_etf_return_risk.png   return and volatility of the 50 ETFs over three years, by theme   prices
  2_etf_correlation.png   how their weekly returns move together, ordered by theme        prices
  3_system_backtest.png   the system replayed since 2022 against the MSCI World            backtest/results/
  4_learning.png          the learned weights of the technical answers and their hit rate  logs/weights.csv
  5_strategies.png        the three strategies' starting mixes against the MSCI World      prices, data/profiles.py
Then data_exploration.md (a few sentences per figure and the key numbers), etf_return_risk.csv (figure 1 as a
table), and code_appendix.pdf from code_appendix.md with its test count brought up to date.
"""
import argparse
import logging
import re
import subprocess
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from agents import learning, model_portfolios  # noqa: E402
from data import config as settings  # noqa: E402
from data.profiles import STRATEGIES  # noqa: E402
from data.universe import load_universe  # noqa: E402

ROOT = settings.ROOT
OUT = ROOT / "docs" / "midterm"
RESULTS = ROOT / "backtest" / "results"
YEAR = 252
YEARS = 3  # the window of figures 1 and 2
SINCE = date(2021, 12, 31)  # figures 3 and 5 start on the last close of 2021, like the system backtest
MIN_WEEKS = 50  # a correlation needs about a year of weekly returns in common
HIT_WINDOW = 120  # sessions in the rolling hit rate of figure 4 (about six months)
DPI = 200

# The 20 themes of the Belfius list, in four groups, in the order of both ETF figures.
GROUPS = {
    "Broad equity": ["Global equity (benchmark)", "US equity", "European equity", "Home market", "Emerging markets",
                     "Size & style factors", "Income strategies"],
    "Sectors and themes": ["Equity / Tech", "Frontier tech", "Health care", "Financials", "Defensive sector",
                           "Energy transition", "Energy & commodities", "Industrial & strategic metals"],
    "Bonds, gold and cash": ["Fixed income / Bonds", "Precious metals", "Money market (cash)"],
    "Complex products": ["Leveraged & inverse", "Digital assets"],
}
ANSWERS = {"trend": "Trend", "mean_reversion": "Mean reversion", "momentum": "Momentum"}
THIRDS = {name: 1 / 3 for name in ANSWERS}  # how the three answers started before the team kept trend alone
VARIANTS = {"current": "The system now: the trend rule alone",
            "learned_three": "Until 18 Nov: the three answers, learned"}
SHORT = {"current": "Trend alone", "learned_three": "Three learned"}

# Colours (the dataviz reference palette, light mode): three series slots validated together, a neutral for the
# benchmark, ink for text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BENCH = "#3d3c39"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
UP, DOWN, NEUTRAL_BAR = "#2a78d6", "#e34948", "#898781"
DIVERGING = LinearSegmentedColormap.from_list(
    "correlation", [(0, "#8f2222"), (0.25, "#ec8f8e"), (0.5, "#f0efec"), (0.75, "#6da7ec"), (1, "#104281")])


# ---------------------------------------------------------------------------------------------------- the numbers

def ordered(universe: pd.DataFrame) -> pd.DataFrame:
    """The ETFs in the order of the figures: by group, then theme, then their place in the Belfius list. A theme
    outside GROUPS goes last, in a group of its own."""
    themes = [theme for members in GROUPS.values() for theme in members]
    group_of = {theme: group for group, members in GROUPS.items() for theme in members}
    out = universe.assign(group=universe["theme"].map(group_of).fillna("Other"),
                          rank=universe["theme"].map({t: i for i, t in enumerate(themes)}).fillna(len(themes)))
    return out.sort_values(["rank", "id"]).drop(columns="rank").reset_index(drop=True)


def closes(prices: pd.DataFrame) -> pd.DataFrame:
    """Adjusted close in EUR per day (rows, as timestamps) and ticker (columns)."""
    out = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def return_and_risk(prices: pd.DataFrame, universe: pd.DataFrame, years: int = YEARS) -> pd.DataFrame:
    """Per ETF over the last `years` of prices (less when it has fewer): yearly return, yearly volatility of daily
    returns, the first close used and whether the window is complete. One row per ETF with prices, figure order."""
    table = closes(prices)
    end = table.index.max()
    start = end - pd.DateOffset(years=years)
    window = table[table.index >= start]
    rows = []
    for etf in ordered(universe).itertuples():
        series = window[etf.ticker].dropna() if etf.ticker in window else pd.Series(dtype=float)
        if len(series) < 2:
            continue
        span = (series.index[-1] - series.index[0]).days / 365.25
        rows.append({"ticker": etf.ticker, "theme": etf.theme, "group": etf.group,
                     "first": series.index[0].date().isoformat(), "last": series.index[-1].date().isoformat(),
                     "years": round(span, 2), "complete": bool(series.index[0] <= start + pd.Timedelta(days=7)),
                     "ret_year": (series.iloc[-1] / series.iloc[0]) ** (1 / span) - 1,
                     "vol_year": series.pct_change().dropna().std() * np.sqrt(YEAR)})
    return pd.DataFrame(rows)


def weekly_correlation(prices: pd.DataFrame, tickers: list[str], years: int = YEARS) -> pd.DataFrame:
    """Correlation of weekly returns (Friday to Friday) over the last `years`, in the order given. Weekly, because
    the 50 ETFs trade on exchanges that close at different hours: a daily correlation would understate the
    link between a US and a European market. Pairs with less than MIN_WEEKS weeks in common are left empty."""
    table = closes(prices)
    window = table[table.index >= table.index.max() - pd.DateOffset(years=years)]
    weekly = window.resample("W-FRI").last().pct_change(fill_method=None)
    return weekly.reindex(columns=tickers).corr(min_periods=MIN_WEEKS)


def drawdown(values: pd.Series) -> pd.Series:
    """How far below its last peak a value is, at each point (0 at a peak, -0.2 for 20% below)."""
    return values / values.cummax() - 1


def book_numbers(book: pd.DataFrame, summary: pd.Series) -> dict:
    """The key numbers of one daily book: its own path, and its row of system_backtest_daily_summary.csv."""
    return {"start": f"{day(book.index.min())}", "end": f"{day(book.index.max())}",
            "value": book["value"].iloc[-1], "bench": book["bench"].iloc[-1], "ret": summary["ret_year"],
            "bench_ret": summary["bench_ret_year"], "vol": summary["vol_year"], "bench_vol": summary["bench_vol_year"],
            "dd": drawdown(book["value"]).min(), "bench_dd": drawdown(book["bench"]).min(),
            "sharpe": summary["sharpe"], "bench_sharpe": summary["bench_sharpe"], "sharpe_h1": summary["sharpe_h1"],
            "sharpe_h2": summary["sharpe_h2"], "years": summary["years"],
            "sharpe_se": sharpe_error(summary["sharpe"], summary["years"]), "beta": summary["beta"],
            "orders": int(summary["orders"]), "broken": int(summary["broken_rules"])}


def backtest_book(daily: pd.DataFrame, variant: str = "current") -> pd.DataFrame:
    """The daily book of one variant of the system backtest: value and benchmark (the same money in WORLD)."""
    book = daily[daily["variant"] == variant].copy()
    book["date"] = pd.to_datetime(book["date"])
    return book.set_index("date")[["value", "bench", "riskfree", "invested"]]


def hit_rates(weights: pd.DataFrame, level: str = "technical", window: int = HIT_WINDOW) -> pd.DataFrame:
    """Per session (rows) and expert of a level of logs/weights.csv (columns): the share of the last `window`
    sessions in which following the expert earned more than nothing (beat the average ETF, for the technical
    answers). A session in which an expert made no call (result missing or exactly 0) does not count."""
    rows = weights[(weights["level"] == level) & weights["result"].notna() & (weights["result"] != 0)]
    hit = rows.assign(hit=(rows["result"] > 0).astype(float)).pivot_table(index="date", columns="name", values="hit")
    hit.index = pd.to_datetime(hit.index)
    return hit.sort_index().rolling(window, min_periods=window // 2).mean()


def learned(weights: pd.DataFrame) -> pd.DataFrame:
    """Per expert of every level: its weight in the last session logged, the sessions it was judged on and how
    often following it earned more than nothing."""
    rows = []
    for (level, name), frame in weights.groupby(["level", "name"], sort=False):
        frame = frame.sort_values("date", kind="stable")
        judged = frame["result"].dropna()
        judged = judged[judged != 0]
        rows.append({"level": level, "name": name, "since": frame["date"].min(), "weight": frame.iloc[-1]["weight"],
                     "first_weight": frame.iloc[0]["weight"], "date": frame.iloc[-1]["date"],
                     "sessions": len(judged), "hit_rate": float((judged > 0).mean()) if len(judged) else np.nan})
    return pd.DataFrame(rows)


def replayed(answers: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """The three technical answers learned from 1/3 each, session by session: the rule the system ran until 18 November
    2026, replayed by agents/learning.py itself on logs/technical_answers.csv (which still records all three).
    Format of logs/weights.csv, level 'technical'."""
    frame, _ = learning.daily_weights(learning.answer_results(answers), learning.shares(THIRDS), as_of)
    return frame.assign(level="technical", date=frame["date"].astype(str).str[:10])


def mix_curves(prices: pd.DataFrame, since: date = SINCE) -> pd.DataFrame:
    """Value of 1 at the close of `since` in each strategy's fixed starting mix (agents/model_portfolios.curve:
    back to its weights at the start of each month, an ETF without a price yet counts as cash at 0%) and in the
    MSCI World ETF, day by day. The curves start two weeks earlier, so an ETF that did not trade on `since` itself
    carries its last close into it."""
    window = prices[pd.to_datetime(prices["date"]) >= pd.Timestamp(since) - pd.Timedelta(days=14)]
    out = pd.DataFrame({key: model_portfolios.curve(window, key) for key in STRATEGIES})
    out.index = pd.to_datetime(out.index)
    out["world"] = closes(window)[settings.BENCHMARK].ffill().reindex(out.index)
    base = out.index[out.index <= pd.Timestamp(since)].max()
    return out[out.index >= base] / out.loc[base]


def unpriced(prices: pd.DataFrame, strategy: str, day: date) -> list[tuple[str, float, str]]:
    """The ETFs of a strategy's starting mix with no price on `day`: (ticker, weight, first priced day)."""
    first = prices.dropna(subset=["adj_close_eur"]).groupby("ticker")["date"].min()
    out = []
    for ticker, (_, weight) in model_portfolios.weights(strategy).items():
        start = first.get(ticker)
        if start is None or pd.Timestamp(start) > pd.Timestamp(day):
            out.append((ticker, weight, "never" if start is None else str(start)[:10]))
    return out


def path_stats(path: pd.Series, cash: pd.Series) -> dict:
    """Yearly return, yearly volatility, worst fall from a peak and Sharpe ratio over cash of a value path, the way
    agents/model_portfolios.track_record measures a reference portfolio."""
    daily = path.pct_change().dropna()
    over = daily - cash.reindex(daily.index).fillna(0)
    years = len(daily) / YEAR
    return {"ret_year": (path.iloc[-1] / path.iloc[0]) ** (1 / years) - 1, "vol_year": daily.std() * np.sqrt(YEAR),
            "worst_fall": drawdown(path).min(), "sharpe": over.mean() / over.std() * np.sqrt(YEAR), "years": years}


def sharpe_error(sharpe: float, years: float) -> float:
    """The standard error of a Sharpe ratio measured over `years` (Lo, 2002, for independent returns)."""
    return float(np.sqrt((1 + sharpe ** 2 / 2) / years))


# ---------------------------------------------------------------------------------------------------- the figures

def style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
        "legend.fontsize": 7.5, "xtick.major.size": 0, "ytick.major.size": 0, "axes.linewidth": 0.8})


def wrapped(text: str, size: float, width_in: float) -> list[str]:
    """The text cut into lines that fit `width_in` inches at `size` points (an explicit new line is kept)."""
    chars = int(width_in * 72 / (0.53 * size))
    return [line for part in text.split("\n") for line in textwrap.wrap(part, chars) or [""]]


def header(fig, title: str, subtitle: str) -> float:
    """The title and the subtitle at the top of the figure. Returns the figure fraction just below them."""
    width, height = fig.get_figwidth(), fig.get_figheight()
    fig.text(0.2 / width, 1 - 0.18 / height, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    lines = wrapped(subtitle, 8.5, width - 0.4)
    fig.text(0.2 / width, 1 - 0.5 / height, "\n".join(lines), ha="left", va="top", fontsize=8.5, color=INK_2,
             linespacing=1.35)
    return 1 - (0.5 + 0.165 * len(lines) + 0.1) / height


def footer(fig, source: str) -> float:
    """The notes and the source at the bottom of the figure. Returns the figure fraction just above them."""
    width, height = fig.get_figwidth(), fig.get_figheight()
    lines = wrapped(source, 6.5, width - 0.4)
    fig.text(0.2 / width, 0.12 / height, "\n".join(lines), ha="left", va="bottom", fontsize=6.5, color=MUTED,
             linespacing=1.35)
    return (0.12 + 0.125 * len(lines) + 0.1) / height


def save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def day(when) -> str:
    """A date as "3 Jan 2022"."""
    when = pd.Timestamp(when)
    return f"{when.day} {when:%b %Y}"


def signed(value: float, digits: int = 0) -> str:
    """A share with its sign, the minus as a true minus sign (as on the axes)."""
    return f"{value:+.{digits}%}".replace("-", "−")


def spread(values: list[float], gap: float) -> list[float]:
    """Positions for labels at `values`, moved apart so no two are closer than `gap` (order kept)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    shift = (max(placed) - max(values[i] for i in order)) / 2 if placed else 0
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k] - shift
    return out


def end_labels(ax, series: dict[str, tuple[pd.Series, str]], fmt, gap_share: float = 0.055) -> None:
    """A label at the right end of each line: its name and last value, in ink, beside a dot of its colour."""
    low, high = ax.get_ylim()
    names = list(series)
    lasts = [float(series[n][0].dropna().iloc[-1]) for n in names]
    ys = spread(lasts, (high - low) * gap_share)
    x = max(s.dropna().index[-1] for s, _ in series.values())
    for name, last, y in zip(names, lasts, ys):
        line, colour = series[name]
        ax.scatter([line.dropna().index[-1]], [last], s=14, color=colour, zorder=5, edgecolor=SURFACE, linewidth=1)
        ax.annotate(f"{name}  {fmt(last)}", xy=(x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK, annotation_clip=False)


def legend_above(ax, ncols: int) -> None:
    """The legend on the right, just above the plot, on the line of the panel's title."""
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=ncols, borderaxespad=0.3, handlelength=1.6)


def fig_etfs(table: pd.DataFrame, out: Path, end: str) -> Path:
    """Figure 1: two bar panels sharing the rows, return on the left, volatility on the right, ETFs by theme."""
    y, ys, theme_rows, group_rows, last = 0.0, [], {}, {}, (None, None)
    for row in table.itertuples():
        if last[1] is not None and row.group != last[1]:
            y += 1.6
        elif last[0] is not None and row.theme != last[0]:
            y += 0.55
        group_rows.setdefault(row.group, y)
        theme_rows.setdefault(row.theme, []).append(y)
        ys.append(y)
        last = (row.theme, row.group)
        y += 1
    width, height = 8.6, 11.4
    fig = plt.figure(figsize=(width, height))
    start = pd.Timestamp(end) - pd.DateOffset(years=YEARS)
    top = header(fig, "The 50 ETFs: three years of return and risk, by theme",
                 f"Adjusted closes in EUR, {day(start)} to {day(pd.Timestamp(end))}. Return compounded a "
                 "year; volatility of the daily returns, a year. Blue: a gain, red: a loss. The dark line is the "
                 "MSCI World ETF (WORLD), the benchmark.")
    short = table[~table["complete"]]
    note = "; ".join(f"{t} from {pd.Timestamp(f):%b %Y}" for t, f in zip(short["ticker"], short["first"]))
    bottom = footer(fig, f"* Priced for less than three years, figures over what there is: {note}.\n"
                         "Source: Yahoo Finance and EODHD prices (data/data_agent.py); backtest/report_figures.py. "
                         "As a table: docs/midterm/etf_return_risk.csv.")
    top, bottom = top - 0.62 / height, bottom + 0.3 / height
    left, width_ret, width_vol = 0.335, 0.33, 0.27
    ax_ret = fig.add_axes((left, bottom, width_ret, top - bottom))
    ax_vol = fig.add_axes((left + width_ret + 0.035, bottom, width_vol, top - bottom), sharey=ax_ret)
    colours = [UP if r >= 0 else DOWN for r in table["ret_year"]]
    ax_ret.barh(ys, table["ret_year"], height=0.68, color=colours, zorder=3)
    ax_vol.barh(ys, table["vol_year"], height=0.68, color=NEUTRAL_BAR, zorder=3)
    world = table[table["ticker"] == settings.BENCHMARK].iloc[0]
    for ax, value, text in ((ax_ret, world["ret_year"], f"MSCI World {signed(world['ret_year'])}"),
                            (ax_vol, world["vol_year"], f"MSCI World {world['vol_year']:.0%}")):
        ax.axvline(value, color=BENCH, linewidth=0.9, zorder=4)
        ax.annotate(text, xy=(value, -1.4), xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=INK, annotation_clip=False)
        ax.grid(axis="x", zorder=0)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis="x", labelsize=7, labeltop=True, labelbottom=True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax_ret.axvline(0, color=AXIS, linewidth=0.8, zorder=2)
    ax_ret.set_title("Return a year", loc="left", fontsize=9, fontweight="bold", pad=16)
    ax_vol.set_title("Volatility a year (risk)", loc="left", fontsize=9, fontweight="bold", pad=16)
    labels = [f"{t} *" if not c else t for t, c in zip(table["ticker"], table["complete"])]
    ax_ret.set_yticks(ys, labels, fontsize=7)
    ax_ret.set_ylim(y, -2.2)
    plt.setp(ax_vol.get_yticklabels(), visible=False)
    for theme, rows in theme_rows.items():
        ax_ret.annotate(theme, xy=(0.2 / width, rows[0]), xycoords=("figure fraction", "data"), va="center",
                        fontsize=7, color=INK_2)
    for group, row in group_rows.items():
        ax_ret.annotate(group.upper(), xy=(0.2 / width, row - 1.0), xycoords=("figure fraction", "data"),
                        va="center", fontsize=7.5, fontweight="bold", color=INK)
    return save(fig, out / "1_etf_return_risk.png")


def fig_correlation(corr: pd.DataFrame, table: pd.DataFrame, out: Path, end: str) -> Path:
    """Figure 2: the correlation matrix of weekly returns, ETFs by group and theme, white lines between groups."""
    n = len(corr)
    width, height = 9.2, 9.6
    fig = plt.figure(figsize=(width, height))
    start = pd.Timestamp(end) - pd.DateOffset(years=YEARS)
    top = header(fig, "How the 50 ETFs move together",
                 f"Correlation of weekly returns (Friday to Friday, adjusted closes in EUR), {start:%b %Y} to "
                 f"{pd.Timestamp(end):%b %Y}. Blue: they move together; red: against each other; grey: no link.")
    bottom = footer(fig, "Weekly rather than daily returns: the ETFs trade on exchanges that close at different "
                         f"hours. A pair needs {MIN_WEEKS} weeks in common, otherwise it is left blank.\n"
                         "Source: Yahoo Finance and EODHD prices; backtest/report_figures.py.")
    left, labels_below, room_right = 1.3, 1.0, 1.05
    side = min(width - left - room_right, (top - bottom) * height - labels_below - 0.3)
    ax = fig.add_axes((left / width, bottom + labels_below / height, side / width, side / height))
    image = ax.imshow(corr.to_numpy(dtype=float), cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=5.8)
    ax.set_yticks(range(n), corr.index, fontsize=5.8)
    groups = table.set_index("ticker").reindex(corr.index)
    for k in range(1, n):
        if groups["group"].iloc[k] != groups["group"].iloc[k - 1]:
            ax.axhline(k - 0.5, color=SURFACE, linewidth=2.2)
            ax.axvline(k - 0.5, color=SURFACE, linewidth=2.2)
        elif groups["theme"].iloc[k] != groups["theme"].iloc[k - 1]:
            ax.axhline(k - 0.5, color=SURFACE, linewidth=0.5)
            ax.axvline(k - 0.5, color=SURFACE, linewidth=0.5)
    for group in groups["group"].unique():
        rows = np.flatnonzero((groups["group"] == group).to_numpy())
        ax.text((rows[0] + rows[-1]) / 2, -1.2, group, ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=INK)
    for edge in ax.spines.values():
        edge.set_visible(False)
    box = ax.get_position()
    bar = fig.add_axes((box.x1 + 0.3 / width, box.y0 + box.height * 0.25, 0.16 / width, box.height * 0.5))
    colourbar = fig.colorbar(image, cax=bar, ticks=[-1, -0.5, 0, 0.5, 1])
    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(labelsize=7)
    colourbar.set_label("Correlation", fontsize=7.5, color=INK_2)
    return save(fig, out / "2_etf_correlation.png")


def fig_backtest(books: dict[str, pd.DataFrame], out: Path) -> Path:
    """Figure 3: the value of the system's book (and of the earlier settings when given) and of the same money in
    WORLD, then how far each fell from its last peak."""
    fig, (top_ax, low) = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.35]})
    first = books["current"]
    start, end = first.index.min(), first.index.max()
    top = header(fig, "The system replayed since 2022 against the MSCI World",
                 f"One book of €10,000 from {day(start)} to {day(end)}: decisions every Monday and Thursday "
                 "under the competition's rules, at most 5 orders a day. Code only and walk-forward: each decision "
                 "uses only what was known that morning. The LLM agents are not replayed: an LLM already knows what "
                 "happened.")
    bottom = footer(fig, "Source: backtest/results/system_backtest_daily.csv (python -m backtest.system_backtest "
                         "--daily), variants " + ", ".join(f"'{k}'" for k in books) + ". MSCI World: the same "
                         "€10,000 in the WORLD ETF.")
    fig.subplots_adjust(left=0.08, right=0.82, top=top - 0.3 / 6.6, bottom=bottom + 0.3 / 6.6, hspace=0.16)
    lines = {key: (book["value"], SERIES[i], VARIANTS.get(key, key), SHORT.get(key, key))
             for i, (key, book) in enumerate(books.items())}
    lines["world"] = (first["bench"], BENCH, "MSCI World", "MSCI World")
    falls = []
    for key, (line, colour, name, _) in lines.items():
        main = key == "current"
        top_ax.plot(line.index, line, color=colour, linewidth=1.7 if main else 1.2, label=name, zorder=4 if main else 3)
        fall = drawdown(line)
        low.plot(fall.index, fall, color=colour, linewidth=1.3 if main else 1.0, zorder=4 if main else 3)
        falls.append((fall.idxmin(), fall.min(), colour))
    for (when, value, colour), y in zip(falls, spread([v for _, v, _ in falls], 0.022)):
        low.scatter([when], [value], s=12, color=colour, zorder=5, edgecolor=SURFACE, linewidth=1)
        low.annotate(f"worst {signed(value)}", xy=(when, y), xytext=(6, 0), textcoords="offset points", fontsize=7,
                     color=INK, va="center")
    top_ax.axhline(10_000, color=AXIS, linewidth=0.8)
    top_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    top_ax.set_title("Value of the book", loc="left", fontsize=8.5, fontweight="bold")
    end_labels(top_ax, {short: (line, colour) for line, colour, _, short in lines.values()}, lambda v: f"€{v:,.0f}")
    top_ax.legend(loc="upper left", ncols=1, handlelength=1.6)
    low.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    low.axhline(0, color=AXIS, linewidth=0.8)
    low.set_title("Fall from the last peak", loc="left", fontsize=8.5, fontweight="bold")
    for ax in (top_ax, low):
        ax.grid(axis="y")
        ax.spines["left"].set_visible(False)
    return save(fig, out / "3_system_backtest.png")


def fig_learning(three: pd.DataFrame, hits: pd.DataFrame, out: Path) -> Path:
    """Figure 4: the weight the learning rule gave each technical answer, session by session, then its rolling hit
    rate, with the day the team kept the trend rule alone."""
    tech = three.pivot_table(index="date", columns="name", values="weight")
    tech.index = pd.to_datetime(tech.index)
    names = [n for n in ANSWERS if n in tech.columns] + [n for n in tech.columns if n not in ANSWERS]
    colour = {name: SERIES[i % len(SERIES)] for i, name in enumerate(names)}
    fig, (top_ax, low) = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    top = header(fig, "What the learning found: the trend answer earned its weight",
                 f"The technical agent's three answers started at 1/3 each on {day(tech.index.min())}. After "
                 f"every session each weight is multiplied by e^({settings.LEARNING_SPEED:g} × what following that "
                 f"answer earned), then the weights are put back to sum to 100%, each between "
                 f"{settings.WEIGHT_MIN:.0%} and {settings.WEIGHT_MAX:.0%} (agents/learning.py, as live). On 19 Nov "
                 "2026 the team kept the trend rule alone: the backtest of figure 3 favoured it.")
    bottom = footer(fig, "Source: logs/technical_answers.csv (every answer of every session since 2022, still "
                         "recorded), replayed with agents/learning.py (answer_results, daily_weights) from 1/3 each: "
                         "the weights logs/weights.csv held until 18 Nov 2026. A weight grows with the size of the "
                         "wins, a hit rate only counts them.")
    fig.subplots_adjust(left=0.08, right=0.82, top=top - 0.3 / 6.8, bottom=bottom + 0.3 / 6.8, hspace=0.2)
    for bound, label in ((settings.WEIGHT_MIN, "floor"), (settings.WEIGHT_MAX, "cap")):
        top_ax.axhline(bound, color=AXIS, linewidth=0.8)
        top_ax.annotate(f"{label} {bound:.0%}", xy=(tech.index.min(), bound), xytext=(2, 3),
                        textcoords="offset points", fontsize=6.5, color=MUTED)
    for name in names:
        top_ax.plot(tech.index, tech[name], color=colour[name], linewidth=1.4, label=ANSWERS.get(name, name))
        if name in hits:
            low.plot(hits.index, hits[name], color=colour[name], linewidth=1.2)
    top_ax.set_ylim(0, 0.85)
    low.axhline(0.5, color=BENCH, linewidth=0.8)
    low.annotate("a coin flip", xy=(hits.index.min(), 0.5), xytext=(2, 3), textcoords="offset points", fontsize=6.5,
                 color=MUTED)
    for ax in (top_ax, low):
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y")
        ax.spines["left"].set_visible(False)
    end_labels(top_ax, {ANSWERS.get(n, n): (tech[n], colour[n]) for n in names}, lambda v: f"{v:.0%}", 0.08)
    end_labels(low, {ANSWERS.get(n, n): (hits[n], colour[n]) for n in names if n in hits}, lambda v: f"{v:.0%}", 0.12)
    legend_above(top_ax, len(names))
    top_ax.set_title("Weight of each answer in the technical score", loc="left", fontsize=8.5, fontweight="bold")
    low.set_title(f"Hit rate: share of the last {HIT_WINDOW} sessions in which its calls beat the average ETF",
                  loc="left", fontsize=8.5, fontweight="bold")
    return save(fig, out / "4_learning.png")


def fig_strategies(curves: pd.DataFrame, out: Path, notes: str) -> Path:
    """Figure 5: what 100 became in each strategy's starting mix and in the MSCI World."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    top = header(fig, "The three strategies' starting mixes against the MSCI World",
                 f"What 100 became from {day(curves.index.min())} to {day(curves.index.max())} in each "
                 "strategy's fixed starting mix (data/profiles.py), put back to its weights at the start of every "
                 "month. Adjusted closes in EUR, no costs.")
    bottom = footer(fig, f"{notes}\nSource: agents/model_portfolios.py (curve, the same as the customer app's "
                         "chart); backtest/report_figures.py.")
    fig.subplots_adjust(left=0.07, right=0.8, top=top - 0.3 / 5.2, bottom=bottom + 0.3 / 5.2)
    lines = {STRATEGIES[key]["label"]: (curves[key] * 100, SERIES[i]) for i, key in enumerate(STRATEGIES)}
    lines["MSCI World"] = (curves["world"] * 100, BENCH)
    for name, (line, colour) in lines.items():
        ax.plot(line.index, line, color=colour, linewidth=1.2 if name == "MSCI World" else 1.5, label=name)
    ax.axhline(100, color=AXIS, linewidth=0.8)
    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)
    ax.set_title("Value of 100", loc="left", fontsize=8.5, fontweight="bold")
    end_labels(ax, lines, lambda v: f"{v:.0f}")
    legend_above(ax, 4)
    return save(fig, out / "5_strategies.png")


# ---------------------------------------------------------------------------------------------------- the page

def pct(value: float, sign: bool = False) -> str:
    return f"{value:+.1%}" if sign else f"{value:.1%}"


def page(numbers: dict) -> str:
    """docs/midterm/data_exploration.md, from the numbers `report` computed."""
    n = numbers
    etfs, corr, strat = n["etfs"], n["corr"], n["strategies"]
    book, old = n["books"]["current"], n["books"].get("learned_three")
    three = n["three"].set_index("name")
    top_answer = three["weight"].idxmax()
    hit_text = ", ".join(f"{ANSWERS.get(k, k).lower()} {pct(three.loc[k, 'hit_rate'])}"
                         for k in [*[a for a in ANSWERS if a in three.index],
                                   *[a for a in three.index if a not in ANSWERS]])
    calm = [strat[k]["ret_year"] for k in [*STRATEGIES, "world"]]
    falls = [strat[k]["worst_fall"] for k in [*STRATEGIES, "world"]]
    ordered_risk = calm == sorted(calm) and falls == sorted(falls, reverse=True)
    lines = [
        "# Data exploration and preliminary results",
        "",
        f"Written by `python -m backtest.report_figures` on {n['made']}: run it again and this page and the figures "
        f"are rewritten. Prices: adjusted closes in EUR of the 50 ETFs (Yahoo Finance and EODHD, `data/data_agent.py`) "
        f"up to {day(n['end'])}. Backtest: `backtest/results/`. Learning: `logs/technical_answers.csv` and "
        "`logs/weights.csv`.",
        "",
        "## 1. The 50 ETFs over three years",
        "",
        "![Return and volatility of the 50 ETFs](1_etf_return_risk.png)",
        "",
        f"Over the three years to {day(n['end'])}, the median ETF returned {pct(etfs['median_ret'])} a year with a "
        f"volatility of {pct(etfs['median_vol'])}; the MSCI World ETF returned {pct(etfs['world_ret'])} for "
        f"{pct(etfs['world_vol'])}. Returns run from {etfs['worst']} ({pct(etfs['worst_ret'], True)} a year) to "
        f"{etfs['best']} ({pct(etfs['best_ret'], True)}), and risk from {etfs['calmest']} "
        f"({pct(etfs['calmest_vol'])}) to {etfs['wildest']} ({pct(etfs['wildest_vol'])}): the list mixes very "
        "different kinds of risk, which is why each strategy caps themes and complex products. "
        f"{etfs['short_n']} ETFs have less than three years of prices: their figures cover what there is.",
        "",
        "## 2. How the ETFs move together",
        "",
        "![Correlation of weekly returns](2_etf_correlation.png)",
        "",
        f"Weekly returns, because the ETFs trade on exchanges that close at different hours. The median correlation "
        f"between two ETFs is {corr['median']:.2f}, and {corr['above_07_n']} pairs of the {corr['pairs']:,} are above "
        f"0.7 ({pct(corr['above_07'])}), the closest {corr['closest']}: inside such a pair, holding both spreads "
        "little risk. The inverse ETFs are the only ones that move against the rest (the most negative pair, "
        f"{corr['min_pair']} at {corr['min']:.2f}, is an index and its own inverse). "
        f"Bonds are at {corr['bonds_equity']:.2f} on average against broad equity, and gold at "
        f"{corr['gold_world']:.2f} against the MSCI World: they are what spreads the risk of the Defensive and "
        "Balanced mixes.",
        "",
        "## 3. The system replayed since 2022",
        "",
        "![The system against the MSCI World](3_system_backtest.png)",
        "",
        f"One book of €10,000 from {book['start']} to {book['end']}, code only (an LLM cannot be replayed on the "
        "past: it already knows what happened), decisions every Monday and Thursday under the competition's rules, "
        f"each from what was known that morning. The system ends at €{book['value']:,.0f} against "
        f"€{book['bench']:,.0f} in the MSCI World: {pct(book['ret'], True)} a year against "
        f"{pct(book['bench_ret'], True)}, with a volatility of {pct(book['vol'])} against {pct(book['bench_vol'])} "
        f"and a worst fall of {pct(book['dd'])} against {pct(book['bench_dd'])}. Its Sharpe ratio is "
        f"{book['sharpe']:.2f} against {book['bench_sharpe']:.2f}, "
        f"{book['sharpe_h1']:.2f} in the first half of the history and {book['sharpe_h2']:.2f} in the second"
        + (": most of the edge is recent" if book["sharpe_h2"] > 1.5 * book["sharpe_h1"] else "")
        + f". Over {book['years']:.1f} years a Sharpe ratio is known to about ±{book['sharpe_se']:.2f}, so the gap "
          "with the market is a sign, not a proof.",
    ]
    if old:
        lines += ["", f"The three technical answers learned together, as the system ran until 18 November, end at "
                      f"€{old['value']:,.0f} with a Sharpe ratio of {old['sharpe']:.2f} and a worst fall of "
                      f"{pct(old['dd'])}. The trend rule alone "
                      + ("did better in both halves of the history" if book["sharpe_h1"] > old["sharpe_h1"]
                         and book["sharpe_h2"] > old["sharpe_h2"] else "did better over the whole history, not in "
                         "both halves" if book["sharpe"] > old["sharpe"] else "did not do better")
                      + (", with a smaller fall" if book["dd"] > old["dd"] else "")
                      + ", and it is one rule to explain instead of three weights: on 19 November the team kept it "
                        "alone (`agents/strategy_config.py`)."]
    if n.get("windows"):
        w = n["windows"]
        lines += ["", f"Replayed as {w['n']} past competitions (three weeks, five trading days each), the system "
                      f"made {pct(w['mean'], True)} a window on average against {pct(w['bench'], True)} for the "
                      f"MSCI World, beat it in {pct(w['beat'])} of the windows, and its worst window was "
                      f"{pct(w['worst'], True)} (`system_backtest_summary.csv`)."]
    lines += [
        "",
        "## 4. What the learning found",
        "",
        "![Learned weights and hit rates](4_learning.png)",
        "",
        f"The learning rule (`agents/learning.py`) multiplies each weight by e^({settings.LEARNING_SPEED:g} × what "
        "following that expert earned) after every session. Starting the three technical answers at 1/3 each on "
        f"{day(three['since'].min())}, it gave the most weight to {ANSWERS.get(top_answer, top_answer).lower()} "
        f"({pct(three.loc[top_answer, 'weight'])} by {day(three.loc[top_answer, 'date'])}). Session by session, each "
        f"answer beats the average ETF about as often as a coin flip ({hit_text} of the sessions it made calls): "
        "the trend answer earned its weight by the size of its wins, not by being right more often. The rule still "
        "weighs the four agents and the LLM's portfolio against the code's, from their first live score: their "
        "record is a few sessions long (table).",
        "",
        "| Level | Expert | Weight now | Since | Sessions judged | Hit rate |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in n["learned"].itertuples():
        hit = "" if np.isnan(row.hit_rate) else pct(row.hit_rate)
        lines.append(f"| {row.level} | {ANSWERS.get(row.name, row.name)} | {pct(row.weight)} | {row.since} | "
                     f"{row.sessions} | {hit} |")
    lines += [
        "",
        "## 5. The three strategies' starting mixes",
        "",
        "![The three strategies against the MSCI World](5_strategies.png)",
        "",
        "Each strategy's fixed starting mix (`data/profiles.py`), put back to its weights at the start of every "
        "month: the reference portfolio a customer sees before our engine publishes a version of it. "
        + " ".join(f"{STRATEGIES[k]['label']} returned {pct(v['ret_year'], True)} a year with a worst fall of "
                   f"{pct(v['worst_fall'])}." for k, v in strat.items() if k != "world")
        + f" The MSCI World returned {pct(strat['world']['ret_year'], True)} with a worst fall of "
          f"{pct(strat['world']['worst_fall'])}"
        + (": the calmer the mix, the smaller both the fall and the return. " if ordered_risk else ". ")
        + n["unpriced_text"],
        "",
        "## Key numbers",
        "",
        "| Measure | Value | Source |",
        "|---|---:|---|",
        f"| ETFs with prices | {etfs['count']} of 50 | prices |",
        f"| Median return / volatility a year, 3 years | {pct(etfs['median_ret'])} / {pct(etfs['median_vol'])} | "
        "prices |",
        f"| MSCI World return / volatility a year, 3 years | {pct(etfs['world_ret'])} / {pct(etfs['world_vol'])} | "
        "prices |",
        f"| Median correlation of two ETFs (weekly) | {corr['median']:.2f} | prices |",
        f"| Pairs above 0.7 | {pct(corr['above_07'])} | prices |",
        f"| System: value of €10,000, {book['start']} to {book['end']} | €{book['value']:,.0f} (MSCI World "
        f"€{book['bench']:,.0f}) | system_backtest_daily.csv |",
        f"| System: return / volatility a year | {pct(book['ret'], True)} / {pct(book['vol'])} (MSCI World "
        f"{pct(book['bench_ret'], True)} / {pct(book['bench_vol'])}) | system_backtest_daily_summary.csv |",
        f"| System: Sharpe ratio (± its error) | {book['sharpe']:.2f} ± {book['sharpe_se']:.2f} (MSCI World "
        f"{book['bench_sharpe']:.2f}) | system_backtest_daily_summary.csv |",
        f"| System: worst fall | {pct(book['dd'])} (MSCI World {pct(book['bench_dd'])}) | system_backtest_daily.csv |",
        f"| System: share of the market's moves (beta) | {book['beta']:.2f} | system_backtest_daily_summary.csv |",
        f"| System: orders over the whole history / trading rules broken | {book['orders']:,} / {book['broken']} | "
        "system_backtest_daily_summary.csv |",
    ]
    if old:
        lines.append(f"| Three answers learned (until 18 Nov): Sharpe / worst fall | {old['sharpe']:.2f} / "
                     f"{pct(old['dd'])} | system_backtest_daily_summary.csv |")
    if n.get("windows"):
        w = n["windows"]
        lines.append(f"| Past competitions replayed / share that beat the MSCI World | {w['n']} / {pct(w['beat'])} | "
                     "system_backtest_summary.csv |")
    if "trend" in three.index:
        lines.append(f"| Trend answer's weight, learned from 1/3 | {pct(three.loc['trend', 'weight'])} | "
                     "technical_answers.csv, agents/learning.py |")
    for key, v in strat.items():
        name = "MSCI World" if key == "world" else STRATEGIES[key]["label"]
        lines.append(f"| {name} since {n['strategies_since']}: return / volatility / worst fall / Sharpe | "
                     f"{pct(v['ret_year'], True)} / {pct(v['vol_year'])} / {pct(v['worst_fall'])} / {v['sharpe']:.2f} "
                     "| prices, data/profiles.py |")
    lines += ["", "## Limits of these numbers", "", *[f"- {limit}" for limit in n["limits"]], ""]
    return "\n".join(lines)


def test_count() -> int | None:
    """How many tests pytest collects, or None when it cannot tell."""
    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True,
                          text=True)
    found = re.search(r"(\d+) tests? collected", done.stdout)
    return int(found.group(1)) if found else None


def appendix(out: Path, count=test_count) -> Path | None:
    """code_appendix.md with its test count brought up to date (`count`: how to count them), and its PDF
    (app/pdf.py). Nothing when `out` holds no code_appendix.md."""
    source = out / "code_appendix.md"
    if not source.exists():
        return None
    text = source.read_text()
    tests = count()
    if tests is not None:
        text = re.sub(r"\*\*\d[\d,]* automated tests\*\*", f"**{tests:,} automated tests**", text)
        source.write_text(text)
    from app.pdf import to_pdf
    target = out / "code_appendix.pdf"
    target.write_bytes(to_pdf(text))
    return target


def report(prices: pd.DataFrame, out: Path = OUT, logs: Path | None = None, results: Path = RESULTS) -> dict:
    """Every figure and the page, into `out`. Returns the numbers behind them."""
    logs = logs or settings.LOGS_DIR
    out.mkdir(parents=True, exist_ok=True)
    style()
    universe = load_universe()
    end = str(pd.to_datetime(prices["date"]).max().date())
    numbers = {"end": end, "made": date.today().isoformat(), "limits": []}

    table = return_and_risk(prices, universe)
    table.to_csv(out / "etf_return_risk.csv", index=False, float_format="%.4f")
    short = table[~table["complete"]]
    world = table.set_index("ticker").loc[settings.BENCHMARK]
    numbers["etfs"] = {
        "count": len(table), "median_ret": table["ret_year"].median(), "median_vol": table["vol_year"].median(),
        "world_ret": world["ret_year"], "world_vol": world["vol_year"],
        "best": table.loc[table["ret_year"].idxmax(), "ticker"], "best_ret": table["ret_year"].max(),
        "worst": table.loc[table["ret_year"].idxmin(), "ticker"], "worst_ret": table["ret_year"].min(),
        "calmest": table.loc[table["vol_year"].idxmin(), "ticker"], "calmest_vol": table["vol_year"].min(),
        "wildest": table.loc[table["vol_year"].idxmax(), "ticker"], "wildest_vol": table["vol_year"].max(),
        "short_n": len(short),
        "short": ", ".join(f"{t} from {pd.Timestamp(f):%b %Y}" for t, f in zip(short["ticker"], short["first"]))}
    figures = [fig_etfs(table, out, end)]

    corr = weekly_correlation(prices, table["ticker"].tolist())
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().dropna()
    groups = table.set_index("ticker")
    bonds = groups.index[groups["theme"] == "Fixed income / Bonds"]
    broad = groups.index[groups["group"] == "Broad equity"]
    low = pairs.idxmin()
    closest = pairs.sort_values(ascending=False).head(3)
    numbers["corr"] = {"median": pairs.median(), "above_07": (pairs > 0.7).mean(), "min": pairs.min(),
                       "above_07_n": int((pairs > 0.7).sum()), "pairs": len(pairs),
                       "closest": ", ".join(f"{a} and {b} ({v:.2f})" for (a, b), v in closest.items()),
                       "min_pair": f"{low[0]} and {low[1]}",
                       "bonds_equity": float(np.nanmean(corr.loc[bonds, broad].to_numpy(dtype=float))),
                       "gold_world": corr.loc["GOLD", settings.BENCHMARK] if "GOLD" in corr else np.nan}
    figures.append(fig_correlation(corr, table, out, end))

    daily = pd.read_csv(results / "system_backtest_daily.csv")
    summary = pd.read_csv(results / "system_backtest_daily_summary.csv").set_index("variant")
    books = {key: backtest_book(daily, key) for key in ["current", *[k for k in VARIANTS if k != "current"]]
             if key in set(daily["variant"])}
    numbers["books"] = {key: book_numbers(book, summary.loc[key]) for key, book in books.items()}
    windows_file = results / "system_backtest_summary.csv"
    if windows_file.exists():  # the 3-week windows, only when they were run on the same system as the daily book
        windows = pd.read_csv(windows_file).set_index("variant")
        if "current" in windows.index and windows.loc["current", "what"] == summary.loc["current", "what"]:
            w = windows.loc["current"]
            numbers["windows"] = {"n": int(w["windows"]), "mean": w["mean_ret"], "bench": w["mean_bench"],
                                  "beat": w["beat_bench"], "worst": w["worst_ret"]}
    figures.append(fig_backtest(books, out))

    answers = pd.read_csv(logs / "technical_answers.csv")
    three = replayed(answers, date.fromisoformat(str(answers["date"].max())[:10]))
    numbers["three"] = learned(three)
    numbers["learned"] = learned(pd.read_csv(logs / "weights.csv"))
    figures.append(fig_learning(three, hit_rates(three), out))

    curves = mix_curves(prices)
    cash = closes(prices)[model_portfolios.CASH].pct_change(fill_method=None).fillna(0)
    numbers["strategies"] = {key: path_stats(curves[key], cash) for key in [*STRATEGIES, "world"]}
    gaps = {key: unpriced(prices, key, SINCE) for key in STRATEGIES}
    share = {key: sum(w for _, w, _ in found) for key, found in gaps.items()}
    late = sorted({(t, first) for found in gaps.values() for t, _, first in found}, key=lambda x: x[1])
    note = ("Before its first price an ETF counts as cash at 0%: " + ", ".join(
        f"{t} (priced from {pd.Timestamp(f):%b %Y})" for t, f in late) + "; at the start that is "
        + ", ".join(f"{share[k]:.0%} of {STRATEGIES[k]['label']}" for k in STRATEGIES) + ".") if late else ""
    numbers["unpriced_text"] = note
    numbers["strategies_since"] = f"{curves.index.min():%b %Y}"
    figures.append(fig_strategies(curves, out, note))

    cash_first = prices.loc[prices["ticker"] == model_portfolios.CASH, "date"].min()
    numbers["limits"] = [
        f"The risk-free rate is the OVERNIGHT CASH ETF, priced only from {day(cash_first)}: before then it counts as "
        "0%, which flatters every Sharpe ratio here a little (the ECB deposit rate reached 4% in 2023).",
        "Some ETFs have a shorter price history than the fund itself (the price source's listing): "
        f"{numbers['etfs']['short']}.",
        "The backtest replays the code only: the Macro/News agents, the LLM's technical reading and the LLM's "
        "portfolio are judged forward, by the learning step, from their first live day (9 November 2026).",
        "No transaction costs, as in the competition; orders fill at the last close before their day "
        "(`FILL_AT_PREVIOUS_CLOSE`, our reading of an open rule).",
    ]
    (out / "data_exploration.md").write_text(page(numbers))
    numbers["figures"] = figures
    numbers["appendix"] = appendix(out)
    return numbers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prices", type=Path, help="prices already saved (Prices format) instead of fetching them")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.prices:
        prices = pd.read_csv(args.prices)
    else:
        from data import data_agent
        prices = data_agent.fetch_prices(SINCE - timedelta(days=30), date.today())
    numbers = report(prices, args.out)
    for path in numbers["figures"] + [args.out / "data_exploration.md", numbers["appendix"]]:
        if path:
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
