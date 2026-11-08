"""Altair charts for the terminal cockpit: thin marks, hairline grid, mono axis labels, tooltips.

Blue = up / buy, red = down / sell, dark gray = neutral (colour-blind-validated dark steps).
"""
import altair as alt
import pandas as pd

from app.ui import MONO, T

BAR = 16  # bar thickness in px (<= 24)
ROW = 22  # heat-map row height, enough for every ETF label
FONT = MONO.split(",")[0].strip("'")


def finish(chart, height: int) -> alt.Chart:
    """Terminal styling shared by every chart."""
    return chart.properties(height=height, background="transparent").configure_view(stroke=None).configure_axis(
        gridColor=T["line"], domainColor=T["line"], tickColor=T["line"], labelColor=T["muted"],
        titleColor=T["muted"], labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11,
        titleFontWeight="normal",
    ).configure_legend(labelColor=T["ink"], titleColor=T["muted"], labelFont=FONT, titleFont=FONT,
                       titleFontWeight="normal").configure_text(font=FONT)


def heatmap(df: pd.DataFrame, columns: list[str]) -> alt.Chart:
    """ETF x source grid of scores in [-1, 1]."""
    order = df.drop_duplicates("ticker")["ticker"].tolist()
    chart = alt.Chart(df).mark_rect(cornerRadius=2).encode(
        x=alt.X("source:N", sort=columns, title=None,
                axis=alt.Axis(orient="top", labelAngle=0, domain=False, ticks=False)),
        y=alt.Y("ticker:N", sort=order, title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=200, domain=False, ticks=False)),
        color=alt.Color("score:Q", title="score",
                        scale=alt.Scale(domain=[-1, 0, 1], range=[T["negative"], T["neutral"], T["positive"]],
                                        interpolate="lab"),
                        legend=alt.Legend(orient="top", gradientLength=200, direction="horizontal")),
        tooltip=[alt.Tooltip("ticker:N", title="ETF"), alt.Tooltip("theme:N", title="Theme"),
                 alt.Tooltip("source:N", title="Source"), alt.Tooltip("score:Q", format="+.2f"),
                 alt.Tooltip("confidence:Q", format=".2f"), alt.Tooltip("rationale:N")],
    )
    return finish(chart, max(200, ROW * len(order)))


def ranked(df: pd.DataFrame, value: str, label: str, color: str, domain: list, fmt: str, tooltip: list) -> alt.Chart:
    """Ranked horizontal bars of one colour, value at the tip."""
    positive = domain[1] > 0
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort="-x" if positive else "x", title=None,
                                        axis=alt.Axis(labelLimit=200, domain=False, ticks=False, grid=False)))
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=3, color=color).encode(
        x=alt.X(f"{value}:Q", scale=alt.Scale(domain=domain), title=None, axis=None), tooltip=tooltip)
    text = base.mark_text(align="left" if positive else "right", dx=6 if positive else -6,
                          color=T["ink"], fontSize=11).encode(x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=fmt))
    return finish(bars + text, max(120, 26 * len(df)))


def factor_bars(df: pd.DataFrame) -> alt.Chart:
    """Macro views from most bullish to most bearish; faded bars = low confidence."""
    order = df.sort_values("direction", ascending=False)["name"].tolist()
    base = alt.Chart(df).encode(y=alt.Y("name:N", sort=order, title=None,
                                        axis=alt.Axis(labelLimit=220, domain=False, ticks=False, grid=False)))
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=3).encode(
        x=alt.X("direction:Q", scale=alt.Scale(domain=[-1, 1]), title=None,
                axis=alt.Axis(format="+.1f", values=[-1, -0.5, 0, 0.5, 1])),
        color=alt.condition("datum.direction >= 0", alt.value(T["positive"]), alt.value(T["negative"])),
        opacity=alt.Opacity("confidence:Q", scale=alt.Scale(domain=[0, 1], range=[0.3, 1]), legend=None),
        tooltip=[alt.Tooltip("name:N", title="Factor"), alt.Tooltip("direction:Q", format="+.2f"),
                 alt.Tooltip("confidence:Q", format=".0%"), alt.Tooltip("meaning:N", title="+1 means")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T["rule"]).encode(x="x:Q")
    return finish(bars + zero, max(160, 26 * len(df)))


def lines(df: pd.DataFrame, x: str, y: str, series: str, y_title: str, y_format: str,
          zero_based: bool = False) -> alt.Chart:
    """2 px lines, one per series (max 3, fixed colour order), legend on top, value at each line end."""
    names = list(dict.fromkeys(df[series]))
    color = alt.Color(f"{series}:N", scale=alt.Scale(domain=names, range=T["series"][:len(names)]),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(df).encode(
        x=alt.X(f"yearmonthdate({x}):T", title=None,
                axis=alt.Axis(format="%a %d %b", tickCount={"interval": "day", "step": 1}, labelAngle=0, grid=False)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_format, tickCount=5),
                scale=alt.Scale(zero=zero_based)),
        color=color,
    )
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    line = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    points = base.mark_point(filled=True, size=60, stroke=None).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:T", title="Date", format="%a %d %b %Y"), alt.Tooltip(f"{series}:N", title=""),
                 alt.Tooltip(f"{y}:Q", format=y_format, title=y_title)],
    ).add_params(hover)
    last = df.sort_values(x).groupby(series).tail(1)
    labels = alt.Chart(last).mark_text(align="left", dx=8, color=T["ink"], fontSize=11).encode(
        x=f"yearmonthdate({x}):T", y=f"{y}:Q", text=alt.Text(f"{y}:Q", format=y_format))
    return finish(line + points + labels, 300)


def hbars(df: pd.DataFrame, value: str, label: str, value_format: str, tooltip: list,
          rule_at: float | None = None, domain: tuple = (0, 1)) -> alt.Chart:
    """One-series horizontal bars with the value at the tip and an optional reference rule (e.g. 50%)."""
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort="-x", title=None,
                                        axis=alt.Axis(labelLimit=260, domain=False, ticks=False, grid=False)))
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=3, color=T["series"][0]).encode(
        x=alt.X(f"{value}:Q", title=None, axis=alt.Axis(format=value_format, tickCount=5),
                scale=alt.Scale(domain=list(domain))), tooltip=tooltip)
    text = base.mark_text(align="left", dx=6, color=T["ink"], fontSize=11).encode(
        x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=value_format))
    layers = bars + text
    if rule_at is not None:
        layers += alt.Chart(pd.DataFrame({"x": [rule_at]})).mark_rule(color=T["rule"]).encode(x="x:Q")
    return finish(layers, max(110, 28 * len(df)))
