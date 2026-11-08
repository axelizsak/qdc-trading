"""Altair charts for the cockpit. Thin marks, hairline grid, tooltips, colours from app.ui.tokens().

Charts render with Streamlit's theme, so axes, text and grid follow light or dark mode.
"""
import altair as alt
import pandas as pd

BAR = 18  # bar thickness in px (<= 24)
ROW = 22  # heat-map row height in px, enough for every ETF label


def heatmap(df: pd.DataFrame, columns: list[str], tok: dict) -> alt.Chart:
    """ETF x source grid of scores in [-1, 1]; diverging blue (buy) / red (sell) around a gray midpoint."""
    order = df.drop_duplicates("ticker")["ticker"].tolist()
    return alt.Chart(df).mark_rect(cornerRadius=2, stroke=None).encode(
        x=alt.X("source:N", sort=columns, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("ticker:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
        color=alt.Color("score:Q", title="Score",
                        scale=alt.Scale(domain=[-1, 0, 1], range=[tok["negative"], tok["neutral"], tok["positive"]],
                                        interpolate="lab"),
                        legend=alt.Legend(orient="top", gradientLength=220)),
        tooltip=[alt.Tooltip("ticker:N", title="ETF"), alt.Tooltip("theme:N", title="Theme"),
                 alt.Tooltip("source:N", title="Source"), alt.Tooltip("score:Q", format="+.2f"),
                 alt.Tooltip("confidence:Q", format=".2f"), alt.Tooltip("rationale:N")],
    ).properties(height=max(200, ROW * len(order)))


def diverging_bars(df: pd.DataFrame, value: str, label: str, tok: dict, tooltip: list) -> alt.LayerChart:
    """Horizontal bars from a zero baseline, blue when positive, red when negative, domain [-1, 1]."""
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort=df[label].tolist(), title=None,
                                        axis=alt.Axis(labelLimit=220)))
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=4).encode(
        x=alt.X(f"{value}:Q", scale=alt.Scale(domain=[-1, 1]), title=None, axis=alt.Axis(format="+.1f")),
        color=alt.condition(f"datum.{value} >= 0", alt.value(tok["positive"]), alt.value(tok["negative"])),
        tooltip=tooltip,
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=tok["rule"], strokeWidth=1).encode(x="x:Q")
    return (bars + zero).properties(height=max(160, 28 * len(df)))


def lines(df: pd.DataFrame, x: str, y: str, series: str, tok: dict, y_title: str, y_format: str,
          zero_based: bool = False) -> alt.LayerChart:
    """2 px lines, one per series (max 3, fixed colour order), legend on top plus end labels."""
    names = list(dict.fromkeys(df[series]))
    color = alt.Color(f"{series}:N", scale=alt.Scale(domain=names, range=tok["series"][:len(names)]),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(df).encode(
        x=alt.X(f"yearmonthdate({x}):T", title=None,
                axis=alt.Axis(format="%a %d %b", tickCount={"interval": "day", "step": 1}, labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_format), scale=alt.Scale(zero=zero_based)),
        color=color,
    )
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    line = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    points = base.mark_point(filled=True, size=70, stroke=None).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:T", title="Date", format="%a %d %b %Y"), alt.Tooltip(f"{series}:N", title=""),
                 alt.Tooltip(f"{y}:Q", format=y_format, title=y_title)],
    ).add_params(hover)
    last = df.sort_values(x).groupby(series).tail(1)
    labels = alt.Chart(last).mark_text(align="left", dx=8, color=tok["ink"]).encode(
        x=f"yearmonthdate({x}):T", y=f"{y}:Q", text=alt.Text(f"{y}:Q", format=y_format))
    return (line + points + labels).properties(height=300)


def hbars(df: pd.DataFrame, value: str, label: str, tok: dict, value_format: str, tooltip: list,
          rule_at: float | None = None, domain: tuple | None = (0, 1)) -> alt.LayerChart:
    """One-series horizontal bars (slot 1) with the value at the tip, optional reference rule."""
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=4, color=tok["series"][0]).encode(
        x=alt.X(f"{value}:Q", title=None, axis=alt.Axis(format=value_format, tickCount=5),
                scale=alt.Scale(domain=list(domain)) if domain else alt.Undefined), tooltip=tooltip)
    text = base.mark_text(align="left", dx=4, color=tok["ink"]).encode(x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=value_format))
    layers = bars + text
    if rule_at is not None:
        layers += alt.Chart(pd.DataFrame({"x": [rule_at]})).mark_rule(color=tok["rule"]).encode(x="x:Q")
    return layers.properties(height=max(120, 28 * len(df)))
