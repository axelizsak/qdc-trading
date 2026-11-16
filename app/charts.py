"""Altair charts of the cockpit: light, thin marks, hairline grid, the value written on the mark, tooltips.

Blue = up / buy, red = down / sell, grey = neutral.
"""
import altair as alt
import pandas as pd

from app.ui import FONT, T

BAR = 16  # bar thickness in px


def finish(chart, height: int) -> alt.Chart:
    """Styling shared by every chart."""
    return chart.properties(height=height, background="transparent").configure_view(stroke=None).configure_axis(
        gridColor=T["line"], domainColor=T["line"], tickColor=T["line"], labelColor=T["muted"],
        titleColor=T["muted"], labelFont=FONT, titleFont=FONT, labelFontSize=12, titleFontSize=12,
        titleFontWeight="normal",
    ).configure_legend(labelColor=T["ink"], titleColor=T["muted"], labelFont=FONT, titleFont=FONT,
                       titleFontWeight="normal", labelFontSize=12).configure_text(font=FONT)


def diverging(df: pd.DataFrame, value: str, label: str, tooltip: list, opacity: str | None = None,
              fmt: str = "+.2f") -> alt.Chart:
    """Scores in [-1, 1] as horizontal bars, highest first: blue to the right, red to the left, the value at the tip.
    `opacity`: a column in [0, 1] (e.g. confidence) that fades the weak ones."""
    order = df.sort_values(value, ascending=False)[label].tolist()
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort=order, title=None,
                                        axis=alt.Axis(labelLimit=220, labelOverlap=False, domain=False, ticks=False,
                                                      grid=False)))
    encode = dict(x=alt.X(f"{value}:Q", scale=alt.Scale(domain=[-1, 1]), title=None,
                          axis=alt.Axis(format="+.1f", values=[-1, -0.5, 0, 0.5, 1])),
                  color=alt.condition(f"datum['{value}'] >= 0", alt.value(T["positive"]), alt.value(T["negative"])),
                  tooltip=tooltip)
    if opacity:
        encode["opacity"] = alt.Opacity(f"{opacity}:Q", scale=alt.Scale(domain=[0, 1], range=[0.35, 1]), legend=None)
    bars = base.mark_bar(size=BAR, cornerRadius=3).encode(**encode)
    text = base.mark_text(dx=alt.expr(f"datum['{value}'] >= 0 ? 8 : -8"), fontSize=11, color=T["ink"],
                          align=alt.expr(f"datum['{value}'] >= 0 ? 'left' : 'right'")).encode(
        x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=fmt))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T["rule"]).encode(x="x:Q")
    return finish(bars + text + zero, max(140, 26 * len(df)))


def hbars(df: pd.DataFrame, value: str, label: str, value_format: str, tooltip: list,
          rule_at: float | None = None, domain: tuple = (0, 1), color: str | None = None) -> alt.Chart:
    """One-series horizontal bars, highest first, the value at the tip, an optional reference rule (e.g. 50%)."""
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort="-x", title=None,
                                        axis=alt.Axis(labelLimit=260, labelOverlap=False, domain=False, ticks=False,
                                                      grid=False)))
    bars = base.mark_bar(size=BAR, cornerRadiusEnd=3, color=color or T["series"][0]).encode(
        x=alt.X(f"{value}:Q", title=None, axis=alt.Axis(format=value_format, tickCount=5),
                scale=alt.Scale(domain=list(domain))), tooltip=tooltip)
    text = base.mark_text(align="left", dx=6, color=T["ink"], fontSize=11).encode(
        x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=value_format))
    layers = bars + text
    if rule_at is not None:
        layers += alt.Chart(pd.DataFrame({"x": [rule_at]})).mark_rule(color=T["rule"], strokeDash=[4, 3]).encode(x="x:Q")
    return finish(layers, max(120, 34 * len(df)))


def lines(df: pd.DataFrame, x: str, y: str, series: str, y_title: str, y_format: str, zero_based: bool = False,
          x_format: str = "%d %b", colors: list[str] | None = None, height: int = 300) -> alt.Chart:
    """2 px lines, one per series, legend on top if several, the last value written at each line end."""
    names = list(dict.fromkeys(df[series]))
    palette = colors or (T["series"] + ["#9333ea", "#0891b2", "#64748b"])
    color = alt.Color(f"{series}:N", scale=alt.Scale(domain=names, range=palette[:len(names)]),
                      legend=alt.Legend(orient="top", title=None) if len(names) > 1 else None)
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format=x_format, labelAngle=0, grid=False, tickCount=6)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_format, tickCount=5), scale=alt.Scale(zero=zero_based)),
        color=color,
    )
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    line = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    points = base.mark_point(filled=True, size=50, stroke=None).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:T", title="Date", format="%a %d %b %Y"), alt.Tooltip(f"{series}:N", title=""),
                 alt.Tooltip(f"{y}:Q", format=y_format, title=y_title)],
    ).add_params(hover)
    last = df.sort_values(x).groupby(series).tail(1)
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=11).encode(
        x=f"{x}:T", y=f"{y}:Q", text=alt.Text(f"{y}:Q", format=y_format), color=color)
    return finish(line + points + labels, height)


KIND_COLORS = {"code": ("#f4f5f7", "#9ca3af"), "llm": ("#eff6ff", "#2563eb"), "you": ("#fffbeb", "#d97706")}
BOX_W, BOX_H = 0.84, 0.62


def flow_diagram(nodes: pd.DataFrame, edges: list[dict], selected: str | None) -> alt.Chart:
    """The day as clickable boxes. `nodes`: step, label (lines split by '\\n'), kind (code | llm | you), x, y.
    `edges`: {"points": [(x, y), ...], "style": "flow" | "back" | "next"}: a polyline with an arrowhead at its end.
    Clicking a box or its label selects its step (selections "box" and "label")."""
    boxes = nodes.assign(x1=nodes["x"] - BOX_W / 2, x2=nodes["x"] + BOX_W / 2,
                         y1=nodes["y"] - BOX_H / 2, y2=nodes["y"] + BOX_H / 2,
                         fill=nodes["kind"].map(lambda k: KIND_COLORS[k][0]),
                         stroke=nodes["kind"].map(lambda k: KIND_COLORS[k][1]),
                         width=[3 if s == selected else 1.2 for s in nodes["step"]])
    segments, heads = [], []
    for edge in edges:
        points = edge["points"]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            segments.append({"x": x1, "y": y1, "x2": x2, "y2": y2, "style": edge["style"]})
        (x1, y1), (x2, y2) = points[-2], points[-1]
        angle = 90 if x2 > x1 else 270 if x2 < x1 else 180 if y2 < y1 else 0
        heads.append({"x": x2, "y": y2, "angle": angle, "style": edge["style"]})
    x_scale = alt.Scale(domain=[nodes["x"].min() - 0.6, nodes["x"].max() + 0.6], nice=False)
    y_scale = alt.Scale(domain=[nodes["y"].min() - 0.9, nodes["y"].max() + 0.9], nice=False)
    hidden = dict(title=None, axis=None)
    color = alt.Color("style:N", scale=alt.Scale(domain=["flow", "back", "next"],
                                                 range=[T["rule"], T["wait"], T["accent"]]), legend=None)
    lines_ = alt.Chart(pd.DataFrame(segments)).mark_rule(strokeWidth=1.5).encode(
        x=alt.X("x:Q", scale=x_scale, **hidden), y=alt.Y("y:Q", scale=y_scale, **hidden), x2="x2:Q", y2="y2:Q",
        color=color, strokeDash=alt.condition("datum.style == 'flow'", alt.value([1, 0]), alt.value([5, 4])))
    arrows = alt.Chart(pd.DataFrame(heads)).mark_point(shape="triangle", filled=True, size=70, opacity=1).encode(
        x=alt.X("x:Q", scale=x_scale, **hidden), y=alt.Y("y:Q", scale=y_scale, **hidden),
        angle=alt.Angle("angle:Q", scale=None), color=color)
    box = alt.selection_point(name="box", fields=["step"], on="click")
    label = alt.selection_point(name="label", fields=["step"], on="click")
    rects = alt.Chart(boxes).mark_rect(cornerRadius=8, cursor="pointer").encode(
        x=alt.X("x1:Q", scale=x_scale, **hidden), x2="x2:Q", y=alt.Y("y1:Q", scale=y_scale, **hidden), y2="y2:Q",
        fill=alt.Fill("fill:N", scale=None), stroke=alt.Stroke("stroke:N", scale=None),
        strokeWidth=alt.StrokeWidth("width:Q", scale=None),
        tooltip=[alt.Tooltip("name:N", title="Step"), alt.Tooltip("hint:N", title="")]).add_params(box)
    texts = alt.Chart(boxes).mark_text(lineBreak="\n", fontSize=12, fontWeight=500, color=T["ink"],
                                       cursor="pointer").encode(
        x=alt.X("x:Q", scale=x_scale, **hidden), y=alt.Y("y:Q", scale=y_scale, **hidden), text="label:N",
        tooltip=[alt.Tooltip("name:N", title="Step"), alt.Tooltip("hint:N", title="")]).add_params(label)
    return finish(lines_ + rects + texts + arrows, 300)
