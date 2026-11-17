"""The executive summary as a PowerPoint deck, for the cockpit's download button.

It reads the signed summary (Markdown, the format graph/pipeline.py writes) and lays out its numbers on nine slides:
orders, stance, the four portfolios, agent weights, risk board, outlook, what to re-check, assumptions. Nothing is
calculated here beyond sums and shares of figures the summary already holds. A slide whose section is missing or
unreadable is left out, and the page works without the deck (the Markdown and the PDF are still offered).
"""
import io
import re

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from agents import portfolio_config as pm_cfg


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


INK, MUTED, BODY = _rgb("1F2328"), _rgb("6B7280"), _rgb("374151")
ACCENT, ACCENT_L = _rgb("2563EB"), _rgb("DBE6FD")
BG, CARD, LINE, WHITE, NAVY, GREY = _rgb("F5F6F8"), _rgb("FFFFFF"), _rgb("E3E6EB"), _rgb("FFFFFF"), _rgb("111827"), _rgb("C7CDD6")
WARN, WARN_L, WARN_LINE = _rgb("B45309"), _rgb("FDF1DD"), _rgb("F1D9AE")
GOOD, GOOD_L = _rgb("15803D"), _rgb("DDF3E4")
BAD, BAD_L, BAD_LINE = _rgb("B91C1C"), _rgb("FDE4E4"), _rgb("F4BDBD")
FONT = "Calibri"
SERIES = [_rgb("D5DAE1"), _rgb("A9B3C1"), _rgb("7C8AA0"), ACCENT]  # four portfolios: the final one stands out


# ---------------------------------------------------------------------------------------------- reading the summary

def _clean(text: str) -> str:
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def _split_row(line: str) -> list[str]:
    return [_clean(c) for c in line.strip().strip("|").split("|")]


def _is_rule(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*", line))


def _tables(lines: list[str]) -> list[list[list[str]]]:
    """Every Markdown table of the lines, header first."""
    out, block = [], []
    for line in lines + [""]:
        if line.strip().startswith("|"):
            block.append(line)
            continue
        if len(block) > 1 and _is_rule(block[1]):
            out.append([_split_row(l) for l in block if not _is_rule(l)])
        block = []
    return out


def _section(lines: list[str], start: str) -> list[str]:
    """The lines under the first heading that begins with `start`, up to the next heading of the same or higher level."""
    for i, line in enumerate(lines):
        m = re.match(r"(#+)\s+(.*)", line)
        if m and m.group(2).lower().startswith(start.lower()):
            level, body = len(m.group(1)), []
            for nxt in lines[i + 1:]:
                n = re.match(r"(#+)\s", nxt)
                if n and len(n.group(1)) <= level:
                    break
                body.append(nxt)
            return body
    return []


def _bullets(lines: list[str]) -> list[str]:
    return [_clean(m.group(1)) for l in lines if (m := re.match(r"\s*[-*+]\s+(.*)", l))]


def _paragraphs(lines: list[str]) -> list[str]:
    out, cur = [], []
    for line in lines + [""]:
        if line.strip() and not re.match(r"\s*([#|>]|[-*+]\s)", line):
            cur.append(line.strip())
        elif cur:
            out.append(_clean(" ".join(cur)))
            cur = []
    return out


def _num(text: str) -> float | None:
    m = re.search(r"[-+−]?\d[\d,]*\.?\d*", text.replace("−", "-"))
    return float(m.group(0).replace(",", "")) if m else None


def _table_named(tables: list, first_cell: str) -> list[list[str]]:
    return next((t for t in tables if t and t[0][0].lower() == first_cell.lower()), [])


def _short(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    stop = max(head.rfind(". "), head.rfind("; "))
    if stop >= limit * 0.4:  # end on a full sentence when one fits
        return head[:stop + 1]
    return head.rsplit(" ", 1)[0].rstrip(",;: ") + "…"


class _Doc:
    """What the deck needs from the summary, each part empty when it is not there."""

    def __init__(self, markdown: str):
        lines = markdown.splitlines()
        self.tables = _tables(lines)
        head = next((l for l in lines if l.startswith("# ")), "")
        self.date = _clean(head.split("·", 1)[1]) if "·" in head else _clean(head.lstrip("# "))
        self.orders = _table_named(self.tables, "Side")
        self.portfolios = next((t for t in self.tables if t[0][:2] == ["ETF", "Now"]), [])
        self.agents = _table_named(self.tables, "Agent")
        self.areas = _table_named(self.tables, "Area")
        self.metrics = _table_named(self.tables, "Metric")
        self.scenarios = _table_named(self.tables, "Scenario")
        memo = _section(lines, "Portfolio Manager")
        self.memo_head = next((_clean(l.lstrip("# ")) for l in lines if l.startswith("### Portfolio Manager")), "")
        paragraphs = _paragraphs(memo)
        self.stance = next((p for p in paragraphs if not p.startswith(("Recommendation", "Portfolio value", "LLM cost"))
                            and len(p) > 80), "")
        self.market_line = next((p for p in paragraphs if p.startswith("Portfolio value")), "")
        self.after = _bullets(_section(lines, "After today"))
        self.consider = _bullets(_section(lines, "To consider"))
        self.assumptions = _bullets(_section(lines, "Assumptions"))
        self.risk_line = next((p for p in _paragraphs(_section(lines, "Risk board")) if "breach" in p), "")
        self.window = next((p for p in _paragraphs(_section(lines, "Where the window ends"))), "")
        self.events = []
        for b in _bullets(lines):
            m = re.match(r"(\d{4}-\d{2}-\d{2}):\s*(.*)", b)
            if m:
                text, _, rest = m.group(2).partition(" Affects:")
                affects = rest.split(" Source:")[0]
                self.events.append((m.group(1), text.strip(), len([a for a in affects.split(",") if a.strip()])))
        self.costs = [l for l in lines if l.startswith("LLM cost")]
        cash = re.search(r"Cash:\s*€?([\d,]+)\s*now,\s*€?([\d,]+)\s*after", markdown)
        self.cash = (float(cash.group(1).replace(",", "")), float(cash.group(2).replace(",", ""))) if cash else None

    def metric(self, name: str) -> tuple[float | None, float | None]:
        """(value, reference level) of a risk metric."""
        for row in self.metrics[1:]:
            if row[0] == name and len(row) >= 3:
                return _num(row[1]), _num(row[2]) if row[2] not in ("—", "-", "") else None
        return None, None

    def final_weights(self) -> list[tuple[str, float]]:
        if not self.portfolios:
            return []
        col = self.portfolios[0].index("Final") if "Final" in self.portfolios[0] else None
        if col is None:
            return []
        rows = [(r[0], (_num(r[col]) or 0.0) / 100) for r in self.portfolios[1:] if len(r) > col]
        return sorted([r for r in rows if r[1] > 0], key=lambda r: -r[1])

    def risk_counts(self) -> tuple[int, int] | None:
        m = re.search(r"(\d+)\s+breach.*?(\d+)\s+warning", self.risk_line)
        return (int(m.group(1)), int(m.group(2))) if m else None


# ---------------------------------------------------------------------------------------------- drawing

class _Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.n = 0
        self.footer = ""

    def add(self, fill=BG):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = fill
        return slide

    def drop(self) -> None:
        ids = self.prs.slides._sldIdLst
        last = ids[-1]
        self.prs.part.drop_rel(last.rId)
        ids.remove(last)
        self.n -= 1

    def page(self, title: str, kicker: str):
        s = self.add()
        box(s, 0, 0, 13.333, 0.1, ACCENT, None, MSO_SHAPE.RECTANGLE)
        text(s, 0.7, 0.42, 11.9, 0.3, kicker.upper(), 12, True, ACCENT)
        text(s, 0.7, 0.74, 11.9, 0.7, title, 28, True)
        text(s, 0.7, 7.08, 9, 0.25, f"QDC Trading · {self.footer}", 10, color=MUTED)
        text(s, 11.9, 7.08, 0.73, 0.25, str(self.n), 10, color=MUTED, align=PP_ALIGN.RIGHT)
        return s

    def save(self) -> bytes:
        out = io.BytesIO()
        self.prs.save(out)
        return out.getvalue()


def text(s, x, y, w, h, content, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=0):
    frame_box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = frame_box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(content if isinstance(content, list) else [content]):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        run = p.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color


def box(s, x, y, w, h, fill=CARD, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    shape_ = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shape_.adjustments[0] = radius
    shape_.fill.solid()
    shape_.fill.fore_color.rgb = fill
    if line is None:
        shape_.line.fill.background()
    else:
        shape_.line.color.rgb = line
        shape_.line.width = Pt(0.75)
    shape_.shadow.inherit = False


def kpi(s, x, y, w, value, label, note="", color=INK, h=1.3):
    box(s, x, y, w, h)
    text(s, x + 0.22, y + 0.14, w - 0.44, 0.6, value, 30, True, color)
    text(s, x + 0.22, y + 0.74, w - 0.44, 0.28, label, 13, True)
    if note:
        text(s, x + 0.22, y + 0.99, w - 0.44, 0.25, _short(note, 40), 11, color=MUTED)


def pill(s, x, y, w, label, fg, bg):
    box(s, x, y, w, 0.3, bg, None, radius=0.5)
    text(s, x, y, w, 0.3, label, 11, True, fg, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def table(s, x, y, widths, rows, size=12, aligns=None, row_h=0.34):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows)))
    tbl = shape.table
    props = shape._element.graphic.graphicData.tbl.tblPr
    props.set("firstRow", "0")
    props.set("bandRow", "0")
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD if r % 2 else BG
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if aligns and aligns[c] == "r" else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = value
            run.font.name, run.font.size, run.font.bold = FONT, Pt(size), r == 0
            run.font.color.rgb = MUTED if r == 0 else (ACCENT if value in ("BUY", "SELL") and c == 0 else INK)


def _bars_chart(s, x, y, w, h, categories, series, number_format, horizontal=False, legend=False, size=12, maximum=None):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    chart = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.has_title = False
    chart.font.name, chart.font.size = FONT, Pt(size)
    chart.has_legend = legend
    if legend:
        chart.legend.position, chart.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
    chart.category_axis.format.line.color.rgb = LINE
    chart.value_axis.minimum_scale = 0
    if maximum:
        chart.value_axis.maximum_scale = maximum
    if horizontal:
        chart.category_axis.reverse_order = True
        chart.value_axis.visible = False
        chart.value_axis.has_major_gridlines = False
        plot = chart.plots[0]
        plot.has_data_labels = True
        plot.data_labels.number_format = number_format
        plot.data_labels.number_format_is_linked = False
        plot.data_labels.font.size, plot.data_labels.font.bold = Pt(size + 1), True
        plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        chart.plots[0].gap_width = 45
    else:
        chart.value_axis.major_gridlines.format.line.color.rgb = LINE
        chart.value_axis.format.line.fill.background()
        chart.value_axis.tick_labels.number_format = number_format
        chart.value_axis.tick_labels.number_format_is_linked = False
        chart.value_axis.tick_labels.font.size = Pt(size - 1)
        chart.plots[0].gap_width, chart.plots[0].overlap = 55, -4
    return chart


# ---------------------------------------------------------------------------------------------- the slides

def _title_slide(deck: _Deck, doc: _Doc) -> None:
    if not (doc.orders or doc.portfolios or doc.areas):
        raise ValueError("not an executive summary")
    s = deck.add(NAVY)
    box(s, 0, 0, 0.35, 7.5, ACCENT, None, MSO_SHAPE.RECTANGLE)
    text(s, 1.0, 1.0, 9, 0.4, "QDC TRADING · BELFIUS x QUANTUMBLACK ETF CHALLENGE", 13, True, _rgb("93B4F8"))
    text(s, 1.0, 2.2, 11, 1.2, "Executive summary", 60, True, WHITE)
    note = re.search(r"\((.*)\)", doc.memo_head)
    text(s, 1.0, 3.45, 11, 0.5, _short(f"{doc.date}" + (f" · {note.group(1)}" if note else ""), 110), 22, color=_rgb("D1D5DB"))
    invested = sum(w for _, w in doc.final_weights())
    chips = []
    if len(doc.orders) > 1:
        sides = {r[0].upper() for r in doc.orders[1:]}
        chips.append((str(len(doc.orders) - 1), "ETF buys" if sides == {"BUY"} else "orders"))
    if invested:
        chips.append((f"{invested:.0%}", "invested"))
    if (counts := doc.risk_counts()):
        chips += [(str(counts[0]), "limit breaches"), (str(counts[1]), "warnings")]
    for i, (big, small) in enumerate(chips[:4]):
        x = 1.0 + i * 2.75
        box(s, x, 4.9, 2.5, 1.45, _rgb("1F2937"), _rgb("374151"))
        text(s, x + 0.25, 5.02, 2.0, 0.7, big, 36, True, WHITE)
        text(s, x + 0.25, 5.8, 2.0, 0.3, small, 14, color=_rgb("9CA3AF"))


def _orders_slide(deck: _Deck, doc: _Doc) -> None:
    rows = doc.orders
    if len(rows) < 2:
        raise ValueError("no orders")
    n = len(rows) - 1
    weights = doc.final_weights()
    invested = sum(w for _, w in weights) or (1 - doc.cash[1] / doc.cash[0] if doc.cash else None)
    sides = {r[0].upper() for r in rows[1:]}
    word = "buys" if sides == {"BUY"} else "orders"
    title = f"{n} {word} put {invested:.0%} of the portfolio to work" if invested else f"{n} {word} to submit"
    s = deck.page(title, "Orders to submit")
    if weights:
        shown = weights[:5]
        text(s, 0.7, 1.7, 6.6, 0.3, "Target share of the portfolio", 14, True)
        chart = _bars_chart(s, 0.6, 2.0, 6.8, 0.42 * len(shown) + 0.3, [t for t, _ in shown],
                            [("Target share", [round(w * 100, 1) for _, w in shown])], '0.0"%"', True, size=13,
                            maximum=max(w for _, w in shown) * 100 * 1.2)
        chart.series[0].format.fill.solid()
        chart.series[0].format.fill.fore_color.rgb = ACCENT
    cols = [0, 1, 2, 3, 4]
    header = [rows[0][c] if c < len(rows[0]) else "" for c in cols]
    body = [[(r[c] if c < len(r) else "") for c in cols] for r in rows[1:7]]
    top = 6.75 - 0.34 * (len(body) + 1) - 0.1
    table(s, 0.7, top, [0.8, 2.0, 1.1, 1.4, 1.3], [header] + body, 12, ["l", "l", "r", "r", "r"])
    text(s, 0.7, 6.8, 6.6, 0.25, "Fills at the last close before the day.", 10, color=MUTED)
    box(s, 8.0, 1.75, 4.65, 2.6)
    text(s, 8.3, 1.95, 4.1, 0.3, "WHERE THE MONEY GOES", 12, True, ACCENT)
    if invested:
        bar_x, bar_w = 8.3, 4.05
        box(s, bar_x, 2.45, bar_w, 0.5, GREY, None, MSO_SHAPE.RECTANGLE)
        box(s, bar_x, 2.45, bar_w * min(invested, 1), 0.5, ACCENT, None, MSO_SHAPE.RECTANGLE)
        text(s, bar_x + 0.15, 2.45, 3.0, 0.5, f"Invested {invested:.1%}", 14, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
        if invested < 0.93:
            text(s, bar_x + bar_w * invested, 2.45, bar_w * (1 - invested), 0.5, f"{1 - invested:.0%}", 12, True, INK,
                 PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    if doc.cash:
        text(s, 8.3, 3.2, 2.0, 0.4, f"€{doc.cash[0]:,.0f}", 28, True)
        text(s, 8.3, 3.7, 2.0, 0.3, "cash now", 12, color=MUTED)
        text(s, 10.5, 3.2, 1.9, 0.4, f"€{doc.cash[1]:,.0f}", 28, True, ACCENT)
        text(s, 10.5, 3.7, 1.9, 0.3, "cash after the orders", 12, color=MUTED)
    kpi(s, 8.0, 4.6, 2.2, f"{n} of 5", "Orders used", "hard rule: at most 5")
    largest = next((re.search(r"Largest position (.+?) at (\d+)%", b) for b in doc.after if "Largest position" in b), None)
    if largest:
        kpi(s, 10.45, 4.6, 2.2, f"{largest.group(2)}%", "Largest position", _short(largest.group(1), 22))
    text(s, 8.0, 6.1, 4.65, 0.7, "The orders are the ones that cut the distance to the target portfolio the most.", 12, color=MUTED)


def _stance_slide(deck: _Deck, doc: _Doc) -> None:
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", doc.stance) if x]
    if len(sentences) < 2:
        raise ValueError("no stance")
    if len(sentences) > 5:
        sentences = sentences[:4] + [" ".join(sentences[4:])]
    s = deck.page("The stance: what the Portfolio Manager recommends", "Why: the Portfolio Manager's memo")
    width = (11.95 - 0.1 * (len(sentences) - 1)) / len(sentences)
    for i, sentence in enumerate(sentences):
        x = 0.7 + i * (width + 0.1)
        box(s, x, 1.85, width, 3.55)
        box(s, x + 0.22, 2.07, 0.5, 0.5, ACCENT, None, MSO_SHAPE.OVAL)
        text(s, x + 0.22, 2.07, 0.5, 0.5, str(i + 1), 16, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        text(s, x + 0.22, 2.85, width - 0.44, 2.45, _short(sentence, 300 if len(sentences) > 4 else 360), 13.5 if width > 2 else 12.5, color=BODY)
    facts = []
    if (m := re.search(r"score of WORLD\):\s*([+-][\d.]+)", doc.market_line)):
        invest = re.search(r"invest (\d+)%", doc.market_line)
        facts.append((m.group(1), "market view (score of WORLD)" + (f": invest {invest.group(1)}%" if invest else "")))
    if (m := re.search(r"counts for (\d+)%", doc.market_line)):
        facts.append((f"{m.group(1)}%", "weight of the LLM's portfolio in the final blend"))
    vol = next((re.search(r"volatility of the portfolio:\s*([\d.]+)%", b) for b in doc.after if "olatility" in b), None)
    if vol:
        facts.append((f"{vol.group(1)}%", "estimated volatility a year"))
    if facts:
        box(s, 0.7, 5.65, 11.95, 1.2, NAVY, None)
        step = 11.95 / len(facts)
        for i, (big, small) in enumerate(facts):
            x = 1.0 + i * step
            text(s, x, 5.82, 1.5, 0.6, big, 28, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
            text(s, x + 1.4, 5.82, step - 1.7, 0.9, small, 12, color=_rgb("D1D5DB"), anchor=MSO_ANCHOR.MIDDLE)


def _portfolios_slide(deck: _Deck, doc: _Doc) -> None:
    table_ = doc.portfolios
    if len(table_) < 2:
        raise ValueError("no portfolios")
    names = [c for c in table_[0][1:] if c != "Now"]
    rows = table_[1:9]
    series = [(c, [round(_num(r[table_[0].index(c)]) or 0.0, 1) for r in rows]) for c in names]
    s = deck.page("Four portfolios blend into one target", "Portfolio construction")
    box(s, 0.7, 1.7, 7.7, 5.2)
    chart = _bars_chart(s, 0.8, 1.85, 7.5, 4.7, [r[0] for r in rows], series, '0"%"', False, True)
    tint = SERIES[-len(series):] if len(series) <= len(SERIES) else SERIES
    for ser, color in zip(chart.series, tint):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = color
    text(s, 0.95, 6.55, 7.2, 0.25, "Share of the portfolio value. What is not shown stays in cash.", 10, color=MUTED)
    anchor = pm_cfg.ANCHOR.capitalize()
    text(s, 8.8, 1.75, 3.9, 0.3, "HOW THE FINAL IS BUILT", 12, True, ACCENT)
    trust = re.search(r"counts for (\d+)%", doc.market_line)
    if trust:
        share = int(trust.group(1))
        for i, (big, small, color) in enumerate([(f"{share}%", "LLM", ACCENT), (f"{100 - share}%", anchor, _rgb("7C8AA0"))]):
            x = 8.8 + i * 2.0
            box(s, x, 2.15, 1.85, 1.1, color, None)
            text(s, x, 2.25, 1.85, 0.55, big, 26, True, WHITE, PP_ALIGN.CENTER)
            text(s, x, 2.82, 1.85, 0.3, small, 13, False, WHITE, PP_ALIGN.CENTER)
        text(s, 8.8, 3.35, 3.9, 0.35, "= Final portfolio, then hard limits", 14, True)
    facts = [("Baseline", "a simple rule: sets the invested share and is the fallback."),
             ("Optimizer", "trades expected return against the risk of the whole portfolio."),
             ("LLM", "Claude's own proposal, free within Belfius's hard rules."),
             ("Hard limits", f"no ETF above {pm_cfg.MAX_POSITION:.0%}; at most {pm_cfg.MAX_LEVERAGED_SHARE:.0%} in "
                             "leveraged and inverse ETFs.")]
    for i, (head, body) in enumerate(facts):
        y = 3.95 + i * 0.72
        box(s, 8.8, y + 0.04, 0.07, 0.58, ACCENT, None, MSO_SHAPE.RECTANGLE)
        text(s, 9.02, y, 3.65, 0.3, head, 12, True)
        text(s, 9.02, y + 0.22, 3.65, 0.5, body, 11, color=MUTED)


def _agents_slide(deck: _Deck, doc: _Doc) -> None:
    rows = doc.agents
    if len(rows) < 2:
        raise ValueError("no agents")
    rows = rows[1:6]
    moved = any(abs((_num(r[2]) or 0) - (_num(r[1]) or 0)) > 0.005 for r in rows if len(r) > 2)
    title = "Agent weights have started to move" if moved else "Agent weights are still at their start: the record is short"
    s = deck.page(title, "Signals")
    width = (11.95 - 0.15 * (len(rows) - 1)) / len(rows)
    for i, r in enumerate(rows):
        r = (r + [""] * 6)[:6]
        x = 0.7 + i * (width + 0.15)
        weight = _num(r[2])
        right = _num(r[5])
        right = right if right is not None and "n/a" not in r[5] else None
        box(s, x, 1.85, width, 3.5)
        text(s, x + 0.25, 2.05, width - 0.5, 0.35, r[0], 16, True)
        shown = f"{weight * 100:.0f}%" if weight is not None and weight <= 1 else (r[2] or "n/a")
        text(s, x + 0.25, 2.5, 1.5, 0.7, shown, 36, True, ACCENT)
        text(s, x + 1.55, 2.72, width - 1.8, 0.4, "weight today", 10, color=MUTED)
        text(s, x + 0.25, 3.45, width - 0.5, 0.3, f"Sessions judged: {r[3]}", 12)
        text(s, x + 0.25, 3.75, width - 0.5, 0.3, f"Mean result a session: {r[4]}", 12)
        text(s, x + 0.25, 4.15, width - 0.5, 0.25, "Share of calls right", 11, color=MUTED)
        bar = width - 0.5
        box(s, x + 0.25, 4.5, bar, 0.2, BG, LINE, MSO_SHAPE.RECTANGLE)
        if right is not None:
            box(s, x + 0.25, 4.5, bar * min(right, 100) / 100, 0.2, GOOD, None, MSO_SHAPE.RECTANGLE)
            text(s, x + 0.25, 4.78, 1.2, 0.3, f"{right:.0f}%", 14, True, GOOD)
        else:
            text(s, x + 0.25, 4.78, 1.8, 0.3, "no record yet", 13, True, MUTED)
        box(s, x + 0.25 + bar * 0.5, 4.44, 0.03, 0.32, INK, None, MSO_SHAPE.RECTANGLE)
        text(s, x + 0.25 + bar * 0.5 - 0.3, 4.78, 0.65, 0.25, "50%", 10, color=MUTED, align=PP_ALIGN.CENTER)
    sessions = [_num(r[3]) for r in rows if len(r) > 3 and _num(r[3]) is not None]
    note = ["How it learns: after each session an agent's weight is multiplied by e^(speed x what following it earned), "
            "then rescaled to sum to 1, each between the minimum and the maximum weight."]
    if sessions and max(sessions) < 20:
        note.append("With so few sessions judged, none of these figures separates skill from luck.")
    box(s, 0.7, 5.65, 11.95, 1.2, ACCENT_L, None)
    text(s, 1.0, 5.85, 11.4, 1.0, note, 12.5, color=INK, space=4)


def _risk_slide(deck: _Deck, doc: _Doc) -> None:
    areas = doc.areas[1:9]
    counts = doc.risk_counts()
    if not areas and not counts:
        raise ValueError("no risk board")
    b, w = counts or (0, sum(1 for a in areas if a[1].lower() == "warning"))
    s = deck.page(f"The risk board shows {b} breach{'' if b == 1 else 'es'} and {w} warning{'' if w == 1 else 's'}",
                  "Risk Manager")
    vol, _ = doc.metric("volatility")
    vol_bench, _ = doc.metric("volatility_vs_benchmark")
    beta, _ = doc.metric("beta")
    checks = re.search(r"(\d+) check", doc.risk_line)
    tiles = [(str(b), "Breaches", f"{checks.group(1)} checks inside their levels" if checks else "", BAD if b else GOOD),
             (str(w), "Warnings", "information only", WARN if w else GOOD)]
    if beta is not None:
        tiles.append((f"{beta:.2f}", "Beta to the benchmark", f"moves {beta:.2f}x its daily changes", INK))
    if vol is not None:
        tiles.append((f"{vol * 100:.1f}%", "Volatility a year", f"{vol / vol_bench:.1f}x the benchmark" if vol_bench else "", INK))
    for i, (value, label, note, color) in enumerate(tiles[:4]):
        kpi(s, 0.7 + i * 3.0, 1.7, 2.85, value, label, note, color, 1.25)
    for i, row in enumerate(areas):
        row = (row + [""] * 3)[:3]
        col, line_ = i % 4, i // 4
        x, y = 0.7 + col * 3.0, 3.25 + line_ * 1.85
        status = row[1].lower()
        fg, bg, edge = (BAD, BAD_L, BAD_LINE) if status == "breach" else (WARN, WARN_L, WARN_LINE) if status == "warning" else (GOOD, CARD, LINE)
        box(s, x, y, 2.85, 1.7, bg, edge)
        box(s, x, y + 0.12, 0.07, 1.46, fg, None, MSO_SHAPE.RECTANGLE)
        label = "OK" if status == "ok" else row[1].capitalize()
        text(s, x + 0.25, y + 0.14, 1.6, 0.5, row[0], 13, True)
        pill(s, x + 1.75, y + 0.12, 0.95, label, fg, WHITE if status != "ok" else GOOD_L)
        text(s, x + 0.25, y + 0.68, 2.45, 0.95, _short(row[2], 120), 11.5, color=BODY)


def _outlook_slide(deck: _Deck, doc: _Doc) -> None:
    text_ = doc.window
    median = re.search(r"median ([-+−]?[\d.]+)%", text_)
    worst = re.search(r"worst 5% ([-+−]?[\d.]+)%", text_)
    below = re.search(r"(\d+)% chance of ending below", text_)
    behind = re.search(r"(\d+)% chance of ending behind", text_)
    days = re.search(r"(\d+) trading days", text_)
    if not (median or worst or doc.scenarios):
        raise ValueError("no outlook")
    title = (f"Median {median.group(1).replace('-', '−')}% to the end of the window; worst 5% {worst.group(1).replace('-', '−')}%"
             if median and worst else "Where the window ends and what could move it")
    s = deck.page(title, "Where the window ends and what could move it")
    _, worst_level = doc.metric("worst_case_5pct")
    if median:
        value = float(median.group(1).replace("−", "-"))
        kpi(s, 0.7, 1.7, 2.85, f"{value:+.1f}%".replace("-", "−"), "Median outcome", f"{days.group(1)} trading days" if days else "",
            GOOD if value >= 0 else WARN, 1.25)
    if worst:
        value = float(worst.group(1).replace("−", "-"))
        kpi(s, 3.75, 1.7, 2.85, f"{value:+.1f}%".replace("-", "−"), "Worst 5% of paths",
            f"reference level: {worst_level * 100:.0f}%".replace("-", "−") if worst_level is not None else "", WARN, 1.25)
    rows = []
    for r in doc.scenarios[1:6]:
        effect = re.search(r"([-+−]\d+\.?\d*)%", r[2]) if len(r) > 2 else None
        if effect:
            rows.append((_short(r[0], 34), float(effect.group(1).replace("−", "-"))))
    if rows:
        box(s, 0.7, 3.2, 5.9, 3.65)
        text(s, 0.95, 3.35, 5.4, 0.3, "Scenario effects on the portfolio, in %", 13, True)
        axis_x = 5.0
        big_down = max((-v for _, v in rows if v < 0), default=0)
        big_up = max((v for _, v in rows if v > 0), default=0)
        unit = min(1.0, 1.9 / big_down if big_down else 1.0, 1.4 / big_up if big_up else 1.0)
        step = min(0.85, 2.6 / len(rows))
        for i, (name, v) in enumerate(rows):
            y = 3.95 + i * step
            text(s, 0.95, y, 1.9, 0.5, name, 12, anchor=MSO_ANCHOR.MIDDLE)
            length = abs(v) * unit
            x0 = axis_x - length if v < 0 else axis_x
            box(s, x0, y, max(length, 0.02), 0.5, WARN if v < 0 else GOOD, None, MSO_SHAPE.RECTANGLE)
            label = f"{v:+.1f}%".replace("-", "−")
            if v < 0:
                text(s, x0 - 0.75, y, 0.68, 0.5, label, 13, True, WARN, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)
            else:
                text(s, x0 + length + 0.08, y, 0.6, 0.5, label, 13, True, GOOD, anchor=MSO_ANCHOR.MIDDLE)
        box(s, axis_x - 0.01, 3.8, 0.02, 2.75, INK, None, MSO_SHAPE.RECTANGLE)
        text(s, 0.95, 6.5, 5.4, 0.25, "Negative: the portfolio loses; positive: it gains.", 10, color=MUTED)
    gauges = []
    if below:
        gauges.append(("Ends below today's value", int(below.group(1)), doc.metric("prob_below_start")[1]))
    if behind:
        gauges.append(("Ends behind the benchmark", int(behind.group(1)), doc.metric("prob_below_benchmark")[1]))
    if gauges:
        text(s, 7.0, 1.75, 5.6, 0.3, "CHANCES, AGAINST THE REFERENCE LEVEL", 12, True, ACCENT)
    for i, (label, value, level) in enumerate(gauges):
        y = 2.2 + i * 1.5
        box(s, 7.0, y, 5.65, 1.3)
        text(s, 7.25, y + 0.15, 3.6, 0.3, label, 13, True)
        text(s, 10.6, y + 0.1, 1.8, 0.4, f"{value}%", 24, True, INK, PP_ALIGN.RIGHT)
        width = 5.15
        box(s, 7.25, y + 0.68, width, 0.22, BG, LINE, MSO_SHAPE.RECTANGLE)
        box(s, 7.25, y + 0.68, width * value / 100, 0.22, GOOD, None, MSO_SHAPE.RECTANGLE)
        if level is not None:
            level_pct = level * 100
            box(s, 7.25 + width * level - 0.015, y + 0.6, 0.03, 0.38, WARN, None, MSO_SHAPE.RECTANGLE)
            text(s, 7.25 + width * level - 0.6, y + 0.99, 1.2, 0.25, f"level {level_pct:.0f}%", 10, True, WARN, PP_ALIGN.CENTER)
    box(s, 7.0, 5.25, 5.65, 1.6, ACCENT_L, None)
    text(s, 7.25, 5.4, 5.15, 1.4, ["The paths are simulated from recent returns with today's weights held constant.",
                                  "Scenarios: an LLM chose the events and their strength; the Risk Manager calculated the effect."],
         11.5, color=BODY, space=4)


def _recheck_slide(deck: _Deck, doc: _Doc) -> None:
    events = sorted(doc.events)[:5]
    cards = doc.consider[:3]
    if not events and not cards:
        raise ValueError("nothing to re-check")
    s = deck.page("What to re-check before the first trading date", "To consider")
    if events:
        box(s, 0.7, 1.7, 11.95, 2.6)
        text(s, 1.0, 1.85, 6, 0.3, "SCHEDULED EVENTS IN THE WINDOW", 12, True, ACCENT)
        box(s, 1.2, 2.78, 10.95, 0.04, LINE, None, MSO_SHAPE.RECTANGLE)
        step = 11.75 / len(events)
        for i, (day, what, affects) in enumerate(events):
            cx = 0.8 + step * (i + 0.5)
            hot = affects >= 4
            label = f"{int(day[8:])} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(day[5:7]) - 1]}"
            text(s, cx - 1.0, 2.25, 2.0, 0.35, label, 16, True, ACCENT if hot else INK, PP_ALIGN.CENTER)
            box(s, cx - 0.12, 2.68, 0.24, 0.24, ACCENT if hot else GREY, WHITE, MSO_SHAPE.OVAL)
            text(s, cx - step / 2 + 0.1, 3.1, step - 0.2, 1.1, _short(what, 85), 12, color=BODY, align=PP_ALIGN.CENTER)
        text(s, 0.7, 4.38, 11, 0.25, "Dates from the Macro/News agent, not checked independently. Highlighted: events that touch "
                                   "four or more of the holdings.", 10, color=MUTED)
    for i, body in enumerate(cards):
        width = (11.95 - 0.2 * (len(cards) - 1)) / len(cards)
        x = 0.7 + i * (width + 0.2)
        box(s, x, 4.8, width, 2.05)
        box(s, x, 4.8, width, 0.07, ACCENT, None, MSO_SHAPE.RECTANGLE)
        text(s, x + 0.25, 5.05, width - 0.5, 1.7, _short(body, 215), 12.5, color=BODY)


def _assumptions_slide(deck: _Deck, doc: _Doc) -> None:
    items = [a for a in doc.assumptions if not a.lower().startswith("this is a recommendation")][:7]
    if not items:
        raise ValueError("no assumptions")
    s = deck.page("Assumptions and limits", "Read this before acting")
    box(s, 0.7, 1.85, 7.6, 5.0)
    step = min(0.7, 4.6 / len(items))
    for i, item in enumerate(items):
        y = 2.05 + i * step
        box(s, 0.95, y + 0.08, 0.1, 0.1, ACCENT, None, MSO_SHAPE.OVAL)
        text(s, 1.2, y, 6.9, step, _short(item, 190), 12.5, color=BODY)
    box(s, 8.7, 1.85, 3.95, 3.0, NAVY, None)
    text(s, 8.95, 2.1, 3.45, 0.8, "A recommendation, not an instruction", 18, True, WHITE)
    text(s, 8.95, 3.0, 3.45, 1.8, "The human reviews every order before it is submitted. Only a breach of a hard rule stops an "
                                  "order; warnings are information.", 13, color=_rgb("D1D5DB"))
    if doc.costs:
        calls = sum(int(m.group(1)) for c in doc.costs if (m := re.search(r"(\d+) Claude call", c)))
        dollars = sum(float(m.group(1)) for c in doc.costs if (m := re.match(r"LLM cost: \$([\d.]+)", c)))
        box(s, 8.7, 5.05, 3.95, 1.8)
        text(s, 8.95, 5.25, 3.45, 0.35, "Cost of this run", 16, True, ACCENT)
        text(s, 8.95, 5.8, 3.45, 0.9, f"{calls} Claude calls, about ${dollars:.2f} in total.", 14, color=BODY)


def to_pptx(markdown: str) -> bytes:
    """The executive summary as a deck (bytes of a .pptx). A slide that cannot be built is left out."""
    doc = _Doc(markdown)
    deck = _Deck()
    deck.footer = f"Executive summary, {doc.date}" if doc.date else "Executive summary"
    built = 0
    for build in (_title_slide, _orders_slide, _stance_slide, _portfolios_slide, _agents_slide, _risk_slide,
                  _outlook_slide, _recheck_slide, _assumptions_slide):
        before = len(deck.prs.slides)
        try:
            build(deck, doc)
            built += 1
        except Exception:
            while len(deck.prs.slides) > before:  # never leave half a slide behind
                deck.drop()
    if built == 0:
        raise ValueError("nothing in this text can be shown as slides")
    return deck.save()
