"""Markdown to PDF, for the documents the cockpit offers as downloads (memo, summary, reports).

A small renderer of its own, not an HTML converter: the agents' texts hold stray tags (<cite>, half-open ones) that
an HTML parser refuses. Here every tag is dropped and the text is kept.
"""
import math
import re

from fpdf import FPDF
from fpdf.enums import TableBordersLayout, TableCellFillMode
from fpdf.fonts import FontFace

INK, MUTED, ACCENT, SOFT, LINE = (31, 35, 40), (107, 114, 128), (37, 99, 235), (244, 245, 247), (229, 231, 235)
WARN, BAD = (180, 83, 9), (185, 28, 28)  # amber and red for a warning or a breach in a table
MM_PER_CHAR, CELL_PAD = 1.4, 5.0  # width of a character at 8 pt, and the padding of a cell, in mm
LINE_H = 4.2

# The built-in PDF fonts only know Latin-1: swap the usual typography for plain look-alikes.
_PLAIN = {"→": "->", "←": "<-", "↑": "up", "↓": "down", "–": "-", "—": "-", "−": "-", "‘": "'", "’": "'", "“": '"',
          "”": '"', "…": "...", "≥": ">=", "≤": "<=", "✓": "ok", "✗": "x", "✕": "x", "▲": "up", "▼": "down", "•": "·",
          "×": "x", "≈": "~", " ": " ", " ": " "}
HEADING = {1: (17, 4), 2: (13, 3), 3: (11, 2), 4: (10, 1.5)}  # size in pt, space above in mm


def _clean(text: str) -> str:
    """One line of Markdown as text for the PDF: links, code marks and every HTML tag gone, Latin-1 only."""
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    text = re.sub(r"\(\s*cite\b[^>]*>", "", text)  # a half-open "(cite index=...>" the agent wrote
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"€\s*(?=[\d(])", "EUR ", text).replace("€", "EUR")
    for char, plain in _PLAIN.items():
        text = text.replace(char, plain)
    return text.encode("latin-1", "replace").decode("latin-1").strip()


def _cells(line: str) -> list[str]:
    return [_clean(c) for c in line.strip().strip("|").split("|")]


def _is_rule(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*", line)) and "-" in line


class _Doc(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"QDC Trading - page {self.page_no()}", align="C")


def _len(cell: str) -> float:
    """Width of a text in characters; capitals count wider than lower-case letters."""
    return len(cell) + 0.4 * sum(c.isupper() for c in cell)


def _word(cell: str) -> float:
    """Length of the longest word, which a column can never be narrower than."""
    return min(40.0, max((_len(w) for w in re.split(r"[\s/]+", cell.replace("**", "")) if w), default=0.0))


def _widths(rows: list[list[str]], total: float) -> list[float]:
    """Column widths: short columns (names, numbers) fit their text, long ones share what is left, and no column
    is thinner than its longest word."""
    n = len(rows[0])
    body = rows[1:] or rows
    need = [max(_len(r[i]) for r in body) for i in range(n)]
    # a column is never thinner than its longest word; the bold header and the bold status words need more room
    floor = [max(max(_word(r[i]) for r in body) * 1.6, _word(rows[0][i]) * 1.7) + CELL_PAD for i in range(n)]
    natural = [max(need[i], _word(rows[0][i])) * MM_PER_CHAR + CELL_PAD for i in range(n)]
    if sum(natural) <= total:  # everything fits on one line: spread the spare room
        return [w + (total - sum(natural)) / n for w in natural]
    fixed = [i for i in range(n) if need[i] <= 24]
    flex = [i for i in range(n) if i not in fixed]
    widths = [need[i] * 1.3 + CELL_PAD if i in fixed else natural[i] for i in range(n)]
    if flex:
        room = max(total - sum(natural[i] for i in fixed), 0)
        for i in flex:
            widths[i] = room * need[i] ** 0.7 / sum(need[j] ** 0.7 for j in flex)
    # lift thin columns to their floor and take it back from the wide ones
    for _ in range(n):
        thin = [i for i in range(n) if widths[i] < floor[i] - 1e-6]
        if not thin:
            break
        lack = sum(floor[i] - widths[i] for i in thin)
        rest = [i for i in flex if i not in thin and widths[i] > floor[i]]  # the long columns give way first
        rest = rest or [i for i in range(n) if i not in thin and widths[i] > floor[i]]
        room = sum(widths[i] - floor[i] for i in rest)
        for i in thin:
            widths[i] = floor[i]
        if room <= 0:
            break
        for i in rest:
            widths[i] -= (widths[i] - floor[i]) * min(1.0, lack / room)
    scale = total / sum(widths)
    return [w * scale for w in widths]


def _parse_table(lines: list[str]) -> tuple[list[list[str]], list[str]]:
    rows = [_cells(l) for l in lines if not _is_rule(l)]
    rule = next((l for l in lines if _is_rule(l)), "")
    n = len(rows[0])
    aligns = ["RIGHT" if c.strip().endswith(":") else "LEFT" for c in rule.strip().strip("|").split("|")]
    return [(r + [""] * n)[:n] for r in rows], (aligns + ["LEFT"] * n)[:n]


def _table_height(rows: list[list[str]], total: float) -> float:
    widths = _widths(rows, total)
    height = 0.0
    for r in rows:
        lines = max(math.ceil(1.1 * _len(c) / max(1.0, (w - CELL_PAD) / MM_PER_CHAR)) or 1 for c, w in zip(r, widths))
        height += lines * LINE_H + 3.4
    return height + 4


def _fit(pdf: _Doc, height: float, limit: float) -> None:
    """Start a new page when `height` does not fit below the cursor (a block longer than `limit` may split)."""
    if pdf.get_y() > pdf.t_margin + 2 and pdf.get_y() + min(height, limit) > pdf.page_break_trigger:
        pdf.add_page()


def _table(pdf: _Doc, lines: list[str]) -> None:
    rows, aligns = _parse_table(lines)
    widths = _widths(rows, pdf.epw)
    _fit(pdf, _table_height(rows, pdf.epw), limit=75)
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(*INK)
    pdf.ln(1)
    with pdf.table(col_widths=widths, text_align=tuple(aligns), line_height=LINE_H, v_align="TOP",
                   borders_layout=TableBordersLayout.HORIZONTAL_LINES, cell_fill_color=SOFT,
                   cell_fill_mode=TableCellFillMode.ROWS, padding=(1.6, 2), markdown=True,
                   headings_style=FontFace(emphasis="BOLD", color=MUTED, fill_color=(255, 255, 255))) as table:
        for r_index, r in enumerate(rows):
            row = table.row()
            for cell in r:
                word = cell.lower().strip("*")
                style = None
                if r_index and word in ("warning", "breach"):
                    style = FontFace(emphasis="BOLD", color=WARN if word == "warning" else BAD)
                row.cell(cell, style=style)
    pdf.ln(3)


def _paragraph(pdf: _Doc, text: str, indent: float = 0, bullet: str = "") -> None:
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(*INK)
    pdf.set_x(pdf.l_margin + indent)
    if bullet:
        pdf.set_text_color(*MUTED)
        pdf.cell(4, 5, bullet)
        pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5, text, markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.2)


def _heading(pdf: _Doc, level: int, text: str, ahead: list[str]) -> None:
    """A heading that never stays alone at the foot of a page: it moves with what follows it."""
    size, above = HEADING[level]
    follows = _table_height(_parse_table(ahead)[0], pdf.epw) if ahead else 24
    _fit(pdf, above + size * 0.5 + 4 + follows, limit=48)
    pdf.ln(above)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(*(ACCENT if level == 1 else INK))
    pdf.multi_cell(0, size * 0.5, text.replace("**", ""), align="L", new_x="LMARGIN", new_y="NEXT")
    if level <= 2:
        pdf.set_draw_color(*LINE)
        pdf.line(pdf.l_margin, pdf.get_y() + 0.5, pdf.w - pdf.r_margin, pdf.get_y() + 0.5)
    pdf.ln(2)


def to_pdf(text: str) -> bytes:
    """The Markdown text as a PDF: headings, paragraphs, lists, tables, bold and italic; links as plain text."""
    pdf = _Doc(format="A4")
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or re.fullmatch(r"[-*_]{3,}", line):
            if line:
                pdf.set_draw_color(*LINE)
                pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
                pdf.ln(3)
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) > 1 and _is_rule(block[1]):
                _table(pdf, block)
            else:
                for b in block:
                    _paragraph(pdf, _clean(b))
        elif (m := re.match(r"(#{1,6})\s+(.*)", line)):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            ahead = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                ahead.append(lines[j])
                j += 1
            ahead = ahead if len(ahead) > 1 and _is_rule(ahead[1]) else []
            _heading(pdf, min(len(m.group(1)), 4), _clean(m.group(2)), ahead)
            i += 1
        elif (m := re.match(r"(\s*)([-*+]|\d+[.)])\s+(.*)", raw)):
            depth = len(m.group(1).expandtabs(4)) // 2
            _paragraph(pdf, _clean(m.group(3)), indent=3 + 5 * depth,
                       bullet="·" if m.group(2)[0] in "-*+" else m.group(2))
            i += 1
        elif line.startswith(">"):
            _paragraph(pdf, _clean(line.lstrip("> ")), indent=4)
            i += 1
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|\||[-*+]\s|\d+[.)]\s|>)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            _paragraph(pdf, _clean(" ".join(para)))
    return bytes(pdf.output())
