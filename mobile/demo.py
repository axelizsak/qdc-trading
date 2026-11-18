"""The online demo of the customer app: a folder of static files that needs no server, for GitHub Pages.

Run:  python -m mobile.demo _site        (GitHub does it at every merge into main: .github/workflows/pages.yml)

The phone screens are copied as they are. What the server would answer is prepared here once (demo.json): the demo
customers, the reference portfolio of every strategy under every set of customer rules, every ETF with its sheet and
its chart from the latest morning run. The rules are not written a second time in JavaScript: the page runs
agents/profiler.py (the coach, the risk level, the check) and mobile/wallet.py (what is open, what is held) itself,
in the browser (mobile/static/demo.js). The demo holds no key and calls no model, and nothing a visitor enters leaves
their browser.
"""
import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

from agents import profiler
from data import config as settings
from data.profiles import STRATEGIES
from mobile import server
from mobile.customers import CUSTOMERS, known

ROOT = Path(__file__).resolve().parent.parent
# The Python files the page runs in the browser. They need nothing but the standard library and each other.
BROWSER_FILES = ["agents/profiler.py", "data/lessons.py", "data/profiles.py", "mobile/customers.py", "mobile/wallet.py"]
NOTE = ("Student prototype for a university data challenge. Not a Belfius product. The customers are fictional and "
        "nothing you enter leaves this page.")


def answers(shared: Path | None = None) -> dict:
    """Everything the page reads instead of asking a server."""
    shared = settings.LOGS_DIR if shared is None else shared
    rules = list(profiler.QUESTIONS["rules"]["answers"])
    chosen = [list(some) for size in range(len(rules) + 1) for some in combinations(rules, size)]
    references = {}
    for strategy in STRATEGIES:
        # references[strategy][rules joined by "+"]: the portfolio for EUR 1; the page multiplies by the amount
        references[strategy] = {"+".join(some): {k: v for k, v in server.reference(strategy, 1, some, shared).items()
                                                 if k != "history"} for some in chosen}
    sheets = {ticker: server.etf_sheet(shared, ticker) for ticker in server.universe()}
    return {"start": server.start(shared), "known": {customer: known(customer, True) for customer in CUSTOMERS},
            "rules": rules, "python": BROWSER_FILES, "references": references,
            "strategy_history": {strategy: server.reference(strategy, 1, [], shared)["history"] for strategy in STRATEGIES},
            "etfs": server.catalogue(shared),
            "sheets": {ticker: {key: value for key, value in sheet.items() if key != "rows"}
                       for ticker, sheet in sheets.items() if sheet}}


def build(out: Path, shared: Path | None = None) -> None:
    """Writes the demo into the folder `out`."""
    shutil.copytree(server.STATIC, out, dirs_exist_ok=True)
    for name in BROWSER_FILES:
        (out / "py" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, out / "py" / name)
    (out / "demo.json").write_text(json.dumps(answers(shared)), encoding="utf-8")
    page = (out / "index.html").read_text(encoding="utf-8")
    app = '<script type="module" src="app.js"></script>'
    note = ('<p style="position:fixed;left:0;right:0;bottom:0;margin:0;padding:6px 12px;font-size:11px;line-height:1.3;'
            f'text-align:center;color:#555;background:rgba(255,255,255,.92);z-index:50">{NOTE}</p>')
    if app not in page:
        raise RuntimeError("index.html no longer loads app.js the way the demo expects")
    # demo.js first: it must answer the app's requests before the app sends the first one
    (out / "index.html").write_text(page.replace(app, f'{note}\n  <script src="demo.js"></script>\n  {app}'), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the online demo of the customer app into a folder.")
    parser.add_argument("out", type=Path, help="the folder to write")
    args = parser.parse_args(argv)
    build(args.out)
    print(f"Demo written to {args.out}: {len(list(args.out.rglob('*')))} files.")


if __name__ == "__main__":
    main()
