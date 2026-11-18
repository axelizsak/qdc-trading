"""The online demo of the customer app (mobile/demo.py): the same screens, with the server's answers prepared."""
import ast
import json
import re
import sys

import pytest

from data import mock
from data.profiles import STRATEGIES
from graph import morning
from mobile import demo, server


@pytest.fixture
def site(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    mock.mock_etf_sheets().to_csv(shared / morning.SHEETS, index=False)
    mock.mock_chart_history().to_csv(shared / morning.HISTORY, index=False)
    demo.build(tmp_path / "site", shared)
    return tmp_path / "site"


def test_the_demo_is_the_app_with_the_servers_answers_prepared(site):
    page = (site / "index.html").read_text()
    assert page.index('src="demo.js"') < page.index('src="app.js"') and demo.NOTE in page  # the stand-in comes first
    assert (site / "app.js").read_text() == (server.STATIC / "app.js").read_text()  # the same screens
    data = json.loads((site / "demo.json").read_text())
    assert data["start"] == json.loads(json.dumps(server.start()))
    assert set(data["known"]) == {customer["id"] for customer in data["start"]["customers"]}
    assert set(data["sheets"]) == {"GOLD", "WORLD"} and "rows" not in data["sheets"]["WORLD"]  # the mock's two sheets
    assert data["sheets"]["WORLD"]["lines"][0].keys() == {"block", "name", "label", "value", "state", "detail"}
    assert data["sheets"]["WORLD"]["history"] and data["sheets"]["WORLD"]["reading"]
    assert len(data["etfs"]) == 50 and "lock" not in data["etfs"][0]  # the page adds the locks with mobile/wallet.py
    assert set(data["strategy_history"]) == set(STRATEGIES) and all(data["strategy_history"].values())
    for name in data["python"]:  # the page runs the repo's own files
        assert (site / "py" / name).read_text() == (demo.ROOT / name).read_text()


def test_there_is_a_reference_for_every_strategy_and_every_set_of_rules(site):
    data = json.loads((site / "demo.json").read_text())
    references = data["references"]
    assert set(references) == set(STRATEGIES) and all(len(refs) == 2 ** len(data["rules"]) for refs in references.values())
    for rules in references["balanced"]:  # the page looks a reference up by its rules, in the order of data["rules"]
        assert rules == "+".join(rule for rule in data["rules"] if rule in rules.split("+"))
    bold = references["opportunity"]
    expected = {k: v for k, v in server.reference("opportunity", 1, [], site.parent / "shared").items() if k != "history"}
    assert bold[""] == json.loads(json.dumps(expected)) and "history" not in bold[""]  # one chart per strategy, apart
    assert {line["ticker"] for line in bold["weapons+crypto"]["left_out"]} == {"DEFENSE", "BITCOIN"}
    assert sum(line["weight"] for line in bold["weapons+crypto"]["lines"]) == pytest.approx(1)


def test_the_stand_in_answers_every_route_of_the_server(tmp_path):
    script = (server.STATIC / "demo.js").read_text()
    routes = {route.path for route in server.create_app(logs=tmp_path, shared=tmp_path).routes if route.path.startswith("/api/")}
    assert set(re.findall(r'^    "(/api/\w+)": async', script, flags=re.M)) == routes
    assert "demo.js" not in (server.STATIC / "index.html").read_text()  # the real app does not load it


def test_what_the_browser_runs_needs_only_the_standard_library():
    """The page runs these files as they are, with nothing installed: any other import at the top would break it."""
    own = {name.removesuffix(".py").replace("/", ".") for name in demo.BROWSER_FILES}
    for name in demo.BROWSER_FILES:
        for node in ast.parse((demo.ROOT / name).read_text()).body:
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                assert module in own or module.split(".")[0] in sys.stdlib_module_names, f"{name} imports {module}"
