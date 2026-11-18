"""What a customer may buy, and what they hold: plain rules on plain data.

ReBel is execution-only: the customer places every order. These rules only say what is open to them (their strategy,
the check on complex products, their own rules) and compare what they hold with their strategy's reference portfolio.
The comparison points to kinds of assets, never to an order.

Standard library only: the online demo runs this file in the browser as it is (mobile/demo.py), so the rules exist
once.
"""
from agents.profiler import QUESTIONS
from data.profiles import COMPLEX_FAMILIES, FAMILY_SLEEVE, SLEEVES, STRATEGIES

# The customer's rules (the coach's last question): which ETFs each one leaves out. etf: {"ticker", "family"}.
RULED_OUT = {
    "weapons": lambda etf: etf["ticker"] == "DEFENSE",
    "crypto": lambda etf: etf["family"] == "crypto",
    "leverage": lambda etf: etf["family"] == "leveraged_inverse",
    "fossil": lambda etf: etf["ticker"] in ("WORLD ENERGY", "OIL & GAS E&P"),
}
CLOSE, FAR = 0.05, 0.15  # a sleeve this close to the reference is fine; beyond FAR it is a gap
MIN_ORDER_EUR = 1


def rule_name(rule: str) -> str:
    return QUESTIONS["rules"]["answers"][rule][0]


def ruled_out_by(etf: dict, rules: list[str] | None) -> str | None:
    """The customer's rule that leaves this ETF out, or None."""
    return next((rule for rule in rules or [] if RULED_OUT[rule](etf)), None)


def lock(etf: dict, strategy: str, unlocked: bool, rules: list[str] | None = None) -> str:
    """Why the customer cannot buy this ETF, in words, or "" when they can."""
    rule = ruled_out_by(etf, rules)
    if rule:
        return f"You ruled out {rule_name(rule).lower()}"
    if etf["family"] in COMPLEX_FAMILIES:
        if not STRATEGIES[strategy]["complex"]:
            return f"Not part of the {STRATEGIES[strategy]['label']} risk level"
        if not unlocked:
            return "Pass the 3-question check to unlock"
    return ""


def decorate(lines: list[dict], strategy: str, unlocked: bool, rules: list[str] | None = None) -> list[dict]:
    """The same lines, each with `lock`: why it cannot be bought, or ""."""
    return [{**line, "lock": lock(line, strategy, unlocked, rules)} for line in lines]


def check_order(etf: dict | None, eur, strategy: str, unlocked: bool, rules: list[str] | None = None) -> str:
    """What is wrong with an order, in words, or "" when it can be placed."""
    if etf is None:
        return "Unknown ETF"
    if isinstance(eur, bool) or not isinstance(eur, (int, float)) or not eur >= MIN_ORDER_EUR:
        return f"The amount must be at least EUR {MIN_ORDER_EUR}"
    return lock(etf, strategy, unlocked, rules)


def sleeve_of(etf: dict, reference: list[dict]) -> str:
    """Where an ETF sits: its sleeve in the reference portfolio, else the sleeve of its family, else the themes."""
    for line in reference:
        if line["ticker"] == etf["ticker"]:
            return line["sleeve"]
    return FAMILY_SLEEVE.get(etf["family"], "satellites")


def holdings(orders: list[dict], last: dict[str, float], etfs: dict[str, dict], reference: list[dict]) -> dict:
    """What the customer holds, valued at the last close, compared with the reference portfolio.

    orders: the customer's orders, each {"ticker", "eur", "units"} (units None when no price was known: then the
    position is worth what was paid). last: ticker -> last close. etfs: ticker -> {"ticker", "name", "theme",
    "family"}. reference: the lines of the strategy's reference portfolio, each {"ticker", "sleeve", "weight"}.
    """
    book = {}
    for order in orders:
        line = book.setdefault(order["ticker"], {"eur": 0.0, "units": 0.0, "priced": True})
        line["eur"] += order["eur"]
        if order.get("units"):
            line["units"] += order["units"]
        else:
            line["priced"] = False
    lines = []
    for ticker, line in book.items():
        etf = etfs[ticker]
        value = line["units"] * last[ticker] if line["priced"] and ticker in last else line["eur"]
        lines.append({"ticker": ticker, "name": etf["name"], "theme": etf["theme"], "family": etf["family"],
                      "sleeve": sleeve_of(etf, reference), "invested": round(line["eur"], 2), "value": round(value, 2),
                      "gain": value / line["eur"] - 1 if line["eur"] else 0.0})
    value = sum(line["value"] for line in lines)
    invested = sum(line["invested"] for line in lines)
    for line in lines:
        line["weight"] = line["value"] / value if value else 0.0
    lines.sort(key=lambda line: -line["value"])
    target = {key: sum(line["weight"] for line in reference if line["sleeve"] == key) for key in SLEEVES}
    held = {key: sum(line["weight"] for line in lines if line["sleeve"] == key) for key in SLEEVES}
    health, ideas = [], []
    if lines:
        for key, name in SLEEVES.items():
            gap = held[key] - target[key]
            if not held[key] and not target[key]:
                continue
            state = "ok" if abs(gap) <= CLOSE else "watch" if abs(gap) <= FAR else "gap"
            health.append({"sleeve": key, "name": name, "held": held[key], "target": target[key], "state": state})
            if gap < -CLOSE:
                ideas.append(f"{name}: about {round(target[key] * 100)}% would match your reference, you hold "
                             f"{round(held[key] * 100)}%")
            elif gap > CLOSE:
                ideas.append(f"{name}: {round(held[key] * 100)}% of what you hold, against {round(target[key] * 100)}% "
                             "in your reference")
    return {"lines": lines, "invested": round(invested, 2), "value": round(value, 2),
            "gain": value / invested - 1 if invested else 0.0,
            "sleeves": [{"key": key, "name": name, "weight": held[key]} for key, name in SLEEVES.items() if held[key]],
            "health": health, "ideas": ideas}


def mix_history(orders: list[dict], histories: dict[str, list[dict]]) -> list[dict]:
    """What the ETFs the customer holds today, in today's units, were worth on each past date of their charts:
    [{"date", "value"}], from the first date every one of them has a price. A position bought without a known price
    counts at what was paid. Empty when no position has units.

    histories: ticker -> the ETF's chart points, [{"date", "value"}] in date order."""
    units: dict[str, float] = {}
    fixed = 0.0
    for order in orders:
        if order.get("units") and histories.get(order["ticker"]):
            units[order["ticker"]] = units.get(order["ticker"], 0.0) + order["units"]
        else:
            fixed += order["eur"]
    if not units:
        return []
    series = {ticker: {p["date"]: p["value"] for p in histories[ticker]} for ticker in units}
    last: dict[str, float] = {}
    out = []
    for date in sorted({d for points in series.values() for d in points}):
        for ticker, points in series.items():
            if date in points:
                last[ticker] = points[date]
        if len(last) == len(units):
            out.append({"date": date, "value": round(fixed + sum(units[t] * last[t] for t in units), 2)})
    return out
