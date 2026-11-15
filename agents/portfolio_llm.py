"""The Portfolio Manager's judgement call. Owner: Person 3. LLM: yes (Sonnet).

The code has already calculated every number (combined scores, learned agent weights, volatilities, the current
portfolio, the code's baseline portfolio). The LLM reads them and proposes its own portfolio, as freely as it
sees fit, within Belfius's hard rules only:
  - target_weights: its target portfolio (share of the portfolio value per ETF, the rest stays cash),
  - exclude / max_weight: the human's feedback turned into constraints,
  - summary, feedback_response, flags, reasons: words for the investment memo.
The code decides how much the LLM counts (learned from its track record against the baseline), sizes the
orders, and `sanitize` drops anything unknown, so a bad reply can never put an unlisted ETF or a short in the
portfolio.
"""
import json

from agents import macro_llm
from agents import portfolio_config as cfg
from data import config as settings

SYSTEM = """You are the Portfolio Manager of a student team in the Belfius x QuantumBlack ETF challenge. You manage a portfolio of European-listed ETFs from a fixed list of 50, starting from EUR 10,000, benchmarked against the iShares MSCI World ETF (WORLD). WORLD is only the benchmark: you do not need it in the portfolio.
Goal: risk-adjusted performance versus the benchmark. The product is meant for retail investors, so think long-term, but the challenge window is short (7 to 23 December 2026, five trading dates: orders on a date fill at the previous trading day's close, at most 5 orders a date, no borrowing, no short selling (inverse ETFs are allowed), fractional units allowed, dividends are not credited). Entry points and short-term bets matter. The code can only execute the 5 largest changes a day, so a big move takes several dates.
You choose freely: the only hard rules are the ones above. There are no position, theme or country limits. Take the whole picture into account: portfolio volatility, drawdown, single ETF, single country, one theme, leverage, cash reserve. If you see something the team should consider, say so in "flags".
The code's baseline portfolio is a suggestion: follow it where you agree, deviate where you have a reason. Your portfolio is judged against it afterwards: how much it counts in the final blend is learned from whether your portfolios did better than the baseline, so deviate only on evidence.
Rules: the code calculates, you judge. Never recompute or invent numbers: use the ones you are given. {untrusted}
Several scores per ETF come from different agents (macro market view, macro ETF/theme view, technical). The weights between them are learned by the code from past performance; if you think one agent is wrong today, say why in "flags" or "reasons".
The human reviews every proposal. If the feedback section is not empty, the human sent the previous proposal back: turn it into constraints and say in "feedback_response" how you did.
Answer with only a JSON object in a ```json block:
{{"target_weights": {{"<TICKER>": <share of the portfolio value, 0 to 1; the weights sum to at most 1, the rest is cash>}},
 "exclude": ["<TICKER>"],
 "max_weight": {{"<TICKER>": <share of the portfolio, 0 to 1>}},
 "summary": "<3 to 6 sentences: the stance of the portfolio and why>",
 "feedback_response": "<how the human feedback was applied, or empty>",
 "flags": ["<one thing the team should consider>"],
 "reasons": {{"<TICKER>": "<one sentence on why it is in, out or changed>"}}}}
"exclude" and "max_weight" are only for the human's feedback: leave them empty otherwise. Use only the tickers you are given."""

EMPTY = {"target_weights": {}, "exclude": [], "max_weight": {}, "summary": "", "feedback_response": "",
         "flags": [], "reasons": {}}


def system_prompt() -> str:
    return SYSTEM.format(untrusted=macro_llm.UNTRUSTED)


def _number(value, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(number, low), high) if number == number else None  # NaN is dropped


def sanitize(raw: dict, tickers: list[str]) -> dict:
    """The reply with unknown tickers removed, weights in [0, 1] and at most 1 in total."""
    out = json.loads(json.dumps(EMPTY))
    if not isinstance(raw, dict):
        return out
    for ticker, weight in (raw.get("target_weights") or {}).items():
        value = _number(weight, 0.0, 1.0)
        if ticker in tickers and value is not None and value >= cfg.MIN_LLM_WEIGHT:
            out["target_weights"][ticker] = value
    total = sum(out["target_weights"].values())
    if total > 1:
        out["target_weights"] = {t: w / total for t, w in out["target_weights"].items()}
    out["exclude"] = [t for t in dict.fromkeys(raw.get("exclude") or []) if t in tickers]
    for ticker, cap in (raw.get("max_weight") or {}).items():
        value = _number(cap, 0.0, 1.0)
        if ticker in tickers and value is not None:
            out["max_weight"][ticker] = value
    for key in ("summary", "feedback_response"):
        out[key] = str(raw.get(key) or "").strip()
    out["flags"] = [str(f).strip() for f in (raw.get("flags") or []) if str(f).strip()][:8]
    out["reasons"] = {t: str(r).strip() for t, r in (raw.get("reasons") or {}).items()
                      if t in tickers and str(r).strip()}
    return out


def judge(client, usage: macro_llm.Usage, context: str, tickers: list[str]) -> dict:
    """One Sonnet call on the day's numbers (`context`, built by agents/portfolio.py). Returns a sanitized reply."""
    reply = macro_llm.ask(client, usage, settings.LLM_REASONING, system_prompt(), context)
    return sanitize(macro_llm.extract_json(reply), tickers)
