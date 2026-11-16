# backtest

Two kinds of backtest live here.

## The system backtest (`system_backtest.py`)

It replays the whole code-only system on past prices with the competition's rules: the technical agent's rule
score, then the Portfolio Manager's code portfolio, then orders and a book. Every past 3-week window is traded
like the competition: decisions on a Monday, Thursday, Monday, Thursday, Monday, each from the prices known
that morning, orders filled at the last close before the day, at most 5 orders a day, €10,000 to start,
valued against the same money in the benchmark (WORLD).

Nothing that asks Claude is replayed (Macro/News, `technical_llm`, the Portfolio Manager's own portfolio). An
LLM already knows what happened in the past and a web search returns later articles, so their backtest would
look into the future. They are judged forward only, by the review loop.

```bash
python -m backtest.system_backtest                                   # every variant, every window since 2022 (about 10 minutes)
python -m backtest.system_backtest --variants current watch_zero     # only these
python -m backtest.system_backtest --since 2024-01-01
```

**To test an idea:** add one line to `VARIANTS` (a setting to change), and a function to `SCORES` if the idea
is a new way to turn the technical sheet into a score. Every variant runs on the same windows.

**Results** (`results/`, committed):

| File | Content |
|---|---|
| `system_backtest_summary.csv` | One row per variant: mean return of a window, the benchmark's, the excess and its t-value, share of windows that beat the benchmark, worst window, and the quality of the score on its own |
| `system_backtest.csv` | One row per variant and window |
| `system_backtest_signals.csv` | What each answer of the technical sheet was worth over the next 1 to 15 sessions: the return of the ETFs with that answer minus the average ETF. `technical_llm` reads this file before it scores |
| `system_backtest_fill_rule.csv` | One row per past Monday and Thursday: what an order filled at the previous close was worth by the close of the day, for the average ETF and for the ETFs that opened highest |

**How to read them.** Windows start every week and last three, so they overlap: the t-value counts them as one
third as many. Below about 2 in absolute value, a difference is not distinguishable from luck. A variant that
looks better by a tenth of a percent with a t-value under 1 is not a finding.

## What the first full run found (16 November 2026)

252 windows from 2022-01-03 to 2026-10-26, 21 variants, 5,292 simulated competitions. No run broke a trading rule.

**1. The fill rule matters more than anything the agents do.** The team's reading is that an order of day T fills at the close of T-1. If that is right and an order can be sent after the market has opened, the price paid is yesterday's while today's is already visible. On 490 past Mondays and Thursdays, the 5 ETFs with the largest opening gap were already +1.91% above their fill price at the open and ended the day +1.87% above it (82% of them up, worst day -3.5%); the average ETF did +0.04%. Five such days compound to about +10%, against +0.43% for a whole window of the current system. Under the other rule (a fill at the close of T itself) this is zero, and the current system earns +0.25% a window instead of +0.43%. Two questions for Belfius decide it: which close, and until what time an order of a trading day is accepted. The system does not use today's prices: whether to is a choice for the team, and for Belfius to allow.

**2. The code-only system earns the benchmark's return per unit of risk, with a third of its risk.** A window returns +0.43% on average against +0.65% for WORLD. The portfolio takes 0.37 of WORLD's moves: it gains less when the market rises and loses less when it falls (worst window -6.7% against -13.9%). The gap to the benchmark is not distinguishable from luck (t -0.8). The leaderboard ranks on return, so in a rising December this portfolio trails a fully invested one.

**3. No variant is proven better.** Against the current system on the same windows, no t-value reaches 2.

| Variant | What it changes | Return | Worst window | Risk | Return per risk | Against the current system (t) |
|---|---|---:|---:|---:|---:|---:|
| `current` | the system as it is | +0.43% | -6.7% | 2.10% | 0.21 | — |
| `no_timing` | orders never deferred to a later date | +0.45% | -6.7% | 2.10% | 0.21 | +0.01% (+1.3) |
| `less_invested` | 60% invested when the market view is neutral, instead of 85% | +0.31% | -4.5% | 1.69% | 0.18 | -0.12% (-1.6) |
| `cap_20` | one ETF at most 20% instead of 30% | +0.41% | -5.8% | 1.65% | 0.25 | -0.03% (-0.4) |
| `risk_aversion_1` | the optimiser fears risk much less (1 instead of 5) | +0.52% | -6.0% | 2.11% | 0.25 | +0.09% (+1.2) |
| `no_leverage` | no leveraged or inverse ETF at all | +0.47% | -6.7% | 2.12% | 0.22 | +0.04% (+1.1) |
| `benchmark_allowed` | the code portfolios may buy the benchmark ETF itself | +0.42% | -6.7% | 2.13% | 0.20 | -0.01% (-0.3) |
| `skip_overbought` | no entry on an ETF the mean-reversion measures call overbought | +0.52% | -6.7% | 2.09% | 0.25 | +0.09% (+1.1) |
| `dip_and_skip` | both of the above | +0.57% | -9.4% | 2.21% | 0.26 | +0.14% (+1.1) |
| `confirmed_only` | a buy counts only with both confirmations | +0.55% | -4.5% | 1.92% | 0.29 | +0.12% (+1.1) |
| `watch_zero` | a 'watch' signal scores 0 instead of +0.5 | +0.51% | -4.0% | 1.72% | 0.30 | +0.08% (+0.6) |
| `buy_only_confirmed` | only a 'buy' with both confirmations scores | +0.58% | -3.2% | 1.68% | 0.34 | +0.15% (+0.9) |
| benchmark | the same money in WORLD | +0.65% | -13.9% | 2.90% | 0.22 | |

Return and risk are per 3-week window; risk is the spread of a window's return.

**4. The most consistent lead: a "watch" signal should not count as a buy.** The ETFs the rule calls "watch" lagged the average ETF, and scoring them 0 gave a higher return with less risk and a smaller worst window. It is a lead, not a proof.

**5. What each answer of the technical sheet was worth:** the return of the ETFs with that answer over the next sessions, minus the average ETF (* = at least 2 standard errors from zero; with 50 cells tested, two or three stars are expected by luck alone).

| What the sheet says about an ETF | 3 sessions | 10 sessions | 15 sessions |
|---|---:|---:|---:|
| rule: buy | +0.16% * | +0.27% | +0.33% |
| rule: watch | -0.12% * | -0.13% | -0.10% |
| rule: neutral | -0.02% | -0.09% | -0.11% |
| rule: avoid | +0.00% | +0.16% | +0.13% |
| mean reversion: oversold (buy) | -0.01% | -0.20% | -0.28% |
| mean reversion: overbought (avoid) | -0.04% | +0.03% | +0.24% |
| trend strength: strong | +0.06% | +0.04% | +0.11% |
| trend strength: absent | -0.06% * | -0.09% | -0.11% |
| momentum 13612: top fifth | +0.06% | +0.05% | +0.14% |
| momentum 13612: bottom fifth | -0.00% | +0.01% | -0.14% |

The mean-reversion measures (oversold, overbought) showed no short-term value, so `technical_llm` is no longer told to lower a buy because an ETF looks overbought: it is given this table instead.

## The strategy backtests (Achille)

`momentum_signal_test.py`, `momentum_variants.py`, `mean_reversion_test.py`, `combo_backtest.py`: tests of the
technical strategies on their own, on a train period and a later test period. `walk_forward.py` is the common
walk-forward frame for the models of `ml_library/`.
