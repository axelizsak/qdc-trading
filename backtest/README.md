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
third as many. Below about 2 in absolute value, a difference is not distinguishable from luck. The summary also
compares each variant with the current system in the first and in the second half of the history: a variant
"holds" only when it is ahead in both. `system_backtest_signals_by_half.csv` does the same for the signal study.

## What the backtest found (16 November 2026)

252 windows from 2022-01-03 to 2026-10-26, 24 variants, 6,048 simulated competitions. No run broke a trading rule.

**1. The fill rule matters more than anything the agents do.** The rule is that an order of day T fills at the close of T-1 (`FILL_AT_PREVIOUS_CLOSE`). When an order is sent after the market has opened, the price paid is yesterday's while today's is already visible. On 490 past Mondays and Thursdays, the 5 ETFs with the largest opening gap were already +1.91% above their fill price at the open and ended the day +1.87% above it (82% of them up, worst day -3.5%); the average ETF did +0.04%. Five such days compound to about +10%, against +0.43% for a whole window of the current system. The system does not use the day's prices: whether to is a choice for the team. Under the other rule (a fill at the close of T itself) this is zero, and the current system earns +0.25% a window instead of +0.43%.

**2. The current system earns the benchmark's return per unit of risk, with a third of its risk.** A window returns +0.43% on average against +0.65% for WORLD. The portfolio takes 0.37 of WORLD's moves: it gains less when the market rises and loses less when it falls (worst window -6.7% against -13.9%). It lagged in the first half of the history (-0.09% a window) and led in the second (+0.95%). The leaderboard ranks on return, so in a rising December this portfolio trails a fully invested one.

**3. A variant counts only when it is ahead in both halves of the history.** One that is ahead in a single half has found a kind of market, not an improvement. These hold:

| Variant | What it changes | Return | Worst window | Risk | Against the current system (t) | First half | Second half | Holds |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `buys_plus_benchmark` | 'watch' scores 0 and the idle budget goes into the benchmark ETF, up to 60% | +0.69% | -8.2% | 2.23% | +0.26% (+1.7) | +0.37% | +0.15% | yes |
| `skip_overbought` | no entry on an ETF the mean-reversion measures call overbought | +0.52% | -6.7% | 2.09% | +0.09% (+1.1) | +0.15% | +0.04% | yes |
| `risk_aversion_1` | the optimiser fears risk much less (1 instead of 5) | +0.52% | -6.0% | 2.11% | +0.09% (+1.2) | +0.17% | +0.02% | yes |
| `idle_to_benchmark_60` | idle budget goes into the benchmark ETF, up to 60% of the portfolio | +0.52% | -9.0% | 2.18% | +0.09% (+1.6) | +0.13% | +0.04% | yes |
| `idle_to_benchmark` | idle budget goes into the benchmark ETF, up to 30% of the portfolio | +0.48% | -7.3% | 2.15% | +0.05% (+1.1) | +0.07% | +0.03% | yes |
| `no_leverage` | no leveraged or inverse ETF at all | +0.47% | -6.7% | 2.12% | +0.04% (+1.1) | +0.07% | +0.02% | yes |
| `risk_aversion_2` | the optimiser fears risk less (2 instead of 5) | +0.46% | -6.6% | 2.15% | +0.03% (+0.5) | +0.03% | +0.04% | yes |
| `no_timing` | orders never deferred to a later date | +0.45% | -6.7% | 2.10% | +0.01% (+1.3) | +0.01% | +0.02% | yes |

These looked better over the whole history and do not hold: their gain came from the first half, when markets were weak and being less invested helped.

| Variant | What it changes | Return | Worst window | Risk | Against the current system (t) | First half | Second half | Holds |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `buy_only_confirmed` | only a 'buy' with both confirmations scores | +0.58% | -3.2% | 1.68% | +0.15% (+0.9) | +0.46% | -0.17% | no |
| `dip_and_skip` | both of the above | +0.57% | -9.4% | 2.21% | +0.14% (+1.1) | +0.32% | -0.05% | no |
| `confirmed_only` | a buy counts only with both confirmations | +0.55% | -4.5% | 1.92% | +0.12% (+1.1) | +0.39% | -0.15% | no |
| `watch_zero` | a 'watch' signal scores 0 instead of +0.5 | +0.51% | -4.0% | 1.72% | +0.08% (+0.6) | +0.28% | -0.12% | no |
| `cap_20` | one ETF at most 20% instead of 30% | +0.41% | -5.8% | 1.65% | -0.03% (-0.4) | +0.08% | -0.13% | no |
| `less_invested` | 60% invested when the market view is neutral, instead of 85% | +0.31% | -4.5% | 1.69% | -0.12% (-1.6) | -0.04% | -0.21% | no |

Return and risk are per 3-week window; risk is the spread of a window's return. The benchmark returns +0.65% with a risk of 2.90% and a worst window of -13.9%.

**4. The one candidate worth applying: `buys_plus_benchmark`.** Only a full "buy" signal counts ("watch" scores 0), and the money the system has no idea for goes into the benchmark ETF instead of cash, up to 60% of the portfolio. It returns +0.69% a window against +0.43% for the current system and +0.65% for the benchmark, with a risk of 2.23%, and it is ahead in both halves (+0.37% and +0.15%). The difference is not yet beyond luck (t +1.7), and its worst window is -8.2% against -6.7%. Scoring "watch" 0 on its own does not hold: the ETFs it removes lag the average ETF, but replacing them with cash costs more than it saves when the market rises. It is three settings: `SIGNAL_SCORES` in `agents/strategy_config.py`, `IDLE_TO_BENCHMARK` and `BENCHMARK_CAP` in `agents/portfolio_config.py`.

**5. What each answer of the technical sheet was worth:** the return of the ETFs with that answer over the next sessions, minus the average ETF (* = at least 2 standard errors from zero; with 50 cells tested, two or three stars are expected by luck alone). The last two columns split the 3-session figure by half.

| What the sheet says about an ETF | 3 sessions | 10 sessions | 15 sessions | 3 sessions, first half | 3 sessions, second half |
|---|---:|---:|---:|---:|---:|
| rule: buy | +0.16% * | +0.27% | +0.33% | +0.09% | +0.19% |
| rule: watch | -0.12% * | -0.13% | -0.10% | -0.07% | -0.14% |
| rule: neutral | -0.02% | -0.09% | -0.11% | -0.01% | -0.03% |
| rule: avoid | +0.00% | +0.16% | +0.13% | +0.00% | -0.01% |
| mean reversion: oversold (buy) | -0.01% | -0.20% | -0.28% | +0.14% | -0.13% |
| mean reversion: overbought (avoid) | -0.04% | +0.03% | +0.24% | -0.08% | -0.00% |
| trend strength: strong | +0.06% | +0.04% | +0.11% | -0.04% | +0.15% |
| trend strength: absent | -0.06% * | -0.09% | -0.11% | +0.05% | -0.14% |
| momentum 13612: top fifth | +0.06% | +0.05% | +0.14% | +0.05% | +0.07% |
| momentum 13612: bottom fifth | -0.00% | +0.01% | -0.14% | -0.02% | +0.01% |

Only two answers point the same way in both halves with some size: a "buy" is worth a little, a "watch" costs a little. Trend strength and the mean-reversion measures change sign from one half to the other. So `technical_llm` is no longer told to lower a buy because an ETF looks overbought: it is given the table instead.

## The strategy backtests (Achille)

`momentum_signal_test.py`, `momentum_variants.py`, `mean_reversion_test.py`, `combo_backtest.py`: tests of the
technical strategies on their own, on a train period and a later test period. `walk_forward.py` is the common
walk-forward frame for the models of `ml_library/`.
