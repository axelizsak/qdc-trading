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
python -m backtest.system_backtest --daily                           # one book over the whole history, valued every day
```

**Daily mode (`--daily`).** The windows answer "what would our competition result have been". The daily mode
answers "is the strategy good": one book from 2022 to today, never reset to cash, a decision every Monday and
Thursday with the same rules, valued at every close. The Sharpe ratio comes from daily returns above the
overnight rate (the `OVERNIGHT CASH` ETF), and two variants are compared on the same days, with a t-value on
the gap between their Sharpe ratios. Results: `system_backtest_daily.csv` (one row per variant and close) and
`system_backtest_daily_summary.csv`.

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

252 windows from 2022-01-03 to 2026-10-26, 23 variants, 5,796 simulated competitions. No run broke a trading rule.

**1. The fill rule matters more than anything the agents do.** The rule is that an order of day T fills at the close of T-1 (`FILL_AT_PREVIOUS_CLOSE`). When an order is sent after the market has opened, the price paid is yesterday's while today's is already visible. On 490 past Mondays and Thursdays, the 5 ETFs with the largest opening gap were already +1.91% above their fill price at the open and ended the day +1.87% above it (82% of them up, worst day -3.5%); the average ETF did +0.04%. Five such days compound to about +10%, against +0.69% for a whole window of the system. The system does not use the day's prices: whether to is a choice for the team. Under the other rule (a fill at the close of T itself) this is zero, and the system earns +0.55% a window instead of +0.69%.

**2. The settings were changed once, on this evidence.** Until 16 November a "watch" signal scored +0.5 and the money the system had no idea for stayed in cash. Now only a full "buy" signal counts, and idle money follows the benchmark ETF, up to 60% of the portfolio.

| Per 3-week window | Return | Risk | Sharpe (a year) | Worst window | Share of the market's moves |
|---|---:|---:|---:|---:|---:|
| The system now | +0.69% | 2.23% | 1.38 | -8.2% | 0.53 |
| The settings before 16 November | +0.43% | 2.10% | 0.91 | -6.7% | 0.37 |
| The benchmark (WORLD) | +0.65% | 2.90% | 0.99 | -13.9% | 1.00 |

The new settings are ahead of the old ones in both halves of the history (+0.37% and +0.15% a window), by +0.26% overall (t +1.7: likely, not proven). The price is a deeper worst window. Scoring "watch" 0 on its own was tested first and withdrawn: the ETFs it removes lag the average ETF, but replacing them with cash cost return when the market rose. The Sharpe ratios rest on under five years of history and are known to about 0.6 either way.

**3. A variant counts only when it is ahead in both halves of the history.** One that is ahead in a single half has found a kind of market, not an improvement. Against the system as it is now, these hold, all by small margins that are not beyond luck:

| Variant | What it changes | Return | Worst window | Risk | Against the current system (t) | First half | Second half | Holds |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `skip_overbought` | no entry on an ETF the mean-reversion measures call overbought | +0.80% | -8.2% | 2.34% | +0.11% (+1.4) | +0.09% | +0.13% | yes |
| `more_invested` | 99% invested when the market view is neutral | +0.79% | -8.9% | 2.26% | +0.10% (+1.7) | +0.03% | +0.16% | yes |
| `dip_and_skip` | both of the above | +0.76% | -9.7% | 2.49% | +0.07% (+0.7) | +0.12% | +0.03% | yes |
| `risk_aversion_1` | the optimiser fears risk much less (1 instead of 5) | +0.73% | -8.2% | 2.22% | +0.04% (+0.9) | +0.03% | +0.05% | yes |
| `buy_only` | only 'buy' scores: 'watch' and 'avoid' score 0 | +0.70% | -8.2% | 2.23% | +0.00% (+1.3) | +0.01% | +0.00% | yes |

And these do not:

| Variant | What it changes | Return | Worst window | Risk | Against the current system (t) | First half | Second half | Holds |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `before_16_nov` | the settings before 16 November 2026: 'watch' scores +0.5, idle budget stays in cash | +0.43% | -6.7% | 2.10% | -0.26% (-1.7) | -0.37% | -0.15% | no |
| `idle_in_cash` | idle budget stays in cash instead of following the benchmark | +0.51% | -4.0% | 1.72% | -0.18% (-1.3) | -0.09% | -0.27% | no |
| `benchmark_cap_30` | the benchmark top-up stops at 30% of the portfolio instead of 60% | +0.62% | -5.4% | 1.92% | -0.08% (-1.0) | -0.05% | -0.10% | no |
| `less_invested` | 60% invested when the market view is neutral, instead of 85% | +0.54% | -7.6% | 1.82% | -0.15% (-1.8) | -0.02% | -0.29% | no |
| `confirmed_only` | a buy counts only with both confirmations | +0.75% | -8.3% | 2.17% | +0.06% (+0.8) | +0.15% | -0.03% | no |
| `buy_only_confirmed` | only a 'buy' with both confirmations scores | +0.75% | -8.3% | 2.17% | +0.06% (+0.8) | +0.16% | -0.03% | no |

Return and risk are per 3-week window; risk is the spread of a window's return. Nothing further was applied: stacking small unproven gains is how a backtest gets fooled.

**4. What each answer of the technical sheet was worth:** the return of the ETFs with that answer over the next sessions, minus the average ETF (* = at least 2 standard errors from zero; with 50 cells tested, two or three stars are expected by luck alone). The last two columns split the 3-session figure by half.

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

**5. The daily book (16 November 2026): Sharpe ratio from daily returns.** 4.9 years, 508 decision days, 23
variants, no rule broken. Sharpe = return a year above the overnight rate (+0.9% a year on average) / volatility a year.

| | Sharpe | First half | Second half | Return a year | Volatility | Worst fall |
|---|---:|---:|---:|---:|---:|---:|
| The system now | 1.18 | 0.63 | 1.72 | +13.3% | 10.3% | -10.1% |
| The settings before 16 November | 0.74 | 0.00 | 1.48 | +8.2% | 10.3% | -19.7% |
| The benchmark (WORLD) | 0.73 | 0.53 | 0.93 | +10.6% | 14.0% | -21.7% |

No variant beats the system now by a Sharpe gap with a t-value of 2. The best, `skip_overbought` (1.32, t +1.0),
is ahead only in the first half. A Sharpe ratio measured on 4.9 years is known to about 0.6 either way.

**Letting the backtest pick the settings does not help.** Each year (or quarter), take the variant with the best
Sharpe ratio so far and trade it in the next period: from 2023 on, that gives a Sharpe ratio of 1.19 (picked every
year) or 1.36 (every quarter), against 1.35 for the settings before 16 November kept fixed. The best variant of the
past is rarely the best of the next year. (The system now scores 1.70 over the same years, but its settings were
chosen after seeing this history, so that number flatters it.)

## The strategy backtests (Achille)

`momentum_signal_test.py`, `momentum_variants.py`, `mean_reversion_test.py`, `combo_backtest.py`: tests of the
technical strategies on their own, on a train period and a later test period. `walk_forward.py` is the common
walk-forward frame for the models of `ml_library/`.
