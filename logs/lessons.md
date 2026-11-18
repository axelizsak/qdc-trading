# Lessons journal

Written by the review loop (graph/review_loop.py) from the logs and the prices: facts with their numbers, no opinion. Newest review first. The agents read it before they decide.

## Review of 2026-11-18

**Weights** (each session, a weight is multiplied by e^(2 x what following it earned in that session); start -> today

- The technical agent's three answers: mean_reversion 33% -> 15% (-0.014% a session over 1188, right 49%); momentum 33% -> 24% (+0.006% a session over 1247, right 54%); trend 33% -> 61% (+0.043% a session over 1242, right 53%)
- The four agents: macro_etf 25% -> 25% (+0.235% a session over 4, right 75%); macro_market 25% -> 25% (+0.221% a session over 6, right 67%); technical_det 25% -> 25% (+0.191% a session over 7, right 57%); technical_llm 25% -> 25% (+1.275% a session over 1, right 100%)
- The LLM's portfolio (llm) against the code's (code): code 40% -> 40% (+0.181% a session over 1, right 100%); llm 60% -> 60% (+0.200% a session over 1, right 100%)

**Where each agent is right** (each score judged on the 3 sessions that followed it; calls with a score of at least 0.1)

- macro_etf: calls up right 26% (n=34), calls down right 76% (n=58)
- macro_market: calls up right 37% (n=35), calls down right 60% (n=143)
- technical_det: calls up right 47% (n=78), calls down right 73% (n=89)
- technical_llm: calls up right 11% (n=9), calls down right 86% (n=14)

**Largest misses of the last judged day** (17 Nov)

- technical_det +0.79 on BEL 20: -2.0% over 1 session
- technical_det +0.70 on HEALTH CARE: -1.9% over 1 session
- technical_det +0.72 on EUROPE 600: -1.8% over 1 session

**ETF analysts by theme** (calls right / calls made, best first)

- Digital assets 12/12, Precious metals 8/10, Leveraged & inverse 12/16, Size & style factors 9/13, ..., Emerging markets 6/10, Fixed income / Bonds 7/13, Equity / Tech 5/10, Defensive sector 3/7

**Macro views by factor** (views right / views judged, best first)

- gold_silver 8/9, crypto 8/10, copper_metals 6/8, geopolitics_defence 5/7, ..., china_em 4/7, growth_recession 4/7, uranium 2/5, eurusd 1/9

## Review of 2026-11-16

**Weights** (each session, a weight is multiplied by e^(2 x what following it earned in that session); start -> today

- The technical agent's three answers: mean_reversion 33% -> 16% (-0.013% a session over 1186, right 49%); momentum 33% -> 24% (+0.005% a session over 1245, right 54%); trend 33% -> 60% (+0.042% a session over 1240, right 53%)
- The four agents: macro_etf 25% -> 25% (+0.157% a session over 3, right 67%); macro_market 25% -> 25% (+0.194% a session over 5, right 60%); technical_det 50% -> 50% (+0.005% a session over 5, right 60%)
- Claude's portfolio (llm) against the code's (code): code 40% -> 40% (no session judged yet); llm 60% -> 60% (no session judged yet)

**Where each agent is right** (each score judged on the 3 sessions that followed it; calls with a score of at least 0.1)

- macro_etf: calls up right 32% (n=22), calls down right 62% (n=34)
- macro_market: calls up right 42% (n=24), calls down right 38% (n=86)
- technical_det: calls up right 75% (n=52), calls down right 66% (n=61)

**Largest misses of the last judged day** (13 Nov)

- technical_det -1.00 on PALLADIUM LVG: +3.1% over 1 session
- macro_etf +0.30 on NASDAQ 3X: -6.7% over 1 session
- technical_det -1.00 on S&P 500 SHORT: +1.6% over 1 session

**ETF analysts by theme** (calls right / calls made, best first)

- Digital assets 6/6, Precious metals 4/5, Size & style factors 5/7, Leveraged & inverse 4/7

**Macro views by factor** (views right / views judged, best first)

- copper_metals 4/5, crypto 4/6, tech_ai_semis 4/6, uranium 2/3, ..., inflation 2/6, china_em 1/4, oil 1/4, eurusd 0/5

**Macro sources**

- Weak regular sources, for a person to decide: calendar, cb_tone, container_freight, gas_storage, open_meteo, rates_credit

## Review of 2026-11-15

Every score is judged on the 3 sessions that followed it. The technical rule score is also recomputed for the last 60 quote days.

**Agent weights** (starting weight -> weight the evidence supports)

- technical_det: 50% -> 51% (60 days judged, rank correlation +0.01, hit rate 49%)
- macro_market: 25% -> 24% (6 days judged, rank correlation +0.00, hit rate 37%)
- macro_etf: 25% -> 25% (3 days judged, rank correlation +0.02, hit rate 55%)

**Where each agent is right** (calls with a score of at least 0.1)

- macro_etf: calls up right 32% (n=22), calls down right 62% (n=34)
- macro_market: calls up right 42% (n=24), calls down right 38% (n=86)
- technical_det: calls up right 49% (n=196), calls down right 59% (n=132)

**Largest misses of the last judged day** (13 Nov)

- technical_det -1.00 on PALLADIUM LVG: +3.1% over 1 session
- macro_etf +0.30 on NASDAQ 3X: -6.7% over 1 session
- technical_det -1.00 on S&P 500 SHORT: +1.6% over 1 session

**ETF analysts by theme** (calls right / calls made, best first)

- Digital assets 6/6, Precious metals 4/5, Size & style factors 5/7, Leveraged & inverse 4/7

**Macro views by factor** (views right / views judged, best first)

- copper_metals 4/5, crypto 4/6, tech_ai_semis 4/6, uranium 2/3, ..., inflation 2/6, china_em 1/4, oil 1/4, eurusd 0/5

**Macro sources**

- Weak regular sources, for a person to decide: calendar, cb_tone, container_freight, gas_storage, open_meteo, rates_credit
