# Lessons journal

Written by the review loop (graph/review_loop.py) from the logs and the prices: facts with their numbers, no opinion. Newest review first. The agents read it before they decide.

## Review of 2026-11-16

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
