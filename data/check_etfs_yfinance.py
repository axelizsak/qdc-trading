"""
Vérifie que les 50 ETF du QDC (liste Belfius) sont disponibles sur yfinance.
Usage :  python data/check_etfs_yfinance.py   (depuis la racine du repo)
Sortie : data/etf_yahoo_map.csv (ticker Yahoo retenu par ISIN + statut)
"""
from pathlib import Path

import yfinance as yf
import pandas as pd

ETFS = [
    ("WORLD","IE00B4L5Y983"),("S&P 500","IE00B5BMR087"),("EUROPE 600","LU0908500753"),
    ("EURO STOXX 50","IE00B53L3W79"),("IBEX 35","FR0010655746"),("BEL 20","FR0000021842"),
    ("WORLD SMALL CAP","IE00BF4RFH31"),("WORLD MIN VOL","IE00B8FHGS14"),("DIVIDEND LEADERS","NL0011683594"),
    ("NASDAQ INCOME","IE000U9J8HX9"),("NASDAQ","IE00B53SZB19"),("SEMIS","IE000I8KRLL9"),
    ("CYBERSEC","IE00BYPLS672"),("BIOTECH","IE00BQ70R696"),("ROBOTICS","IE00BYZK4552"),
    ("SPACE","IE000YU9K6K2"),("QUANTUM","IE0007Y8Y157"),("EURO BANKS","LU1829219390"),
    ("HEALTH CARE","IE00BM67HK77"),("EMMKT","IE00BKM4GZ66"),("CHINA LC","IE00B02KXK85"),
    ("INDIA","IE00BZCQB185"),("KOREA","LU1900066975"),("BRAZIL","IE00B0M63516"),
    ("CHINA A","IE00BQT3WG13"),("WORLD ENERGY","IE00BM67HM91"),("OIL & GAS E&P","IE00B6R51Z18"),
    ("URANIUM","IE000M7V94E1"),("CLEAN ENERGY","IE00BK5BCH80"),("HYDROGEN","FR0010930644"),
    ("COPPER MINERS","IE0003Z9E2Y3"),("RARE EARTHS","IE0002PG6CA6"),("GOLD","IE00B4ND3602"),
    ("GOLD MINERS","IE00BQQP9F84"),("SILVER MINERS","IE000UL6CLP7"),("HIGHY BOND","LU1109942653"),
    ("US TREASURY 20Y","LU1407890620"),("GLOBAL AGG BOND","IE00BG47KH54"),("EUR GOVT BOND","IE00B4WXJJ64"),
    ("OVERNIGHT CASH","LU1190417599"),("DEFENSE","IE000YYE6WK5"),("STAPLES","IE00BM67HN09"),
    ("BITCOIN","XS2940466316"),("CRYPTO EQUITY","IE00BMDKNW35"),("STOXX SHORT","FR0010424143"),
    ("COPPER LVG","IE00B8JVMZ80"),("PALLADIUM LVG","IE00B94QLN63"),("NASDAQ 3X","IE00BLRPRL42"),
    ("S&P 500 SHORT","LU0322251520"),("EURO STOXX 2X","FR0010468983"),
]

rows = []
for name, isin in ETFS:
    try:
        quotes = yf.Search(isin, max_results=10).quotes
    except Exception as e:
        quotes = []
    cands = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue
        try:
            h = yf.Ticker(sym).history(period="1y", auto_adjust=False)
            cur = yf.Ticker(sym).fast_info.get("currency")
        except Exception:
            continue
        if len(h) == 0:
            continue
        cands.append({"symbol": sym, "exchange": q.get("exchange"), "currency": cur,
                      "days_1y": len(h), "last_date": h.index[-1].date()})
    # préférence : cotation en EUR, puis le plus d'historique
    cands.sort(key=lambda c: (c["currency"] == "EUR", c["days_1y"]), reverse=True)
    best = cands[0] if cands else None
    status = "OK" if best and best["days_1y"] >= 200 else ("PARTIEL" if best else "INTROUVABLE")
    rows.append({"name": name, "isin": isin, "status": status,
                 "yahoo_symbol": best and best["symbol"], "currency": best and best["currency"],
                 "days_1y": best and best["days_1y"], "last_date": best and best["last_date"],
                 "autres_symboles": ", ".join(c["symbol"] for c in cands[1:])})
    print(f"{status:12} {name:18} {isin}  -> {best and best['symbol']} ({best and best['currency']}, {best and best['days_1y']} j)")

df = pd.DataFrame(rows)
out = Path(__file__).resolve().parent / "etf_yahoo_map.csv"
df.to_csv(out, index=False)
print("\n", df["status"].value_counts().to_string())
print(f"\nRésultat écrit dans {out}")
