"""
Trouve, pour chacun des 50 ETF du QDC, le symbole Yahoo sur la bourse de référence de Belfius
(Xetra pour presque tous, Euronext Bruxelles pour BEL 20, Euronext Milan pour COPPER LVG
et PALLADIUM LVG). C'est le cours de clôture de cette bourse qui sert à exécuter et valoriser
les ordres.

Usage :  python data/check_etfs_yfinance.py   (depuis la racine du repo)
Sortie : data/etf_yahoo_map.csv (symbole Yahoo retenu par ETF + statut)

Statuts :
  OK            trouvé sur la bourse de référence, au moins 200 jours sur un an
  PARTIEL       trouvé sur la bourse de référence, mais moins de 200 jours
  AUTRE BOURSE  pas trouvé sur la bourse de référence : une autre cotation est retenue
  INTROUVABLE   aucune cotation avec des données
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from data.universe import load_universe, reference_exchange, yahoo_suffix  # noqa: E402

# Symboles que la recherche par ISIN ne trouve pas sur la bourse de référence.
# Ils sont testés en plus des résultats de la recherche.
EXTRA_CANDIDATES = {
    "EURO STOXX 50": ["SXRT.DE"],
    "CHINA A": ["36BZ.DE"],
    "COPPER MINERS": ["4COP.DE"],
    "OVERNIGHT CASH": ["CSH2.DE", "LYXO.DE"],
}


def candidates(ticker: str, isin: str) -> list[str]:
    try:
        quotes = yf.Search(isin, max_results=10).quotes
    except Exception:
        quotes = []
    found = [q["symbol"] for q in quotes if q.get("symbol")]
    return list(dict.fromkeys(EXTRA_CANDIDATES.get(ticker, []) + found))


def describe(symbol: str) -> dict | None:
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="1y", auto_adjust=False)
        currency = t.fast_info.get("currency")
    except Exception:
        return None
    if len(h) == 0:
        return None
    return {"symbol": symbol, "currency": currency, "days_1y": len(h), "last_date": h.index[-1].date()}


rows = []
for etf in load_universe().itertuples():
    suffix = yahoo_suffix(etf.ticker)
    cands = [c for c in (describe(s) for s in candidates(etf.ticker, etf.isin)) if c]
    # Préférence : bourse de référence, puis cotation en EUR, puis le plus d'historique.
    cands.sort(key=lambda c: (c["symbol"].endswith(suffix), c["currency"] == "EUR", c["days_1y"]), reverse=True)
    best = cands[0] if cands else None
    if best is None:
        status = "INTROUVABLE"
    elif not best["symbol"].endswith(suffix):
        status = "AUTRE BOURSE"
    else:
        status = "OK" if best["days_1y"] >= 200 else "PARTIEL"
    rows.append({"name": etf.ticker, "isin": etf.isin, "exchange": reference_exchange(etf.ticker),
                 "status": status, "yahoo_symbol": best and best["symbol"],
                 "currency": best and best["currency"], "days_1y": best and best["days_1y"],
                 "last_date": best and best["last_date"],
                 "autres_symboles": ", ".join(c["symbol"] for c in cands[1:])})
    print(f"{status:12} {etf.ticker:18} {etf.isin}  -> {best and best['symbol']} "
          f"({best and best['currency']}, {best and best['days_1y']} j)")

df = pd.DataFrame(rows)
out = Path(__file__).resolve().parent / "etf_yahoo_map.csv"
df.to_csv(out, index=False)
print("\n", df["status"].value_counts().to_string())
print("\nÀ vérifier :")
print(df[df["status"] != "OK"][["name", "exchange", "status", "yahoo_symbol", "autres_symboles"]].to_string(index=False))
print(f"\nRésultat écrit dans {out}")
