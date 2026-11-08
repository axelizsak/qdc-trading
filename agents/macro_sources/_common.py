"""Helpers shared by the collectors: evidence rows, history context, HTTP client."""
from datetime import date

import httpx
import numpy as np
import pandas as pd


def http_client() -> httpx.Client:
    """HTTP client with a descriptive User-Agent."""
    agent = "qdc-trading/0.1 (QTEM Data Challenge student project)"
    return httpx.Client(headers={"User-Agent": agent}, timeout=30, follow_redirects=True)


def evidence(as_of: date, factor: str, source: str, kind: str, summary: str,
             value: float | None = None, url: str = "") -> dict:
    """One row in the Evidence format."""
    return {"date": as_of.isoformat(), "factor": factor, "source": source, "kind": kind,
            "value": np.nan if value is None else float(value), "summary": summary, "url": url}


def context(series: pd.Series, lookback: int = 252) -> dict:
    """Latest value of a series, set against its own recent history (code computes, the LLM interprets)."""
    s = pd.to_numeric(series, errors="coerce").dropna().tail(lookback)
    if s.empty:
        raise ValueError("empty series")
    last = float(s.iloc[-1])
    std = float(s.std()) if len(s) > 1 else 0.0
    return {
        "last": last,
        "z": (last - float(s.mean())) / std if std > 0 else 0.0,
        "pct": float((s <= last).mean() * 100),  # percentile within the window
        "chg_5": last - float(s.iloc[-6]) if len(s) > 5 else np.nan,
        "n": len(s),
    }


def describe(name: str, ctx: dict, unit: str = "", digits: int = 2) -> str:
    """'VIX/VIX3M = 0.92 (z +1.3, 85th percentile of the last 252 obs, 5-obs change +0.05)'."""
    chg = "" if np.isnan(ctx["chg_5"]) else f", 5-obs change {ctx['chg_5']:+.{digits}f}{unit}"
    return (f"{name} = {ctx['last']:.{digits}f}{unit} (z {ctx['z']:+.1f}, {ctx['pct']:.0f}th percentile "
            f"of the last {ctx['n']} obs{chg})")
