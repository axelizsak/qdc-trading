"""Two-week temperature forecast against the 10-year normal, Europe and US (Open-Meteo, no key).

Cold spells raise gas and power demand; the effect on energy ETFs is indirect, to be tested.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from agents.macro_sources._common import evidence, http_client

SOURCE = "open_meteo"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
REGIONS = {
    "Europe": {"Berlin": (52.52, 13.40), "Paris": (48.85, 2.35), "Milan": (45.46, 9.19), "Amsterdam": (52.37, 4.90)},
    "US": {"Chicago": (41.88, -87.63), "New York": (40.71, -74.01), "Dallas": (32.78, -96.80)},
}
DAYS = 14
YEARS = 10


def _daily(http, url: str, params: dict) -> pd.Series:
    response = http.get(url, params={**params, "daily": "temperature_2m_mean", "timezone": "UTC"})
    response.raise_for_status()
    daily = response.json()["daily"]
    return pd.Series(daily["temperature_2m_mean"], index=pd.to_datetime(daily["time"]), dtype=float)


def _city_anomaly(http, lat: float, lon: float, as_of: date) -> tuple[float, float]:
    """(forecast mean - normal mean, std of the yearly means) over the next DAYS days."""
    coords = {"latitude": lat, "longitude": lon}
    forecast = _daily(http, FORECAST, {**coords, "forecast_days": DAYS + 1}).iloc[1:DAYS + 1]
    history = _daily(http, ARCHIVE, {**coords, "start_date": date(as_of.year - YEARS, 1, 1).isoformat(),
                                     "end_date": (as_of - timedelta(days=7)).isoformat()})
    yearly = []
    for years_back in range(1, YEARS + 1):
        start = forecast.index[0] - pd.DateOffset(years=years_back)
        window = history[(history.index >= start) & (history.index < start + pd.Timedelta(days=DAYS))]
        if len(window) == DAYS:
            yearly.append(window.mean())
    if len(yearly) < 3:
        raise ValueError("not enough history for a normal")
    return float(forecast.mean() - np.mean(yearly)), float(np.std(yearly))


def collect(as_of: date, http=None) -> list[dict]:
    http = http or http_client()
    rows = []
    for region, cities in REGIONS.items():
        anomalies, spreads = zip(*[_city_anomaly(http, lat, lon, as_of) for lat, lon in cities.values()])
        anomaly, spread = float(np.mean(anomalies)), float(np.mean(spreads))
        z = anomaly / spread if spread > 0 else 0.0
        rows.append(evidence(
            as_of, "energy_weather", SOURCE, "number",
            f"{region} ({', '.join(cities)}): next {DAYS} days forecast {anomaly:+.1f}°C against the "
            f"{YEARS}-year normal for these dates (z {z:+.1f}; negative = colder, more heating demand)",
            anomaly, "https://open-meteo.com"))
    return rows
