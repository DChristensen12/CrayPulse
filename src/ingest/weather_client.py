"""
weather_client.py
─────────────────────────────────────────────────────────────
NWS weather data source for Pulse.

Pulls observations from the LBNL1 station (or whatever NWS_STATION_ID is set to)
and returns a DataFrame of reliable weather features. No API key needed; only a
descriptive User-Agent in headers.

Important quirk of the NWS observations endpoint: it serves a rolling window of
roughly the last 7 days at non-airport stations like LBNL1. You CANNOT use this
for historical backfill — only for live operation. For training history you'd
need a different source (e.g. WeatherAPI.com, which Night Heron already uses).
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import requests
from config.config import Config

logger = logging.getLogger(__name__)

# Properties we pull from NWS observations. Each entry says: (NWS property name,
# output column name, NWS unit, conversion function to canonical unit).
# Canonical units: temperature in °C, pressure in hPa, wind in km/h, humidity in %.
_NWS_PROPERTIES = [
    # (nws_name,                out_name,        convert)
    ("temperature",             "air_temp_c",    lambda v: v),
    ("dewpoint",                "dewpoint_c",    lambda v: v),
    ("relativeHumidity",        "humidity_pct",  lambda v: v),
    ("windSpeed",               "wind_kmh",      lambda v: v),
    ("windDirection",           "wind_dir_deg",  lambda v: v),
    ("windGust",                "wind_gust_kmh", lambda v: v),
    ("barometricPressure",      "pressure_hpa",  lambda v: v / 100.0 if v is not None else None),
]


def fetch_nws_weather(
    start_time,
    end_time,
    station_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all reliable weather features from the NWS station over [start_time, end_time].

    Returns a DataFrame with a UTC DatetimeIndex and one column per property
    in _NWS_PROPERTIES. Empty DataFrame on any failure.

    Note: NWS serves only a ~7-day rolling window for personal stations like LBNL1.
    Requests for longer windows will silently return only what's available.
    """
    station = station_id or Config.NWS_STATION_ID
    base_url = f"https://api.weather.gov/stations/{station}/observations"
    headers = {
        "User-Agent": Config.NWS_USER_AGENT,
        "Accept": "application/geo+json",
    }
    params = {
        "start": _ensure_utc_suffix(start_time),
        "end":   _ensure_utc_suffix(end_time),
        "limit": 500,
    }

    all_features = []
    next_url = base_url
    page = 0

    while next_url and page < 50:  # hard cap as safety net
        page += 1
        try:
            resp = requests.get(
                next_url,
                headers=headers,
                params=(params if page == 1 else None),
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"NWS fetch failed on page {page}: {e}")
            break

        if resp.status_code != 200:
            logger.error(
                f"NWS API error {resp.status_code} for station {station}: "
                f"{resp.text[:200]}"
            )
            break

        data = resp.json()
        all_features.extend(data.get("features", []) or [])

        # pagination.next is a string URL when present; defend against null.
        pagi = data.get("pagination") or {}
        next_url = pagi.get("next") if isinstance(pagi, dict) else None
        if not isinstance(next_url, str) or not next_url.startswith("http"):
            next_url = None

    if not all_features:
        logger.warning(f"No NWS observations returned for station {station}.")
        return pd.DataFrame()

    # Flatten properties → rows
    records = []
    for feat in all_features:
        props = feat.get("properties", {}) or {}
        ts = props.get("timestamp")
        if not ts:
            continue

        row = {"datetime": ts}
        for nws_name, out_name, convert in _NWS_PROPERTIES:
            measurement = props.get(nws_name)
            if isinstance(measurement, dict):
                raw_val = measurement.get("value")
            else:
                raw_val = measurement
            row[out_name] = convert(raw_val) if raw_val is not None else None
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for col in [out for _, out, _ in _NWS_PROPERTIES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any column that's entirely null. At LBNL1 that's typically nothing
    # in this list, but at other stations (or after sensor failures) it cleans
    # up unusable features.
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        logger.info(f"NWS: dropping fully-null columns at {station}: {null_cols}")
        df = df.drop(columns=null_cols)

    logger.info(
        f"NWS: fetched {len(df):,} observations from {station} "
        f"({df.index.min()} → {df.index.max()}) "
        f"with features: {list(df.columns)}"
    )
    return df


def _ensure_utc_suffix(t):
    """Make sure a timestamp string is recognizable as UTC by the NWS API."""
    if isinstance(t, datetime):
        s = t.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        s = str(t)
    if not (s.endswith("Z") or "+00" in s or "-00" in s):
        return s + "Z"
    return s
