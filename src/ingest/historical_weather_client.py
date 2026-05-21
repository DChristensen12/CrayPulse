"""
historical_weather_client.py
─────────────────────────────────────────────────────────────
Open-Meteo Historical Archive data source for training-time weather.

The NWS observations endpoint only retains ~7 days at LBNL1, which makes it
unsuitable for backfilling 30+ days of training data. Open-Meteo's archive
goes back to 1940 and is free with no API key required.

The returned columns match what weather_client.py produces, so the model
sees consistent features regardless of which source supplied them:
  air_temp_c, dewpoint_c, humidity_pct, wind_kmh, wind_dir_deg,
  wind_gust_kmh, pressure_hpa

Trade-off vs LBNL1: Open-Meteo is hourly (not 15-min) and reanalyzed (not
directly observed). Values are typically within 1-2°C of LBNL1 for the same
location and hour, close enough that a model trained on Open-Meteo
generalizes cleanly to LBNL1 at inference time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from config.config import Config

logger = logging.getLogger(__name__)

# Berkeley coordinates, roughly central to the SCMG sensor sites.
# All 5 sites are within ~1.5 miles of this point and Berkeley weather is
# spatially uniform at that scale.
_BERKELEY_LAT = 37.873
_BERKELEY_LON = -122.260

_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Map Open-Meteo's parameter names to our canonical column names.
# Open-Meteo wind speed comes in km/h when wind_speed_unit=kmh is set.
# Pressure is hPa (= mb) by default.
_OPEN_METEO_VARIABLES = [
    ("temperature_2m",        "air_temp_c"),
    ("dew_point_2m",          "dewpoint_c"),
    ("relative_humidity_2m",  "humidity_pct"),
    ("wind_speed_10m",        "wind_kmh"),
    ("wind_direction_10m",    "wind_dir_deg"),
    ("wind_gusts_10m",        "wind_gust_kmh"),
    ("pressure_msl",          "pressure_hpa"),
]


def fetch_open_meteo_weather(
    start_time,
    end_time,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fetch historical hourly weather from Open-Meteo's archive endpoint.

    Returns a DataFrame with a UTC DatetimeIndex and the same columns
    weather_client.fetch_nws_weather produces, so data_loader can merge
    them interchangeably.

    Parameters
    ----------
    start_time, end_time : datetime | str
        Inclusive bounds. Strings should be ISO-8601 dates or datetimes.
    latitude, longitude : float, optional
        Default Berkeley (37.873, -122.260).
    """
    lat = latitude if latitude is not None else _BERKELEY_LAT
    lon = longitude if longitude is not None else _BERKELEY_LON

    # Open-Meteo expects start_date and end_date as YYYY-MM-DD strings.
    if isinstance(start_time, datetime):
        start_date = start_time.strftime("%Y-%m-%d")
    else:
        start_date = str(start_time)[:10]
    if isinstance(end_time, datetime):
        end_date = end_time.strftime("%Y-%m-%d")
    else:
        end_date = str(end_time)[:10]

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start_date,
        "end_date":         end_date,
        "hourly":           ",".join(om_name for om_name, _ in _OPEN_METEO_VARIABLES),
        "wind_speed_unit":  "kmh",
        "timezone":         "UTC",
    }

    try:
        resp = requests.get(_OPEN_METEO_URL, params=params, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo request failed: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        logger.error(
            f"Open-Meteo returned {resp.status_code}: {resp.text[:300]}"
        )
        return pd.DataFrame()

    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        logger.warning(
            f"Open-Meteo returned no observations for "
            f"{start_date} to {end_date}"
        )
        return pd.DataFrame()

    # Build the DataFrame. Each variable in Open-Meteo's response is a
    # parallel array indexed by the 'time' array.
    rows = {"datetime": times}
    for om_name, our_name in _OPEN_METEO_VARIABLES:
        rows[our_name] = hourly.get(om_name, [None] * len(times))

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for _, col in _OPEN_METEO_VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop fully-null columns (rare but possible if Open-Meteo lacks coverage)
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        logger.info(f"Open-Meteo: dropping fully-null columns: {null_cols}")
        df = df.drop(columns=null_cols)

    logger.info(
        f"Open-Meteo: fetched {len(df):,} hourly observations "
        f"({df.index.min()} → {df.index.max()}) "
        f"for ({lat}, {lon}) "
        f"with features: {list(df.columns)}"
    )
    return df
