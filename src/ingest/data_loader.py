import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
from config.config import Config
from src.ingest.api_client import fetch_network_snapshot


# Columns that should never be treated as model features, even if numeric.
# - EnviroDIY_Mayfly_Batt: sensor health telemetry, not a creek measurement
# - Unnamed: 0, delta, Precip_Max: pandas/CSV artifacts from earlier exports
_NON_FEATURE_COLUMNS = {
    "Unnamed: 0",
    "delta",
    "Precip_Max",
    "EnviroDIY_Mayfly_Batt",
}

# When the requested data window exceeds this many days, NWS LBNL1 cannot
# cover it (it only retains ~7 days). Fall back to Open-Meteo's historical
# archive, which goes back decades. Both sources produce the same column
# names, so the model never sees the difference.
_NWS_HISTORICAL_LIMIT_DAYS = 5

# Weather columns that are accumulated quantities (summed over the preceding
# hour), not instantaneous readings. When we spread an hourly value across the
# finer creek cadence, these must be divided so the pieces re-sum to the
# original hourly total. Everything else (temperature, solar) is instantaneous
# and gets copied across the sub-hourly rows unchanged.
_ACCUMULATED_WEATHER_COLUMNS = {
    "rain_mm",
}


def load_and_preprocess_data(
    file_path=Config.DATA_FILE,
    force_download=False,
    days=30,
    data_source="api",
):
    """
    Load creek data, caching to CSV. Fetches fresh data when the cache is
    missing or force_download=True.

    The cache is a rolling window: new fetches are merged with existing data
    rather than overwriting it, then deduplicated on (datetime, location)
    and trimmed to Config.ROLLING_WINDOW_DAYS. This means the on-disk file
    becomes a usable working set for debugging, offline analysis, and
    --mode update retrains, while staying bounded in size.

    data_source: "api" pulls from the public REST API (3 sensor features).
                 "sql" pulls from the production MySQL database (richer features).
    """
    if not os.path.exists(file_path) or force_download:
        source_label = "API" if data_source == "api" else "SQL database"
        if force_download:
            print(f"Refresh requested. Fetching last {days} days from {source_label}...")
        else:
            print(f"Local file '{file_path}' not found. Pulling from {source_label}...")

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        if data_source == "sql":
            from src.ingest.sql_client import fetch_network_snapshot_sql
            df_raw = fetch_network_snapshot_sql(
                start_time=start_date.isoformat(),
                end_time=end_date.isoformat(),
            )
        else:
            df_raw = fetch_network_snapshot(
                start_time=start_date.isoformat(),
                end_time=end_date.isoformat(),
            )

        if df_raw.empty:
            print(f"CRITICAL ERROR: No data retrieved from {source_label}.")
            if not os.path.exists(file_path):
                sys.exit(1)
            print("Falling back to existing local file.")
        else:
            # Rename raw MMW column names to Pulse's internal names.
            # Only sensors that the API/SQL paths actually expose are listed
            # here; the API tier delivers 3 features (cond, depth, temp) plus
            # battery. The SQL tier may deliver more — any extra columns just
            # pass through with their raw names and get used as model features.
            column_mapping = {
                "Meter_Hydros21_Cond":  "conductivity",
                "Meter_Hydros21_Depth": "depth",
                "Meter_Hydros21_Temp":  "temperature",
                "timestamp":            "datetime",
                "station_id":           "location",
            }
            df_raw = df_raw.rename(columns=column_mapping)

            if Config.USE_NWS_WEATHER:
                df_raw = _merge_weather(df_raw, start_date, end_date, days)

            # ─── Append to rolling cache ─────────────────────────────────
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            combined = _merge_with_cache(df_raw, file_path)
            combined = _trim_to_rolling_window(combined, Config.ROLLING_WINDOW_DAYS)
            combined.to_csv(file_path, index=False)
            print(
                f"Cache updated: {len(combined):,} rows on disk "
                f"({combined['datetime'].min()} → {combined['datetime'].max()})"
            )

    # ─── Load from cache and finalize ──────────────────────────────────────
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    print(f"--- Data Loading Report ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Range: {df.index.min()} to {df.index.max()}")

    # Report which sites are present in the dataset and which are missing.
    # Missing sites are normal (sensors can be down), but worth surfacing.
    sites_present = sorted(df["location"].unique()) if "location" in df.columns else []
    expected = set(Config.LOCATIONS)
    missing = sorted(expected - set(sites_present))
    print(f"Sites present ({len(sites_present)}): {sites_present}")
    if missing:
        print(f"Sites missing (offline this window): {missing}")

    # Feature selection: every numeric column that isn't location or in our
    # exclude set. New columns from SQL or NWS get picked up automatically.
    feature_cols = [
        col for col in df.columns
        if col != "location"
        and col not in _NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        print("CRITICAL ERROR: No numeric feature columns found in dataset.")
        sys.exit(1)

    df_featured = df[["location"] + feature_cols].copy()

    print(f"Active features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"---------------------------\n")

    return df_featured, df, Config.LOCATIONS


def _merge_with_cache(df_new, file_path):
    """
    Combine freshly-fetched data with whatever's already in the cache file,
    deduplicating on (datetime, location). Newer rows win on conflict — this
    lets late-arriving data correct earlier values if the upstream system
    backfilled or recomputed something.
    """
    if not os.path.exists(file_path):
        return df_new

    try:
        existing = pd.read_csv(file_path)
        existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
    except Exception as e:
        # Cache file is corrupted or has incompatible schema. Don't crash —
        # just start fresh with the new data. The old file gets overwritten.
        print(f"[WARN] Could not read existing cache ({e}). Starting fresh.")
        return df_new

    # Ensure schema compatibility. If columns drifted (e.g. you added a new
    # weather feature), the union of columns gets used and missing values
    # become NaN — pandas handles this fine.
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True)

    # Dedupe: keep the most recent version of each (datetime, location) row.
    # 'keep="last"' relies on df_new having been concatenated after existing.
    combined = combined.drop_duplicates(
        subset=["datetime", "location"], keep="last"
    )
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def _trim_to_rolling_window(df, window_days):
    """
    Keep only the last `window_days` of data. Anchored on the latest timestamp
    in the dataset (not 'now'), so the cache stays useful even if a fetch
    happens long after the last real observation.
    """
    if df.empty or window_days is None or window_days <= 0:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    cutoff = df["datetime"].max() - pd.Timedelta(days=window_days)
    trimmed = df[df["datetime"] >= cutoff]

    dropped = len(df) - len(trimmed)
    if dropped > 0:
        print(f"Trimmed {dropped:,} rows older than {window_days} days.")
    return trimmed


def _estimate_rows_per_hour(creek_datetimes):
    """
    Figure out how many creek rows fall in a typical hour, from the median
    gap between consecutive timestamps. The creek reports every 15 minutes,
    so this is normally 4, but computing it keeps the rain disaggregation
    correct if the cadence ever changes (e.g. a sensor switched to 5-min).

    Returns a positive integer, defaulting to 4 if the cadence can't be
    determined (too few rows, or all timestamps identical).
    """
    times = pd.Series(pd.to_datetime(creek_datetimes, utc=True)).drop_duplicates().sort_values()
    if len(times) < 2:
        return 4
    median_gap = times.diff().dropna().median()
    if pd.isna(median_gap) or median_gap.total_seconds() <= 0:
        return 4
    minutes = median_gap.total_seconds() / 60.0
    rows_per_hour = round(60.0 / minutes)
    return max(1, int(rows_per_hour))


def _merge_weather(df_raw, start_date, end_date, days):
    """
    Choose the appropriate weather source based on the window size and merge
    it onto df_raw. NWS has 15-min resolution but only ~7 days of history;
    Open-Meteo has hourly resolution but unlimited history.

    Both sources return the same column names, so downstream code doesn't
    need to know which one ran.

    Rain disaggregation: Open-Meteo precipitation is an accumulation, the total
    millimeters that fell during the preceding hour, not an instantaneous
    reading. When we spread that one hourly value across the creek's four
    15-minute rows, we must divide it so the four pieces re-sum to the original
    hourly total. Copying the full hourly value to all four rows (the old
    behavior) inflated cumulative rainfall ~4x, which both skewed the rain_mm
    model feature and made the rain-aware threshold trip on a quarter of the
    intended rainfall. Temperature and solar radiation are instantaneous and
    are copied across the four rows unchanged.

    If you ever switch the model or the rain-aware threshold to use sub-hourly
    rain INTENSITY (rate) rather than a windowed SUM, this even split is no
    longer right; you'd want a plain forward-fill of the rate instead. The
    division here is specifically correct for sum-based use.
    """
    if days <= _NWS_HISTORICAL_LIMIT_DAYS:
        from src.ingest.weather_client import fetch_nws_weather
        print(f"Fetching NWS weather for station {Config.NWS_STATION_ID} "
              f"(window: {days}d, source: live observations)...")
        df_weather = fetch_nws_weather(start_date.isoformat(), end_date.isoformat())
    else:
        from src.ingest.historical_weather_client import fetch_open_meteo_weather
        print(f"Fetching Open-Meteo historical weather "
              f"(window: {days}d, source: reanalysis archive)...")
        df_weather = fetch_open_meteo_weather(start_date, end_date)

    if df_weather.empty:
        print("Weather data unavailable — proceeding without weather features.")
        return df_raw

    # Collapse to one observation per hour. NWS reports every 15 min so each
    # hour has ~4 obs; Open-Meteo is already hourly. For instantaneous values
    # the hourly mean is right. For accumulated values (rain) we want the
    # hourly sum, not the mean, so the hour's total is preserved before we
    # disaggregate it back out across the creek's sub-hourly rows.
    accumulated_present = [c for c in df_weather.columns if c in _ACCUMULATED_WEATHER_COLUMNS]
    instantaneous_present = [c for c in df_weather.columns if c not in _ACCUMULATED_WEATHER_COLUMNS]

    parts = []
    if instantaneous_present:
        parts.append(df_weather[instantaneous_present].resample("h").mean())
    if accumulated_present:
        parts.append(df_weather[accumulated_present].resample("h").sum())
    df_weather_hourly = pd.concat(parts, axis=1) if parts else df_weather.resample("h").mean()
    # Keep a stable column order matching the source
    df_weather_hourly = df_weather_hourly[[c for c in df_weather.columns if c in df_weather_hourly.columns]]

    # How many creek rows share one weather hour. Used to split accumulated
    # rain evenly so the per-row pieces re-sum to the hourly total.
    rows_per_hour = _estimate_rows_per_hour(df_raw["datetime"])

    # Divide accumulated columns by rows_per_hour BEFORE the merge, so each
    # 15-min row that inherits this hour's value carries its fair share.
    if accumulated_present and rows_per_hour > 1:
        df_weather_hourly = df_weather_hourly.copy()
        for col in accumulated_present:
            df_weather_hourly[col] = df_weather_hourly[col] / rows_per_hour
        print(
            f"Rain disaggregated across {rows_per_hour} sub-hourly rows "
            f"(columns: {accumulated_present}) to preserve windowed sums."
        )

    # Build an hour-bucket key on the creek side and merge.
    creek_dt = pd.to_datetime(df_raw["datetime"], utc=True)
    df_raw["_hour_key"] = creek_dt.dt.floor("h")

    df_merged = df_raw.merge(
        df_weather_hourly,
        how="left",
        left_on="_hour_key",
        right_index=True,
    ).drop(columns=["_hour_key"])

    n_with_weather = (
        df_merged["air_temp_c"].notna().sum()
        if "air_temp_c" in df_merged.columns else 0
    )
    print(
        f"Weather merged — {n_with_weather:,}/{len(df_merged):,} rows "
        f"have weather context "
        f"(features: {list(df_weather_hourly.columns)})"
    )
    return df_merged
