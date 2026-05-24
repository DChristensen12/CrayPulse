import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.config import Config

_RAIN_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "rain_cache")
_RAIN_COL = "rain_mm"
_TIME_CANDIDATES = ["DateTimeUTC", "timestamp", "datetime"]


def _find_time_column(df):
    for c in _TIME_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _estimate_rows_per_hour(times):
    t = pd.Series(pd.to_datetime(times, utc=True)).drop_duplicates().sort_values()
    if len(t) < 2:
        return 4
    gap = t.diff().dropna().median()
    if pd.isna(gap) or gap.total_seconds() <= 0:
        return 4
    minutes = gap.total_seconds() / 60.0
    return max(1, int(round(60.0 / minutes)))


def _cache_path(start_date, end_date):
    return os.path.join(_RAIN_CACHE_DIR, f"rain_{start_date.date()}_{end_date.date()}.csv")


def _get_rain_hourly(start_date, end_date):
    # Import here, not at module top, so an import problem surfaces as a clear
    # runtime error during a fetch rather than killing the script silently.
    from src.ingest.historical_weather_client import fetch_open_meteo_weather

    os.makedirs(_RAIN_CACHE_DIR, exist_ok=True)
    cpath = _cache_path(start_date, end_date)

    if os.path.exists(cpath):
        cached = pd.read_csv(cpath)
        cached["datetime"] = pd.to_datetime(cached["datetime"], utc=True)
        cached = cached.set_index("datetime")
        print(f"    rain from cache: {os.path.basename(cpath)}", flush=True)
        return cached[[_RAIN_COL]]

    print(f"    fetching Open-Meteo rain {start_date.date()} to {end_date.date()} ...", flush=True)
    weather = fetch_open_meteo_weather(start_date, end_date)
    if weather is None or weather.empty or _RAIN_COL not in weather.columns:
        print(f"    fetch returned no usable rain data", flush=True)
        return None

    hourly = weather[[_RAIN_COL]].copy().resample("h").sum()
    out = hourly.reset_index()
    out.columns = ["datetime", _RAIN_COL]
    out.to_csv(cpath, index=False)
    print(f"    cached to {os.path.basename(cpath)} ({len(out)} hours)", flush=True)
    return hourly


def backfill_file(path, overwrite_existing_rain=False):
    fname = os.path.basename(path)
    df = pd.read_csv(path, sep=None, engine="python")

    time_col = _find_time_column(df)
    if time_col is None:
        print(f"  [skip] {fname}: no recognizable timestamp column", flush=True)
        return "skipped_no_time"

    if _RAIN_COL in df.columns and not overwrite_existing_rain:
        existing = pd.to_numeric(df[_RAIN_COL], errors="coerce")
        if existing.notna().any():
            print(f"  [skip] {fname}: already has populated {_RAIN_COL}", flush=True)
            return "skipped_has_rain"

    times = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    valid = times.notna()
    if not valid.any():
        print(f"  [skip] {fname}: no parseable timestamps", flush=True)
        return "skipped_bad_time"

    start_date = times[valid].min().to_pydatetime()
    end_date = times[valid].max().to_pydatetime()
    print(f"  [{fname}] {int(valid.sum())} rows, {start_date.date()} to {end_date.date()}", flush=True)

    hourly_rain = _get_rain_hourly(start_date, end_date)
    if hourly_rain is None:
        print(f"  [fail] {fname}: could not get rain, leaving file unchanged", flush=True)
        return "failed_fetch"

    rows_per_hour = _estimate_rows_per_hour(df.loc[valid, time_col])
    hourly_disagg = hourly_rain.copy()
    if rows_per_hour > 1:
        hourly_disagg[_RAIN_COL] = hourly_disagg[_RAIN_COL] / rows_per_hour

    df["_hour_key"] = times.dt.floor("h")
    hour_lookup = hourly_disagg[_RAIN_COL].to_dict()
    df[_RAIN_COL] = df["_hour_key"].map(hour_lookup)
    df = df.drop(columns=["_hour_key"])
    df[_RAIN_COL] = pd.to_numeric(df[_RAIN_COL], errors="coerce").fillna(0.0)

    filled = int((df[_RAIN_COL] > 0).sum())
    df.to_csv(path, index=False)
    print(f"  [done] {fname}: rain_mm added, {filled} rows with rain > 0, "
          f"disaggregated across {rows_per_hour} rows/hour", flush=True)
    return "backfilled"


def main(target_dirs, overwrite_existing_rain):
    print("=== backfill_rain starting ===", flush=True)
    paths = []
    for d in target_dirs:
        full = d if os.path.isabs(d) else os.path.join(_REPO_ROOT, d)
        found = sorted(glob.glob(os.path.join(full, "*.csv")))
        print(f"  scanning {full}: {len(found)} CSVs", flush=True)
        paths.extend(found)

    if not paths:
        print(f"No CSVs found in {target_dirs}. Check the --dirs paths.", flush=True)
        sys.exit(1)

    print(f"--- Rain backfill: {len(paths)} files total ---\n", flush=True)

    summary = {}
    for p in paths:
        try:
            result = backfill_file(p, overwrite_existing_rain=overwrite_existing_rain)
        except Exception as e:
            print(f"  [error] {os.path.basename(p)}: {e}", flush=True)
            result = "error"
        summary[result] = summary.get(result, 0) + 1
        print(flush=True)

    print("--- Summary ---", flush=True)
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time backfill of Open-Meteo rain onto labeled CSVs")
    parser.add_argument("--dirs", nargs="+", default=["data/anomalies", "data/normal"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(target_dirs=args.dirs, overwrite_existing_rain=args.overwrite)
    