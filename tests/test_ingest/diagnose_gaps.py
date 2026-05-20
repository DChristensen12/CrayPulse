"""
Inspect where NaN values are concentrated in the cached training data.
Helps decide whether long gaps need a longer imputation limit, a smarter
validity gate, or model-level NaN masking.
"""

import pandas as pd

from config.config import Config

df = pd.read_csv(Config.DATA_FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

print(f"Total rows: {len(df):,}")
print(f"Sites: {sorted(df['location'].unique())}")
print(f"Range: {df['datetime'].min()} → {df['datetime'].max()}\n")

feature_cols = [c for c in df.columns if c not in ("datetime", "location")]

# Per-column non-null rate by site
print("Non-null rate per (site, feature):")
for site in sorted(df["location"].unique()):
    site_df = df[df["location"] == site]
    print(f"\n  {site} ({len(site_df):,} rows):")
    for col in feature_cols:
        non_null = site_df[col].notna().sum()
        pct = 100 * non_null / len(site_df) if len(site_df) > 0 else 0
        print(f"    {col:25s}  {non_null:>6,} / {len(site_df):>6,}  ({pct:5.1f}%)")

# Find longest contiguous gaps per (site, feature)
print("\n" + "=" * 70)
print("Longest gap per (site, feature):")
print("=" * 70)
for site in sorted(df["location"].unique()):
    site_df = df[df["location"] == site].sort_values("datetime").reset_index(drop=True)
    print(f"\n  {site}:")
    for col in feature_cols:
        # Find runs of NaN
        is_nan = site_df[col].isna()
        if not is_nan.any():
            print(f"    {col:25s}  no gaps")
            continue
        # Compute gap durations
        gap_groups = (is_nan != is_nan.shift()).cumsum()
        gap_lengths = is_nan.groupby(gap_groups).sum()
        gap_lengths = gap_lengths[gap_lengths > 0]
        if gap_lengths.empty:
            continue
        max_gap_rows = gap_lengths.max()
        max_gap_hours = max_gap_rows * 0.25  # 15-min cadence
        print(f"    {col:25s}  max gap: {max_gap_rows:>4} rows ({max_gap_hours:>5.1f}h), "
              f"{len(gap_lengths)} total gaps")
        