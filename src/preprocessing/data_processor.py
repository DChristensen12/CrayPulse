import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config.config import Config
from src.ingest.data_loader import _NON_FEATURE_COLUMNS


def _impute_short_gaps(df, feature_cols, limit_hours):
    """
    For each (location, feature) pair, linearly interpolate over gaps shorter
    than limit_hours. Longer gaps are left as NaN so the absence-tracking
    logic downstream can treat them as missing rather than fabricated.
    limit_area='inside' prevents extrapolation beyond first/last real obs.
    """
    timestamps = pd.DatetimeIndex(sorted(df.index.unique()))
    if len(timestamps) < 2:
        return df

    median_interval = pd.Series(timestamps).diff().median()
    if pd.isna(median_interval) or median_interval <= pd.Timedelta(0):
        print("[WARN] Could not determine sampling cadence; skipping imputation.")
        return df

    rows_per_hour = pd.Timedelta("1h") / median_interval
    limit_rows = max(1, int(limit_hours * rows_per_hour))

    # Safety: pandas' interpolation breaks if limit_rows >= per-site array length.
    min_site_rows = df.groupby("location").size().min()
    max_safe_limit = max(1, min_site_rows - 2)
    if limit_rows > max_safe_limit:
        print(
            f"[INFO] limit_rows={limit_rows} exceeds smallest site size "
            f"({min_site_rows} rows); capping to {max_safe_limit}."
        )
        limit_rows = max_safe_limit

    print(f"--- Imputing Short Sensor Gaps (limit: {limit_hours}h / {limit_rows} rows) ---")
    df = df.copy()
    total_filled = 0

    for location in df["location"].unique():
        mask = df["location"] == location
        before = int(df.loc[mask, feature_cols].isna().sum().sum())

        site_row_count = mask.sum()
        if site_row_count <= 2:
            print(f"  [{location}] only {site_row_count} rows — skipping interpolation")
            continue

        df.loc[mask, feature_cols] = (
            df.loc[mask, feature_cols]
            .interpolate(method="time", limit=limit_rows, limit_area="inside")
        )

        filled = before - int(df.loc[mask, feature_cols].isna().sum().sum())
        if filled > 0:
            print(f"  [{location}] filled {filled} missing values")
        total_filled += filled

    print(f"[INFO] Imputation complete — {total_filled} values filled across all locations.\n")
    return df


def prepare_sequences_normalized(df_featured, location_to_idx, sequence_length=Config.SEQUENCE_LENGTH):
    """
    Prepare temporal sequences with Z-score normalization, generalized to
    handle missing data consistently regardless of which sensor or site.

    Three classes of "missing":
      1. Permanently absent in this window: an (node, feature) cell has zero
         non-null values anywhere in the loaded data. e.g. footbridge fully
         offline, or Oxford lacking a sensor that's installed elsewhere.
      2. Transiently absent: an (node, feature) cell is NaN at some timesteps
         but valid at others. Includes "came online partway through" sites
         like south_fork_1, mid-window outages, calibration windows, etc.
         Anything not filled by the short-gap imputation is here.
      3. Present: the cell has a real value (or one interpolated over a short
         gap; we treat short-gap imputation as close enough to real).

    All three are zero-filled in the 3D array so the model sees a consistent
    "no signal" representation, and timesteps are validated only against
    cells that are actually present.
    """
    exclude_cols = _NON_FEATURE_COLUMNS | {"location"}
    feature_cols = [
        col for col in df_featured.select_dtypes(include=[np.number]).columns.tolist()
        if col not in exclude_cols
    ]
    print(f"[INFO] Using {len(feature_cols)} features: {', '.join(feature_cols)}")

    # Fill short gaps before anything else. Anything still NaN after this is
    # a long enough gap that we won't pretend we know its value.
    df_featured = _impute_short_gaps(df_featured, feature_cols, Config.IMPUTATION_LIMIT_HOURS)

    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)

    # ─── Detect permanently absent (node, feature) pairs ─────────────────────
    # A cell is permanently absent if it has zero non-null values for that
    # specific (location, feature) across the entire loaded window. This
    # includes both "sensor never installed" and "sensor offline for this
    # entire window" — they're indistinguishable from the data alone and
    # the model treats them identically anyway.
    print("--- Detecting Permanently Absent Sensor Channels ---")
    permanent_absent = set()  # {(node_idx, feat_idx), ...}
    for location, node_idx in location_to_idx.items():
        loc_mask = df_featured["location"] == location
        for feat_idx, feat in enumerate(feature_cols):
            if not loc_mask.any() or df_featured.loc[loc_mask, feat].isna().all():
                permanent_absent.add((node_idx, feat_idx))
                print(f"  {location}/{feat}: no data — permanently absent")
    if not permanent_absent:
        print("  (none)")
    print()

    # ─── Z-score normalization ───────────────────────────────────────────────
    # Fit only on fully-valid rows so NaN from long outages doesn't corrupt
    # the scaler's mean/std statistics.
    all_data = []
    for location in location_to_idx.keys():
        loc_data = df_featured[df_featured["location"] == location][feature_cols].values
        all_data.append(loc_data)
    all_data = np.vstack(all_data)

    valid_rows = ~np.isnan(all_data).any(axis=1)
    scaler = StandardScaler()
    scaler.fit(all_data[valid_rows])

    df_normalized = df_featured.copy()
    for location in location_to_idx.keys():
        loc_mask = df_featured["location"] == location
        if not loc_mask.any():
            print(f"  [{location}] no rows in this window — skipping normalization")
            continue
        df_normalized.loc[loc_mask, feature_cols] = scaler.transform(
            df_featured.loc[loc_mask, feature_cols].values
        )

    # ─── Build 3D array (timesteps, nodes, features) ────────────────────────
    print("--- Building 3D Array ---")
    timestamps_all = sorted(df_normalized.index.unique())
    data_3d = np.full((len(timestamps_all), num_nodes, num_features), np.nan)

    for t_idx, timestamp in enumerate(tqdm(timestamps_all, desc="Pivoting data")):
        t_data = df_normalized.loc[timestamp]
        if isinstance(t_data, pd.Series):
            t_data = t_data.to_frame().T
        for _, row in t_data.iterrows():
            node_idx = location_to_idx[row["location"]]
            data_3d[t_idx, node_idx, :] = row[feature_cols].values

    # ─── Compute transient absence mask ──────────────────────────────────────
    # transient_absent_mask[t, n, f] = True iff (node n, feature f) is NaN at
    # timestep t AND that cell is NOT permanently absent (which already gets
    # zeroed below). These are real outages — sensor down for longer than the
    # imputation limit, but the sensor isn't permanently missing.
    nan_mask = np.isnan(data_3d)
    permanent_mask = np.zeros((num_nodes, num_features), dtype=bool)
    for node_idx, feat_idx in permanent_absent:
        permanent_mask[node_idx, feat_idx] = True

    transient_absent_mask = nan_mask & ~permanent_mask[np.newaxis, :, :]
    n_transient = int(transient_absent_mask.sum())
    n_total_cells = int(nan_mask.size)
    print(
        f"[INFO] Transient absences: {n_transient:,} cells "
        f"({100 * n_transient / n_total_cells:.1f}% of all (t, node, feature) cells)\n"
    )

    # ─── Validity check ──────────────────────────────────────────────────────
    # A timestep is valid if, after zeroing out both permanent and transient
    # absences, no NaN remains. In practice this means: at least one (node,
    # feature) cell at this timestep had real data, and all NaN cells are
    # accounted for as known absences.
    def is_valid_timestep(t_idx):
        t_data = data_3d[t_idx].copy()
        # Zero permanently absent cells
        for node_idx, feat_idx in permanent_absent:
            t_data[node_idx, feat_idx] = 0
        # Zero transiently absent cells at this specific timestep
        t_data[transient_absent_mask[t_idx]] = 0
        return not np.isnan(t_data).any()

    valid_mask = np.array([is_valid_timestep(i) for i in range(len(timestamps_all))])
    print(f"[INFO] Valid timesteps: {valid_mask.sum():,} / {len(valid_mask):,}")

    # Optional safety: a timestep with EVERY node absent is not useful even
    # if it technically passes the check. Require at least one node to have
    # real (non-absent) data at each timestep.
    real_data_per_timestep = ~(
        nan_mask | permanent_mask[np.newaxis, :, :]
    )
    has_any_real_node = real_data_per_timestep.any(axis=(1, 2))
    valid_mask &= has_any_real_node
    print(f"[INFO] Valid timesteps after 'at least one real node' filter: "
          f"{valid_mask.sum():,} / {len(valid_mask):,}")

    # ─── Build sliding-window sequences ──────────────────────────────────────
    print(f"--- Creating Sequences (Length: {sequence_length}) ---")
    sequences = []
    targets = []
    sequence_timestamps = []

    for i in tqdm(range(len(timestamps_all) - sequence_length), desc="Sliding window"):
        if valid_mask[i:i+sequence_length+1].all():
            seq = data_3d[i:i+sequence_length].copy()
            target = data_3d[i+sequence_length].copy()

            # Zero permanent absences across the whole window + target
            for node_idx, feat_idx in permanent_absent:
                seq[:, node_idx, feat_idx] = 0
                target[node_idx, feat_idx] = 0

            # Zero transient absences at each step in the window
            for step_offset in range(sequence_length):
                t_idx = i + step_offset
                seq[step_offset][transient_absent_mask[t_idx]] = 0
            # Zero transient absences at the target timestep
            target[transient_absent_mask[i + sequence_length]] = 0

            sequences.append(seq)
            targets.append(target)
            sequence_timestamps.append(timestamps_all[i + sequence_length])

    print(f"[INFO] Final Sequence Count: {len(sequences):,}")
    return np.array(sequences), np.array(targets), sequence_timestamps, scaler, feature_cols