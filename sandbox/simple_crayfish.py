import os
import sys
import glob
import pickle
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# This script lives in sandbox/, one level below the repo root. Put the repo
# root on the path so the config and src imports resolve the same way they do
# for main.py at the root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.config import Config
from src.utils.graph_utils import create_graph_topology
from src.preprocessing.data_processor import prepare_sequences_normalized
from src.training.trainer import train_temporal_gnn
from src.anomalies.anomaly_detector import compute_anomaly_scores
from src.models.Dusk_Crayfish import DuskCrayfish


# Raw SQL column names to internal names. The three sensor channels that are
# always populated, plus the site and timestamp keys. No weather, no battery,
# no Atlas probe. The two raw schemas we handle differ only in the time column
# name (DateTimeUTC in the anomaly CSVs, timestamp in the per-site exports),
# so both are listed and whichever is present gets mapped.
_RAW_TO_INTERNAL = {
    "Meter_Hydros21_Cond":  "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp":  "temperature",
    "DateTimeUTC":          "datetime",
    "timestamp":            "datetime",
    "site_code":            "location",
}

# Possible names for the timestamp column across the two raw schemas.
_TIME_COLUMN_CANDIDATES = ["DateTimeUTC", "timestamp"]

_THREE_FEATURES = ["conductivity", "depth", "temperature"]

_MODEL_NAME = "simple_crayfish"
_MODEL_DIR = os.path.join(_REPO_ROOT, "models")


def _load_raw_csvs(raw_dir, days):
    """
    Load every raw CSV in raw_dir, map the SQL schema to the internal column
    names, keep only the three sensor features, drop any site that isn't a
    node in the graph, and trim to the last `days` of data.

    Handles two raw schemas without a flag:
      - per-site SQL exports: site_code column present, time column is timestamp
      - anomaly window CSVs:  site_code column present, time column is DateTimeUTC
    The raw exports are also inconsistent about delimiter (some tab-separated,
    some comma), so the delimiter is sniffed per file rather than assumed.

    Site identity always comes from the site_code column. If a file somehow has
    no site_code but its filename matches a graph node, the filename stem is
    used as a fallback location.

    Returns a long-format dataframe indexed by datetime with a 'location'
    column and the three feature columns, shaped exactly the way
    prepare_sequences_normalized expects.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        print(f"ERROR: no CSV files found in {raw_dir}")
        sys.exit(1)

    graph_sites = set(Config.LOCATIONS)
    frames = []
    skipped_offgraph = set()
    skipped_unreadable = []

    for path in paths:
        fname = os.path.basename(path)

        # Raw exports are inconsistent: some are tab-separated, some comma.
        # sep=None with the python engine sniffs the delimiter per file so a
        # TSV doesn't get read as one mangled single-column frame.
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as e:
            skipped_unreadable.append(f"{fname} (parse error: {e})")
            continue

        # Find which time column this file uses
        time_col = next((c for c in _TIME_COLUMN_CANDIDATES if c in df.columns), None)
        if time_col is None:
            skipped_unreadable.append(f"{fname} (no timestamp column)")
            continue

        # Site identity: prefer the site_code column. Fall back to the filename
        # stem only if site_code is absent, which lets bare per-site files work
        # even if an export ever drops the column.
        if "site_code" in df.columns:
            df = df.rename(columns={time_col: "datetime", "site_code": "location"})
        else:
            stem = os.path.splitext(fname)[0]
            df = df.rename(columns={time_col: "datetime"})
            df["location"] = stem

        # Map the sensor columns to internal names
        df = df.rename(columns={
            k: v for k, v in _RAW_TO_INTERNAL.items()
            if k in df.columns and v in _THREE_FEATURES
        })

        present_features = [c for c in _THREE_FEATURES if c in df.columns]
        if not present_features:
            skipped_unreadable.append(f"{fname} (no Hydros21 sensor columns)")
            continue

        keep = ["datetime", "location"] + present_features
        df = df[keep].copy()

        # Coerce types
        for c in present_features:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Drop sites that aren't graph nodes. Proxying them onto a node would
        # distort the spatial structure the GCN learns, so we leave them out
        # and let the processor mark any empty graph node as permanently
        # absent, exactly like footbridge in the main pipeline.
        sites_in_file = set(df["location"].dropna().unique())
        offgraph = sites_in_file - graph_sites
        if offgraph:
            skipped_offgraph |= offgraph
            df = df[df["location"].isin(graph_sites)]

        if not df.empty:
            frames.append(df)

    if skipped_unreadable:
        print(f"[INFO] Skipped unreadable files: {skipped_unreadable}")
    if skipped_offgraph:
        print(f"[INFO] Skipped off-graph sites (not in Config.LOCATIONS): "
              f"{sorted(skipped_offgraph)}")

    if not frames:
        print("ERROR: no rows for any graph site after loading. Nothing to do.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Dedupe on (datetime, location) in case files overlap, newest wins
    combined = combined.drop_duplicates(subset=["datetime", "location"], keep="last")

    # Trim to the last `days` anchored on the latest timestamp present
    if days and days > 0:
        cutoff = combined["datetime"].max() - pd.Timedelta(days=days)
        combined = combined[combined["datetime"] >= cutoff]

    combined = combined.set_index("datetime").sort_index()
    return combined


def _report_loaded(df):
    print(f"--- Data Loading Report (simple, 3 features) ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Range: {df.index.min()} to {df.index.max()}")
    sites_present = sorted(df["location"].unique())
    missing = sorted(set(Config.LOCATIONS) - set(sites_present))
    print(f"Sites present ({len(sites_present)}): {sites_present}")
    if missing:
        print(f"Sites missing (not in this data): {missing}")
    feature_cols = [c for c in _THREE_FEATURES if c in df.columns]
    print(f"Active features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"------------------------------------------------\n")


def _compute_system_scores(errors, feature_cols):
    """
    Conductivity-only scoring, averaged across nodes. Same Model All Alert One
    strategy main.py uses, kept identical so this script's detections mean the
    same thing.
    """
    if "conductivity" in feature_cols:
        cond_idx = feature_cols.index("conductivity")
        print(f"[INFO] Scoring on conductivity error (feature index {cond_idx}), averaged across nodes")
        return np.mean(errors[:, :, cond_idx], axis=1)
    print("[WARN] 'conductivity' not in feature_cols — falling back to mean across all features.")
    return np.mean(errors, axis=(1, 2))


def main(raw_dir, days, mode):
    os.makedirs(_MODEL_DIR, exist_ok=True)
    weights_path = os.path.join(_MODEL_DIR, f"{_MODEL_NAME}_weights.pt")
    metadata_path = os.path.join(_MODEL_DIR, f"{_MODEL_NAME}_metadata.pkl")

    print("--- Simple Crayfish (3-feature pipeline, no weather) ---")
    print(f"Mode:    {mode.upper()}")
    print(f"Raw dir: {raw_dir}")
    print(f"Device:  {Config.DEVICE}\n")

    # ─── Load and preprocess ─────────────────────────────────────────────────
    df_featured = _load_raw_csvs(raw_dir, days)
    _report_loaded(df_featured)

    edge_index, _, location_to_idx = create_graph_topology()

    sequences, targets, timestamps, scaler, feature_cols = prepare_sequences_normalized(
        df_featured, location_to_idx, Config.SEQUENCE_LENGTH
    )

    if len(sequences) == 0:
        print("ERROR: No valid sequences could be built. Try a longer --days window,")
        print("       or check that the raw files share overlapping timestamps.")
        sys.exit(1)

    num_node_features = sequences.shape[3]
    model = DuskCrayfish(num_node_features=num_node_features).to(Config.DEVICE)

    # ─── Resolve mode (inference falls back to train if no weights) ──────────
    if mode == "inference":
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            model.load_state_dict(
                torch.load(weights_path, map_location=Config.DEVICE, weights_only=True)
            )
        else:
            print("No weights found — switching to fresh train.")
            mode = "train"

    # ─── Split ───────────────────────────────────────────────────────────────
    if mode == "inference":
        train_seq, train_tgt = None, None
        test_seq, test_tgt = sequences, targets
        test_timestamps = timestamps
    else:
        split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
        train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
        train_tgt, test_tgt = targets[:split_idx], targets[split_idx:]
        test_timestamps = timestamps[split_idx:]

    # ─── Train ───────────────────────────────────────────────────────────────
    trained_threshold = None
    if mode == "train":
        print("Commencing model optimization...")
        _, _, trained_threshold = train_temporal_gnn(
            model,
            train_seq,
            train_tgt,
            edge_index,
            val_sequences=test_seq,
            val_targets=test_tgt,
            feature_cols=feature_cols,
        )
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")

        with open(metadata_path, "wb") as f:
            pickle.dump({
                "scaler": scaler,
                "feature_cols": feature_cols,
                "location_to_idx": location_to_idx,
                "threshold": trained_threshold,
                "threshold_percentile": Config.THRESHOLD_PERCENTILE,
            }, f)
        print(f"Metadata saved to {metadata_path}")
    else:
        print("Skipping training. Detection only.")

    # ─── Score ───────────────────────────────────────────────────────────────
    model.eval()
    errors, _ = compute_anomaly_scores(model, test_seq, test_tgt, edge_index, Config.DEVICE)
    system_scores = _compute_system_scores(errors, feature_cols)

    # ─── Resolve threshold (trained this run, else metadata, else P99) ───────
    base_threshold = trained_threshold
    if base_threshold is None and os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            saved = pickle.load(f)
        base_threshold = saved.get("threshold")
    if base_threshold is None:
        print(f"[WARN] No trained threshold — falling back to P{Config.THRESHOLD_PERCENTILE} of this run.")
        base_threshold = float(np.percentile(system_scores, Config.THRESHOLD_PERCENTILE))
    else:
        print(f"[INFO] Using trained threshold: {base_threshold:.6f}")

    # ─── Detect (no weather, so no rain adjustment — flat threshold) ─────────
    # This is the one place we intentionally differ from main.py: there is no
    # rain_mm here, so we apply the base threshold directly rather than calling
    # detect_spills_with_rain_adjustment. With no rain data that function would
    # just run flat anyway, so the result is the same.
    spill_flags = system_scores > base_threshold
    spill_count = int(spill_flags.sum())

    print(f"\n--- Detection Summary (simple) ---")
    print(f"Total anomalies detected: {spill_count}")
    print(f"Threshold: {base_threshold:.6f}")
    print(f"----------------------------------\n")

    # Brief per-day breakdown so the output is interpretable
    if spill_count > 0:
        flagged_times = pd.to_datetime(
            [test_timestamps[i] for i in np.where(spill_flags)[0]], utc=True
        )
        from collections import Counter
        by_day = Counter(t.date() for t in flagged_times)
        print("Anomalies by day:")
        for day in sorted(by_day):
            print(f"  {day}  {by_day[day]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple 3-feature Crayfish pipeline (no weather)")
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw_data",
        help="Directory of raw per-site SQL CSVs (uuid, site_code, timestamp, Meter_Hydros21_*)",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Trim to the last N days of data (anchored on the latest timestamp)",
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "inference"],
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, days=args.days, mode=args.mode)
    