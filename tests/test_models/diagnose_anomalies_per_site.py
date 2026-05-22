"""
Break down the model's anomaly scores per (site, feature) for the most
anomalous timesteps. Scores on conductivity only, matching main.py's
"Model All, Alert One" strategy, so the ranking here reflects what the
pipeline actually flags.

The per-cell error grid still shows ALL features so we can see what else
is happening at each flagged moment, but the ranking and threshold
comparison use conductivity error averaged across nodes.

Run after `python main.py --mode inference --data-source api`.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from config.config import Config
from src.ingest.data_loader import load_and_preprocess_data
from src.utils.graph_utils import create_graph_topology
from src.preprocessing.data_processor import prepare_sequences_normalized
from src.anomalies.anomaly_detector import compute_anomaly_scores
from src.models.Dusk_Crayfish import DuskCrayfish


MODEL_NAME = "dusk_crayfish"
MODEL_DIR = "models"
TOP_N = 15


def main():
    weights_path  = os.path.join(MODEL_DIR, f"{MODEL_NAME}_weights.pt")
    metadata_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_metadata.pkl")

    if not os.path.exists(weights_path) or not os.path.exists(metadata_path):
        print(f"Missing model artifacts. Run --mode train first.")
        return

    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)
    threshold = meta["threshold"]
    print(f"Loaded threshold: {threshold:.6f}\n")

    df_featured, df_original, locations = load_and_preprocess_data(
        force_download=False, days=30, data_source="api"
    )
    edge_index, _, location_to_idx = create_graph_topology()

    sequences, targets, timestamps, scaler, feature_cols = prepare_sequences_normalized(
        df_featured, location_to_idx, Config.SEQUENCE_LENGTH
    )
    if len(sequences) == 0:
        print("No sequences built.")
        return

    idx_to_location = {v: k for k, v in location_to_idx.items()}
    n_nodes = len(location_to_idx)

    # Find the conductivity column index so we score the same way main.py does
    if "conductivity" not in feature_cols:
        print("[WARN] conductivity not in feature_cols, falling back to all-features mean")
        cond_idx = None
    else:
        cond_idx = feature_cols.index("conductivity")

    num_node_features = sequences.shape[3]
    model = DuskCrayfish(num_node_features=num_node_features).to(Config.DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    errors, _ = compute_anomaly_scores(
        model, sequences, targets, edge_index, Config.DEVICE
    )
    # errors shape: (n_sequences, n_nodes, n_features)

    # Score on conductivity only, averaged across nodes (matches main.py)
    if cond_idx is not None:
        system_scores = np.mean(errors[:, :, cond_idx], axis=1)
    else:
        system_scores = np.mean(errors, axis=(1, 2))

    timestamps = pd.to_datetime(timestamps, utc=True)
    flags = system_scores > threshold
    n_flagged = int(flags.sum())
    print(f"Total sequences scored: {len(system_scores):,}")
    print(f"Flagged (conductivity score > threshold): {n_flagged:,}\n")

    top_idx = np.argsort(system_scores)[-TOP_N:][::-1]

    print(f"Top {TOP_N} most anomalous timesteps, ranked by CONDUCTIVITY error.")
    print(f"Threshold: {threshold:.4f}\n")
    print("Reading guide:")
    print("  Rows are sites, columns are features.")
    print("  Score = conductivity error averaged across active nodes.")
    print("  The grid shows all features so you can see context, but only")
    print("  the conductivity column drives the score.")
    print("  Footbridge is all zeros (permanently absent this window).\n")

    for rank, seq_idx in enumerate(top_idx, 1):
        ts = timestamps[seq_idx]
        score = system_scores[seq_idx]
        per_cell = errors[seq_idx]  # (n_nodes, n_features)

        print(f"─── Rank {rank}: {ts}   cond_score={score:.4f}   ratio={score/threshold:.2f}× ───")
        feat_header = "  " + " ".join(f"{f[:8]:>10}" for f in feature_cols)
        print(f"  {'site':<14}{feat_header}")

        for node_idx in range(n_nodes):
            site_name = idx_to_location[node_idx]
            row_vals = per_cell[node_idx]
            row_str = " ".join(f"{v:>10.4f}" for v in row_vals)
            # Mark the conductivity cell since that's what matters
            cond_val = row_vals[cond_idx] if cond_idx is not None else row_vals.mean()
            marker = "  ← high cond" if cond_val > threshold else ""
            print(f"  {site_name:<14}  {row_str}{marker}")
        print()

    # Aggregate over events flagged by the CONDUCTIVITY score
    if flags.any():
        flagged_errors = errors[flags]
        site_cond_during_flags = flagged_errors[:, :, cond_idx].mean(axis=0) if cond_idx is not None else flagged_errors.mean(axis=(0, 2))

        print("─── Aggregate: average CONDUCTIVITY error during flagged events ─────")
        print("  (Only events flagged by conductivity score are counted)")
        print("  By site:")
        for node_idx in range(n_nodes):
            site_name = idx_to_location[node_idx]
            val = site_cond_during_flags[node_idx]
            print(f"    {site_name:<14}  {val:.4f}")

        # Temporal breakdown of conductivity-flagged events
        print("\n  Conductivity-flagged anomalies by day:")
        flagged_times = timestamps[flags]
        from collections import Counter
        by_day = Counter(t.date() for t in flagged_times)
        for day in sorted(by_day.keys()):
            count = by_day[day]
            bar = "█" * min(count, 50)
            print(f"    {day}  {count:>3}  {bar}")


if __name__ == "__main__":
    main()
    