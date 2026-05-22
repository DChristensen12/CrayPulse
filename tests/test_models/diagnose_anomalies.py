"""
Break down where the flagged anomalies actually are in time, so we can
tell whether the model is seeing real signal in recent data or just
mis-fitting some part of the training window.

Run after `python main.py --mode inference --data-source api`.
Reads the cached data, loads the saved model + metadata, recomputes the
anomaly scores, and prints a temporal breakdown.
"""

import os
import pickle
from collections import Counter

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


def main():
    weights_path  = os.path.join(MODEL_DIR, f"{MODEL_NAME}_weights.pt")
    metadata_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_metadata.pkl")

    if not os.path.exists(weights_path) or not os.path.exists(metadata_path):
        print(f"Missing model artifacts. Run --mode train first.")
        return

    # Load metadata to get the saved threshold
    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)
    threshold = meta["threshold"]
    print(f"Loaded threshold: {threshold:.6f} "
          f"(P{meta.get('threshold_percentile', 99)} from training)\n")

    # Load whatever's in the cache (don't refetch)
    df_featured, df_original, locations = load_and_preprocess_data(
        force_download=False, days=30, data_source="api"
    )

    edge_index, _, location_to_idx = create_graph_topology()

    sequences, targets, timestamps, scaler, feature_cols = prepare_sequences_normalized(
        df_featured, location_to_idx, Config.SEQUENCE_LENGTH
    )

    if len(sequences) == 0:
        print("No sequences built — cache may be empty.")
        return

    # Load the model
    num_node_features = sequences.shape[3]
    model = DuskCrayfish(num_node_features=num_node_features).to(Config.DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    # Compute anomaly scores using the same logic as main.py
    errors, _ = compute_anomaly_scores(
        model, sequences, targets, edge_index, Config.DEVICE
    )
    system_scores = np.mean(errors, axis=(1, 2))
    flags = system_scores > threshold

    n_total = len(flags)
    n_flagged = int(flags.sum())
    print(f"Total sequences scored: {n_total:,}")
    print(f"Flagged as anomalous:   {n_flagged:,}  ({100 * n_flagged / n_total:.2f}%)\n")

    if n_flagged == 0:
        return

    # ─── Temporal breakdown ──────────────────────────────────────────────────
    # The target timestamp of each sequence is the actual "event time" we're
    # scoring on. Group flagged sequences by date.
    timestamps = pd.to_datetime(timestamps, utc=True)
    flagged_times = timestamps[flags]

    print("─── Anomalies by day ───────────────────────────────────────────")
    by_day = Counter(t.date() for t in flagged_times)
    all_days = sorted(set(t.date() for t in timestamps))
    for day in all_days:
        count = by_day.get(day, 0)
        bar = "█" * count
        marker = " ←" if count > 0 else ""
        print(f"  {day}  {count:>3}  {bar}{marker}")

    # ─── Anomaly score distribution ──────────────────────────────────────────
    print("\n─── Anomaly score distribution ─────────────────────────────────")
    pcts = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    for p in pcts:
        val = float(np.percentile(system_scores, p))
        marker = " ← THRESHOLD" if val > threshold and (p == 99 or all(
            np.percentile(system_scores, q) <= threshold for q in pcts if q < p
        )) else ""
        print(f"  P{p:>5}: {val:.6f}{marker}")
    print(f"  ───")
    print(f"  Trained threshold: {threshold:.6f}")
    print(f"  Mean error:        {float(system_scores.mean()):.6f}")
    print(f"  Std error:         {float(system_scores.std()):.6f}")
    print(f"  Max error:         {float(system_scores.max()):.6f}")

    # ─── Top 10 most anomalous timesteps ─────────────────────────────────────
    print("\n─── Top 10 most anomalous timesteps ─────────────────────────────")
    top_idx = np.argsort(system_scores)[-10:][::-1]
    print(f"  {'rank':>4}  {'timestamp':<28}  {'score':>10}  ratio")
    for rank, idx in enumerate(top_idx, 1):
        ts = timestamps[idx]
        score = system_scores[idx]
        ratio = score / threshold
        print(f"  {rank:>4}  {str(ts):<28}  {score:>10.6f}  {ratio:.2f}×")

    # ─── Per-day vs total context ────────────────────────────────────────────
    print("\n─── Train period vs recent period split ─────────────────────────")
    # Approximate train period as the first 80% of the timestamp range
    # (matches the chronological 80/20 split main.py uses)
    n_train = int(len(timestamps) * 0.8)
    if n_train < len(timestamps):
        train_end = timestamps[n_train - 1]
        train_flags = flags[:n_train]
        val_flags = flags[n_train:]
        print(f"  Train portion (≤ {train_end}):")
        print(f"    {int(train_flags.sum())} flags out of {len(train_flags):,} "
              f"({100 * train_flags.sum() / len(train_flags):.2f}%)")
        print(f"  Validation/recent portion (> {train_end}):")
        print(f"    {int(val_flags.sum())} flags out of {len(val_flags):,} "
              f"({100 * val_flags.sum() / len(val_flags):.2f}%)")


if __name__ == "__main__":
    main()
