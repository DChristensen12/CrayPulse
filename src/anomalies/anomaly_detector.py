import numpy as np
import pandas as pd
import torch
from config.config import Config


def compute_anomaly_scores(model, sequences, targets, edge_index, device):
    """
    Compute reconstruction errors (anomaly scores) for sequences.
    Returns:
        errors: np.ndarray (absolute prediction error per sensor per feature)
        predictions: np.ndarray (raw model outputs)
    """
    model.eval()
    model = model.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        seq_tensor = torch.FloatTensor(sequences).to(device)
        target_tensor = torch.FloatTensor(targets).to(device)

        predictions = model(
            seq_tensor,
            edge_index,
            batch_size=len(sequences),
            num_nodes=sequences.shape[2]
        )

        # Compute absolute reconstruction error (MAE)
        errors = torch.abs(predictions - target_tensor)

    return errors.cpu().numpy(), predictions.cpu().numpy()


def detect_spills_with_rain_adjustment(
    system_anomaly_scores,
    timestamps,
    df_original,
    locations,
    base_threshold=None,
    threshold_percentile=Config.THRESHOLD_PERCENTILE,
    rain_window_hours=Config.RAIN_WINDOW_HOURS,
    rain_threshold_multiplier=Config.RAIN_THRESHOLD_MULTIPLIER,
    rain_amount_threshold=Config.RAIN_AMOUNT_THRESHOLD
):
    """
    Rain-aware spill detection with adaptive thresholding.

    Identifies rain-affected periods with a lookback window, applies a base
    threshold, then scales it up during rain to reduce false positives from
    natural runoff.

    If base_threshold is provided (from training metadata), it's used directly.
    Otherwise falls back to computing it from the current run's scores — this
    is the OLD buggy behavior where ~1% of any window gets flagged by definition.
    Always prefer passing a threshold computed at training time.
    """
    # Resolve the threshold. The trained value (passed in) is the right answer;
    # recomputing on the current run is a fallback that warns loudly.
    if base_threshold is None:
        print(
            f"[WARN] No trained threshold passed in — falling back to "
            f"P{threshold_percentile} of current run's scores. "
            "This is the OLD buggy behavior; retrain to get a stable threshold."
        )
        base_threshold = np.percentile(system_anomaly_scores, threshold_percentile)

    # Decide whether rain adjustment is even possible. We need a rain_mm column
    # with real values, a location column to pick a weather reference site, and
    # at least one location to read from.
    can_adjust_for_rain = (
        'rain_mm' in df_original.columns
        and not df_original['rain_mm'].isna().all()
        and 'location' in df_original.columns
        and len(locations) > 0
        and (df_original['location'] == locations[0]).any()
    )

    if not can_adjust_for_rain:
        print("[INFO] No usable rain_mm data — running without rain adjustment.")
        adjusted_thresholds = np.full(len(timestamps), base_threshold)
        spill_flags = system_anomaly_scores > adjusted_thresholds
        rain_flags = np.zeros(len(timestamps), dtype=bool)

        print(f"--- Detection Summary ---")
        print(f"Total Spills Detected: {spill_flags.sum()}")
        print(f"Rain-Affected Spills: 0 (no rain data)")
        print(f"Dry-Weather Spills: {spill_flags.sum()}")
        print(f"-------------------------\n")

        return spill_flags, rain_flags, adjusted_thresholds

    # Use the first location as the weather reference for rain data
    rain_data = df_original[df_original['location'] == locations[0]][['rain_mm']].copy()

    # Flag each timestamp where cumulative rain in the past rain_window_hours
    # exceeds the amount threshold.
    rain_flags = np.zeros(len(timestamps), dtype=bool)
    for i, ts in enumerate(timestamps):
        lookback_start = ts - pd.Timedelta(hours=rain_window_hours)
        recent_rain = rain_data[(rain_data.index >= lookback_start) & (rain_data.index <= ts)]
        if recent_rain['rain_mm'].sum() > rain_amount_threshold:
            rain_flags[i] = True

    # Apply the multiplier only where rain_flags is True
    adjusted_thresholds = np.where(
        rain_flags,
        base_threshold * rain_threshold_multiplier,
        base_threshold
    )

    # Generate spill flags
    spill_flags = system_anomaly_scores > adjusted_thresholds

    # Summary logging
    print(f"--- Detection Summary ---")
    print(f"Total Spills Detected: {spill_flags.sum()}")
    print(f"Rain-Affected Spills: {(spill_flags & rain_flags).sum()}")
    print(f"Dry-Weather Spills: {(spill_flags & ~rain_flags).sum()}")
    print(f"-------------------------\n")

    return spill_flags, rain_flags, adjusted_thresholds
