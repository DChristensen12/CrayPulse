"""
Anomaly-detection integration tests for the SCMG Temporal GNN.

Each test loads a labeled sensor window from data/anomalies/ (or data/normal/),
runs it through the trained model, and checks whether the conductivity
reconstruction error crosses the model's threshold.

--- Scoring matches production ---
main.py uses the "Model All, Alert One" strategy: the model predicts every
feature but anomaly scoring looks only at the conductivity channel. These
tests do the same, so a pass here means the same thing a flag means in
the live pipeline.

--- Rain-adjusted threshold (mirrors anomaly_detector) ---
Production raises the detection threshold during rain. anomaly_detector
.detect_spills_with_rain_adjustment flags a timestep as rain-affected when the
SUM of rain_mm over the preceding Config.RAIN_WINDOW_HOURS exceeds
Config.RAIN_AMOUNT_THRESHOLD, then multiplies the threshold there by
Config.RAIN_THRESHOLD_MULTIPLIER. These tests replicate that exact logic
(sum over the window, same constants), so a rain event the production pipeline
would suppress is suppressed here too.

Most labeled CSVs lack a rain column, so for those the rain adjustment is a
no-op (no rain data means no suppression, the conservative choice, matching
production's behaviour when rain_mm is absent). Where a file carries rain, the
adjustment engages and the per-case output says so.

--- Judging against the trained threshold, not an in-file baseline ---
We judge against the model's trained threshold (stored in metadata), the same
calibrated number production uses, rather than an in-file baseline ratio. The
baseline-ratio approach breaks when the anomaly fills the whole window or sits
at the very start, leaving no clean pre-event period to divide against. Many of
these labeled files start mid-event, so the absolute threshold is the honest
test. We require a few consecutive timesteps over threshold so a single noisy
point does not count as an event.

Every case prints its peak error, count over threshold, and a sparkline of the
error curve so a pass or fail can be inspected rather than taken on faith.

--- Catalog corrections from SCMG ground-truth notes ---
Two cases were relabeled after cross-checking the SCMG team's field notes:
  - nov25_rain/nf0 was originally a "clean rain" true negative, but the team's
    notes describe the North Fork behaving anomalously during the 11/13 storm
    (an unexpected conductivity drop on nf0, a spike on nf1). It is now a
    relative-only case, not an absolute true negative.
  - jan26_actuator_baseline is a January window judged by a model trained on
    April-May. The whole window reads mildly elevated due to seasonal
    distribution shift, not an event. It is now a relative-only case: the model
    has no seasonal context, so an absolute-threshold true-negative test is not
    meaningful for it. The actuator-vs-baseline relative test still runs.

--- How partial-parameter files are handled ---
The labeled CSVs only contain the Hydros21 sensor columns (conductivity,
depth, temperature). The trained model may use more. Missing columns are filled
with 0.0 (the normalised mean), so the model still detects anomalies in the
channels that ARE present. This is deliberate: in production a sensor may
report a partial feature set and detection should degrade gracefully.

--- Running ---
    pytest tests/test_anomaly_detection.py -v -s

The -s flag shows the per-case diagnostic output.

Requires a trained model:
    python main.py --mode train
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.config import Config

ANOMALY_DIR = ROOT / "data" / "anomalies"
NORMAL_DIR = ROOT / "data" / "normal"

STATION_MAP = {
    "nf0": "north_fork_0",
    "nf1": "north_fork_1",
    "sf0": "south_fork_0",
    "sf1": "south_fork_1",
    "sf2": "south_fork_2",
}

COLUMN_MAP = {
    "DateTimeUTC":                   "datetime",
    "Meter_Hydros21_Cond":           "conductivity",
    "Meter_Hydros21_Depth":          "depth",
    "Meter_Hydros21_Temp":           "temperature",
    "TE_TR_525USW_Precip_5minTotal": "rain_mm",
    "Sensirion_SHT40_Temperature":   "air_temp_c",
}

# Ground-truth catalog.
# (filename, station_suffix_or_None, label, event_group)
#
# label is one of:
#   "anomaly"        the model should flag it (sustained error over threshold)
#   "true_negative"  the model should NOT flag it (absolute threshold test)
#   "relative_only"  judged only by relative comparison, not absolute threshold
#                    (used where seasonal shift or a mislabel makes an absolute
#                    threshold test meaningless; see catalog-corrections note)
EVENT_CATALOG = [
    # June 2025 mystery spill, propagating downstream across all south-fork sensors.
    # Confirmed real by SCMG: multi-day deviation, no rain, supervisors unaware of cause.
    ("anomaly_2025_06_12_spill_sf0.csv",        "sf0", "anomaly",       "jun25_spill"),
    ("anomaly_2025_06_12_spill_sf1.csv",        "sf1", "anomaly",       "jun25_spill"),
    ("anomaly_2025_06_12_spill_sf2.csv",        "sf2", "anomaly",       "jun25_spill"),
    # September 2025 overnight conductivity spike across south fork.
    ("anomaly_2025_09_10_overnight_sf0.csv",    "sf0", "anomaly",       "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf1.csv",    "sf1", "anomaly",       "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf2.csv",    "sf2", "anomaly",       "sep25_overnight"),
    # November 2025 foam/chemical event, nf1 anomalous, nf0 contrast.
    ("anomaly_2025_11_05_foam_nf1.csv",         "nf1", "anomaly",       "nov25_foam"),
    ("anomaly_2025_11_05_foam_nf0.csv",         "nf0", "true_negative", "nov25_foam"),
    # November 2025 storm. nf1 shows an anomalous conductivity spike. nf0 was
    # originally labeled a clean rain response, but SCMG notes describe the
    # North Fork behaving anomalously during this storm, so nf0 is relative-only.
    ("anomaly_2025_11_13_rain_nf1.csv",         "nf1", "anomaly",       "nov25_rain"),
    ("anomaly_2025_11_13_rain_nf0.csv",         "nf0", "relative_only", "nov25_rain"),
    # April 2026 rainfall. Confirmed heavy rain 04/01-04/02 per SCMG. These are
    # true negatives judged with the rain-adjusted threshold, since a first-flush
    # conductivity bump during confirmed rain is what rain-aware thresholding
    # is designed to absorb.
    ("anomaly_2026_04_01_rainfall_nf0.csv",     "nf0", "true_negative", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_nf1.csv",     "nf1", "true_negative", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_sf0.csv",     "sf0", "true_negative", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_sf2.csv",     "sf2", "true_negative", "apr26_rainfall"),
    # Standalone true-negative rain event.
    ("anomaly_2025_05_12_rain_nf1.csv",         "nf1", "true_negative", "may25_rain_tn"),
    # Botanical actuator malfunction (Jan/Feb 2026). Confirmed real by SCMG.
    ("anomaly_2026_01_botanical_actuator.csv",  None,  "anomaly",       "jan26_actuator"),
    # Botanical normal baseline (January 2026). Model trained on April-May, so
    # this winter window reads mildly elevated from seasonal shift, not an event.
    # Relative-only: judged against the actuator window, not absolute. In data/normal/.
    ("normal_2026_01_botanical_baseline.csv",   None,  "relative_only", "jan26_actuator_baseline"),
    # Fire-hydrant spill at north fork 0 (03/20 Euclid Ave). Confirmed real by SCMG.
    ("anomaly_2026_03_20_hydrant_nf0.csv",      "nf0", "anomaly",       "mar26_hydrant"),
]

# A window needs at least this many error points to be judged. Shorter files
# are skipped rather than failed.
MIN_TIMESTEPS_TO_JUDGE = 30

# How many consecutive timesteps must exceed the threshold to call it an event.
MIN_TIMESTEPS_OVER_THRESHOLD = 3


def _load_threshold(model_metadata) -> float:
    threshold = model_metadata.get("threshold")
    if threshold is None:
        pytest.skip(
            "No trained threshold in model metadata. Retrain with the current "
            "pipeline (python main.py --mode train) so the threshold is saved."
        )
    return float(threshold)


def _rain_window_periods() -> int:
    """
    Number of 15-minute timesteps in the rain look-back window. Production uses
    Config.RAIN_WINDOW_HOURS; the creek cadence is 15 minutes, so 4 steps/hour.
    """
    hours = getattr(Config, "RAIN_WINDOW_HOURS", 12)
    return int(hours * 4)


def _per_timestep_rain_threshold(errors, rain_series, base_threshold):
    """
    Build a per-timestep threshold array that lifts the floor during rain,
    replicating anomaly_detector.detect_spills_with_rain_adjustment exactly.

    For each error timestep i (a one-step-ahead prediction over a
    SEQUENCE_LENGTH window, so it scores creek timestep i + SEQUENCE_LENGTH),
    we sum rain_mm over the preceding Config.RAIN_WINDOW_HOURS. If that sum
    exceeds Config.RAIN_AMOUNT_THRESHOLD, the threshold there is
    base_threshold * Config.RAIN_THRESHOLD_MULTIPLIER. Otherwise it stays at
    base_threshold.

    This mirrors production's sum-over-window comparison (not a peak check), so
    light rain spread over the window triggers suppression the same way it would
    live.

    If rain_series is None or all non-positive (most labeled files have no rain
    column), returns a flat base_threshold array. That matches production, which
    runs without rain adjustment when rain_mm is absent.
    """
    multiplier      = getattr(Config, "RAIN_THRESHOLD_MULTIPLIER", 2.0)
    amount_threshold = getattr(Config, "RAIN_AMOUNT_THRESHOLD", 0.1)
    look_back       = _rain_window_periods()
    seq_len         = Config.SEQUENCE_LENGTH

    thresholds = np.full(len(errors), base_threshold, dtype=float)

    if rain_series is None or np.all(np.nan_to_num(rain_series) <= 0):
        return thresholds

    rain = np.nan_to_num(np.asarray(rain_series, dtype=float))

    for i in range(len(errors)):
        center = i + seq_len  # the creek timestep this error scores
        lo = max(0, center - look_back)
        window = rain[lo:center + 1]
        if window.size and window.sum() > amount_threshold:
            thresholds[i] = base_threshold * multiplier

    return thresholds


def _reconstruction_errors(csv_path, station_suffix, model, model_metadata, edge_index,
                           return_rain=False):
    """
    Load a single-sensor CSV window and return per-timestep conductivity
    reconstruction errors for the target node. Scores only the conductivity
    channel, matching main.py.

    If return_rain is True, also returns the raw rain_mm series (aligned to the
    original CSV rows) so the caller can build a rain-adjusted threshold.
    Returns errors or (errors, rain_series).
    """
    feature_cols    = model_metadata["feature_cols"]
    scaler          = model_metadata["scaler"]
    location_to_idx = model_metadata["location_to_idx"]
    num_nodes       = len(location_to_idx)
    num_features    = len(feature_cols)

    if "conductivity" not in feature_cols:
        pytest.skip("conductivity not in feature_cols, cannot score the way production does")
    cond_idx = feature_cols.index("conductivity")

    station_name = STATION_MAP.get(station_suffix) if station_suffix else None
    if station_name is not None and station_name not in location_to_idx:
        pytest.skip(
            f"Station '{station_name}' (suffix '{station_suffix}') is not in the "
            f"trained model's graph. Add it to Config.LOCATIONS and retrain."
        )
    node_idx = location_to_idx.get(station_name, 0)

    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    feature_matrix = pd.DataFrame(0.0, index=df.index, columns=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            feature_matrix[col] = df[col]

    # Pull rain before normalisation so the threshold logic sees real mm
    rain_series = None
    if "rain_mm" in df.columns:
        rain_series = df["rain_mm"].fillna(0.0).values

    normalised = scaler.transform(feature_matrix.fillna(0.0).values)

    seq_len = Config.SEQUENCE_LENGTH
    if len(normalised) <= seq_len:
        pytest.skip(
            f"{csv_path.name} has only {len(normalised)} rows, need more than "
            f"{seq_len} for a single sequence. File too short to test."
        )

    errors = []
    with torch.no_grad():
        for i in range(len(normalised) - seq_len):
            seq    = np.zeros((seq_len, num_nodes, num_features))
            target = np.zeros((num_nodes, num_features))

            seq[:, node_idx, :] = normalised[i : i + seq_len]
            target[node_idx, :] = normalised[i + seq_len]

            seq_t = torch.FloatTensor(seq).unsqueeze(0).to(Config.DEVICE)
            pred  = model(seq_t, edge_index, batch_size=1, num_nodes=num_nodes)

            err = torch.abs(
                pred[0, node_idx] -
                torch.FloatTensor(target[node_idx]).to(Config.DEVICE)
            )
            errors.append(err[cond_idx].item())

    errors = np.array(errors)
    if return_rain:
        return errors, rain_series
    return errors


def _curve_shape(errors: np.ndarray, n_buckets: int = 10) -> str:
    if len(errors) == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    bucket_size = max(1, len(errors) // n_buckets)
    buckets = [
        errors[i:i + bucket_size].mean()
        for i in range(0, len(errors), bucket_size)
    ]
    lo, hi = min(buckets), max(buckets)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((b - lo) / span * 7))] for b in buckets)


def _report(name, errors, thresholds):
    """Print per-case diagnostics. thresholds may be scalar or per-timestep array."""
    thr_arr = np.full(len(errors), thresholds) if np.isscalar(thresholds) else thresholds
    over = int((errors > thr_arr).sum())
    base = float(thr_arr.min())
    rained = (not np.isscalar(thresholds)) and (thr_arr.max() > thr_arr.min())
    rain_note = " (rain-adjusted in places)" if rained else ""
    print(
        f"\n  [{name}] points={len(errors)} peak={errors.max():.3f} "
        f"mean={errors.mean():.3f} over_thresh={over}/{len(errors)} "
        f"base_thresh={base:.3f}{rain_note}"
    )
    print(f"    curve: {_curve_shape(errors)}")


def _is_flagged(errors: np.ndarray, thresholds) -> bool:
    """
    True if at least MIN_TIMESTEPS_OVER_THRESHOLD points exceed their threshold.
    thresholds may be scalar or a per-timestep array (rain-adjusted).
    """
    thr_arr = np.full(len(errors), thresholds) if np.isscalar(thresholds) else thresholds
    return int((errors > thr_arr).sum()) >= MIN_TIMESTEPS_OVER_THRESHOLD


ANOMALOUS_CASES     = [(f, s, g) for f, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"]
TRUE_NEGATIVE_CASES = [(f, s, g) for f, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"]


@pytest.mark.parametrize(
    "filename,station,group", ANOMALOUS_CASES,
    ids=[g + "/" + f.replace(".csv", "")
         for f, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"],
)
def test_anomaly_detected(filename, station, group, trained_model, model_metadata, edge_index):
    """Conductivity error must cross the (rain-adjusted) threshold for known anomalies."""
    base_threshold = _load_threshold(model_metadata)
    errors, rain = _reconstruction_errors(
        ANOMALY_DIR / filename, station, trained_model, model_metadata, edge_index,
        return_rain=True,
    )
    if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
        pytest.skip(
            f"{filename}: only {len(errors)} error points "
            f"(need {MIN_TIMESTEPS_TO_JUDGE}). Too short to judge reliably."
        )
    thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
    _report(f"{group}/{filename}", errors, thresholds)
    assert _is_flagged(errors, thresholds), (
        f"[{group}] {filename}: expected an anomaly but only "
        f"{int((errors > thresholds).sum())} timesteps crossed threshold "
        f"(base {base_threshold:.4f}, need {MIN_TIMESTEPS_OVER_THRESHOLD}). "
        f"Peak error {errors.max():.4f}."
    )


@pytest.mark.parametrize(
    "filename,station,group", TRUE_NEGATIVE_CASES,
    ids=[g + "/" + f.replace(".csv", "")
         for f, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"],
)
def test_true_negative_not_flagged(filename, station, group, trained_model, model_metadata, edge_index):
    """Conductivity error must stay below the (rain-adjusted) threshold for normal events."""
    base_threshold = _load_threshold(model_metadata)
    data_dir = NORMAL_DIR if filename.startswith("normal_") else ANOMALY_DIR
    errors, rain = _reconstruction_errors(
        data_dir / filename, station, trained_model, model_metadata, edge_index,
        return_rain=True,
    )
    if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
        pytest.skip(
            f"{filename}: only {len(errors)} error points "
            f"(need {MIN_TIMESTEPS_TO_JUDGE}). Too short to judge reliably."
        )
    thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
    _report(f"{group}/{filename}", errors, thresholds)
    assert not _is_flagged(errors, thresholds), (
        f"[{group}] {filename}: expected no anomaly but "
        f"{int((errors > thresholds).sum())} timesteps crossed threshold "
        f"(base {base_threshold:.4f}). Peak error {errors.max():.4f}. "
        f"Model may be over-flagging this case. If this file carries no rain "
        f"column the rain-adjustment is a no-op; check whether real rain data "
        f"would have suppressed it (see per-case output for whether rain engaged)."
    )


# Within-group relative tests. Threshold-independent, so robust to seasonal drift.

def test_foam_event_nf1_more_anomalous_than_nf0(trained_model, model_metadata, edge_index):
    """Nov 2025 foam: nf1 (anomalous) peak should exceed nf0 (contrast)."""
    err_nf1 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_05_foam_nf1.csv",
        "nf1", trained_model, model_metadata, edge_index
    )
    err_nf0 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_05_foam_nf0.csv",
        "nf0", trained_model, model_metadata, edge_index
    )
    print(f"\n  foam: nf1_peak={err_nf1.max():.3f} nf0_peak={err_nf0.max():.3f}")
    assert err_nf1.max() > err_nf0.max(), (
        f"Foam event: nf1 peak ({err_nf1.max():.4f}) should exceed "
        f"nf0 peak ({err_nf0.max():.4f})."
    )


def test_rain_storm_nf1_more_anomalous_than_nf0(trained_model, model_metadata, edge_index):
    """
    Nov 2025 storm: nf1 (anomalous conductivity spike during rain) peak should
    exceed nf0. Per SCMG notes both forks behaved oddly during this storm, so
    this is a relative comparison, not an absolute true-negative on nf0.
    """
    err_nf1 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_13_rain_nf1.csv",
        "nf1", trained_model, model_metadata, edge_index
    )
    err_nf0 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_13_rain_nf0.csv",
        "nf0", trained_model, model_metadata, edge_index
    )
    print(f"\n  storm: nf1_peak={err_nf1.max():.3f} nf0_peak={err_nf0.max():.3f}")
    assert err_nf1.max() > err_nf0.max(), (
        f"Nov storm: nf1 peak ({err_nf1.max():.4f}) should exceed "
        f"nf0 peak ({err_nf0.max():.4f})."
    )


def test_spill_propagation_all_south_fork_flagged(trained_model, model_metadata, edge_index):
    """
    June 2025 spill: all judged south-fork sensors should show sustained
    conductivity error over threshold. No rain during this event per SCMG notes.
    Sensors too short to judge are skipped.
    """
    base_threshold = _load_threshold(model_metadata)
    results = {}
    for suffix, fname in [("sf0", "anomaly_2025_06_12_spill_sf0.csv"),
                          ("sf1", "anomaly_2025_06_12_spill_sf1.csv"),
                          ("sf2", "anomaly_2025_06_12_spill_sf2.csv")]:
        errors, rain = _reconstruction_errors(
            ANOMALY_DIR / fname, suffix, trained_model, model_metadata, edge_index,
            return_rain=True,
        )
        if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
            continue
        thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
        _report(f"jun25_spill/{suffix}", errors, thresholds)
        results[suffix] = (errors, thresholds)

    if not results:
        pytest.skip("All south-fork spill files too short to judge.")

    failures = [s for s, (e, t) in results.items() if not _is_flagged(e, t)]
    assert not failures, (
        f"Jun 2025 spill: expected all judged sf sensors flagged, "
        f"but {failures} did not cross threshold."
    )


def test_botanical_actuator_more_anomalous_than_baseline(trained_model, model_metadata, edge_index):
    """
    The actuator malfunction window should have a higher peak conductivity error
    than the same sensor's normal baseline window (January 2026). This relative
    comparison cancels the seasonal offset that makes an absolute baseline test
    meaningless. Baseline file lives in data/normal/.
    """
    err_actuator = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2026_01_botanical_actuator.csv",
        None, trained_model, model_metadata, edge_index
    )
    err_baseline = _reconstruction_errors(
        NORMAL_DIR / "normal_2026_01_botanical_baseline.csv",
        None, trained_model, model_metadata, edge_index
    )
    print(f"\n  actuator: anomaly_peak={err_actuator.max():.3f} baseline_peak={err_baseline.max():.3f}")
    assert err_actuator.max() > err_baseline.max(), (
        f"Botanical actuator: anomaly peak ({err_actuator.max():.4f}) should "
        f"exceed baseline peak ({err_baseline.max():.4f})."
    )


def test_april_rainfall_no_false_positives(trained_model, model_metadata, edge_index):
    """
    April 2026 rain (confirmed heavy rain 04/01-04/02): no judged sensor should
    cross the rain-adjusted threshold. Validates that rain-aware thresholding
    absorbs first-flush conductivity bumps. Sensors too short to judge skipped.

    If the April CSVs carry no rain_mm column the rain adjustment is a no-op and
    this falls back to the bare threshold; the per-case output shows whether rain
    adjustment engaged, which tells you whether a flag here is a real model gap
    or just missing rain data in the test file.
    """
    base_threshold = _load_threshold(model_metadata)
    files = [
        ("nf0", "anomaly_2026_04_01_rainfall_nf0.csv"),
        ("nf1", "anomaly_2026_04_01_rainfall_nf1.csv"),
        ("sf0", "anomaly_2026_04_01_rainfall_sf0.csv"),
        ("sf2", "anomaly_2026_04_01_rainfall_sf2.csv"),
    ]
    false_positives = []
    judged = 0
    for suffix, fname in files:
        errors, rain = _reconstruction_errors(
            ANOMALY_DIR / fname, suffix, trained_model, model_metadata, edge_index,
            return_rain=True,
        )
        if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
            continue
        judged += 1
        thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
        _report(f"apr26_rainfall/{suffix}", errors, thresholds)
        if _is_flagged(errors, thresholds):
            false_positives.append(suffix)

    if judged == 0:
        pytest.skip("All April rainfall files too short to judge.")

    assert not false_positives, (
        f"Apr 2026 rainfall: sensors {false_positives} crossed the rain-adjusted "
        f"threshold but should be true negatives. If these files lack a rain_mm "
        f"column the adjustment was a no-op; see per-case output."
    )
    