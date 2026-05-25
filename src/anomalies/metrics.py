import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Direction codes used in the signature table and in the observed changes.
# UP means the parameter rose during the event relative to baseline.
# DOWN means it fell. FLAT means no significant change.
# INDET means the table does not assign this parameter a direction for this
# pollutant, because it genuinely does not discriminate (e.g. rain temperature
# can go either way). INDET cells are skipped during matching, neither
# rewarded nor penalized, so a non-diagnostic parameter never drags down the
# score for the correct pollutant.
UP = "up"
DOWN = "down"
FLAT = "flat"
INDET = "indeterminate"

# Magnitude tag. Most signature cells are ordinary moves. A few in the table
# are called out as major (oil and sewage crashing dissolved oxygen, oil
# tanking floating conductivity). We record the tag so a future version can
# reward a strong observed move more heavily, but the current matcher only
# uses direction, treating MAJOR the same as a normal move.
NORMAL = "normal"
MAJOR = "major"

# The discriminating channels. These are the parameters that actually separate
# pollutant types in the signature table. Conductivity and temperature alone
# collapse most types together (rain, tapwater, and oil all drop conductivity;
# sewage and fertilizer both raise it), so without at least one of these the
# classifier cannot honestly name a type. Used by the diagnosis gate.
_DISCRIMINATING = ("dissolved_oxygen", "ph", "floating_conductivity")

# A best match scoring below this fraction of agreeing parameters is treated as
# a poor fit, which triggers the possible-new-type verdict. Half means fewer
# than half the comparable parameters agreed with the closest known signature.
_NEW_TYPE_SCORE_FLOOR = 0.5

# The diagnosis gate. At least this many discriminating channels must be
# populated before the classifier will commit to a named pollutant. One is the
# honest floor; below it, any named type is really a conductivity-and-temperature
# guess, so the result is reported as undetermined instead.
_MIN_DISCRIMINATING_FOR_DIAGNOSIS = 1


# The signature table, transcribed from the Water Quality team's Table 1a.
# Each pollutant maps each parameter to (direction, magnitude). This is the
# one place to edit if the table changes. The classifier reads only from here.
#
# Parameters, in the order the team defined them:
#   temperature, dissolved_oxygen, ph, conductivity, floating_conductivity
#
# Depth is intentionally left out of the per-pollutant signatures. Depth rises
# with added water volume, so it tracks how a pollutant is delivered (runoff
# during rain raises it, a concentrated point-source discharge may not) rather
# than which pollutant it is. It is a useful event confirmation signal but a
# poor type discriminator, which matches the team's own conclusion that an
# increase in depth indicates that some spill occurred without saying which.
# To bring depth in as a runoff-vs-point-source hint later, uncomment the
# depth entries below and add "depth" to _PARAMETERS.
_SIGNATURES = {
    "rain": {
        "temperature":           (INDET, NORMAL),   # depends on rain and air temp
        "dissolved_oxygen":      (UP,    NORMAL),   # increased turbulence
        "ph":                    (DOWN,  NORMAL),   # rain is slightly acidic
        "conductivity":          (DOWN,  NORMAL),   # more volume dilutes solutes
        "floating_conductivity": (DOWN,  NORMAL),   # more volume dilutes solutes
        # "depth":               (UP,    NORMAL),   # rain adds volume
    },
    "tapwater": {
        "temperature":           (DOWN,  NORMAL),   # Berkeley tap is ~13C, cooler
        "dissolved_oxygen":      (DOWN,  NORMAL),   # tap has less DO, has chloramine
        "ph":                    (UP,    NORMAL),   # Berkeley tap is ~9.4
        "conductivity":          (DOWN,  NORMAL),   # more volume dilutes solutes
        "floating_conductivity": (DOWN,  NORMAL),   # more volume dilutes solutes
        # "depth":               (UP,    NORMAL),   # adds volume, but may be small
    },
    "oil": {
        "temperature":           (UP,    NORMAL),   # reduces evaporative cooling
        "dissolved_oxygen":      (DOWN,  MAJOR),    # blocks gas exchange, kills plants
        "ph":                    (DOWN,  NORMAL),   # CO2 from decomposition
        "conductivity":          (DOWN,  NORMAL),   # oil is a poor conductor
        "floating_conductivity": (DOWN,  MAJOR),    # floats, hits surface hardest
        # "depth":               (INDET, NORMAL),   # delivery dependent
    },
    "sewage": {
        "temperature":           (UP,    NORMAL),   # sewage warmer than creek
        "dissolved_oxygen":      (DOWN,  MAJOR),    # decomposer bacteria consume DO
        "ph":                    (INDET, NORMAL),   # cleaners raise it, ammonia lowers it
        "conductivity":          (UP,    NORMAL),   # chlorides, phosphates, nitrates
        "floating_conductivity": (UP,    NORMAL),   # chlorides, phosphates, nitrates
        # "depth":               (UP,    NORMAL),   # adds volume, often point source
    },
    "fertilizer": {
        "temperature":           (FLAT,  NORMAL),   # fertilizer itself does not move temp
        "dissolved_oxygen":      (DOWN,  NORMAL),   # algae die-off, bacteria consume DO
        "ph":                    (INDET, NORMAL),   # algae raise it, ammoniacal runoff lowers it
        "conductivity":          (UP,    NORMAL),   # chlorides, phosphates, nitrates
        "floating_conductivity": (UP,    NORMAL),   # chlorides, phosphates, nitrates
        # "depth":               (UP,    NORMAL),   # arrives as runoff
    },
}

_PARAMETERS = [
    "temperature",
    "dissolved_oxygen",
    "ph",
    "conductivity",
    "floating_conductivity",
    # "depth",
]

_FEATURE_ALIASES = {
    "temperature": "temperature",
    "conductivity": "conductivity",
    "dissolved_oxygen": "dissolved_oxygen",
    "AtlasSci_DO": "dissolved_oxygen",
    "ph": "ph",
    "pH": "ph",
    "AtlasSci_pH": "ph",
    "floating_conductivity": "floating_conductivity",
    "AtlasSci_FloatCond": "floating_conductivity",
    "depth": "depth",
}

_CHANGE_THRESHOLD_STD = 1.0


def _observed_direction(baseline_vals, event_vals):
    """
    Decide whether a parameter went UP, DOWN, or FLAT from baseline to event.

    Uses the baseline's own variability as the yardstick. The event mean has to
    differ from the baseline mean by more than _CHANGE_THRESHOLD_STD baseline
    standard deviations to count as a move. This makes the threshold adapt to
    each parameter's natural noise instead of using one fixed number for very
    different scales like temperature and conductivity.

    Returns one of UP, DOWN, FLAT, or None if there is not enough data to tell.
    """
    b = pd.to_numeric(pd.Series(baseline_vals), errors="coerce").dropna()
    e = pd.to_numeric(pd.Series(event_vals), errors="coerce").dropna()
    if len(b) < 2 or len(e) < 1:
        return None

    baseline_mean = b.mean()
    baseline_std = b.std()
    event_mean = e.mean()

    if pd.isna(baseline_std) or baseline_std == 0:
        if baseline_mean == 0:
            return FLAT if event_mean == 0 else (UP if event_mean > 0 else DOWN)
        rel = (event_mean - baseline_mean) / abs(baseline_mean)
        if abs(rel) < 0.05:
            return FLAT
        return UP if rel > 0 else DOWN

    shift = (event_mean - baseline_mean) / baseline_std
    if abs(shift) < _CHANGE_THRESHOLD_STD:
        return FLAT
    return UP if shift > 0 else DOWN


def _resolve_parameter(column_name):
    """Map a dataframe column to a signature parameter name, or None."""
    return _FEATURE_ALIASES.get(column_name)


def classify_event(baseline_df, event_df):
    """
    Classify a detected anomaly by matching observed parameter changes against
    the pollutant signature table, and decide on one of three verdicts:
    a named pollutant, undetermined (not enough discriminating data to name a
    type), or possible new type (the data is good enough to judge but matches
    no known signature well).

    baseline_df and event_df cover the period just before the event and the
    event window itself. Whichever signature parameters are present and
    populated get used, the rest are skipped, so this runs today on the few
    populated sensors and sharpens once dissolved oxygen, pH, and floating
    conductivity report, with no code change.

    Verdict logic:
      - If no discriminating channel (DO, pH, floating conductivity) is
        populated, the result is "undetermined": the system cannot honestly
        separate pollutant types on conductivity and temperature alone, so it
        declines to name one rather than guessing. It still reports the leading
        candidate as a hint, clearly marked as not a diagnosis.
      - If discriminating data is available but even the best match scores below
        the floor, or the top score is a tie across several pollutants with no
        separation, the result is "possible_new_type": the event is real and
        judgeable but does not look like anything in the table, which is exactly
        the case worth surfacing for a human.
      - Otherwise the result is the best-matching named pollutant.

    Returns a dict with the verdict, the named type (or None), the ranked
    matches, and the diagnostic context.
    """
    observed = {}
    for col in event_df.columns:
        param = _resolve_parameter(col)
        if param is None or param not in _PARAMETERS:
            continue
        if col not in baseline_df.columns:
            continue
        direction = _observed_direction(baseline_df[col].values, event_df[col].values)
        if direction is not None:
            observed[param] = direction

    available_params = sorted(observed.keys())
    discriminating_available = [p for p in available_params if p in _DISCRIMINATING]

    results = []
    for pollutant, signature in _SIGNATURES.items():
        comparable = 0
        agreements = 0
        per_param = {}
        for param in _PARAMETERS:
            sig_dir, _sig_mag = signature.get(param, (INDET, NORMAL))
            if sig_dir == INDET:
                per_param[param] = "skipped (not diagnostic)"
                continue
            if param not in observed:
                per_param[param] = "skipped (no data)"
                continue
            comparable += 1
            if observed[param] == sig_dir:
                agreements += 1
                per_param[param] = f"match ({observed[param]})"
            else:
                per_param[param] = f"differ (saw {observed[param]}, expected {sig_dir})"

        score = (agreements / comparable) if comparable > 0 else 0.0
        results.append({
            "pollutant": pollutant,
            "score": score,
            "agreements": agreements,
            "comparable": comparable,
            "per_param": per_param,
        })

    results.sort(key=lambda r: (r["score"], r["comparable"]), reverse=True)

    top_score = results[0]["score"]
    tied = [r["pollutant"] for r in results if r["score"] == top_score and top_score > 0]

    # ─── Decide the verdict ──────────────────────────────────────────────────
    if len(discriminating_available) < _MIN_DISCRIMINATING_FOR_DIAGNOSIS:
        # Not enough channels to honestly name a type. Decline to diagnose.
        verdict = "undetermined"
        named_type = None
        verdict_note = (
            "Cannot name a pollutant type. None of the discriminating channels "
            "(dissolved oxygen, pH, floating conductivity) are populated, and "
            "conductivity with temperature alone cannot separate the candidates. "
            "Leading candidate below is a hint only, not a diagnosis."
        )
    elif top_score < _NEW_TYPE_SCORE_FLOOR:
        # Enough data to judge, but nothing fits. This is the case to surface.
        verdict = "possible_new_type"
        named_type = None
        verdict_note = (
            f"Does not match any known spill signature well (best score "
            f"{top_score:.2f}, below {_NEW_TYPE_SCORE_FLOOR:.2f}). This may be a "
            f"new or unclassified type of event and is worth a closer look."
        )
    elif len(tied) > 1:
        # Several pollutants fit equally and the data did not separate them.
        verdict = "possible_new_type"
        named_type = None
        verdict_note = (
            f"Top match is an unresolved tie between {', '.join(tied)} (all at "
            f"score {top_score:.2f}). The available channels did not separate "
            f"them, so no single type can be named; may also be a new type."
        )
    else:
        verdict = "diagnosed"
        named_type = results[0]["pollutant"]
        verdict_note = (
            f"Best match is {named_type} (score {top_score:.2f}, "
            f"{results[0]['agreements']}/{results[0]['comparable']} parameters agreed)."
        )

    # Confidence note about channel coverage, independent of the verdict.
    if len(discriminating_available) == 0:
        confidence = "none (no discriminating channels populated)"
    elif len(discriminating_available) == 1:
        confidence = f"moderate (one discriminating channel: {discriminating_available[0]})"
    else:
        confidence = f"good ({len(discriminating_available)} discriminating channels)"

    return {
        "verdict": verdict,
        "named_type": named_type,
        "verdict_note": verdict_note,
        "ranked": results,
        "available_parameters": available_params,
        "discriminating_available": discriminating_available,
        "observed_directions": observed,
        "confidence": confidence,
        "top_candidates": tied,
    }


def format_classification(result):
    """Turn a classify_event result into a readable report."""
    lines = []
    lines.append("--- Spill Type Classification ---")
    lines.append(f"Verdict: {result['verdict'].upper().replace('_', ' ')}")
    if result["named_type"]:
        lines.append(f"Diagnosed type: {result['named_type']}")
    lines.append(result["verdict_note"])
    lines.append("")
    lines.append(f"Parameters available: {result['available_parameters'] or 'none'}")
    lines.append("Observed changes: " + (
        ", ".join(f"{p}={d}" for p, d in result["observed_directions"].items())
        if result["observed_directions"] else "none"
    ))
    lines.append(f"Channel confidence: {result['confidence']}")
    lines.append("")
    lines.append("Ranked matches:")
    for r in result["ranked"]:
        if r["comparable"] == 0:
            lines.append(f"  {r['pollutant']:12s} no comparable parameters")
            continue
        lines.append(
            f"  {r['pollutant']:12s} score {r['score']:.2f} "
            f"({r['agreements']}/{r['comparable']} parameters agreed)"
        )
    lines.append("")
    lines.append("Detail for leading candidate:")
    top = result["ranked"][0]
    for param, verdict in top["per_param"].items():
        lines.append(f"  {param:22s} {verdict}")
    lines.append("---------------------------------")
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Case A: only conductivity and temperature, as in real data today. Should
    # come back UNDETERMINED, because no discriminating channel is present.
    print("CASE A: conductivity up, temperature flat, no discriminating channels")
    baseline = pd.DataFrame({
        "conductivity": rng.normal(300, 5, 50),
        "temperature":  rng.normal(15, 0.3, 50),
    })
    event = pd.DataFrame({
        "conductivity": rng.normal(360, 5, 20),
        "temperature":  rng.normal(15.1, 0.3, 20),
    })
    print(format_classification(classify_event(baseline, event)))
    print()

    # Case B: discriminating channels present and a clean sewage signature.
    # Conductivity up, temperature up, DO crashes. Should DIAGNOSE sewage.
    print("CASE B: sewage signature with DO present")
    baseline = pd.DataFrame({
        "conductivity":     rng.normal(300, 5, 50),
        "temperature":      rng.normal(15, 0.3, 50),
        "dissolved_oxygen": rng.normal(8, 0.2, 50),
    })
    event = pd.DataFrame({
        "conductivity":     rng.normal(360, 5, 20),  # up
        "temperature":      rng.normal(16.5, 0.3, 20), # up
        "dissolved_oxygen": rng.normal(4, 0.2, 20),   # major down
    })
    print(format_classification(classify_event(baseline, event)))
    print()

    # Case C: discriminating channel present but the pattern fits nothing.
    # DO up, conductivity up, temperature up: no single signature matches well.
    # Should come back POSSIBLE NEW TYPE.
    print("CASE C: contradictory pattern with a discriminating channel")
    baseline = pd.DataFrame({
        "conductivity":     rng.normal(300, 5, 50),
        "temperature":      rng.normal(15, 0.3, 50),
        "dissolved_oxygen": rng.normal(8, 0.2, 50),
    })
    event = pd.DataFrame({
        "conductivity":     rng.normal(360, 5, 20),  # up
        "temperature":      rng.normal(16.5, 0.3, 20), # up
        "dissolved_oxygen": rng.normal(11, 0.2, 20),  # up, which no up-conductivity type expects
    })
    print(format_classification(classify_event(baseline, event)))
    