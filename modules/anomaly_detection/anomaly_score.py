# modules/anomaly_detection/anomaly_score.py
"""Anomaly scoring utilities for FewVision.

This module converts nearest-neighbour distances (returned by the similarity
engine) into human-readable anomaly scores, confidence values, and labels.

Design notes
------------
* This module performs **no image I/O** and has **no side effects**.
* All threshold values are sourced from :mod:`config`; nothing is hardcoded.
* The scoring function is intentionally kept pure so it can be unit-tested
  without any filesystem or model dependencies.
* Labels are designed to be extensible — adding a new tier only requires
  updating ``config.ANOMALY_THRESHOLDS``.

Anomaly score interpretation
-----------------------------
The raw anomaly score is derived from the minimum nearest-neighbour distance
(i.e. the distance to the single closest reference embedding).  A lower
distance → lower anomaly score → more normal.

The score is normalised to ``[0, 100]`` using a sigmoid-like mapping so that
very small distances produce near-zero scores and very large distances
saturate toward 100.

Label thresholds (configured in ``config.ANOMALY_THRESHOLDS``)
---------------------------------------------------------------
``normal_max``
    Distances ≤ this value → label ``"Normal"``
``suspicious_max``
    Distances ≤ this value (and > ``normal_max``) → label ``"Suspicious"``
Anything beyond ``suspicious_max`` → label ``"Anomalous"``

Public API
----------
compute_anomaly_score(distances, nearest_index) → dict
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger("fewvision.anomaly_detection.score")


# ---------------------------------------------------------------------------
# Config helper (avoids re-importing at module level for testability)
# ---------------------------------------------------------------------------

def _get_thresholds() -> dict:
    """Return anomaly thresholds from config with safe fallbacks.

    Returns
    -------
    dict
        ``{"normal_max": float, "suspicious_max": float}``
    """
    try:
        import config as _cfg
        thresholds = getattr(_cfg, "ANOMALY_THRESHOLDS", {})
    except (ImportError, AttributeError):
        thresholds = {}

    return {
        "normal_max": float(thresholds.get("normal_max", 0.20)),
        "suspicious_max": float(thresholds.get("suspicious_max", 0.50)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_score(distance: float) -> float:
    """Map a raw distance to a normalised anomaly score in ``[0, 100]``.

    Uses the formula: ``score = 100 * (1 - exp(-alpha * distance))``
    where ``alpha`` controls how quickly the score rises.  At distance=0,
    score=0 (perfectly normal).  Score asymptotically approaches 100.

    Parameters
    ----------
    distance : float
        The minimum nearest-neighbour distance (non-negative).

    Returns
    -------
    float
        Anomaly score in ``[0.0, 100.0]``.
    """
    alpha = 5.0   # controls the rise-rate; tunable via future config extension
    return round(100.0 * (1.0 - math.exp(-alpha * max(0.0, distance))), 2)


def _assign_label(min_distance: float, thresholds: dict) -> str:
    """Assign a human-readable anomaly label based on the minimum distance.

    Parameters
    ----------
    min_distance : float
        Distance to the nearest reference embedding.
    thresholds : dict
        ``{"normal_max": float, "suspicious_max": float}``

    Returns
    -------
    str
        One of ``"Normal"``, ``"Suspicious"``, or ``"Anomalous"``.
    """
    if min_distance <= thresholds["normal_max"]:
        return "Normal"
    if min_distance <= thresholds["suspicious_max"]:
        return "Suspicious"
    return "Anomalous"


def _compute_confidence(min_distance: float, thresholds: dict, label: str) -> float:
    """Estimate confidence in the assigned label.

    Confidence is measured as how far the distance is from the nearest
    decision boundary, normalised to ``[0.0, 1.0]``.

    * **Normal** — distance is below ``normal_max``; confidence grows as
      distance → 0.
    * **Suspicious** — distance is between the two thresholds; confidence
      peaks at the midpoint.
    * **Anomalous** — distance exceeds ``suspicious_max``; confidence
      grows as distance increases beyond the boundary.

    Parameters
    ----------
    min_distance : float
        Distance to the nearest reference embedding.
    thresholds : dict
        ``{"normal_max": float, "suspicious_max": float}``
    label : str
        The assigned label.

    Returns
    -------
    float
        Confidence estimate in ``[0.0, 1.0]``.
    """
    normal_max = thresholds["normal_max"]
    suspicious_max = thresholds["suspicious_max"]

    if label == "Normal":
        # 1.0 at distance=0, falls toward 0.5 at the boundary
        ratio = min_distance / normal_max if normal_max > 0 else 0.0
        return round(max(0.5, 1.0 - 0.5 * ratio), 4)

    if label == "Suspicious":
        span = suspicious_max - normal_max
        if span <= 0:
            return 0.5
        # Confidence peaks at midpoint, lower near both boundaries
        midpoint = normal_max + span / 2.0
        deviation = abs(min_distance - midpoint) / (span / 2.0)
        return round(max(0.5, 1.0 - 0.5 * deviation), 4)

    # Anomalous
    excess = min_distance - suspicious_max
    # Confidence rises logarithmically as distance increases beyond boundary
    if suspicious_max > 0:
        confidence = min(1.0, 0.5 + 0.5 * math.log1p(excess / suspicious_max))
    else:
        confidence = min(1.0, 0.5 + 0.5 * math.log1p(excess))
    return round(confidence, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_anomaly_score(
    distances: Sequence[float],
    nearest_index: int,
) -> dict:
    """Convert nearest-neighbour distances into a structured anomaly result.

    This function is the single entry-point for scoring.  It is **pure** —
    no I/O, no model calls — making it trivially testable.

    Parameters
    ----------
    distances : Sequence[float]
        Distances to the ``k`` nearest reference embeddings, sorted ascending
        (index 0 = nearest).  Must contain at least one value.
    nearest_index : int
        Index into the memory bank of the nearest reference embedding.

    Returns
    -------
    dict
        ``{
            "score": float,            # 0–100 normalised anomaly score
            "confidence": float,       # 0–1 label confidence
            "label": str,              # "Normal" | "Suspicious" | "Anomalous"
            "nearest_reference": int,  # memory bank index of nearest neighbour
            "min_distance": float,     # raw distance to nearest neighbour
            "mean_distance": float,    # mean over all k distances
            "thresholds_used": dict,   # thresholds applied (for traceability)
        }``

    Raises
    ------
    ValueError
        If *distances* is empty.
    """
    distances_arr = np.asarray(distances, dtype=np.float64)
    if distances_arr.size == 0:
        raise ValueError("distances must contain at least one value.")

    thresholds = _get_thresholds()
    min_dist = float(distances_arr[0])   # already sorted ascending by caller
    mean_dist = float(np.mean(distances_arr))

    score = _normalise_score(min_dist)
    label = _assign_label(min_dist, thresholds)
    confidence = _compute_confidence(min_dist, thresholds, label)

    result = {
        "score": score,
        "confidence": confidence,
        "label": label,
        "nearest_reference": int(nearest_index),
        "min_distance": round(min_dist, 6),
        "mean_distance": round(mean_dist, 6),
        "thresholds_used": thresholds,
    }

    logger.debug(
        "Anomaly score computed — label=%s score=%.2f confidence=%.4f min_dist=%.6f",
        label, score, confidence, min_dist,
    )

    return result
