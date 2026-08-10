# modules/anomaly_detection/temporal_aggregation.py
"""Temporal aggregation utilities for multi-frame (video) anomaly inspection.

Combines per-frame anomaly scores produced by the DINOv2 + PatchCore pipeline
into a single video-level score and computes temporal persistence metrics.

All functions are **pure** — no I/O, no model calls — making them trivially
testable and reusable across different inspection front-ends.

Supported aggregation methods
-----------------------------
mean    : np.mean(scores)
median  : np.median(scores)
max     : np.max(scores)
p95     : np.percentile(scores, 95)
top_k   : mean of the K highest scores

Public API
----------
aggregate_frame_scores(scores, method, k) → float
compute_temporal_persistence(frame_scores, threshold) → dict
make_video_status(video_score, anomalous_count, min_anomalous_frames,
                  anomaly_persistence, temporal_threshold) → str
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger("fewvision.anomaly_detection.temporal_aggregation")

# ---------------------------------------------------------------------------
# Supported aggregation method names
# ---------------------------------------------------------------------------
SUPPORTED_METHODS = frozenset({"mean", "median", "max", "p95", "top_k"})


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_frame_scores(
    scores: List[float],
    method: str = "top_k",
    k: int = 5,
) -> float:
    """Combine a list of per-frame anomaly scores into a single video score.

    Parameters
    ----------
    scores : List[float]
        Anomaly scores for each processed frame, each in [0, 100].
    method : str
        One of ``"mean"``, ``"median"``, ``"max"``, ``"p95"``, ``"top_k"``.
    k : int
        Number of top scores to average when ``method == "top_k"``.

    Returns
    -------
    float
        The aggregated video-level anomaly score in [0, 100].

    Raises
    ------
    ValueError
        If *scores* is empty or *method* is not recognised.
    """
    if not scores:
        raise ValueError("scores must contain at least one value.")

    arr = np.asarray(scores, dtype=np.float64)
    method = method.strip().lower()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Supported: {sorted(SUPPORTED_METHODS)}"
        )

    if method == "mean":
        result = float(np.mean(arr))
    elif method == "median":
        result = float(np.median(arr))
    elif method == "max":
        result = float(np.max(arr))
    elif method == "p95":
        result = float(np.percentile(arr, 95))
    elif method == "top_k":
        k_clamped = max(1, min(k, len(arr)))
        top_scores = np.sort(arr)[::-1][:k_clamped]
        result = float(np.mean(top_scores))
    else:
        # Unreachable — guarded by SUPPORTED_METHODS check above.
        result = float(np.mean(arr))

    logger.debug(
        "Aggregated %d frame scores via '%s' (k=%d) → %.4f",
        len(scores), method, k, result,
    )
    return round(result, 4)


# ---------------------------------------------------------------------------
# Temporal persistence
# ---------------------------------------------------------------------------

def compute_temporal_persistence(
    frame_scores: List[float],
    threshold: float,
) -> dict:
    """Compute how persistently an anomaly appears across processed frames.

    Parameters
    ----------
    frame_scores : List[float]
        Per-frame anomaly scores (not normalised to [0,100] — these are the
        raw ``max_patch_score`` values from PatchCore, in [0, ~2]).
    threshold : float
        Score threshold above which a frame is considered anomalous
        (should match ``config.PATCH_THRESHOLD``).

    Returns
    -------
    dict
        ``{
            "anomalous_frames": int,
            "total_frames": int,
            "anomaly_persistence": float,   # ratio in [0.0, 1.0]
        }``
    """
    total = len(frame_scores)
    if total == 0:
        return {
            "anomalous_frames": 0,
            "total_frames": 0,
            "anomaly_persistence": 0.0,
        }

    anomalous = sum(1 for s in frame_scores if s >= threshold)
    persistence = anomalous / total

    return {
        "anomalous_frames": anomalous,
        "total_frames": total,
        "anomaly_persistence": round(persistence, 6),
    }


# ---------------------------------------------------------------------------
# Final video status decision
# ---------------------------------------------------------------------------

def make_video_status(
    video_score: float,
    anomalous_count: int,
    min_anomalous_frames: int,
    anomaly_persistence: float,
    temporal_threshold: float,
    patch_threshold_score: float,
) -> str:
    """Determine the final NORMAL / SUSPICIOUS / ANOMALY status of a video.

    Decision logic (all conditions must be true for ANOMALY):
    1. video_score >= patch_threshold_score * 100   (score-based)
    2. anomalous_count >= min_anomalous_frames       (count-based)
    3. anomaly_persistence >= temporal_threshold     (persistence-based)

    If score indicates anomaly but the frame count or persistence is too low,
    the result is SUSPICIOUS.

    Parameters
    ----------
    video_score : float
        Aggregated video-level anomaly score [0, 100].
    anomalous_count : int
        Number of frames individually flagged as anomalous.
    min_anomalous_frames : int
        Minimum number of anomalous frames required.
    anomaly_persistence : float
        Ratio of anomalous to total processed frames.
    temporal_threshold : float
        Minimum persistence ratio required for ANOMALY.
    patch_threshold_score : float
        The raw patch distance threshold from config (PATCH_THRESHOLD).
        Converted to [0,100] scale internally.

    Returns
    -------
    str
        One of ``"NORMAL"``, ``"SUSPICIOUS"``, or ``"ANOMALY"``.
    """
    # Convert patch threshold to same [0,100] scale as video_score.
    # video_score is derived from max_patch_score which lives in distance units;
    # a rough calibration: score > 50 correlates with above-threshold distance.
    score_threshold_100 = min(100.0, patch_threshold_score * 100.0)

    score_flag = video_score >= score_threshold_100
    count_flag = anomalous_count >= min_anomalous_frames
    persistence_flag = anomaly_persistence >= temporal_threshold

    if score_flag and count_flag and persistence_flag:
        return "ANOMALY"
    if score_flag or count_flag:
        return "SUSPICIOUS"
    return "NORMAL"
