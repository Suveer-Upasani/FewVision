# adaptive_augmentation.py
"""Unified adaptive augmentation decision engine.

Uses the suitability score to scale augmentation count:
  - suitability > 85  → up to 12 augmentations
  - suitability > 60  → up to 8 augmentations
  - otherwise         → up to 4 augmentations
"""

from typing import List
from models import QualityMetrics, ContentMetrics

# Thresholds – heuristics for demo data
BLUR_HIGH = 200.0
BRIGHTNESS_LOW = 80.0
BRIGHTNESS_HIGH = 180.0
CONTRAST_LOW = 40.0
ORIENTATION_ANGLE = 10.0
COVERAGE_LOW = 30.0


def _add(aug_list: List[str], aug: str) -> None:
    if aug not in aug_list:
        aug_list.append(aug)


def decide_augmentations(
    quality: QualityMetrics,
    content: ContentMetrics,
    suitability_score: float = 50.0,
) -> List[str]:
    """Return a list of augmentation identifiers.

    Parameters
    ----------
    quality : QualityMetrics
        Quality analysis result.
    content : ContentMetrics
        Content analysis result.
    suitability_score : float
        Combined suitability score (0–100) that scales the augmentation count.

    Returns
    -------
    list[str]
        Augmentation identifiers recognised by ``augmentations.apply_augmentation``.
    """
    augments: List[str] = []

    # ---- Brightness ----
    if quality.brightness < BRIGHTNESS_LOW:
        _add(augments, "brightness_up")
    elif quality.brightness > BRIGHTNESS_HIGH:
        _add(augments, "brightness_down")

    # ---- Contrast ----
    if quality.contrast < CONTRAST_LOW:
        _add(augments, "contrast_up")

    # ---- Blur / Sharpness ----
    if quality.blur > BLUR_HIGH:
        _add(augments, "sharpen")

    # ---- Orientation ----
    if abs(content.orientation) > ORIENTATION_ANGLE:
        _add(augments, "rotate_small")

    # ---- Always apply ----
    _add(augments, "horizontal_flip")

    # ---- Low coverage ----
    if content.object_coverage < COVERAGE_LOW:
        _add(augments, "center_crop")

    # ---- Exposure fixes ----
    if quality.underexposed_pct > 5.0:
        _add(augments, "brightness_up")
    if quality.overexposed_pct > 5.0:
        _add(augments, "brightness_down")

    # Conflict Resolution
    if "brightness_up" in augments and "brightness_down" in augments:
        if quality.brightness < 128.0:
            augments.remove("brightness_down")
        else:
            augments.remove("brightness_up")

    # ---- Scale by suitability ----
    if suitability_score > 85:
        max_augs = 12
    elif suitability_score > 60:
        max_augs = 8
    else:
        max_augs = 4

    # Pad with generic augmentations if we have room
    generic_pool = [
        "vertical_flip",
        "rotate_small",
        "gaussian_noise",
        "color_jitter",
        "random_crop",
        "elastic_transform",
        "grid_distortion",
        "perspective",
        "motion_blur",
        "channel_shuffle",
    ]
    for aug in generic_pool:
        if len(augments) >= max_augs:
            break
        # Skip contradictory blurs if sharpen is already selected
        if aug == "motion_blur" and "sharpen" in augments:
            continue
        _add(augments, aug)

    return augments[:max_augs]
