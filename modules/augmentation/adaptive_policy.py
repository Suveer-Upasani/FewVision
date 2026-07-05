# modules/augmentation/adaptive_policy.py
"""Adaptive augmentation decision engine for the FewVision pipeline.

Analyses :class:`~modules.utils.models.QualityMetrics` and
:class:`~modules.utils.models.ContentMetrics` for a single image and returns
an ordered list of augmentation identifiers that should be applied during
batch generation.

The number of augmentations is scaled by the image's suitability score:

  - suitability > 85  → up to 12 augmentations
  - suitability > 60  → up to 8 augmentations
  - otherwise         → up to 4 augmentations
"""

from typing import List

from modules.utils.models import QualityMetrics, ContentMetrics

# Thresholds – heuristics calibrated for industrial inspection images
BLUR_HIGH = 200.0
BRIGHTNESS_LOW = 80.0
BRIGHTNESS_HIGH = 180.0
CONTRAST_LOW = 40.0
ORIENTATION_ANGLE = 10.0
COVERAGE_LOW = 30.0


def _add(aug_list: List[str], aug: str) -> None:
    """Append *aug* to *aug_list* only if it is not already present."""
    if aug not in aug_list:
        aug_list.append(aug)


def decide_augmentations(
    quality: QualityMetrics,
    content: ContentMetrics,
    suitability_score: float = 50.0,
) -> List[str]:
    """Return a list of augmentation identifiers for a single image.

    This function is a **pure policy engine** — it makes decisions only.
    It does **not** modify images. The returned identifiers are consumed by
    :func:`~modules.augmentation.augmentations.generate_batch`.

    Parameters
    ----------
    quality : QualityMetrics
        Quality analysis result for the image.
    content : ContentMetrics
        Content analysis result for the image.
    suitability_score : float
        Combined suitability score (0–100) used to scale augmentation count.

    Returns
    -------
    list[str]
        Ordered list of augmentation identifiers, e.g.::

            ["brightness_up", "contrast_up", "horizontal_flip"]
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

    # Conflict resolution: remove contradictory brightness adjustments
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

    # Pad with generic augmentations if capacity remains
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
        # Avoid contradictory: do not add motion_blur if sharpen is selected
        if aug == "motion_blur" and "sharpen" in augments:
            continue
        _add(augments, aug)

    return augments[:max_augs]
