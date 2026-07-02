# adaptive_augmentation.py
"""Decision engine for adaptive augmentation.

Given quality and content analysis results for an image, this module selects a list of augmentation identifiers that should be applied.
"""

from typing import List, Dict

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

def decide_augmentations(quality: Dict, content: Dict) -> List[str]:
    """Return a list of augmentation identifiers.

    Rules (example implementation):
    * brightness_up / brightness_down depending on image brightness
    * contrast_up when contrast is low
    * sharpen when blur is high
    * rotate_small when orientation deviates >10°
    * horizontal_flip always added
    * center_crop when object coverage is low
    """
    augments: List[str] = []
    bright = quality.get("brightness", 0)
    if bright < BRIGHTNESS_LOW:
        _add(augments, "brightness_up")
    elif bright > BRIGHTNESS_HIGH:
        _add(augments, "brightness_down")
    contrast = quality.get("contrast", 0)
    if contrast < CONTRAST_LOW:
        _add(augments, "contrast_up")
    blur = quality.get("blur", 0)
    if blur > BLUR_HIGH:
        _add(augments, "sharpen")
    orientation = content.get("orientation", 0.0)
    if abs(orientation) > ORIENTATION_ANGLE:
        _add(augments, "rotate_small")
    _add(augments, "horizontal_flip")
    coverage = content.get("object_coverage", 100.0)
    if coverage < COVERAGE_LOW:
        _add(augments, "center_crop")
    return augments
