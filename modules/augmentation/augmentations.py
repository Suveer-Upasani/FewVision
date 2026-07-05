# modules/augmentation/augmentations.py
"""Image augmentation engine for the FewVision pipeline.

Provides two public functions:

* :func:`apply_augmentation` – Apply a single named augmentation to one image.
* :func:`generate_batch`     – Generate N augmented images from a source image
                               using a policy list from :mod:`adaptive_policy`.
"""

import os
import random
import logging
import cv2
import numpy as np
import albumentations as A

logger = logging.getLogger("fewvision")

# ---------------------------------------------------------------------------
# Valid transform pool and categories
# ---------------------------------------------------------------------------

VALID_TRANSFORMS = {
    "brightness_up", "brightness_down", "contrast_up", "sharpen", "rotate_small",
    "horizontal_flip", "center_crop", "vertical_flip", "gaussian_noise", "color_jitter",
    "random_crop", "elastic_transform", "grid_distortion", "perspective", "motion_blur",
    "channel_shuffle",
}

GEOMETRIC = {
    "rotate_small", "horizontal_flip", "vertical_flip", "perspective",
    "center_crop", "random_crop", "elastic_transform", "grid_distortion",
}

NON_GEOMETRIC = {
    "brightness_up", "brightness_down", "contrast_up", "sharpen",
    "gaussian_noise", "color_jitter", "motion_blur", "channel_shuffle",
}

CONSERVATIVE_FALLBACK = ["horizontal_flip", "rotate_small"]


# ---------------------------------------------------------------------------
# Internal transform factory
# ---------------------------------------------------------------------------

def _get_transform(name: str) -> A.BasicTransform:
    """Map an augmentation identifier to an Albumentations transform.

    Parameters
    ----------
    name : str
        Augmentation identifier (as returned by :func:`~adaptive_policy.decide_augmentations`).

    Returns
    -------
    A.BasicTransform
        Configured Albumentations transform.
    """
    if name == "brightness_up":
        return A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=0, p=1.0)
    if name == "brightness_down":
        return A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=0, p=1.0)
    if name == "contrast_up":
        return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.2, 0.5), p=1.0)
    if name == "sharpen":
        return A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
    if name == "rotate_small":
        return A.Rotate(limit=10, p=1.0)
    if name == "horizontal_flip":
        return A.HorizontalFlip(p=1.0)
    if name == "center_crop":
        return A.NoOp()   # handled dynamically in generate_batch
    if name == "vertical_flip":
        return A.VerticalFlip(p=1.0)
    if name == "gaussian_noise":
        return A.GaussNoise(p=1.0)
    if name == "color_jitter":
        return A.ColorJitter(p=1.0)
    if name == "random_crop":
        return A.NoOp()   # handled dynamically in generate_batch
    if name == "elastic_transform":
        return A.ElasticTransform(p=1.0)
    if name == "grid_distortion":
        return A.GridDistortion(p=1.0)
    if name == "perspective":
        return A.Perspective(p=1.0)
    if name == "motion_blur":
        return A.MotionBlur(p=1.0)
    if name == "channel_shuffle":
        return A.ChannelShuffle(p=1.0)
    return A.NoOp()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_augmentation(image_path: str, aug_name: str) -> np.ndarray:
    """Load *image_path* and apply the augmentation identified by *aug_name*.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    aug_name : str
        Augmentation identifier (e.g. ``"brightness_up"``).

    Returns
    -------
    np.ndarray
        Augmented image in BGR format (compatible with OpenCV).

    Raises
    ------
    FileNotFoundError
        If the source image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Center crop needs dynamic dimensions based on the image size
    if aug_name == "center_crop":
        h, w = img.shape[:2]
        crop_h, crop_w = int(0.7 * h), int(0.7 * w)
        transform = A.CenterCrop(height=crop_h, width=crop_w, p=1.0)
    else:
        transform = _get_transform(aug_name)
    augmented = transform(image=img)
    return augmented["image"]


def generate_batch(
    image_path: str,
    output_dir: str,
    num_images: int = 10,
    augmentations: list | None = None,
) -> list[str]:
    """Generate a batch of augmented images from a single source image.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    output_dir : str
        Directory where augmented images will be saved.
    num_images : int
        Number of augmented images to produce (default: 10).
    augmentations : list[str] or None
        Policy list from :func:`~adaptive_policy.decide_augmentations`.
        If ``None``, a conservative generic pipeline is used as a fallback.

    Returns
    -------
    list[str]
        Paths to the generated image files.

    Raises
    ------
    FileNotFoundError
        If the source image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    generated_paths: list[str] = []

    # CASE A: Legacy fallback — no policy supplied
    if augmentations is None:
        pipeline = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.8),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])
        for i in range(num_images):
            augmented = pipeline(image=img)
            out_path = os.path.join(output_dir, f"{base_name}_aug_{i + 1}.png")
            cv2.imwrite(out_path, augmented["image"])
            generated_paths.append(out_path)
        return generated_paths

    # CASE B: Explicit policy supplied
    valid_pool = [name for name in augmentations if name in VALID_TRANSFORMS]
    if not valid_pool:
        logger.warning("No valid augmentations in policy — using conservative fallback.")
        valid_pool = CONSERVATIVE_FALLBACK

    h, w = img.shape[:2]

    for i in range(num_images):
        # Stochastically select 1 or 2 transforms per image
        num_transforms = random.choice([1, 2])
        sample_names = random.sample(valid_pool, min(num_transforms, len(valid_pool)))

        # Enforce max 1 geometric + max 1 non-geometric per image
        geom_selected = [n for n in sample_names if n in GEOMETRIC]
        non_geom_selected = [n for n in sample_names if n in NON_GEOMETRIC]

        active_names: list[str] = []
        if geom_selected:
            active_names.append(geom_selected[0])
        if non_geom_selected:
            active_names.append(non_geom_selected[0])

        aug_img = img.copy()
        for name in active_names:
            if name == "center_crop":
                crop_h, crop_w = int(0.7 * h), int(0.7 * w)
                transform = A.CenterCrop(height=crop_h, width=crop_w, p=1.0)
                aug_img = transform(image=aug_img)["image"]
            elif name == "random_crop":
                crop_h, crop_w = int(0.8 * h), int(0.8 * w)
                transform = A.RandomCrop(height=crop_h, width=crop_w, p=1.0)
                aug_img = transform(image=aug_img)["image"]
            else:
                transform = _get_transform(name)
                aug_img = transform(image=aug_img)["image"]

        out_path = os.path.join(output_dir, f"{base_name}_aug_{i + 1}.png")
        cv2.imwrite(out_path, aug_img)
        generated_paths.append(out_path)

    return generated_paths
