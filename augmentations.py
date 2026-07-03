# augmentations.py
"""Utility functions for applying image augmentations using Albumentations.

The module provides a single public function :func:`apply_augmentation` that
receives the path to an image and an augmentation identifier (as returned
by ``adaptive_augmentation.decide_augmentations``) and returns the augmented
image as a NumPy array.
"""

import cv2
import numpy as np
import albumentations as A


def _get_transform(name: str):
    """Map an augmentation identifier to an Albumentations transform.

    Supported identifiers correspond to those produced by
    ``adaptive_augmentation.decide_augmentations``.
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
        # Handled specially in ``apply_augmentation`` because it needs the image size.
        return A.NoOp()
    return A.NoOp()


def apply_augmentation(image_path: str, aug_name: str) -> np.ndarray:
    """Load ``image_path`` and apply the augmentation identified by ``aug_name``.

    Parameters
    ----------
    image_path: str
        Path to the source image.
    aug_name: str
        Identifier of the augmentation (e.g. ``"brightness_up"``).

    Returns
    -------
    np.ndarray
        The augmented image in BGR format (compatible with OpenCV).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Center crop needs dynamic dimensions based on the image size.
    if aug_name == "center_crop":
        h, w = img.shape[:2]
        crop_h, crop_w = int(0.7 * h), int(0.7 * w)
        transform = A.CenterCrop(height=crop_h, width=crop_w, p=1.0)
    else:
        transform = _get_transform(aug_name)
    augmented = transform(image=img)
    return augmented["image"]


def generate_batch(image_path: str, output_dir: str, num_images: int = 10) -> list:
    """Generate a batch of augmented images to simulate different angles and conditions.
    
    This uses Albumentations to apply random spatial and lighting transformations.
    """
    import os
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Base spatial augmentation pipeline for varying camera angles
    pipeline = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.8),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2), # Occasional slight focus loss
    ])
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    generated_paths = []
    
    for i in range(num_images):
        augmented = pipeline(image=img)
        aug_img = augmented["image"]
        
        out_filename = f"{base_name}_aug_{i+1}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        cv2.imwrite(out_path, aug_img)
        generated_paths.append(out_path)
        
    return generated_paths
