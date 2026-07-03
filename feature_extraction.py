# feature_extraction.py
"""Member 2 Task: Feature Extraction using a pre-trained ResNet50 backbone.

This module takes every image from the augmented_dataset/ directory,
passes them through a pre-trained ResNet50 model (with the final classification
layer removed), and extracts 2048-dimensional feature embeddings.

These embeddings are the mathematical "fingerprints" of each product image
and serve as the direct input for the Few-Shot Learning model (Member 3).

Output:
  - features.npy     -> Shape [N, 2048] float32 array of feature vectors.
  - labels.npy       -> Shape [N,] array of filenames matching each vector row.
"""

import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
AUGMENTED_DIR = "augmented_dataset"
OUTPUT_FEATURES = "features.npy"
OUTPUT_LABELS   = "labels.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization — required because ResNet50 was trained on ImageNet
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),           # ResNet50 standard input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],          # ImageNet mean
        std=[0.229, 0.224, 0.225]            # ImageNet std deviation
    ),
])


# -----------------------------------------------------------------------
# Build backbone (ResNet50 minus the final classification layer)
# -----------------------------------------------------------------------
_BACKBONE_CACHE = None

def build_backbone() -> nn.Module:
    """Load pre-trained ResNet50 and strip the final FC layer.
    
    By removing the final fully-connected classification head, the model
    now outputs a 2048-dim feature vector (embedding) instead of class logits.
    This technique is called 'feature extraction' and is the foundation
    of transfer learning.
    """
    global _BACKBONE_CACHE
    if _BACKBONE_CACHE is None:
        print("Loading ResNet50 backbone...")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Strip the final classification layer — output is now 2048-dim embeddings
        backbone.fc = nn.Identity()
        backbone.eval()
        backbone.to(DEVICE)
        _BACKBONE_CACHE = backbone
    return _BACKBONE_CACHE


# -----------------------------------------------------------------------
# Extract features from a directory of images
# -----------------------------------------------------------------------
def extract_features(image_dir: str) -> tuple:
    """Process all images in image_dir and return (feature_matrix, labels).

    Parameters
    ----------
    image_dir : str
        Path to the directory containing augmented product images.

    Returns
    -------
    features : np.ndarray, shape [N, 2048]
    labels   : list of str (filenames)
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in valid_ext
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    print(f"Extracting features from {len(image_paths)} images...")  

    backbone = build_backbone()

    all_features = []
    all_labels   = []

    with torch.no_grad():  # No gradient computation needed — only inference
        for i, img_path in enumerate(image_paths, 1):
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB
                tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # Add batch dim

                # Forward pass through backbone — output shape: [1, 2048]
                embedding = backbone(tensor)
                all_features.append(embedding.squeeze(0).cpu().numpy())
                all_labels.append(img_path.name)

                if i % 10 == 0 or i == len(image_paths):
                    print(f"  {i}/{len(image_paths)} images processed...")  

            except Exception as e:
                print(f"  [WARN] Skipping {img_path.name}: {e}")

    features_matrix = np.array(all_features, dtype=np.float32)
    return features_matrix, all_labels


# -----------------------------------------------------------------------
# Save output and run summary
# -----------------------------------------------------------------------
def run(image_dir: str = AUGMENTED_DIR):
    """Main entry point for Member 2's feature extraction task."""
    features, labels = extract_features(image_dir)

    np.save(OUTPUT_FEATURES, features)
    np.save(OUTPUT_LABELS, np.array(labels))

    print(f"Feature extraction complete — shape: {features.shape} saved to {OUTPUT_FEATURES}")

    return features, labels


if __name__ == "__main__":
    import sys
    img_dir = sys.argv[1] if len(sys.argv) > 1 else AUGMENTED_DIR
    run(img_dir)
