# models/feature_extraction.py
"""Feature extraction module — future scope.

This module is a stub placeholder for future implementation.

Planned implementations:
  - ResNet50 feature extraction (torchvision)
  - Vision Transformer (ViT) embeddings
  - CLIP embeddings (openai/clip)

Usage (future):
    from models.feature_extraction import FeatureExtractor
    extractor = FeatureExtractor(backbone='resnet50')
    features = extractor.extract(image_dir)

Note: Requires ``torch`` and ``torchvision`` which are listed as optional
dependencies in requirements.txt. Install with:
    pip install torch torchvision
"""

from __future__ import annotations


class FeatureExtractor:
    """Placeholder for future feature extraction implementations.

    Parameters
    ----------
    backbone : str
        Backbone architecture. One of ``"resnet50"``, ``"vit"``, ``"clip"``.
    device : str
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(self, backbone: str = "resnet50", device: str = "cpu"):
        self.backbone = backbone
        self.device = device
        raise NotImplementedError(
            "Feature extraction is not yet implemented. "
            "This module is a stub for future development."
        )

    def extract(self, image_dir: str) -> tuple:
        """Extract feature vectors from all images in *image_dir*.

        Parameters
        ----------
        image_dir : str
            Directory containing images to process.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            ``(features, labels)`` arrays.
        """
        raise NotImplementedError


def run(image_dir: str) -> None:
    """CLI entry point (future implementation)."""
    raise NotImplementedError(
        "Feature extraction pipeline not yet implemented."
    )
