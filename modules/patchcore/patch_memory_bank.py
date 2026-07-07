# modules/patchcore/patch_memory_bank.py
"""Memory bank to store, save, and load normalized local reference patch embeddings."""

from __future__ import annotations

import json
import logging
import os
import numpy as np
import cv2

import config
from modules.patchcore.patch_extractor import PatchExtractor
from modules.utils.file_utils import ensure_dir

logger = logging.getLogger("fewvision.patchcore.memory_bank")


class PatchMemoryBank:
    """Memory bank for PatchCore industrial anomaly localization.

    Stores, saves, loads, and searches reference patch embeddings and metadata.
    """

    def __init__(
        self,
        extractor: PatchExtractor | None = None,
        patch_size: int = 14,
    ) -> None:
        self.extractor = extractor or PatchExtractor(patch_size=patch_size)
        self.patch_size = patch_size
        self._embeddings: np.ndarray | None = None  # (total_patches, dim), L2-normalised
        self._metadata: list[dict] = []  # list of patch metadata dicts
        self._is_built = False

    @property
    def is_built(self) -> bool:
        """True if the memory bank is built or loaded."""
        return self._is_built

    def build(self, session_id: str) -> PatchMemoryBank:
        """Construct the patch memory bank from augmented images.

        Parameters
        ----------
        session_id : str
            Session identifier containing the augmented images.

        Returns
        -------
        PatchMemoryBank
            self
        """
        logger.info("Building PatchCore Memory Bank for session %s...", session_id)

        aug_dir = os.path.join(config.AUGMENTED_FOLDER, session_id)
        if not os.path.isdir(aug_dir):
            raise FileNotFoundError(f"Augmented images directory not found: {aug_dir}")

        # Collect augmented images using the pipeline's helper
        from modules.pipeline.pipeline import _collect_augmented_images
        aug_paths = _collect_augmented_images(aug_dir)
        if not aug_paths:
            raise ValueError(f"No augmented images found in {aug_dir}")

        logger.info("Loading %d augmented images for patch extraction...", len(aug_paths))
        images = []
        valid_paths = []
        for p in aug_paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
                valid_paths.append(p)
            else:
                logger.warning("Could not read image: %s", p)

        if not images:
            raise ValueError("No valid images could be loaded.")

        # Batch extract patch tokens: (N, 196, dim)
        logger.info("Extracting patch embeddings...")
        patch_embeddings = self.extractor.extract_batch(
            images,
            batch_size=config.EXTRACTION_BATCH_SIZE,
        )

        n_images = len(images)
        dim = patch_embeddings.shape[-1]
        logger.info("Extracted shape: %s from %d images", patch_embeddings.shape, n_images)

        # Build metadata for each patch
        metadata_all = []
        for idx, path in enumerate(valid_paths):
            fname = os.path.basename(path)
            stem = os.path.splitext(fname)[0]

            # Reconstruct original source image name
            if "_aug_" in stem:
                orig_stem = stem.split("_aug_")[0]
            else:
                orig_stem = stem

            # Search upload folder to find correct extension
            orig_name = None
            upload_dir = os.path.join(config.UPLOAD_FOLDER, session_id)
            if os.path.isdir(upload_dir):
                for f in os.listdir(upload_dir):
                    if os.path.splitext(f)[0] == orig_stem:
                        orig_name = f
                        break
            if not orig_name:
                orig_name = orig_stem + ".png"  # fallback

            for patch_idx in range(196):
                row = patch_idx // 14
                col = patch_idx % 14
                metadata_all.append({
                    "original_image": orig_name,
                    "patch_index": patch_idx,
                    "row": row,
                    "column": col,
                    "augmentation_source": fname,
                })

        # Reshape to (N * 196, dim) and normalize
        flat_embeddings = patch_embeddings.reshape(-1, dim)
        logger.info("L2 normalizing patch embeddings...")
        norms = np.linalg.norm(flat_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._embeddings = (flat_embeddings / norms).astype(np.float32)
        self._metadata = metadata_all
        self._is_built = True

        logger.info("PatchCore Memory Bank built with %d patches.", self._embeddings.shape[0])
        return self

    def save(self, session_id: str) -> None:
        """Save the memory bank to data/memory_bank/{session_id}/patchcore/."""
        if not self._is_built:
            raise RuntimeError("Memory bank must be built before saving.")

        out_dir = ensure_dir(os.path.join(config.MEMORY_BANK_FOLDER, session_id, "patchcore"))
        
        np.save(os.path.join(out_dir, "memory.npy"), self._embeddings)
        with open(os.path.join(out_dir, "patch_metadata.json"), "w") as f:
            json.dump(self._metadata, f, indent=2)

        logger.info("PatchCore Memory Bank saved to %s", out_dir)

    def load(self, session_id: str) -> PatchMemoryBank:
        """Load the memory bank from disk."""
        out_dir = os.path.join(config.MEMORY_BANK_FOLDER, session_id, "patchcore")
        npy_path = os.path.join(out_dir, "memory.npy")
        meta_path = os.path.join(out_dir, "patch_metadata.json")

        if not os.path.isfile(npy_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(f"PatchCore Memory Bank files not found for session {session_id}")

        self._embeddings = np.load(npy_path)
        with open(meta_path) as f:
            self._metadata = json.load(f)

        self._is_built = True
        logger.info("PatchCore Memory Bank loaded: %d patches.", self._embeddings.shape[0])
        return self
