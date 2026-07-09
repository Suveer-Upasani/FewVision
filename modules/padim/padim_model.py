# modules/padim/padim_model.py
"""Isolated PaDiM (Patch Distribution Modeling) anomaly detection model core.

Fits location-wise Gaussian distributions (means and covariance matrices)
over reference patch features and computes Mahalanobis distance maps for queries.
"""

from __future__ import annotations

import json
import os
import logging
import numpy as np

logger = logging.getLogger("fewvision.padim.model")


class PaDiMModel:
    """PaDiM anomaly detection and localization model.

    Parameters
    ----------
    d : int
        Reduced feature dimension (number of channels to select, default: 100).
    regularization : float
        Covariance regularization parameter (default: 0.01).
    covariance_method : {"empirical", "diagonal", "shrunk"}
        Method to estimate and regularize covariances (default: "empirical").
    random_seed : int
        Random seed for deterministic channel selection (default: 42).
    """

    def __init__(
        self,
        d: int = 100,
        regularization: float = 0.01,
        covariance_method: str = "empirical",
        random_seed: int = 42,
    ) -> None:
        self.d = d
        self.regularization = regularization
        self.covariance_method = covariance_method.lower().strip()
        self.random_seed = random_seed

        if self.covariance_method not in {"empirical", "diagonal", "shrunk"}:
            raise ValueError(
                f"Unknown covariance method '{covariance_method}'. "
                "Valid options are: 'empirical', 'diagonal', 'shrunk'."
            )

        # Fitted parameters
        self.means: np.ndarray | None = None               # (H, W, d)
        self.inv_covs: np.ndarray | None = None            # (H, W, d, d)
        self.selected_channels: np.ndarray | None = None    # (d,)
        self.grid_shape: tuple[int, int] | None = None     # (H, W)
        self.original_dim: int | None = None
        self.reduced_dim: int | None = None
        self.num_reference_observations: int | None = None
        self.localization_threshold: float | None = None
        self.is_fitted = False

    def _validate_input_features(self, features: np.ndarray) -> None:
        """Validate input feature arrays for NaN/Inf values and data types."""
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a numpy ndarray.")

        if np.isnan(features).any():
            raise ValueError("Input features contain NaN values.")

        if np.isinf(features).any():
            raise ValueError("Input features contain infinite (Inf) values.")

    def fit(self, reference_features: np.ndarray) -> None:
        """Fit location-wise Gaussian distributions over reference spatial features.

        Parameters
        ----------
        reference_features : np.ndarray
            Reference spatial features of shape (N, H, W, D).
        """
        self._validate_input_features(reference_features)

        if reference_features.ndim != 4:
            raise ValueError(
                f"Expected reference features to be 4D array of shape (N, H, W, D), "
                f"got shape {reference_features.shape}"
            )

        n, h, w, D = reference_features.shape
        if n < 1:
            raise ValueError("Reference features must contain at least one observation.")

        # Determine target dimension d
        d = min(self.d, D)
        self.reduced_dim = d
        self.original_dim = D
        self.grid_shape = (h, w)
        self.num_reference_observations = n

        # Deterministically select channels
        rng = np.random.RandomState(self.random_seed)
        self.selected_channels = rng.choice(D, size=d, replace=False)
        self.selected_channels.sort()

        logger.info(
            "Fitting PaDiM model on %d observations with grid (%d, %d). "
            "Reducing dimension from %d to %d.",
            n, h, w, D, d
        )

        # Allocate arrays
        self.means = np.zeros((h, w, d), dtype=np.float32)
        self.inv_covs = np.zeros((h, w, d, d), dtype=np.float32)

        # Compute stats per grid cell
        for r in range(h):
            for c in range(w):
                # Extract features for current grid location across all reference images
                # Shape: (N, d)
                Y = reference_features[:, r, c, self.selected_channels]

                # Compute mean vector
                mean_vec = np.mean(Y, axis=0)
                self.means[r, c, :] = mean_vec

                # Compute covariance matrix
                if n < 2:
                    cov = np.zeros((d, d), dtype=np.float32)
                else:
                    cov = np.cov(Y, rowvar=False)
                    cov = np.atleast_2d(cov)

                # Regularize covariance matrix
                if self.covariance_method == "empirical":
                    cov = cov + self.regularization * np.eye(d, dtype=np.float32)
                elif self.covariance_method == "diagonal":
                    cov = np.diag(np.diag(cov)) + self.regularization * np.eye(d, dtype=np.float32)
                elif self.covariance_method == "shrunk":
                    mean_var = np.trace(cov) / d if d > 0 else 0.0
                    cov = (1.0 - self.regularization) * cov + self.regularization * mean_var * np.eye(d, dtype=np.float32)

                # Compute stable inverse covariance
                try:
                    inv_cov = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    logger.warning(
                        "Singular covariance at grid (%d, %d). Falling back to pseudo-inverse.",
                        r, c
                    )
                    inv_cov = np.linalg.pinv(cov)

                self.inv_covs[r, c, :, :] = inv_cov

        self.is_fitted = True
        logger.info("PaDiM model fitting complete.")

    def score(self, query_features: np.ndarray) -> np.ndarray:
        """Compute the raw Mahalanobis anomaly distance map for a query image.

        Parameters
        ----------
        query_features : np.ndarray
            Query spatial feature map of shape (H, W, D).

        Returns
        -------
        np.ndarray
            Raw anomaly distance map of shape (H, W), float32.
        """
        if not self.is_fitted:
            raise RuntimeError("PaDiMModel must be fitted before scoring.")

        self._validate_input_features(query_features)

        if query_features.ndim != 3:
            raise ValueError(
                f"Expected query features to be 3D array of shape (H, W, D), "
                f"got shape {query_features.shape}"
            )

        h, w, D = query_features.shape
        if (h, w) != self.grid_shape or D != self.original_dim:
            raise ValueError(
                f"Incompatible query features. Expected grid {self.grid_shape} and "
                f"dimension {self.original_dim}, got grid {(h, w)} and dimension {D}"
            )

        # Select channels
        # Shape: (H, W, d)
        query_sel = query_features[:, :, self.selected_channels]

        # Fully vectorized Mahalanobis distance calculation
        # diff shape: (H, W, d)
        diff = query_sel - self.means

        # Compute: dist[h, w] = sqrt( diff[h, w]^T * inv_covs[h, w] * diff[h, w] )
        # Using np.einsum for high-speed computation matching C-level speed
        # Step 1: Multiply diff by inv_covs (H, W, d) x (H, W, d, d) -> (H, W, d)
        temp = np.einsum("hwi,hwij->hwj", diff, self.inv_covs)
        # Step 2: Dot product with diff (H, W, d) x (H, W, d) -> (H, W)
        dist_sq = np.einsum("hwj,hwj->hw", temp, diff)

        # Clip values below zero to handle floating-point precision bounds
        dist = np.sqrt(np.clip(dist_sq, 0.0, None))

        return dist.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist model state and metadata to a .npz file.

        Parameters
        ----------
        path : str
            Target file path (should end with .npz).
        """
        if not self.is_fitted:
            raise RuntimeError("Only fitted models can be saved.")

        metadata = {
            "d": self.d,
            "regularization": self.regularization,
            "covariance_method": self.covariance_method,
            "random_seed": self.random_seed,
            "grid_shape": list(self.grid_shape) if self.grid_shape else None,
            "original_dim": self.original_dim,
            "reduced_dim": self.reduced_dim,
            "num_reference_observations": self.num_reference_observations,
            "localization_threshold": self.localization_threshold,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        np.savez(
            path,
            means=self.means,
            inv_covs=self.inv_covs,
            selected_channels=self.selected_channels,
            metadata=json.dumps(metadata),
        )
        logger.info("PaDiM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> PaDiMModel:
        """Load a persisted PaDiMModel from a .npz file.

        Parameters
        ----------
        path : str
            Path to the saved .npz file.

        Returns
        -------
        PaDiMModel
            Loaded and fitted model instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(path)
        metadata = json.loads(str(data["metadata"]))

        # Instantiate model
        model = cls(
            d=metadata["d"],
            regularization=metadata["regularization"],
            covariance_method=metadata["covariance_method"],
            random_seed=metadata["random_seed"],
        )

        model.means = data["means"]
        model.inv_covs = data["inv_covs"]
        model.selected_channels = data["selected_channels"]
        model.grid_shape = tuple(metadata["grid_shape"]) if metadata["grid_shape"] else None
        model.original_dim = metadata["original_dim"]
        model.reduced_dim = metadata["reduced_dim"]
        model.num_reference_observations = metadata["num_reference_observations"]
        model.localization_threshold = metadata.get("localization_threshold")
        model.is_fitted = True

        logger.info("PaDiM model loaded successfully from %s", path)
        return model
