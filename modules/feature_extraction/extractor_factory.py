# modules/feature_extraction/extractor_factory.py
"""Factory for creating feature extractor instances.

Provides a single ``get_extractor`` entry-point that maps a string name to
a concrete :class:`~modules.feature_extraction.base_extractor.BaseExtractor`
subclass.  Pipeline code should **always** use this factory instead of
importing extractors directly.

Usage
-----
>>> from modules.feature_extraction.extractor_factory import get_extractor
>>> extractor = get_extractor()          # uses config.FEATURE_EXTRACTOR
>>> extractor = get_extractor("dinov2")  # explicit

Extending
---------
To add a new extractor:

1. Create ``modules/feature_extraction/my_extractor.py`` implementing
   :class:`BaseExtractor`.
2. Add an entry to :data:`REGISTRY` below.
3. Update ``config.FEATURE_EXTRACTOR`` (or the env-var) to use it.

No pipeline code changes are required.
"""

from __future__ import annotations

import logging

from modules.feature_extraction.base_extractor import BaseExtractor

logger = logging.getLogger("fewvision.feature_extraction")

# ---------------------------------------------------------------------------
# Registry — maps string identifiers to extractor classes (lazy import)
# ---------------------------------------------------------------------------
# Importers are stored as callables that return the class.  This avoids
# loading torch at import time for deployments that don't use extraction.

def _dinov2_class() -> type[BaseExtractor]:
    from modules.feature_extraction.dinov2_extractor import DINOv2Extractor
    return DINOv2Extractor


def _vit_class() -> type[BaseExtractor]:
    from modules.feature_extraction.vit_extractor import ViTExtractor
    return ViTExtractor


def _resnet50_class() -> type[BaseExtractor]:
    from modules.feature_extraction.resnet_extractor import ResNet50Extractor
    return ResNet50Extractor


REGISTRY: dict[str, callable] = {
    "dinov2": _dinov2_class,
    "vit": _vit_class,
    "resnet50": _resnet50_class,
    # Future extractors — uncomment and implement when ready:
    # "clip":     lambda: _import("modules.feature_extraction.clip_extractor", "CLIPExtractor"),
}


def get_extractor(
    name: str | None = None,
    **kwargs,
) -> BaseExtractor:
    """Create and return a feature extractor instance.

    Parameters
    ----------
    name : str or None
        Extractor identifier (e.g. ``"dinov2"``).  When ``None``, the value
        from ``config.FEATURE_EXTRACTOR`` is used.
    **kwargs
        Additional keyword arguments forwarded to the extractor constructor
        (e.g. ``model_variant``, ``device``).

    Returns
    -------
    BaseExtractor
        Uninitialised extractor instance.  Call ``.load_model()`` before
        invoking ``.extract()`` or ``.extract_batch()``.

    Raises
    ------
    ValueError
        If ``name`` is not found in :data:`REGISTRY`.

    Examples
    --------
    >>> extractor = get_extractor("dinov2", model_variant="dinov2_vitb14")
    >>> extractor.load_model()
    >>> embedding = extractor.extract(image)
    """
    if name is None:
        try:
            import config as _cfg
            name = getattr(_cfg, "FEATURE_EXTRACTOR", "dinov2")
        except ImportError:
            name = "dinov2"

    name = name.lower().strip()
    if "/" in name:
        parts = name.split("/", 1)
        name = parts[0]
        kwargs.setdefault("model_variant", parts[1])
        
    # Backwards compatibility fix for memory banks saved with the "resnet/resnet50" bug
    if name == "resnet":
        name = "resnet50"

    if name not in REGISTRY:
        available = ", ".join(f'"{k}"' for k in REGISTRY)
        raise ValueError(
            f"Unknown extractor '{name}'. "
            f"Available extractors: {available}."
        )

    cls_factory = REGISTRY[name]
    cls = cls_factory()

    # Inject model_variant from config if not overridden in kwargs
    if name == "dinov2" and "model_variant" not in kwargs:
        try:
            import config as _cfg
            kwargs.setdefault(
                "model_variant",
                getattr(_cfg, "DINOV2_MODEL_VARIANT", "dinov2_vits14"),
            )
        except ImportError:
            pass
    elif name == "vit" and "model_variant" not in kwargs:
        try:
            import config as _cfg
            kwargs.setdefault(
                "model_variant",
                getattr(_cfg, "VIT_MODEL_VARIANT", "vit_b_16"),
            )
        except ImportError:
            pass
    elif name == "resnet50" and "model_variant" not in kwargs:
        try:
            import config as _cfg
            kwargs.setdefault(
                "model_variant",
                getattr(_cfg, "RESNET50_MODEL_VARIANT", "resnet50"),
            )
        except ImportError:
            pass

    extractor = cls(**kwargs)
    logger.info("Created extractor: %s", extractor)
    return extractor


def list_extractors() -> list[str]:
    """Return all registered extractor names."""
    return list(REGISTRY.keys())
