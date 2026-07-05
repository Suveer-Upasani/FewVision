# config.py
"""FewVision application configuration.

All tunable constants live here. Import this module anywhere configuration
is needed — never hard-code paths or limits in other modules.
"""

import os

# ---------------------------------------------------------------------------
# Base directory (project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_FOLDER = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(DATA_FOLDER, "uploads")
AUGMENTED_FOLDER = os.path.join(DATA_FOLDER, "augmented")
REPORTS_FOLDER = os.path.join(DATA_FOLDER, "reports")
LOGS_FOLDER = os.path.join(DATA_FOLDER, "logs")
TEMP_FOLDER = os.path.join(DATA_FOLDER, "temp")
EMBEDDINGS_FOLDER = os.path.join(DATA_FOLDER, "embeddings")

# ---------------------------------------------------------------------------
# Flask settings
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB = 64
MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024
SECRET_KEY = os.environ.get("FEWVISION_SECRET_KEY", "fewvision-dev-secret-change-in-prod")
DEBUG = os.environ.get("FEWVISION_DEBUG", "true").lower() == "true"
PORT = int(os.environ.get("FEWVISION_PORT", "5005"))

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUGMENTED_IMAGES_PER_SOURCE = 10
MIN_IMAGES = 1
MAX_IMAGES = 50

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
# Set to True to run DINOv2 feature extraction after augmentation.
# On first run the model weights (~85 MB) are downloaded automatically.
FEATURE_EXTRACTION_ENABLED = os.environ.get("FEWVISION_FEATURE_EXTRACTION", "true").lower() == "true"

# Extractor to use. See modules/feature_extraction/extractor_factory.py REGISTRY.
# Options: "dinov2"  (future: "clip", "resnet50", "vit")
FEATURE_EXTRACTOR = os.environ.get("FEWVISION_EXTRACTOR", "dinov2")

# DINOv2 model variant. Options:
#   dinov2_vits14 — 384-dim, fast (default)
#   dinov2_vitb14 — 768-dim, better quality
#   dinov2_vitl14 — 1024-dim, best quality (slow)
DINOV2_MODEL_VARIANT = os.environ.get("FEWVISION_DINOV2_VARIANT", "dinov2_vits14")

# Number of images per forward pass (increase if GPU memory allows)
EXTRACTION_BATCH_SIZE = int(os.environ.get("FEWVISION_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
