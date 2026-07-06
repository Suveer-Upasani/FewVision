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

# ViT model variant. Options:
#   vit_b_16 — 768-dim, standard ImageNet-1K pretrained ViT (default)
VIT_MODEL_VARIANT = os.environ.get("FEWVISION_VIT_VARIANT", "vit_b_16")

# Number of images per forward pass (increase if GPU memory allows)
EXTRACTION_BATCH_SIZE = int(os.environ.get("FEWVISION_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

# ---------------------------------------------------------------------------
# Memory Bank
# ---------------------------------------------------------------------------
# Set to True (default) to automatically build the Memory Bank after feature
# extraction.  Disable to skip this stage (e.g. during quick dev runs).
ENABLE_MEMORY_BANK = os.environ.get("FEWVISION_MEMORY_BANK", "true").lower() == "true"

# Directory where memory bank data is persisted.
# Each session gets its own subdirectory: data/memory_bank/{session_id}/
MEMORY_BANK_FOLDER = os.path.join(DATA_FOLDER, "memory_bank")

# ---------------------------------------------------------------------------
# Similarity Engine
# ---------------------------------------------------------------------------
# Metric used for nearest-neighbour search in the Memory Bank.
# Options:
#   "cosine"    — cosine distance (recommended; works best with L2-normalised
#                 DINOv2 / ViT embeddings)
#   "euclidean" — L2 distance (useful if embeddings are not pre-normalised)
SIMILARITY_METRIC = os.environ.get("FEWVISION_SIMILARITY_METRIC", "cosine")

# Number of nearest neighbours returned by MemoryBank.search() and
# top_k_neighbors().  Increase for more context during anomaly scoring.
TOP_K_NEIGHBORS = int(os.environ.get("FEWVISION_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Anomaly Scoring Thresholds
# ---------------------------------------------------------------------------
# Distance thresholds that map nearest-neighbour distances to anomaly labels.
#
#   distance ≤ normal_max     → "Normal"
#   normal_max < dist ≤ suspicious_max → "Suspicious"
#   distance > suspicious_max → "Anomalous"
#
# These are cosine distances in [0, 2].  With L2-normalised embeddings and
# DINOv2, typical intra-class distances are < 0.2.  Adjust after calibration
# on your specific product images.
ANOMALY_THRESHOLDS: dict = {
    "normal_max": float(os.environ.get("FEWVISION_THRESH_NORMAL", "0.20")),
    "suspicious_max": float(os.environ.get("FEWVISION_THRESH_SUSPICIOUS", "0.50")),
}

# ---------------------------------------------------------------------------
# Inference Pipeline
# ---------------------------------------------------------------------------
# Directory where inference results are persisted
INFERENCE_FOLDER = os.path.join(DATA_FOLDER, "inference")

# Master enable switch for inference
ENABLE_INFERENCE = os.environ.get("FEWVISION_INFERENCE", "true").lower() == "true"

# Default configurations for similarity searches in inference
DEFAULT_TOP_K = int(os.environ.get("FEWVISION_DEFAULT_TOP_K", "5"))
DEFAULT_SIMILARITY = os.environ.get("FEWVISION_DEFAULT_SIMILARITY", "cosine")
DEFAULT_MEMORY_BANK = os.environ.get("FEWVISION_DEFAULT_MEMORY_BANK", MEMORY_BANK_FOLDER)
MAX_TEST_IMAGES = int(os.environ.get("FEWVISION_MAX_TEST_IMAGES", "20"))
