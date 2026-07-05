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
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
