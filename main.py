#!/usr/bin/env python3
"""FewVision Pipeline — Main Orchestrator.

Thin entry point that delegates to specialised modules:
  1. quality.py               → Image quality analysis
  2. content_analysis.py      → Content & scene analysis
  3. adaptive_augmentation.py  → Augmentation decision engine
  4. report_generator.py      → Per-image visual reports
  5. dataset_analyzer.py      → Dataset-level analytics
"""

import os
import csv
import json
import logging
from models import AnalysisResult
from quality import ImageQualityChecker
from content_analysis import ContentAnalyzer
from adaptive_augmentation import decide_augmentations
from report_generator import generate_image_report
from dataset_analyzer import analyze_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPORTS_DIR = "reports"
LOG_DIR = "logs"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

logger = logging.getLogger("fewvision")


# ---------------------------------------------------------------------------
# Suitability score (#12)
# ---------------------------------------------------------------------------
def _suitability(quality_score: float, content_score: float) -> tuple[float, str]:
    """Compute combined suitability score and rating."""
    score = round(0.5 * quality_score + 0.5 * content_score, 2)
    if score >= 75:
        rating = "Ready"
    elif score >= 50:
        rating = "Marginal"
    else:
        rating = "Unsuitable"
    return score, rating


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def process_image(image_path: str) -> AnalysisResult:
    """Run quality + content analysis, compute suitability, decide augmentations."""
    logger.info("Processing %s", image_path)

    q = ImageQualityChecker(image_path).analyze()
    c = ContentAnalyzer(image_path).analyze()

    suit_score, suit_rating = _suitability(q.quality_score, c.content_score)
    augs = decide_augmentations(q, c, suitability_score=suit_score)

    return AnalysisResult(
        image=os.path.basename(image_path),
        image_path=image_path,
        quality=q,
        content=c,
        suitability_score=suit_score,
        suitability_rating=suit_rating,
        augmentations=augs,
    )


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------
def _write_csv(results: list[AnalysisResult], path: str) -> None:
    if not results:
        return
    rows = [r.to_flat_dict() for r in results]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written to %s", path)


def _write_json(results: list[AnalysisResult], path: str) -> None:
    rows = [r.to_flat_dict() for r in results]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("JSON written to %s", path)


# ---------------------------------------------------------------------------
# Folder processing
# ---------------------------------------------------------------------------
def process_folder(folder_path: str) -> None:
    """Process every supported image in *folder_path*."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    image_paths = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    )

    if not image_paths:
        print("No supported images found.")
        logger.warning("No images found in %s", folder_path)
        return

    results: list[AnalysisResult] = []

    for idx, img_path in enumerate(image_paths, 1):
        try:
            print(f"[{idx}/{len(image_paths)}] Analysing {os.path.basename(img_path)} ...", end=" ")
            result = process_image(img_path)
            results.append(result)

            # Generate per-image report
            report_path = generate_image_report(result, REPORTS_DIR)
            print(
                f"[OK]  Quality={result.quality.quality_score:.0f}  "
                f"Content={result.content.content_score:.0f}  "
                f"Suit={result.suitability_score:.0f} [{result.suitability_rating}]"
            )
            logger.info(
                "OK  %s  quality=%.1f  content=%.1f  suit=%.1f  [%s]",
                result.image,
                result.quality.quality_score,
                result.content.content_score,
                result.suitability_score,
                result.suitability_rating,
            )
        except Exception as exc:
            print(f"[X]  Error: {exc}")
            logger.error("FAIL  %s  %s", img_path, exc, exc_info=True)

    # Write CSV + JSON
    _write_csv(results, os.path.join(REPORTS_DIR, "report.csv"))
    _write_json(results, os.path.join(REPORTS_DIR, "report.json"))

    # Dataset-level analytics
    if results:
        analytics = analyze_dataset(results, REPORTS_DIR)
        print(f"Distribution plot : {analytics['distribution_plot']}")
        print(f"Duplicates CSV    : {analytics['duplicates_csv']}")
        print(f"Embedding preview : {analytics['embedding_plot']}")

    # -----------------------------------------------------------------------
    # Machine Learning Pipeline (Member 1, 2, 3)
    # -----------------------------------------------------------------------
    # Member 1 Task: Adaptive Augmentation
    print("\nStarting Adaptive Augmentation batch generation...")
    from augmentations import generate_batch
    import shutil
    
    aug_dir = os.path.join(os.getcwd(), "augmented_dataset")
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(aug_dir, exist_ok=True)
    
    for r in results:
        img_path = r.image_path
        try:
            generate_batch(img_path, output_dir=aug_dir, num_images=10)
        except Exception as e:
            print(f"[X]  Augmentation failed for {img_path}: {e}")
    print(f"Augmented dataset saved to: {aug_dir}")

    # Member 2 Task: Feature Extraction
    print("\nStarting Feature Extraction via ResNet50...")
    try:
        from feature_extraction import run as extract_features_run
        extract_features_run(image_dir=aug_dir)
    except ImportError:
        print("[X]  torch/torchvision not installed. Run: pip install torch torchvision")
    except Exception as e:
        print(f"[X]  Feature extraction failed: {e}")

    # Member 3 Task: Few-Shot Classification
    print("\nPrototypical Network classifier...")
    try:
        from few_shot_model import run as few_shot_run
        few_shot_run()
    except Exception as e:
        print(f"[X]  Few-shot classification failed: {e}")

    print(f"\nAll reports written to ./{REPORTS_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    import sys

    default = os.path.join(os.getcwd(), "images")

    if len(sys.argv) > 1:
        # Path provided as CLI argument:  python main.py /path/to/images
        folder = sys.argv[1]
    else:
        # Interactive prompt (press Enter for default)
        folder = input(f"Enter dataset folder path (default {default}): ").strip() or default

    if not os.path.isdir(folder):
        print(f"Folder does not exist: {folder}")
    else:
        process_folder(folder)
