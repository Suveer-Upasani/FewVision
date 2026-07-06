# modules/pipeline/pipeline.py
"""FewVision pipeline orchestrator.

This module is the heart of the application. It coordinates every stage of
the preprocessing workflow:

    1. Image Quality Analysis       → :mod:`modules.quality.quality`
    2. Content Analysis             → :mod:`modules.content.content_analysis`
    3. Suitability Scoring          → computed here
    4. Adaptive Augmentation Policy → :mod:`modules.augmentation.adaptive_policy`
    5. Report Generation            → :mod:`modules.reporting.report_generator`
    6. Batch Augmentation           → :mod:`modules.augmentation.augmentations`
    7. Dataset Analytics            → :mod:`modules.reporting.dataset_analytics`
    8. Feature Extraction           → :mod:`modules.feature_extraction`
    9. Memory Bank Construction     → :mod:`modules.anomaly_detection.memory_bank`

The pipeline intentionally contains **no Flask code** and **no ML training
logic**. It is a pure Python orchestrator that can be called from the Flask
app, a CLI, or tests.

Future extensions (PatchCore, PaDiM, grading) can be added by extending
:func:`process_dataset` without touching any other module.
"""

import os
import csv
import json
import logging

import config
from modules.utils.models import AnalysisResult, DatasetResult
from modules.utils.file_utils import ensure_dir, new_session_id, create_zip
from modules.quality.quality import ImageQualityChecker
from modules.content.content_analysis import ContentAnalyzer
from modules.augmentation.adaptive_policy import decide_augmentations
from modules.augmentation.augmentations import generate_batch
from modules.reporting.report_generator import generate_image_report
from modules.reporting.dataset_analytics import analyze_dataset
from modules.feature_extraction.extractor_factory import get_extractor
from modules.feature_extraction.embedding_database import save_embeddings

logger = logging.getLogger("fewvision")


# ---------------------------------------------------------------------------
# Suitability scoring
# ---------------------------------------------------------------------------

def _suitability(quality_score: float, content_score: float) -> tuple[float, str]:
    """Compute a combined suitability score and rating for a single image.

    Parameters
    ----------
    quality_score : float
        Quality score (0–100) from the quality module.
    content_score : float
        Content score (0–100) from the content analysis module.

    Returns
    -------
    tuple[float, str]
        ``(score, rating)`` where rating is one of ``"Ready"``,
        ``"Marginal"``, or ``"Unsuitable"``.
    """
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
    """Run the full per-image analysis pipeline.

    Executes quality analysis, content analysis, suitability scoring, and
    augmentation policy generation for a single image.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file.

    Returns
    -------
    AnalysisResult
        Complete analysis result for the image.
    """
    logger.info("Processing image: %s", image_path)

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
# CSV / JSON export helpers
# ---------------------------------------------------------------------------

def _write_csv(results: list[AnalysisResult], path: str) -> None:
    """Write analysis results to a CSV file."""
    if not results:
        return
    rows = [r.to_flat_dict() for r in results]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written: %s", path)


def _write_json(results: list[AnalysisResult], path: str) -> None:
    """Write analysis results to a JSON file."""
    rows = [r.to_flat_dict() for r in results]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("JSON written: %s", path)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _collect_augmented_images(aug_dir: str) -> list[str]:
    """Return sorted absolute paths of all images in *aug_dir*."""
    valid_ext = config.VALID_EXTENSIONS
    return sorted(
        os.path.join(aug_dir, f)
        for f in os.listdir(aug_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    )


def _build_embedding_metadata(
    aug_image_paths: list[str],
    results: list[AnalysisResult],
) -> list[dict]:
    """Build per-image metadata dicts for embedding_database.save_embeddings.

    Maps each augmented image back to its source analysis result (matched
    by stem prefix) to attach quality and content scores.

    Parameters
    ----------
    aug_image_paths : list[str]
        Absolute paths to augmented images.
    results : list[AnalysisResult]
        Per-image analysis results from the preprocessing pipeline.

    Returns
    -------
    list[dict]
        One metadata dict per augmented image.
    """
    # Build a fast lookup: source stem → AnalysisResult
    source_map: dict[str, AnalysisResult] = {
        os.path.splitext(r.image)[0]: r for r in results
    }

    metadata_list: list[dict] = []
    for img_path in aug_image_paths:
        fname = os.path.basename(img_path)
        stem  = os.path.splitext(fname)[0]

        # Augmented files are named like ``bearing001_aug_3.png``; match
        # back to the source ``bearing001`` stem.
        source_result: AnalysisResult | None = None
        for src_stem, res in source_map.items():
            if stem == src_stem or stem.startswith(src_stem + "_aug"):
                source_result = res
                break

        if source_result is not None:
            entry = {
                "filename": fname,
                "source_image": source_result.image,
                "quality_score": source_result.quality.quality_score,
                "quality_rating": source_result.quality.quality_rating,
                "content_score": source_result.content.content_score,
                "suitability_score": source_result.suitability_score,
                "suitability_rating": source_result.suitability_rating,
                "augmentations_applied": source_result.augmentations,
                "is_augmented": stem != os.path.splitext(source_result.image)[0],
            }
        else:
            entry = {"filename": fname}

        metadata_list.append(entry)

    return metadata_list


# ---------------------------------------------------------------------------
# Dataset pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    upload_dir: str,
    session_id: str | None = None,
) -> DatasetResult:
    """Run the full dataset preprocessing pipeline.

    Steps:
    1. Collect all images from *upload_dir*.
    2. Run per-image analysis (quality + content + suitability + policy).
    3. Generate per-image visual reports.
    4. Run dataset-level analytics.
    5. Generate augmented images for all analysis results.
    6. Package results into a :class:`~modules.utils.models.DatasetResult`.

    Parameters
    ----------
    upload_dir : str
        Directory containing the uploaded images.
    session_id : str, optional
        Session identifier. A new one is generated if not provided.

    Returns
    -------
    DatasetResult
        Full pipeline result including per-image analyses, analytics,
        and paths to augmented/report directories.
    """
    if session_id is None:
        session_id = new_session_id()

    # Session-scoped output directories
    report_dir = ensure_dir(os.path.join(config.REPORTS_FOLDER, session_id))
    aug_dir = ensure_dir(os.path.join(config.AUGMENTED_FOLDER, session_id))

    # Collect valid images
    valid_ext = config.VALID_EXTENSIONS
    image_paths = sorted(
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    )

    if not image_paths:
        logger.warning("No supported images found in: %s", upload_dir)
        return DatasetResult(session_id=session_id)

    results: list[AnalysisResult] = []

    # --- Step 1: Per-image analysis + reports ---
    for idx, img_path in enumerate(image_paths, 1):
        try:
            logger.info("[%d/%d] Analysing %s", idx, len(image_paths),
                        os.path.basename(img_path))
            result = process_image(img_path)
            generate_image_report(result, report_dir)
            results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", img_path, exc, exc_info=True)

    if not results:
        logger.error("All images failed processing in session %s", session_id)
        return DatasetResult(session_id=session_id)

    # --- Step 2: Export CSV + JSON reports ---
    _write_csv(results, os.path.join(report_dir, "report.csv"))
    _write_json(results, os.path.join(report_dir, "report.json"))

    # --- Step 3: Dataset analytics ---
    analytics = analyze_dataset(results, report_dir)

    # --- Step 4: Generate augmented dataset ---
    augmented_count = 0
    for result in results:
        try:
            generated = generate_batch(
                image_path=result.image_path,
                output_dir=aug_dir,
                num_images=config.AUGMENTED_IMAGES_PER_SOURCE,
                augmentations=result.augmentations,
            )
            augmented_count += len(generated)
        except Exception as exc:
            logger.error("Augmentation failed for %s: %s", result.image, exc)

    # --- Step 5: Count suitability ratings ---
    ready = sum(1 for r in results if r.suitability_rating == "Ready")
    marginal = sum(1 for r in results if r.suitability_rating == "Marginal")
    unsuitable = sum(1 for r in results if r.suitability_rating == "Unsuitable")

    logger.info(
        "Session %s — preprocessing complete: %d analysed, %d augmented",
        session_id, len(results), augmented_count,
    )

    # --- Step 6: Feature extraction (DINOv2) ---
    embedding_path: str = ""
    embedding_count: int = 0

    if config.FEATURE_EXTRACTION_ENABLED:
        try:
            logger.info("Starting feature extraction for session %s …", session_id)

            aug_image_paths = _collect_augmented_images(aug_dir)

            if not aug_image_paths:
                logger.warning("No augmented images found in %s — skipping extraction.", aug_dir)
            else:
                # Load model once — reused for entire session
                extractor = get_extractor(config.FEATURE_EXTRACTOR)
                extractor.load_model()

                # Load images as BGR numpy arrays
                import cv2 as _cv2
                images_bgr = []
                valid_paths = []
                for p in aug_image_paths:
                    img = _cv2.imread(p)
                    if img is not None:
                        images_bgr.append(img)
                        valid_paths.append(p)
                    else:
                        logger.warning("Could not read augmented image: %s", p)

                if images_bgr:
                    embeddings = extractor.extract_batch(
                        images_bgr,
                        batch_size=config.EXTRACTION_BATCH_SIZE,
                    )
                    filenames = [os.path.basename(p) for p in valid_paths]
                    metadata  = _build_embedding_metadata(valid_paths, results)

                    embedding_path = save_embeddings(
                        session_id=session_id,
                        embeddings=embeddings,
                        filenames=filenames,
                        metadata=metadata,
                        extractor_info=extractor.info,
                    )
                    embedding_count = len(filenames)

                    logger.info(
                        "Feature extraction complete — %d embeddings, dim=%d, saved to %s",
                        embedding_count, extractor.embedding_dim, embedding_path,
                    )

        except Exception as exc:
            logger.error(
                "Feature extraction failed for session %s: %s",
                session_id, exc, exc_info=True,
            )
            # Non-fatal: pipeline continues, embedding fields stay empty

    # --- Step 7: Memory Bank construction ---
    memory_bank_path: str = ""
    memory_bank_count: int = 0
    memory_bank_dim: int = 0

    if config.ENABLE_MEMORY_BANK and embedding_count > 0:
        try:
            from modules.anomaly_detection.memory_bank import MemoryBank

            bank = MemoryBank()
            bank.build(session_id)
            memory_bank_path = bank.save(session_id)
            memory_bank_count = bank.size()
            memory_bank_dim = bank.embedding_dimension()

            logger.info(
                "Memory Bank created successfully — %d embeddings, dim=%d, path=%s",
                memory_bank_count, memory_bank_dim, memory_bank_path,
            )

        except Exception as exc:
            logger.error(
                "Memory Bank construction failed for session %s: %s",
                session_id, exc, exc_info=True,
            )
            # Non-fatal: pipeline continues, memory bank fields stay empty

    return DatasetResult(
        results=results,
        analytics=analytics,
        augmented_dir=aug_dir,
        report_dir=report_dir,
        session_id=session_id,
        total_images=len(results),
        augmented_count=augmented_count,
        ready_count=ready,
        marginal_count=marginal,
        unsuitable_count=unsuitable,
        embedding_path=embedding_path,
        embedding_count=embedding_count,
        memory_bank_path=memory_bank_path,
        memory_bank_count=memory_bank_count,
        memory_bank_dim=memory_bank_dim,
    )



def create_augmented_zip(session_id: str) -> str:
    """Create a ZIP archive of the augmented dataset for a session.

    Parameters
    ----------
    session_id : str
        Session identifier.

    Returns
    -------
    str
        Absolute path to the created ZIP file.

    Raises
    ------
    FileNotFoundError
        If the augmented directory for this session does not exist.
    """
    aug_dir = os.path.join(config.AUGMENTED_FOLDER, session_id)
    if not os.path.isdir(aug_dir):
        raise FileNotFoundError(f"No augmented dataset for session: {session_id}")

    temp_dir = ensure_dir(config.TEMP_FOLDER)
    zip_path = os.path.join(temp_dir, f"fewvision_{session_id}.zip")
    return create_zip(aug_dir, zip_path)
