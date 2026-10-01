# app.py
"""FewVision Flask Application.

This module contains **only** Flask routes and application bootstrap code.
All image processing, augmentation, and reporting logic is delegated to
the pipeline module.

Routes
------
GET  /                               Upload page
POST /api/upload                     Accept images and run pipeline
GET  /dashboard/<session_id>         Analysis dashboard
POST /api/generate/<session_id>      Generate augmented dataset ZIP
GET  /api/download/<session_id>      Stream ZIP file download
GET  /results/<session_id>           Results summary page
GET  /reports/<session_id>/<filename> Serve report images
GET  /api/embeddings/<session_id>    Embedding database metadata (JSON)
GET  /api/embeddings/<session_id>/download  Download embeddings.npy
GET  /api/memory-bank/<session_id>   Memory Bank metadata (JSON)
"""

import io
import json
import os
import logging
import datetime
import uuid
import base64
import threading
import time
import warnings

# Suppress benign non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"dinov2\..*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.hub")
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

try:
    import cv2.utils.logging as cvlog
    cvlog.setLogLevel(cvlog.LOG_LEVEL_SILENT)
except Exception:
    pass

import cv2
import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
    redirect,
    url_for,
    session,
    Response,
)
from werkzeug.utils import secure_filename

import config
from modules.utils.file_utils import ensure_dir, new_session_id, clear_dir
from modules.pipeline.pipeline import process_dataset, create_augmented_zip
from modules.reporting.pdf_report import generate_pdf_report

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns
    -------
    Flask
        Configured application instance.
    """
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY
    # Allow the larger of image limit and video limit so video uploads succeed.
    video_limit = getattr(config, "MAX_VIDEO_UPLOAD_SIZE_MB", 512) * 1024 * 1024
    app.config["MAX_CONTENT_LENGTH"] = max(config.MAX_CONTENT_LENGTH, video_limit)

    # Ensure all data directories exist at startup
    ensure_dir(config.UPLOAD_FOLDER)
    ensure_dir(config.AUGMENTED_FOLDER)
    ensure_dir(config.REPORTS_FOLDER)
    ensure_dir(config.LOGS_FOLDER)
    ensure_dir(config.TEMP_FOLDER)
    ensure_dir(config.MEMORY_BANK_FOLDER)
    ensure_dir(config.INFERENCE_FOLDER)
    ensure_dir(getattr(config, "FEEDBACK_FOLDER", os.path.join(config.DATA_FOLDER, "feedback")))
    ensure_dir(getattr(config, "CAPTURES_FOLDER", os.path.join(config.DATA_FOLDER, "inspection_results")))


    # ---------------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------------
    logging.basicConfig(
        filename=os.path.join(config.LOGS_FOLDER, "app.log"),
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )
    app_logger = logging.getLogger("fewvision")

    # ---------------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------------
    def _allowed_file(filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in config.VALID_EXTENSIONS

    def _allowed_video_file(filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in getattr(config, "VALID_VIDEO_EXTENSIONS", {".mp4", ".mov", ".avi", ".mkv"})

    def _is_video_upload(filename: str) -> bool:
        """Return True if the uploaded file should be treated as a video."""
        return _allowed_video_file(filename)

    # ---------------------------------------------------------------------------
    # Page 1: Upload
    # ---------------------------------------------------------------------------

    @app.route("/")
    def index():
        """Render the image upload page."""
        from modules.anomaly_detection.memory_bank import list_memory_bank_sessions, load_memory_bank_summary
        sessions = list_memory_bank_sessions()

        active_session_id = session.get("session_id")
        if not active_session_id and sessions:
            active_session_id = sessions[-1]

        mb_summary = None
        if active_session_id and active_session_id in sessions:
            try:
                mb_summary = load_memory_bank_summary(active_session_id)
                npy_path = os.path.join(config.MEMORY_BANK_FOLDER, active_session_id, "memory.npy")
                if os.path.isfile(npy_path):
                    mtime = os.path.getmtime(npy_path)
                    mb_summary["created_time"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    mb_summary["created_time"] = "Unknown"
            except Exception:
                pass

        return render_template("index.html", mb_summary=mb_summary)

    # ---------------------------------------------------------------------------
    # API: Upload + Run Pipeline
    # ---------------------------------------------------------------------------

    @app.route("/api/upload", methods=["POST"])
    def upload():
        """Accept uploaded images, run the analysis pipeline, redirect to dashboard.

        Expects a multipart form with key ``files`` containing one or more images.

        Returns
        -------
        JSON response with session_id and redirect URL on success,
        or an error message on failure.
        """
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files")
        valid_files = [f for f in files if f and _allowed_file(f.filename)]

        if not valid_files:
            return jsonify({"error": "No valid image files found"}), 400

        if len(valid_files) > config.MAX_IMAGES:
            return jsonify({
                "error": f"Too many files. Maximum is {config.MAX_IMAGES} images."
            }), 400

        # Create a session-scoped upload directory
        sid = new_session_id()
        upload_dir = ensure_dir(os.path.join(config.UPLOAD_FOLDER, sid))

        try:
            # Save uploaded files
            for f in valid_files:
                filename = secure_filename(f.filename)
                f.save(os.path.join(upload_dir, filename))

            # Get selected extractor from form, default to None (will use config)
            extractor_name = request.form.get("extractor")

            # Run the full pipeline
            app_logger.info("Starting pipeline for session %s (%d images, extractor=%s)", sid, len(valid_files), extractor_name or config.FEATURE_EXTRACTOR)
            dataset_result = process_dataset(upload_dir, session_id=sid, extractor_name=extractor_name)

            # Serialise results for the session store
            session["session_id"] = sid
            session["total_images"] = dataset_result.total_images
            session["augmented_count"] = dataset_result.augmented_count
            session["ready_count"] = dataset_result.ready_count
            session["marginal_count"] = dataset_result.marginal_count
            session["unsuitable_count"] = dataset_result.unsuitable_count
            session["report_dir"] = dataset_result.report_dir
            session["memory_bank_count"] = dataset_result.memory_bank_count
            session["memory_bank_dim"] = dataset_result.memory_bank_dim
            session["memory_bank_path"] = dataset_result.memory_bank_path

            # Build per-image data for the dashboard
            images_data = []
            for r in dataset_result.results:
                images_data.append({
                    "image": r.image,
                    "quality": r.quality.to_dict(),
                    "content": r.content.to_dict(),
                    "suitability_score": r.suitability_score,
                    "suitability_rating": r.suitability_rating,
                    "augmentations": r.augmentations,
                    "report_url": url_for(
                        "serve_report",
                        session_id=sid,
                        filename=f"report_{os.path.splitext(r.image)[0]}.png",
                    ),
                })
            session["images_data"] = images_data

            return jsonify({
                "success": True,
                "session_id": sid,
                "redirect": url_for("dashboard", session_id=sid),
            })

        except Exception as exc:
            app_logger.error("Pipeline error for session %s: %s", sid, exc, exc_info=True)
            # Clean up partial upload
            try:
                clear_dir(upload_dir)
            except Exception:
                pass
            return jsonify({"error": str(exc)}), 500

    # ---------------------------------------------------------------------------
    # Page 2: Analysis Dashboard
    # ---------------------------------------------------------------------------

    @app.route("/dashboard/<session_id>")
    def dashboard(session_id: str):
        """Render the analysis dashboard for a completed pipeline session.

        Parameters
        ----------
        session_id : str
            Session identifier returned by the upload route.
        """
        from modules.anomaly_detection.memory_bank import load_memory_bank_summary

        images_data = session.get("images_data", [])
        summary = {
            "session_id": session.get("session_id", session_id),
            "total_images": session.get("total_images", 0),
            "ready_count": session.get("ready_count", 0),
            "marginal_count": session.get("marginal_count", 0),
            "unsuitable_count": session.get("unsuitable_count", 0),
        }

        # Attempt to load memory bank details from disk
        mb_summary = None
        try:
            mb_summary = load_memory_bank_summary(session_id)
        except Exception:
            pass

        if mb_summary and mb_summary.get("count", 0) > 0:
            memory_bank = {
                "count": mb_summary["count"],
                "dim": mb_summary["embedding_dim"],
                "path": mb_summary["location"],
                "enabled": config.ENABLE_MEMORY_BANK,
                "metric": mb_summary.get("similarity_metric", config.SIMILARITY_METRIC),
                "status": "Ready",
            }
        else:
            memory_bank = {
                "count": session.get("memory_bank_count", 0),
                "dim": session.get("memory_bank_dim", 0),
                "path": session.get("memory_bank_path", ""),
                "enabled": config.ENABLE_MEMORY_BANK,
                "metric": config.SIMILARITY_METRIC,
                "status": "Ready" if session.get("memory_bank_count", 0) > 0 else "Not built",
            }

        # Robustly calculate original and augmented image counts for Workflow 2 status card
        original_count = summary["total_images"]
        augmented_count = session.get("augmented_count", 0)
        extractor_name = config.FEATURE_EXTRACTOR

        try:
            # Load metadata for extractor info
            meta_path = os.path.join(config.MEMORY_BANK_FOLDER, session_id, "memory_metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    m_data = json.load(f)
                extractor_name = m_data.get("extractor_info", {}).get("extractor_name", extractor_name)

            # Look up metadata of embeddings to check precise counts
            emb_meta_path = os.path.join(config.EMBEDDINGS_FOLDER, session_id, "metadata.json")
            if os.path.isfile(emb_meta_path):
                with open(emb_meta_path) as f:
                    emb_meta = json.load(f)
                augmented_count = len(emb_meta)
                sources = {item.get("source_image") for item in emb_meta if item.get("source_image")}
                if sources:
                    original_count = len(sources)
        except Exception:
            pass

        memory_bank["original_count"] = original_count
        memory_bank["augmented_count"] = augmented_count
        memory_bank["extractor"] = extractor_name

        return render_template(
            "dashboard.html",
            images=images_data,
            summary=summary,
            session_id=session_id,
            memory_bank=memory_bank,
        )

    # ---------------------------------------------------------------------------
    # API: Generate ZIP
    # ---------------------------------------------------------------------------

    @app.route("/api/generate/<session_id>", methods=["POST"])
    def generate(session_id: str):
        """Create the augmented dataset ZIP for download.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        JSON response with download URL or error.
        """
        try:
            zip_path = create_augmented_zip(session_id)
            return jsonify({
                "success": True,
                "download_url": url_for("download", session_id=session_id),
                "augmented_count": session.get("augmented_count", 0),
            })
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            app_logger.error("ZIP creation error for session %s: %s", session_id, exc)
            return jsonify({"error": str(exc)}), 500

    # ---------------------------------------------------------------------------
    # API: Download ZIP
    # ---------------------------------------------------------------------------

    @app.route("/api/download/<session_id>")
    def download(session_id: str):
        """Stream the augmented dataset ZIP file.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        zip_path = os.path.join(config.TEMP_FOLDER, f"fewvision_{session_id}.zip")
        if not os.path.isfile(zip_path):
            return jsonify({"error": "ZIP file not found. Please generate first."}), 404

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"fewvision_augmented_{session_id}.zip",
            mimetype="application/zip",
        )

    # ---------------------------------------------------------------------------
    # Page 3: Results (served via dashboard redirect after generation)
    # ---------------------------------------------------------------------------

    @app.route("/results/<session_id>")
    def results(session_id: str):
        """Render the results page after dataset generation.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        summary = {
            "session_id": session_id,
            "total_images": session.get("total_images", 0),
            "augmented_count": session.get("augmented_count", 0),
            "ready_count": session.get("ready_count", 0),
            "marginal_count": session.get("marginal_count", 0),
            "unsuitable_count": session.get("unsuitable_count", 0),
        }
        return render_template("results.html", summary=summary, session_id=session_id)

    # ---------------------------------------------------------------------------
    # Static file serving
    # ---------------------------------------------------------------------------

    @app.route("/reports/<session_id>/<filename>")
    def serve_report(session_id: str, filename: str):
        """Serve a generated report image.

        Parameters
        ----------
        session_id : str
            Session identifier.
        filename : str
            Report image filename.
        """
        report_dir = os.path.join(config.REPORTS_FOLDER, session_id)
        return send_file(os.path.join(report_dir, secure_filename(filename)))

    # ---------------------------------------------------------------------------
    # Embedding API routes
    # ---------------------------------------------------------------------------

    @app.route("/api/embeddings/<session_id>")
    def embedding_info(session_id: str):
        """Return embedding database metadata as JSON.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        from modules.feature_extraction.embedding_database import embedding_summary, list_sessions
        sessions = list_sessions()
        if session_id not in sessions:
            return jsonify({"error": f"No embedding database found for session '{session_id}'"}), 404
        summary = embedding_summary(session_id)
        return jsonify(summary)

    @app.route("/api/embeddings/<session_id>/download")
    def download_embeddings(session_id: str):
        """Stream the embeddings.npy binary file for download.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        import config as _cfg
        npy_path = os.path.join(_cfg.EMBEDDINGS_FOLDER, session_id, "embeddings.npy")
        if not os.path.isfile(npy_path):
            return jsonify({"error": "Embeddings not found for this session."}), 404
        return send_file(
            npy_path,
            as_attachment=True,
            download_name=f"fewvision_embeddings_{session_id}.npy",
            mimetype="application/octet-stream",
        )

    # ---------------------------------------------------------------------------
    # Memory Bank API routes
    # ---------------------------------------------------------------------------

    @app.route("/api/memory-bank/<session_id>")
    def memory_bank_info(session_id: str):
        """Return Memory Bank metadata as JSON.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        JSON with keys: session_id, count, embedding_dim, similarity_metric,
        top_k, npy_size_mb, location.
        """
        from modules.anomaly_detection.memory_bank import (
            load_memory_bank_summary,
            list_memory_bank_sessions,
        )
        sessions = list_memory_bank_sessions()
        if session_id not in sessions:
            return jsonify({
                "error": f"No memory bank found for session '{session_id}'"
            }), 404
        summary = load_memory_bank_summary(session_id)
        return jsonify(summary)

    # ---------------------------------------------------------------------------
    # Inference API routes
    # ---------------------------------------------------------------------------

    @app.route("/api/inspect", methods=["POST"])
    def inspect():
        """Accept a test image or video, run inspection, return results.

        Accepts either:
        - One or more image files (PNG/JPG/JPEG) → existing image inspection.
        - A single video file (MP4/MOV/AVI/MKV) → new video inspection.

        The response always includes ``input_type`` (``"image"`` or ``"video"``)
        so clients can choose the appropriate rendering path.
        """
        if not config.ENABLE_INFERENCE:
            return jsonify({"error": "Inference pipeline is disabled."}), 400

        if "files" not in request.files:
            return jsonify({"error": "No test files uploaded."}), 400

        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No test files uploaded."}), 400

        # Detect whether this is a video upload by checking the first file.
        first_filename = files[0].filename if files else ""
        is_video = _is_video_upload(first_filename)

        # Get active session_id
        session_id = request.form.get("session_id") or session.get("session_id")
        if not session_id:
            from modules.anomaly_detection.memory_bank import list_memory_bank_sessions
            sessions = list_memory_bank_sessions()
            if sessions:
                session_id = sessions[-1]
            else:
                return jsonify({"error": "No active session or reference memory bank found. Build reference memory first."}), 400

        run_id = uuid.uuid4().hex[:12]
        temp_run_dir = ensure_dir(os.path.join(config.TEMP_FOLDER, "inference", session_id, run_id))

        if is_video:
            # ----------------------------------------------------------------
            # Video inspection path
            # ----------------------------------------------------------------
            video_enabled = getattr(config, "VIDEO_ENABLED", True)
            if not video_enabled:
                return jsonify({"error": "Video inspection is disabled."}), 400

            video_file = files[0]
            if not _allowed_video_file(video_file.filename):
                return jsonify({"error": "Unsupported video format. Accepted: mp4, mov, avi, mkv."}), 400

            video_filename = secure_filename(video_file.filename)
            video_save_path = os.path.join(temp_run_dir, video_filename)

            try:
                video_file.save(video_save_path)
                app_logger.info(
                    "Saved video upload: %s (session=%s run=%s)",
                    video_filename, session_id, run_id,
                )

                from modules.inference.inference_engine import InferenceEngine
                from modules.inference.video_inspector import VideoInspector

                app_logger.info("Initializing InferenceEngine for session %s", session_id)
                engine = InferenceEngine(session_id)

                inspector = VideoInspector(session_id=session_id, engine=engine)
                video_result = inspector.inspect_video(video_path=video_save_path, run_id=run_id)

                return jsonify({
                    "success": True,
                    "input_type": "video",
                    "session_id": session_id,
                    "run_id": run_id,
                    **video_result.to_dict(),
                })

            except Exception as exc:
                app_logger.error(
                    "Video inspection error for session %s: %s", session_id, exc, exc_info=True
                )
                return jsonify({"error": str(exc)}), 500
            finally:
                # Clean up the temp video file only; frame outputs live in inspection dir.
                try:
                    if os.path.isfile(video_save_path):
                        os.remove(video_save_path)
                    os.rmdir(temp_run_dir)
                except Exception:
                    pass

        else:
            # ----------------------------------------------------------------
            # Image inspection path (existing behaviour, unchanged)
            # ----------------------------------------------------------------
            valid_files = [f for f in files if f and _allowed_file(f.filename)]
            if not valid_files:
                return jsonify({"error": "No valid test image files found."}), 400

            if len(valid_files) > config.MAX_TEST_IMAGES:
                return jsonify({
                    "error": f"Too many test files. Maximum is {config.MAX_TEST_IMAGES} images."
                }), 400

            try:
                saved_paths = []
                for f in valid_files:
                    fname = secure_filename(f.filename)
                    save_path = os.path.join(temp_run_dir, fname)
                    f.save(save_path)
                    saved_paths.append(save_path)

                from modules.inference.inference_engine import InferenceEngine

                app_logger.info("Initializing InferenceEngine for session %s", session_id)
                engine = InferenceEngine(session_id)

                app_logger.info("Running inspection batch for %d images", len(saved_paths))
                results = engine.predict_batch(saved_paths)

                summary = engine.save_run(results)
                results_dict = [r.to_dict() for r in results]

                return jsonify({
                    "success": True,
                    "input_type": "image",
                    "session_id": session_id,
                    "run_id": run_id,
                    "summary": summary,
                    "results": results_dict,
                })

            except Exception as exc:
                app_logger.error(
                    "Inference batch error for session %s: %s", session_id, exc, exc_info=True
                )
                return jsonify({"error": str(exc)}), 500
            finally:
                try:
                    clear_dir(temp_run_dir)
                    os.rmdir(temp_run_dir)
                except Exception:
                    pass

    @app.route("/api/inference/<session_id>")
    def inference_info(session_id: str):
        """Return inference metadata/runs summary for a session."""
        session_inf_dir = os.path.join(config.INFERENCE_FOLDER, session_id)
        if not os.path.isdir(session_inf_dir):
            return jsonify({"session_id": session_id, "runs": []})

        runs = []
        for d in sorted(os.listdir(session_inf_dir)):
            summary_path = os.path.join(session_inf_dir, d, "inspection_summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as f:
                        runs.append(json.load(f))
                except Exception:
                    pass
        return jsonify({
            "session_id": session_id,
            "runs": sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)
        })

    @app.route("/api/inference/<session_id>/download")
    def download_inference_results(session_id: str):
        """Download results.json of the latest inference run for a session."""
        session_inf_dir = os.path.join(config.INFERENCE_FOLDER, session_id)
        if not os.path.isdir(session_inf_dir):
            return jsonify({"error": "No inference runs found for this session."}), 404

        # Find the directory with the latest summary
        latest_run_id = None
        latest_time = None
        for d in os.listdir(session_inf_dir):
            summary_path = os.path.join(session_inf_dir, d, "inspection_summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    ts = summary.get("timestamp", "")
                    if latest_time is None or ts > latest_time:
                        latest_time = ts
                        latest_run_id = d
                except Exception:
                    pass

        if not latest_run_id:
            return jsonify({"error": "No valid runs found."}), 404

        results_path = os.path.join(session_inf_dir, latest_run_id, "results.json")
        if not os.path.isfile(results_path):
            return jsonify({"error": "results.json not found for the latest run."}), 404

        return send_file(
            results_path,
            as_attachment=True,
            download_name=f"fewvision_inference_results_{session_id}_{latest_run_id}.json",
            mimetype="application/json",
        )

    @app.route("/api/inference/<session_id>/report")
    def download_inference_pdf(session_id: str):
        """Generate and download a detailed PDF inspection report."""
        session_inf_dir = os.path.join(config.INFERENCE_FOLDER, session_id)
        if not os.path.isdir(session_inf_dir):
            return jsonify({"error": "No inference runs found for this session."}), 404

        # Find latest run
        latest_run_id = None
        latest_time = None
        for d in os.listdir(session_inf_dir):
            summary_path = os.path.join(session_inf_dir, d, "inspection_summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    ts = summary.get("timestamp", "")
                    if latest_time is None or ts > latest_time:
                        latest_time = ts
                        latest_run_id = d
                except Exception:
                    pass

        if not latest_run_id:
            return jsonify({"error": "No valid runs found."}), 404

        run_dir = os.path.join(session_inf_dir, latest_run_id)
        results_path = os.path.join(run_dir, "results.json")
        summary_path = os.path.join(run_dir, "inspection_summary.json")

        if not os.path.isfile(results_path) or not os.path.isfile(summary_path):
            return jsonify({"error": "Run data not found."}), 404

        try:
            pdf_path = os.path.join(run_dir, "inspection_report.pdf")
            generate_pdf_report(results_path, summary_path, pdf_path)
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"fewvision_report_{session_id}_{latest_run_id}.pdf",
                mimetype="application/pdf",
            )
        except Exception as exc:
            logging.getLogger("fewvision.app").exception("PDF generation failed")
            return jsonify({"error": f"PDF generation failed: {exc}"}), 500

    # ---------------------------------------------------------------------------
    # PatchCore / Defect Localization API routes
    # ---------------------------------------------------------------------------

    @app.route("/inspection/<session_id>/<filename>")
    def serve_inspection_file(session_id: str, filename: str):
        """Serve inspection images, heatmaps, and overlays (flat, image inspection)."""
        directory = os.path.join(config.DATA_FOLDER, "inspection", session_id)
        if not os.path.isdir(directory):
            return jsonify({"error": "No inspection directory found"}), 404
        return send_file(os.path.join(directory, secure_filename(filename)))

    @app.route("/inspection/<session_id>/video/<run_id>/<subdir>/<filename>")
    def serve_video_inspection_file(session_id: str, run_id: str, subdir: str, filename: str):
        """Serve per-frame images, heatmaps, and overlays for video inspection runs.

        URL pattern mirrors the on-disk layout:
          data/inspection/{session_id}/video/{run_id}/{frames|heatmaps|overlays}/{filename}
        """
        allowed_subdirs = {"frames", "heatmaps", "overlays"}
        if subdir not in allowed_subdirs:
            return jsonify({"error": "Invalid subdir."}), 404

        directory = os.path.join(
            config.DATA_FOLDER, "inspection",
            session_id, "video", run_id, subdir
        )
        if not os.path.isdir(directory):
            return jsonify({"error": "Video inspection directory not found."}), 404

        file_path = os.path.join(directory, secure_filename(filename))
        if not os.path.isfile(file_path):
            return jsonify({"error": "File not found."}), 404

        return send_file(file_path)

    @app.route("/api/video-inspect/<session_id>/<run_id>")
    def video_inspect_info(session_id: str, run_id: str):
        """Return the full video inspection result JSON for a given run."""
        results_path = os.path.join(
            config.DATA_FOLDER, "inspection", session_id, "video", run_id, "results.json"
        )
        if not os.path.isfile(results_path):
            return jsonify({"error": "Video inspection results not found."}), 404

        with open(results_path) as f:
            return jsonify(json.load(f))

    @app.route("/api/patchcore/<session_id>")
    def patchcore_results(session_id: str):
        """Return PatchCore localization details for all inspected images in the latest run."""
        session_inf_dir = os.path.join(config.INFERENCE_FOLDER, session_id)
        if not os.path.isdir(session_inf_dir):
            return jsonify({"error": "No inference runs found for this session."}), 404

        # Find latest run
        latest_run_id = None
        latest_time = None
        for d in os.listdir(session_inf_dir):
            summary_path = os.path.join(session_inf_dir, d, "inspection_summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    ts = summary.get("timestamp", "")
                    if latest_time is None or ts > latest_time:
                        latest_time = ts
                        latest_run_id = d
                except Exception:
                    pass

        if not latest_run_id:
            return jsonify({"error": "No valid runs found."}), 404

        results_path = os.path.join(session_inf_dir, latest_run_id, "results.json")
        if not os.path.isfile(results_path):
            return jsonify({"error": "results.json not found for the latest run."}), 404

        with open(results_path) as f:
            results = json.load(f)

        # Formulate output keyed by image name
        patchcore_data = {}
        for r in results:
            if r.get("patchcore_enabled", False):
                patchcore_data[r["image_name"]] = {
                    "heatmap_url": r.get("heatmap_url", ""),
                    "overlay_url": r.get("overlay_url", ""),
                    "original_url": r.get("original_url", ""),
                    "bounding_box": r.get("bounding_box", []),
                    "area_percent": r.get("anomaly_area_percent", 0.0),
                    "max_score": r.get("max_patch_score", 0.0),
                    "centroid": r.get("centroid", []),
                    "top_5_patch_matches": r.get("top_5_patch_matches", [])
                }

        return jsonify(patchcore_data)

    @app.route("/api/inspection/<session_id>/<image_name>")
    def inspection_details(session_id: str, image_name: str):
        """Return the complete inspection JSON for a specific image in the latest run."""
        session_inf_dir = os.path.join(config.INFERENCE_FOLDER, session_id)
        if not os.path.isdir(session_inf_dir):
            return jsonify({"error": "No inference runs found for this session."}), 404

        # Find latest run
        latest_run_id = None
        latest_time = None
        for d in os.listdir(session_inf_dir):
            summary_path = os.path.join(session_inf_dir, d, "inspection_summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    ts = summary.get("timestamp", "")
                    if latest_time is None or ts > latest_time:
                        latest_time = ts
                        latest_run_id = d
                except Exception:
                    pass

        if not latest_run_id:
            return jsonify({"error": "No valid runs found."}), 404

        results_path = os.path.join(session_inf_dir, latest_run_id, "results.json")
        if not os.path.isfile(results_path):
            return jsonify({"error": "results.json not found for the latest run."}), 404

        with open(results_path) as f:
            results = json.load(f)

        for r in results:
            if r["image_name"] == image_name:
                return jsonify(r)

        return jsonify({"error": f"Image '{image_name}' not found in latest run."}), 404

    # ---------------------------------------------------------------------------
    # Live Webcam Inspection API routes
    # ---------------------------------------------------------------------------
    from modules.webcam import (
        WebcamManager,
        list_available_cameras,
        LiveWebcamInspector,
        FeedbackStore,
    )

    webcam_state = {
        "manager": None,
        "inspector": None,
        "feedback_store": FeedbackStore(),
        "stream_thread": None,
        "stream_stop_event": threading.Event(),
        "active_roi": None,
        "auto_roi": False,
        "active_session_id": None,
    }

    def _get_or_init_inspector(sid: str) -> LiveWebcamInspector:
        """Reuse existing LiveWebcamInspector for session or initialize once."""
        if (
            webcam_state["inspector"] is not None
            and webcam_state["inspector"].session_id == sid
        ):
            return webcam_state["inspector"]

        app_logger.info("Initializing LiveWebcamInspector for session %s (Single-Load)", sid)
        inspector = LiveWebcamInspector(sid)
        webcam_state["inspector"] = inspector
        webcam_state["active_session_id"] = sid
        return inspector

    def _webcam_background_inference_loop():
        """Background worker running model inference on latest camera frames for MJPEG stream."""
        app_logger.info("Webcam background inference stream worker started.")
        mgr = webcam_state["manager"]
        while not webcam_state["stream_stop_event"].is_set():
            if mgr is None or not mgr.is_running:
                break
            if mgr.is_paused:
                time.sleep(0.05)
                continue

            success, frame, ts = mgr.get_latest_frame()
            if success and frame is not None and frame.size > 0:
                inspector = webcam_state["inspector"]
                if inspector is not None:
                    try:
                        res = inspector.inspect_frame(
                            frame,
                            roi_spec=webcam_state["active_roi"],
                            auto_roi=webcam_state["auto_roi"],
                        )
                        mgr.update_latest_result(res["_annotated_frame"], res)
                    except Exception as err:
                        app_logger.error("Error in webcam background inference: %s", err)

            # Limit inference rate to avoid burning CPU unnecessarily
            time.sleep(0.04)

        app_logger.info("Webcam background inference stream worker stopped.")

    @app.route("/api/webcam/status")
    def webcam_status():
        """Return status of webcam capture and active model inspector."""
        mgr = webcam_state["manager"]
        mgr_status = mgr.get_status() if mgr is not None else {"is_running": False}

        inspector = webcam_state["inspector"]
        insp_status = {
            "has_inspector": inspector is not None,
            "session_id": webcam_state["active_session_id"],
            "device": getattr(inspector, "device", "CPU") if inspector else "CPU",
            "threshold": inspector.patch_threshold if inspector else config.PATCH_THRESHOLD,
            "total_inspections": inspector.total_inspections if inspector else 0,
            "total_anomalies": inspector.total_anomalies if inspector else 0,
        }

        # Check memory bank readiness
        from modules.anomaly_detection.memory_bank import list_memory_bank_sessions
        available_sessions = list_memory_bank_sessions()

        return jsonify({
            "success": True,
            "camera": mgr_status,
            "inspector": insp_status,
            "available_sessions": available_sessions,
            "active_session": webcam_state["active_session_id"] or session.get("session_id"),
        })

    @app.route("/api/webcam/cameras")
    def webcam_cameras():
        """List physical cameras detected on the host system."""
        try:
            cameras = list_available_cameras(max_probe=4)
            return jsonify({"success": True, "cameras": cameras})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc), "cameras": []}), 500

    @app.route("/api/webcam/start", methods=["POST"])
    def webcam_start():
        """Initialize and start live webcam inspection."""
        data = request.get_json(silent=True) or request.form.to_dict()
        session_id = data.get("session_id") or session.get("session_id")

        if not session_id:
            from modules.anomaly_detection.memory_bank import list_memory_bank_sessions
            sessions = list_memory_bank_sessions()
            if sessions:
                session_id = sessions[-1]
            else:
                return jsonify({
                    "success": False,
                    "error": "No reference memory bank found. Please build reference memory first.",
                }), 400

        camera_index = int(data.get("camera_index", getattr(config, "CAMERA_INDEX", 0)))
        mode = data.get("mode", "hardware").lower()  # "hardware" or "simulated"
        width = int(data.get("width", getattr(config, "FRAME_WIDTH", 1280)))
        height = int(data.get("height", getattr(config, "FRAME_HEIGHT", 720)))
        target_fps = int(data.get("target_fps", getattr(config, "TARGET_FPS", 15)))

        roi = data.get("roi")
        if roi and isinstance(roi, list) and len(roi) == 4:
            webcam_state["active_roi"] = [float(x) for x in roi]
        else:
            webcam_state["active_roi"] = None

        webcam_state["auto_roi"] = str(data.get("auto_roi", "false")).lower() == "true"

        # 1. Initialize Inspector (Model loaded once)
        try:
            _get_or_init_inspector(session_id)
        except Exception as exc:
            app_logger.error("Failed to initialize LiveWebcamInspector: %s", exc)
            return jsonify({
                "success": False,
                "error": f"Failed to initialize anomaly model: {exc}",
            }), 500

        # 2. Stop any existing webcam manager
        if webcam_state["manager"] is not None:
            webcam_state["stream_stop_event"].set()
            if webcam_state["stream_thread"] is not None:
                webcam_state["stream_thread"].join(timeout=1.5)
            webcam_state["manager"].stop()

        # 3. Create and start new WebcamManager
        mgr = WebcamManager(
            camera_index=camera_index,
            width=width,
            height=height,
            target_fps=target_fps,
            mode=mode,
            simulated_folder=getattr(config, "TEST_STREAM_FOLDER", None),
        )

        success = mgr.start()
        if not success:
            return jsonify({
                "success": False,
                "error": mgr.error_message or "Failed to start camera.",
            }), 500

        webcam_state["manager"] = mgr
        webcam_state["stream_stop_event"].clear()

        # 4. Launch background inference thread for MJPEG stream
        stream_thread = threading.Thread(
            target=_webcam_background_inference_loop,
            name="WebcamBackgroundInference",
            daemon=True,
        )
        stream_thread.start()
        webcam_state["stream_thread"] = stream_thread

        return jsonify({
            "success": True,
            "session_id": session_id,
            "camera_status": mgr.get_status(),
        })

    @app.route("/api/webcam/stop", methods=["POST"])
    def webcam_stop():
        """Stop camera capture and background stream."""
        webcam_state["stream_stop_event"].set()
        if webcam_state["stream_thread"] is not None:
            webcam_state["stream_thread"].join(timeout=1.5)
            webcam_state["stream_thread"] = None

        if webcam_state["manager"] is not None:
            webcam_state["manager"].stop()
            webcam_state["manager"] = None

        return jsonify({"success": True, "message": "Camera stopped successfully."})

    @app.route("/api/webcam/pause", methods=["POST"])
    def webcam_pause():
        """Toggle pause state."""
        if webcam_state["manager"] is not None:
            is_paused = webcam_state["manager"].pause()
            return jsonify({"success": True, "is_paused": is_paused})
        return jsonify({"success": False, "error": "Camera is not running."}), 400

    @app.route("/api/webcam/stream")
    def webcam_stream():
        """MJPEG video stream showing real-time inspection visualization."""
        def generate_frames():
            while True:
                mgr = webcam_state["manager"]
                if mgr is None or not mgr.is_running:
                    time.sleep(0.1)
                    continue

                frame = mgr.get_latest_annotated_frame()
                if frame is not None and frame.size > 0:
                    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
                time.sleep(0.04)

        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/webcam/inspect_frame", methods=["POST"])
    def webcam_inspect_frame():
        """Accept a frame from the browser (e.g. via getUserMedia) or client,
        run live inspection, and return full telemetry + base64 overlay.
        """
        # Determine session ID
        session_id = request.form.get("session_id")
        if not session_id and request.is_json:
            session_id = request.json.get("session_id")
        if not session_id:
            session_id = session.get("session_id")
            if not session_id:
                from modules.anomaly_detection.memory_bank import list_memory_bank_sessions
                sessions = list_memory_bank_sessions()
                if sessions:
                    session_id = sessions[-1]
                else:
                    return jsonify({"error": "No reference memory bank found."}), 400

        # Resolve or initialize inspector (model loaded once)
        try:
            inspector = _get_or_init_inspector(session_id)
        except Exception as exc:
            return jsonify({"error": f"Failed to load anomaly model: {exc}"}), 500

        # Decode frame
        img_bgr = None
        if "frame" in request.files:
            file = request.files["frame"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif request.is_json and "frame" in request.json:
            b64_data = request.json["frame"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            try:
                decoded = base64.b64decode(b64_data)
                img_bgr = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
            except Exception as exc:
                return jsonify({"error": f"Invalid base64 frame: {exc}"}), 400
        elif "frame_base64" in request.form:
            b64_data = request.form["frame_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            try:
                decoded = base64.b64decode(b64_data)
                img_bgr = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
            except Exception as exc:
                return jsonify({"error": f"Invalid base64 frame: {exc}"}), 400

        if img_bgr is None or img_bgr.size == 0:
            return jsonify({"error": "No valid image frame received."}), 400

        # Dynamic ROI parameters
        roi = None
        if request.is_json and "roi" in request.json:
            roi = request.json["roi"]
        elif "roi" in request.form:
            try:
                roi = json.loads(request.form["roi"])
            except Exception:
                pass

        auto_roi = False
        if request.is_json:
            auto_roi = bool(request.json.get("auto_roi", False))
        elif "auto_roi" in request.form:
            auto_roi = request.form.get("auto_roi", "false").lower() == "true"

        # Dynamic threshold override if requested
        if request.is_json and "threshold" in request.json:
            try:
                inspector.set_threshold(float(request.json["threshold"]))
            except Exception:
                pass
        elif "threshold" in request.form:
            try:
                inspector.set_threshold(float(request.form["threshold"]))
            except Exception:
                pass

        # Run inspection
        try:
            res = inspector.inspect_frame(img_bgr, roi_spec=roi, auto_roi=auto_roi)

            # Encode annotated frame as base64 JPEG
            annotated_bgr = res.get("_annotated_frame")
            overlay_b64 = ""
            if annotated_bgr is not None:
                _, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            # Encode heatmap ROI as base64 JPEG
            heatmap_roi = res.get("_heatmap_roi")
            heatmap_b64 = ""
            if heatmap_roi is not None:
                _, h_buf = cv2.imencode(".jpg", heatmap_roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
                heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(h_buf).decode("utf-8")

            # Clean output dict
            output = {}
            for k, v in res.items():
                if not k.startswith("_") and not isinstance(v, np.ndarray):
                    output[k] = v

            output["overlay_base64"] = overlay_b64
            output["heatmap_base64"] = heatmap_b64
            output["success"] = True

            return jsonify(output)

        except Exception as exc:
            app_logger.error("Live inspection error: %s", exc, exc_info=True)
            return jsonify({"error": str(exc), "success": False}), 500

    @app.route("/api/webcam/capture", methods=["POST"])
    def webcam_capture():
        """Save a snapshot of the current frame, overlay, heatmap, and metadata JSON."""
        data = request.get_json(silent=True) or request.form.to_dict()

        raw_frame = None
        annotated_frame = None
        heatmap_frame = None

        # Check if frames were transmitted from browser
        if "raw_frame" in data:
            raw_b64 = data["raw_frame"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            raw_frame = cv2.imdecode(np.frombuffer(base64.b64decode(raw_b64), np.uint8), cv2.IMREAD_COLOR)

        if "annotated_frame" in data:
            ann_b64 = data["annotated_frame"]
            if "," in ann_b64:
                ann_b64 = ann_b64.split(",", 1)[1]
            annotated_frame = cv2.imdecode(np.frombuffer(base64.b64decode(ann_b64), np.uint8), cv2.IMREAD_COLOR)

        if "heatmap_frame" in data:
            hm_b64 = data["heatmap_frame"]
            if "," in hm_b64:
                hm_b64 = hm_b64.split(",", 1)[1]
            heatmap_frame = cv2.imdecode(np.frombuffer(base64.b64decode(hm_b64), np.uint8), cv2.IMREAD_COLOR)

        # Fallback to server manager if available
        mgr = webcam_state["manager"]
        if raw_frame is None and mgr is not None:
            _, raw_frame, _ = mgr.get_latest_frame()
        if annotated_frame is None and mgr is not None:
            annotated_frame = mgr.get_latest_annotated_frame()

        if raw_frame is None:
            return jsonify({"success": False, "error": "No frame available to capture."}), 400
        if annotated_frame is None:
            annotated_frame = raw_frame.copy()

        result_meta = data.get("result_meta") or (mgr.get_latest_result() if mgr else {}) or {}

        store = webcam_state["feedback_store"]
        cap_res = store.save_capture(
            raw_frame=raw_frame,
            annotated_frame=annotated_frame,
            heatmap_frame=heatmap_frame,
            result_meta=result_meta,
        )

        return jsonify(cap_res)

    @app.route("/api/webcam/review", methods=["POST"])
    def webcam_review():
        """Save human expert feedback and store labeled sample into feedback dataset."""
        data = request.get_json(silent=True) or request.form.to_dict()

        expert_label = data.get("expert_label", "UNCERTAIN")
        model_prediction = data.get("model_prediction", "UNKNOWN")
        model_score = float(data.get("model_score", 0.0))
        notes = data.get("notes", "")
        quality_score = float(data.get("quality_score", 0.0))
        capture_id = data.get("capture_id")

        frame_bgr = None
        if "frame" in data:
            f_b64 = data["frame"]
            if "," in f_b64:
                f_b64 = f_b64.split(",", 1)[1]
            frame_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(f_b64), np.uint8), cv2.IMREAD_COLOR)

        # Fallback to server manager
        mgr = webcam_state["manager"]
        if frame_bgr is None and mgr is not None:
            _, frame_bgr, _ = mgr.get_latest_frame()

        if frame_bgr is None:
            # Check if capture_id has raw.jpg
            if capture_id:
                for d in os.listdir(webcam_state["feedback_store"].captures_dir):
                    if capture_id in d:
                        raw_file = os.path.join(webcam_state["feedback_store"].captures_dir, d, "raw.jpg")
                        if os.path.isfile(raw_file):
                            frame_bgr = cv2.imread(raw_file)
                            break

        if frame_bgr is None:
            return jsonify({"success": False, "error": "No frame image provided for review."}), 400

        store = webcam_state["feedback_store"]
        review_res = store.record_expert_review(
            frame_bgr=frame_bgr,
            model_prediction=model_prediction,
            model_score=model_score,
            expert_label=expert_label,
            notes=notes,
            quality_score=quality_score,
            capture_id=capture_id,
        )

        return jsonify(review_res)

    @app.route("/api/webcam/history")
    def webcam_history():
        """Retrieve recent inspection history."""
        store = webcam_state["feedback_store"]
        limit = int(request.args.get("limit", 50))
        return jsonify({"success": True, "history": store.get_history(limit=limit)})

    @app.route("/api/webcam/history/clear", methods=["POST"])
    def webcam_history_clear():
        """Clear the inspection history list."""
        store = webcam_state["feedback_store"]
        store.clear_history()
        return jsonify({"success": True, "message": "History cleared."})

    @app.route("/api/webcam/history/export")
    def webcam_history_export():
        """Download inspection history as a CSV file."""
        store = webcam_state["feedback_store"]
        csv_data = store.export_csv()
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-disposition": f"attachment; filename=fewvision_webcam_history_{now_str}.csv"
            },
        )

    @app.route("/api/webcam/threshold", methods=["POST"])
    def webcam_set_threshold():
        """Dynamically update the patch anomaly threshold."""
        data = request.get_json(silent=True) or request.form.to_dict()
        try:
            new_thresh = float(data.get("threshold", 0.5))
            if webcam_state["inspector"] is not None:
                webcam_state["inspector"].set_threshold(new_thresh)
            return jsonify({"success": True, "threshold": new_thresh})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/captures/<capture_id>/<filename>")
    def serve_capture_file(capture_id: str, filename: str):
        """Serve saved capture files (raw.jpg, annotated.jpg, heatmap.jpg, result.json)."""
        safe_id = secure_filename(capture_id)
        safe_file = secure_filename(filename)
        directory = os.path.join(config.CAPTURES_FOLDER, safe_id)
        file_path = os.path.join(directory, safe_file)
        if not os.path.isfile(file_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(file_path)

    @app.route("/feedback/<category>/<filename>")
    def serve_feedback_file(category: str, filename: str):
        """Serve saved human feedback images."""
        if category not in {"normal", "anomaly", "uncertain"}:
            return jsonify({"error": "Invalid category"}), 404
        directory = os.path.join(config.FEEDBACK_FOLDER, category)
        file_path = os.path.join(directory, secure_filename(filename))
        if not os.path.isfile(file_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(file_path)

    return app



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG, port=config.PORT)
