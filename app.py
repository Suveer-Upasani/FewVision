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

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
    redirect,
    url_for,
    session,
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
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Ensure all data directories exist at startup
    ensure_dir(config.UPLOAD_FOLDER)
    ensure_dir(config.AUGMENTED_FOLDER)
    ensure_dir(config.REPORTS_FOLDER)
    ensure_dir(config.LOGS_FOLDER)
    ensure_dir(config.TEMP_FOLDER)
    ensure_dir(config.MEMORY_BANK_FOLDER)
    ensure_dir(config.INFERENCE_FOLDER)

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
        """Accept test images, run the inference engine, return results."""
        if not config.ENABLE_INFERENCE:
            return jsonify({"error": "Inference pipeline is disabled."}), 400

        if "files" not in request.files:
            return jsonify({"error": "No test files uploaded."}), 400

        files = request.files.getlist("files")
        valid_files = [f for f in files if f and _allowed_file(f.filename)]
        if not valid_files:
            return jsonify({"error": "No valid test image files found."}), 400

        if len(valid_files) > config.MAX_TEST_IMAGES:
            return jsonify({
                "error": f"Too many test files. Maximum is {config.MAX_TEST_IMAGES} images."
            }), 400

        # Get active session_id
        session_id = request.form.get("session_id") or session.get("session_id")
        if not session_id:
            from modules.anomaly_detection.memory_bank import list_memory_bank_sessions
            sessions = list_memory_bank_sessions()
            if sessions:
                session_id = sessions[-1]
            else:
                return jsonify({"error": "No active session or reference memory bank found. Build reference memory first."}), 400

        from modules.inference.inference_engine import InferenceEngine

        # Setup unique temp subdirectory for this run
        run_id = uuid.uuid4().hex[:12]
        temp_run_dir = ensure_dir(os.path.join(config.TEMP_FOLDER, "inference", session_id, run_id))

        try:
            # Save uploaded test files temporarily
            saved_paths = []
            for f in valid_files:
                fname = secure_filename(f.filename)
                save_path = os.path.join(temp_run_dir, fname)
                f.save(save_path)
                saved_paths.append(save_path)

            app_logger.info("Initializing InferenceEngine for session %s", session_id)
            engine = InferenceEngine(session_id)

            app_logger.info("Running inspection batch for %d images", len(saved_paths))
            results = engine.predict_batch(saved_paths)

            # Save the run under data/inference/{session_id}/{run_id}/
            summary = engine.save_run(results)

            # Prepare list of dict results to return
            results_dict = [r.to_dict() for r in results]

            return jsonify({
                "success": True,
                "session_id": session_id,
                "run_id": run_id,
                "summary": summary,
                "results": results_dict
            })

        except Exception as exc:
            app_logger.error("Inference batch error for session %s: %s", session_id, exc, exc_info=True)
            return jsonify({"error": str(exc)}), 500
        finally:
            # Cleanup temp run directory
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
        """Serve inspection images, heatmaps, and overlays."""
        directory = os.path.join(config.DATA_FOLDER, "inspection", session_id)
        if not os.path.isdir(directory):
            return jsonify({"error": "No inspection directory found"}), 404
        return send_file(os.path.join(directory, secure_filename(filename)))

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

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG, port=config.PORT)
