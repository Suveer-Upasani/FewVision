# app.py
"""FewVision Flask Application.

This module contains **only** Flask routes and application bootstrap code.
All image processing, augmentation, and reporting logic is delegated to
the pipeline module.

Routes
------
GET  /                          Upload page
POST /api/upload                Accept images and run pipeline
GET  /dashboard/<session_id>    Analysis dashboard
POST /api/generate/<session_id> Generate augmented dataset ZIP
GET  /api/download/<session_id> Stream ZIP file download
GET  /reports/<session_id>/<filename> Serve report images
"""

import os
import logging

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
        return render_template("index.html")

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

            # Run the full pipeline
            app_logger.info("Starting pipeline for session %s (%d images)", sid, len(valid_files))
            dataset_result = process_dataset(upload_dir, session_id=sid)

            # Serialise results for the session store
            session["session_id"] = sid
            session["total_images"] = dataset_result.total_images
            session["augmented_count"] = dataset_result.augmented_count
            session["ready_count"] = dataset_result.ready_count
            session["marginal_count"] = dataset_result.marginal_count
            session["unsuitable_count"] = dataset_result.unsuitable_count
            session["report_dir"] = dataset_result.report_dir

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
        images_data = session.get("images_data", [])
        summary = {
            "session_id": session.get("session_id", session_id),
            "total_images": session.get("total_images", 0),
            "ready_count": session.get("ready_count", 0),
            "marginal_count": session.get("marginal_count", 0),
            "unsuitable_count": session.get("unsuitable_count", 0),
        }
        return render_template(
            "dashboard.html",
            images=images_data,
            summary=summary,
            session_id=session_id,
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

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG, port=config.PORT)
