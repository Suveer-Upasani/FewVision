# FewVision

**Adaptive Quality-Aware Few-Shot Anomaly Detection for Industrial Part Inspection**

FewVision is a production-grade Flask application that automates the preparation of small image datasets (5–20 normal product images) and testing of new product items for few-shot industrial anomaly detection.

---

## Full Pipelines

### 1. Reference Pipeline (Build Reference Memory)

```
Reference Images
       │
       ▼
Quality Assessment
       │
       ▼
Content Analysis
       │
       ▼
Adaptive Augmentation
       │
       ▼
Reference Dataset  ←  augmented images saved to data/augmented/
       │
       ▼
DINOv2 Feature Extraction
       │
       ▼
Patch Embeddings  ←  196 patch embeddings per image (14x14 grid)
       │
       ▼
Patch Memory Bank  ←  saved to data/memory_bank/{session_id}/patchcore/
```

### 2. Inspection Pipeline (Inspect Product)

```
Test Image
       │
       ▼
Patch Embeddings  ←  196 local patch embeddings (14x14 grid)
       │
       ▼
Patch Similarity Search  ←  Cosine / Euclidean matches against Memory Bank
       │
       ▼
Distance Map  ←  reshaped to 14x14 grid, upscaled to original resolution
       │
       ▼
Heatmap  ←  JET colormap visualization overlay
       │
       ▼
Defect Localization  ←  thresholding, contour extraction, bounding boxes, centroid
       │
       ▼
Anomaly Score  ←  composite image-level scoring
       │
       ▼
Inspection Dashboard  ←  interactive Glassmorphism UI
```

---

## Project Structure

```
FewVision/
├── app.py              # Flask routes and application bootstrap
├── config.py           # All configuration constants + env-var overrides
├── requirements.txt
├── README.md
│
├── modules/
│   ├── quality/            # Blur, brightness, contrast, noise, resolution
│   ├── content/            # Background, lighting, object coverage, orientation
│   ├── augmentation/       # Adaptive policy + Albumentations engine
│   ├── reporting/          # Per-image PNG reports + dataset analytics
│   ├── pipeline/           # Orchestrator — coordinates reference pipeline stages
│   ├── feature_extraction/ # Embedding extraction modules
│   ├── anomaly_detection/  # Memory Bank, Similarity Engine, Anomaly Scoring
│   ├── patchcore/          # PatchCore Anomaly Localization (NEW)
│   │   ├── __init__.py            # Module exports
│   │   ├── patch_extractor.py     # DINOv2 196-patch token extraction (14x14 grid)
│   │   ├── patch_memory_bank.py   # Normalized patch memory bank (npy + metadata)
│   │   ├── patch_similarity.py    # Cosine/Euclidean vectorized nearest neighbors
│   │   ├── heatmap.py             # Upscaled colormap (JET) overlay generation
│   │   └── localization.py        # Contour-based bounding boxes & centroid localization
│   ├── inference/          # Inference pipeline modules
│   └── utils/              # Dataclasses, image helpers, file helpers
│
├── models/             # Future: Prototypical Networks, Siamese Networks
├── templates/          # index.html, dashboard.html, results.html
├── static/             # CSS design system + JS modules
├── data/               # Runtime data (gitignored)
│   ├── uploads/
│   ├── augmented/
│   ├── reports/
│   ├── embeddings/     # Embedding databases per session
│   ├── memory_bank/    # Memory Bank per session
│   ├── inference/      # Inspection run results per session & run
│   ├── inspection/     # Heatmaps and overlays per session (NEW)
│   ├── logs/
│   └── temp/
```

---

## PatchCore Architecture

Instead of extracting one global embedding per image, FewVision now extracts **local patch embeddings** from every image.
- **Reference Generation**: We preprocess reference images to `196x196` pixels. Using the DINOv2 backbone with a patch size of 14, we extract a grid of `14x14 = 196` local patch embeddings per image. A dataset of 50 augmented reference images generates a reference bank of `9,800` patches stored in `memory.npy` alongside details in `patch_metadata.json`.
- **Similarity Search**: During inspection, the test image is converted to 196 patch embeddings. We query the reference database using optimized vector search (Cosine or Euclidean) to find the nearest reference neighbor for each patch.
- **Distance Map & Heatmap**: Patch distances are reshaped to a `14x14` grid and upscaled to the original image resolution using bicubic interpolation. We normalize the scores, apply a `cv2.COLORMAP_JET` colormap, and blend it with the original image using `cv2.addWeighted` to generate a premium overlay.
- **Defect Localization**: The upscaled distance map is thresholded at `config.PATCH_THRESHOLD`. We extract contours, compute bounding boxes, calculate the percentage of anomalous area, and identify the centroid of the largest defect.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/FewVision.git
cd FewVision
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run

```bash
python app.py
```

Open **http://localhost:5005** in your browser.

---

## Configuration

All settings are in [`config.py`](config.py). Override via environment variables:

| Variable | Default | Description |
|---|---|---|
| `FEWVISION_PATCHCORE_ENABLED` | `true` | Enable PatchCore localized defect detection |
| `FEWVISION_PATCH_SIZE` | `14` | Transformer patch size |
| `FEWVISION_PATCH_SIMILARITY` | `cosine` | Similarity metric used for patches (`cosine`, `euclidean`) |
| `FEWVISION_PATCH_THRESHOLD` | `0.5` | Defect mask binary threshold |
| `FEWVISION_HEATMAP_ALPHA` | `0.6` | Heatmap overlay transparency |
| `FEWVISION_SAVE_HEATMAPS` | `true` | Save heatmaps and overlays to disk |
| `FEWVISION_EXTRACTOR` | `dinov2` | Active feature extractor (`dinov2`) |

---

## REST API

### Reference Pipeline & Memory Bank
```bash
POST /api/upload                     # Upload reference images and run pipeline
POST /api/generate/<session_id>      # Generate augmented dataset ZIP
GET  /api/download/<session_id>      # Download augmented dataset ZIP
GET  /api/embeddings/<session_id>    # Retrieve embedding session summary
GET  /api/memory-bank/<session_id>   # Retrieve memory bank summary
```

### Inspection / Inference
```bash
POST /api/inspect                    # Run inspection on uploaded test image(s)
GET  /api/inference/<session_id>     # Retrieve all inference runs for a session
GET  /api/inference/<session_id>/download # Download results.json of the latest run
```

### PatchCore Localization (NEW)
```bash
# Retrieve PatchCore localization details for all inspected images in the latest run
GET  /api/patchcore/<session_id>

# Retrieve complete inspection JSON for a specific test image
GET  /api/inspection/<session_id>/<image_name>

# Serve original image, heatmap, and overlay files
GET  /inspection/<session_id>/<filename>
```

---

## Data Directory Layout

### 1. Preprocessing & Reference Memory
- `data/uploads/{session_id}/` — Reference upload images.
- `data/augmented/{session_id}/` — Augmented reference images.
- `data/embeddings/{session_id}/` — Image-level embeddings database.
- `data/memory_bank/{session_id}/patchcore/` — Patch-level memory bank (`memory.npy`, `patch_metadata.json`).

### 2. Inspection Outputs
- `data/inspection/{session_id}/` — Generated `{image_stem}_original.png`, `{image_stem}_heatmap.png`, and `{image_stem}_overlay.png`.
- `data/inference/{session_id}/{run_id}/` — Inspection summary and report files.

---

## Roadmap

| Stage | Status |
|---|---|
| Image Quality Assessment | ✅ Complete |
| Content Analysis | ✅ Complete |
| Adaptive Augmentation | ✅ Complete |
| Feature Extraction (DINOv2 / ViT) | ✅ Complete |
| Memory Bank Setup | ✅ Complete |
| Similarity Search Engine | ✅ Complete |
| Inference & Inspection Pipeline | ✅ Complete |
| PatchCore Anomaly Localization | ✅ Complete |
| Heatmap Visualization | ✅ Complete |
| PaDiM Anomaly Localization | ✅ Complete |
| Product Grading | ✅ Complete |
| Production Live Webcam Inspection | ✅ Complete |

---

## Live Webcam Inspection Mode

FewVision includes a production-grade **Live Webcam Inspection Studio** designed for real-time industrial part quality assurance on continuous manufacturing conveyor belts and benchtop inspection stations.

```mermaid
flowchart TD
    subgraph Offline_Reference_Setup [1. Few-Shot Reference Memory Setup]
        RefImgs[5-20 Normal Industrial Part Images] --> DINO_Ref[DINOv2 Backbone (Single Load)]
        DINO_Ref --> Patch_Embed[Patch-Level Embeddings (14x14 Grid)]
        Patch_Embed --> MemBank[(PatchCore Memory Bank memory.npy)]
    end

    subgraph Live_Webcam_Inference [2. Real-Time Inspection Pipeline]
        Cam[Live Camera Feed / Browser WebRTC / Simulated Stream] --> Quality[Image Quality Assessment (Blur, Exposure, Noise)]
        Quality -- BAD Quality --> Warn["IMAGE QUALITY TOO LOW (Suppress Decision)"]
        Quality -- GOOD / WARNING --> ROI[ROI Cropping (Center 80% / Full / Auto)]
        ROI --> DINO_Live[DINOv2 Feature Extractor (Zero-Reload)]
        DINO_Live --> Live_Patches[Live Patch Embeddings]
        Live_Patches & MemBank --> NN_Search[Vectorized Nearest-Neighbor Search]
        NN_Search --> DistMap[Patch Anomaly Distance Map]
        DistMap --> RawScore[Raw Anomaly Score & Localization]
        RawScore --> Temporal[Temporal Stabilization (Moving Average + Anomaly Voting)]
        Temporal --> Heatmap[JET Colormap Upscaling & Blended Overlay]
        Heatmap --> LiveHUD[Live Industrial HUD & Telemetry Dashboard]
    end

    subgraph Human_In_The_Loop [3. Expert Validation & Feedback Dataset]
        LiveHUD --> Snapshot[Capture Snapshot (raw, annotated, heatmap, json)]
        LiveHUD --> Review{Human Expert Review}
        Review -- Confirm Anomaly / False Negative --> Feedback_Anomaly[data/feedback/anomaly/]
        Review -- Mark Normal / False Positive --> Feedback_Normal[data/feedback/normal/]
        Review -- Uncertain --> Feedback_Uncertain[data/feedback/uncertain/]
        Feedback_Anomaly & Feedback_Normal --> Retrain[Future Model Calibration & Retraining]
    end
```

### 1. Key Capabilities
- **Single-Load Architecture**: The DINOv2 vision transformer and PatchCore memory bank are loaded **once** upon session activation. Zero per-frame memory leaks or model reloads.
- **Dual Camera Operating Modes**:
  1. **Browser Camera (WebRTC `getUserMedia`)**: Runs in the client browser with camera permission prompts and controlled frame sampling (dispatches the next frame only after the backend finishes inference, preventing network congestion).
  2. **Server Camera (OpenCV Hardware / MJPEG)**: Runs on the server host via direct DirectShow (`cv2.CAP_DSHOW` on Windows / V4L2 on Linux) with background buffer flushing.
  3. **Simulated Live Stream**: Simulates a live camera feed using any static dataset folder (e.g. `Tiles_dataset/test/cracks/`) at configurable target FPS for testing and CI/CD without physical camera hardware.
- **In-Memory Frame Quality Gate**: Rejects or flags dark (<30 brightness), overexposed (>220 brightness), low contrast (<20), or severely blurred frames (`cv2.Laplacian < 80.0`). BAD frames immediately show **`IMAGE QUALITY TOO LOW`** and suppress false defect decisions.
- **Industrial Part ROI Selection**: Automatically crops the active inspection region (`center` 80% box, `full` frame, or `auto` morphological part contour detection) to eliminate noisy factory background elements.
- **Temporal Stabilization**: Moving window average (default `N=5`) and anomaly voting ratio (default `threshold=0.6`) eliminate sensor flicker and transient lighting spikes.
- **Real-Time HUD Overlay**: Renders defect localization bounding boxes, centroid coordinates, defect area percentage, measured FPS, and per-stage latency breakdown directly onto the video feed.
- **Human-In-The-Loop Expert Review**: Operators can record expert verdicts (`Confirm Anomaly`, `Mark Normal`, `False Positive`, `False Negative`, `Uncertain`). Images and metadata are segregated into partitioned feedback directories (`data/feedback/{normal,anomaly,uncertain}`) for future model calibration.
- **Snapshot Capture & History Export**: Single-click snapshot saves raw frame (`raw.jpg`), annotated composite (`annotated.jpg`), isolated heatmap (`heatmap.jpg`), and JSON metadata (`result.json`). Inspection history can be browsed live or exported as CSV.

---

### 2. Configuration Settings

Webcam parameters in [`config.py`](config.py):

| Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| `WEBCAM_ENABLED` | `FEWVISION_WEBCAM_ENABLED` | `true` | Master toggle for webcam studio |
| `CAMERA_INDEX` | `FEWVISION_CAMERA_INDEX` | `0` | Default hardware camera index (`0`, `1`, `2`...) |
| `FRAME_WIDTH` | `FEWVISION_FRAME_WIDTH` | `1280` | Requested capture width |
| `FRAME_HEIGHT` | `FEWVISION_FRAME_HEIGHT` | `720` | Requested capture height |
| `TARGET_FPS` | `FEWVISION_TARGET_FPS` | `15` | Target capture rate |
| `TEMPORAL_WINDOW` | `FEWVISION_TEMPORAL_WINDOW` | `5` | Moving window size for temporal stabilization |
| `ANOMALY_VOTE_THRESHOLD` | `FEWVISION_VOTE_THRESHOLD` | `0.6` | Ratio of anomalous frames required in window for verdict |
| `FEEDBACK_FOLDER` | — | `data/feedback` | Root directory for human expert reviewed datasets |
| `CAPTURES_FOLDER` | — | `data/inspection_results` | Directory for saved snapshots and inspection runs |
| `TEST_STREAM_FOLDER` | — | `Tiles_dataset/test/cracks` | Default folder for simulated camera test stream |

#### Image Quality Thresholds:
```python
QUALITY_CONFIG = {
    "blur_threshold": 80.0,       # Minimum Laplacian variance for sharpness
    "brightness_min": 30.0,       # Minimum mean intensity (prevents dark frames)
    "brightness_max": 220.0,      # Maximum mean intensity (prevents glare/overexposure)
    "contrast_threshold": 20.0,   # Minimum standard deviation of pixel intensities
    "noise_threshold": 25.0,      # Maximum MAD Laplacian noise estimate
}
```

---

### 3. Hardware Acceleration & CPU / GPU Compatibility

FewVision automatically detects compute hardware at runtime:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- **CUDA GPU**: Enables CUDA synchronization timing and tensor acceleration (~15–30 ms per frame, 30+ FPS).
- **CPU Fallback**: Full support on standard industrial PCs and edge devices without CUDA. Single-frame CPU latency is typically ~300–450 ms (2.5–3.5 FPS). Controlled frame sampling ensures zero lag or backlog regardless of compute speed.

---

### 4. Running the Live Webcam Studio

#### Step 1: Start the application
```bash
python app.py
```
Open **http://localhost:5005** in your browser and switch to the **Live Inspection Studio** tab.

#### Step 2: Select or Initialize a Reference Session
Ensure an industrial reference dataset is loaded (e.g. session `fa1bfedd30f6`). If you are training a new part:
1. Upload 5–20 normal part images in the **Reference Studio** tab.
2. Click **Run Preparation Pipeline** to build the PatchCore memory bank (`memory.npy`).

#### Step 3: Start Live Inspection
1. **Browser Camera**: Select "Browser Camera (WebRTC)" from the camera selector and click **Start Inspection**. Allow browser webcam permissions when prompted.
2. **Server Hardware Camera**: Select "Camera 0" (or 1, 2) from the list and click **Start Inspection**.
3. **Simulated Stream**: Select "Simulated Stream (Tiles Dataset)" to test defect detection without a camera.

---

### 5. Live Webcam REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/webcam/status` | Current streaming status, active session, FPS, and device |
| `GET` | `/api/webcam/cameras` | List auto-discovered physical camera devices on the host |
| `POST` | `/api/webcam/start` | Start server OpenCV camera or simulated stream |
| `POST` | `/api/webcam/stop` | Stop live capture and release camera resources |
| `POST` | `/api/webcam/pause` | Pause / resume camera capture |
| `GET` | `/api/webcam/stream` | Multi-part MJPEG stream of live annotated camera feed |
| `POST` | `/api/webcam/inspect_frame` | Submit client frame (base64/file) for live inference |
| `POST` | `/api/webcam/capture` | Save snapshot (`raw.jpg`, `annotated.jpg`, `heatmap.jpg`, `result.json`) |
| `POST` | `/api/webcam/review` | Record operator review (`CONFIRM_ANOMALY`, `MARK_NORMAL`, etc.) |
| `GET` | `/api/webcam/history` | Retrieve rolling inspection history log |
| `POST` | `/api/webcam/history/clear` | Clear inspection history |
| `GET` | `/api/webcam/history/export` | Download inspection history as CSV |
| `POST` | `/api/webcam/threshold` | Dynamically calibrate patch anomaly threshold |
| `GET` | `/captures/<capture_id>/<file>` | Serve saved capture snapshot artifacts |
| `GET` | `/feedback/<category>/<file>` | Serve labeled human review dataset images |

---

### 6. Testing

Run the automated test suite:
```bash
# Run live webcam unit and integration tests
python -m pytest tests/test_webcam.py tests/test_webcam_integration.py -v

# Run the complete test suite across the entire application
python -m pytest tests/
```

---

## License

MIT License — see `LICENSE` for details.

