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
| PaDiM Anomaly Localization | 🔜 Upcoming |
| Product Grading | 🔜 Upcoming |

---

## License

MIT License — see `LICENSE` for details.
