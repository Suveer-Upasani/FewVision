# FewVision

**Adaptive Quality-Aware Few-Shot Anomaly Detection for Industrial Part Inspection**

FewVision is a production-grade Flask application that automates the preparation of small image datasets (5–20 normal product images) and testing of new product items for few-shot industrial anomaly detection.

---

## Full Pipelines

### 1. Reference Pipeline (Build Reference Memory)

```
Upload Normal Product Images
         │
         ▼
Image Quality Assessment
(blur · brightness · contrast · noise · resolution · exposure)
         │
         ▼
Content Analysis
(background · lighting · object coverage · orientation · centering)
         │
         ▼
Adaptive Augmentation Policy
(per-image augmentation strategy based on quality + content scores)
         │
         ▼
Reference Dataset  ←  augmented images saved to data/augmented/
         │
         ▼
Feature Extraction  ✅  (implemented)
(DINOv2 [default] / ViT-B/16 · MPS/CUDA/CPU auto-detect)
         │
         ▼
Embedding Database  ✅  (implemented)
(data/embeddings/{session_id}/)
         │
         ▼
Memory Bank  ✅  (implemented)
(L2-normalised · cosine/euclidean similarity · FAISS-ready)
(data/memory_bank/{session_id}/)
```

### 2. Inspection Pipeline (Inspect Product)

```
Upload Test Image(s)
         │
         ▼
Quality Assessment
         │
         ▼
Content Analysis
         │
         ▼
Feature Extraction
(uses the same DINOv2 / ViT extractor used in the Reference Pipeline)
         │
         ▼
Memory Bank Search
(queries memory bank for top-k nearest normal neighbors)
         │
         ▼
Similarity Search
(computes distance / similarity scores)
         │
         ▼
Anomaly Score
(normalises distance values to a composite 0-100 anomaly score)
         │
         ▼
Inspection Result  ←  saved to data/inference/{session_id}/{run_id}/
(predicts label: Normal / Suspicious / Anomalous)
```

---

## Project Structure

```
FewVision/
├── app.py              # Flask routes only — zero processing logic
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
│   │   ├── base_extractor.py      # Abstract interface for all extractors
│   │   ├── preprocessing.py       # Preprocessing transforms (DINOv2 spec)
│   │   ├── dinov2_extractor.py    # DINOv2 ViT-S/14 implementation
│   │   ├── vit_extractor.py       # ViT-B/16 implementation
│   │   ├── extractor_factory.py   # Factory: name → extractor instance
│   │   └── embedding_database.py  # Save / load embedding store
│   ├── anomaly_detection/  # Memory Bank, Similarity Engine, Anomaly Scoring
│   │   ├── __init__.py            # Public API re-exports
│   │   ├── memory_bank.py         # MemoryBank class — build/save/load/search
│   │   ├── similarity.py          # Cosine & Euclidean nearest-neighbour utils
│   │   └── anomaly_score.py       # Distance → score / label conversion
│   ├── inference/          # Inference pipeline modules
│   │   ├── __init__.py            # Public API re-exports
│   │   ├── inference_engine.py    # InferenceEngine orchestrator
│   │   └── inspection_result.py   # InspectionResult data model
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
│   ├── logs/
│   └── temp/
├── tests/
└── docs/
```

---

## Features

- **Image Quality Assessment** — blur, brightness, contrast, noise, resolution, exposure analysis.
- **Content Analysis** — background complexity, lighting, object coverage, orientation.
- **Adaptive Augmentation** — per-image augmentation policy driven by quality + content scores.
- **DINOv2 / ViT Feature Extraction** — self-supervised embeddings with automatic device selection.
- **Embedding Database** — persistent per-session NumPy store with full metadata.
- **Memory Bank Generation** — L2-normalised reference embedding store ready for search.
- **Industrial Product Inspection** — inspect one or multiple test images against a saved reference Memory Bank.
- **Top-K Similarity Search** — configurable Cosine and Euclidean nearest-neighbour search.
- **Anomaly Decision Engine** — distance-to-score normalization with extensible Normal/Suspicious/Anomalous thresholding.
- **FAISS-Ready Design** — swap brute-force similarity calculations for FAISS index search without changing calling code.

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

> **Note:** `torch` and `torchvision` are included in `requirements.txt`.
> DINOv2 model weights (~85 MB) are downloaded automatically on first run
> and cached at `~/.cache/torch/hub/`.

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
| `FEWVISION_SECRET_KEY` | dev key | Flask session secret |
| `FEWVISION_DEBUG` | `true` | Debug mode |
| `FEWVISION_PORT` | `5005` | Server port |
| `FEWVISION_FEATURE_EXTRACTION` | `true` | Run feature extraction after augmentation |
| `FEWVISION_EXTRACTOR` | `dinov2` | Extractor name (`dinov2`, `vit`) |
| `FEWVISION_DINOV2_VARIANT` | `dinov2_vits14` | DINOv2 variant (`dinov2_vits14`, `dinov2_vitb14`) |
| `FEWVISION_VIT_VARIANT` | `vit_b_16` | ViT variant (`vit_b_16`) |
| `FEWVISION_BATCH_SIZE` | `32` | Images per forward pass |
| `FEWVISION_MEMORY_BANK` | `true` | Build Memory Bank after feature extraction |
| `FEWVISION_SIMILARITY_METRIC` | `cosine` | Similarity metric (`cosine`, `euclidean`) |
| `FEWVISION_TOP_K` | `5` | Number of nearest neighbours to return |
| `FEWVISION_THRESH_NORMAL` | `0.20` | Cosine distance threshold for "Normal" label |
| `FEWVISION_THRESH_SUSPICIOUS` | `0.50` | Cosine distance threshold for "Suspicious" label |
| `FEWVISION_INFERENCE` | `true` | Enable inference testing |
| `FEWVISION_DEFAULT_TOP_K` | `5` | Neighbors to return during inspection query |
| `FEWVISION_DEFAULT_SIMILARITY` | `cosine` | Metric used for query embedding searches |
| `FEWVISION_MAX_TEST_IMAGES` | `20` | Limit of test files uploaded per inspection |

---

## REST API

### Reference Pipeline

```bash
# Upload normal product images and run preprocessing
POST /api/upload

# Generate augmented dataset ZIP
POST /api/generate/<session_id>

# Download augmented dataset ZIP
GET /api/download/<session_id>
```

### Embeddings & Memory Bank

```bash
# Retrieve embedding session summary
GET /api/embeddings/<session_id>

# Download raw embeddings binary
GET /api/embeddings/<session_id>/download

# Retrieve memory bank summary
GET /api/memory-bank/<session_id>
```

### Inspection / Inference

```bash
# Run product inspection on uploaded test image(s)
POST /api/inspect
# Parameters (form-data):
#   files: File[] (raw image files)
#   session_id: string (optional session override)

# Retrieve all inference runs for a session
GET /api/inference/<session_id>

# Download results.json of the latest run
GET /api/inference/<session_id>/download
```

---

## Data Directory Layout

Runtime-generated files are stored in `data/` and excluded from source control:

### 1. Uploads and Preprocessing
- `data/uploads/{session_id}/` — Original reference images.
- `data/augmented/{session_id}/` — Augmented images generated by adaptive augmentation policies.
- `data/reports/{session_id}/` — Per-image quality reports and charts.

### 2. Embeddings and Memory Bank
- `data/embeddings/{session_id}/` — Embedding database files (`embeddings.npy`, `filenames.json`, `metadata.json`, `extractor_info.json`).
- `data/memory_bank/{session_id}/` — Reference memory bank files (`memory.npy`, `memory_metadata.json`, `config.json`).

### 3. Inspection Run Outputs
Inference outputs are saved in non-overwriting run directories:
- `data/inference/{session_id}/{run_id}/`
  - `results.json` — Detailed list of predicted labels, anomaly scores, confidence, quality/content metrics, and top-5 neighbors.
  - `inspection_summary.json` — High-level statistics (total checked, normal/suspicious/anomalous counts).
  - `annotated_image.png` — Visual inspection results placeholder.

---

## Architecture & Workflows

FewVision divides processing into two decoupled workflows:

1. **Workflow 1: Build Reference Memory**:
   Normal product images are analyzed, augmented, and embedded into a Reference Memory Bank. This process establishes the baseline profile of "normal" for parts.
   
2. **Workflow 2: Inspect Product**:
   Test images are uploaded and evaluated individually or in batches. The Inference Engine loads the saved Memory Bank once and performs fast nearest-neighbor searches to detect deviations, bypassing the reference pipeline.

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
| PatchCore Anomaly Localization | 🔜 Upcoming |
| PaDiM Anomaly Localization | 🔜 Upcoming |
| Heatmap Visualization | 🔜 Upcoming |
| Product Grading | 🔜 Upcoming |

---

## License

MIT License — see `LICENSE` for details.
