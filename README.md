# FewVision

**Adaptive Quality-Aware Few-Shot Anomaly Detection for Industrial Part Inspection**

FewVision is a production-grade Flask application that automates the preparation of small image datasets (5–20 normal product images) for few-shot industrial anomaly detection. Upload images → analyse quality and content → generate an augmented reference dataset → extract DINOv2 embeddings → embedding database ready for anomaly detection.

---

## Full Pipeline

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
Embedding Database  ←  data/embeddings/{session_id}/
         │
         ▼
[ Anomaly Detection — next stage ]
(Memory Bank / PatchCore / PaDiM)
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
│   ├── pipeline/           # Orchestrator — coordinates all 6 stages
│   ├── feature_extraction/ # Embedding extraction modules
│   │   ├── base_extractor.py      # Abstract interface for all extractors
│   │   ├── preprocessing.py       # Preprocessing transforms (DINOv2 spec)
│   │   ├── dinov2_extractor.py    # DINOv2 ViT-S/14 implementation
│   │   ├── vit_extractor.py       # ViT-B/16 implementation (NEW)
│   │   ├── extractor_factory.py   # Factory: name → extractor instance
│   │   └── embedding_database.py  # Save / load embedding store
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
│   ├── logs/
│   └── temp/
├── tests/
└── docs/
```

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

### DINOv2 model variants

| Variant | Embedding Dim | Params | Notes |
|---|---|---|---|
| `dinov2_vits14` | 384 | 21 M | **Default** — fast, great quality |
| `dinov2_vitb14` | 768 | 86 M | Better quality, ~4× slower |
| `dinov2_vitl14` | 1024 | 307 M | Best quality, slow |

---

## Modules

| Module | Responsibility |
|---|---|
| `modules/quality/` | Blur, brightness, contrast, noise, resolution, exposure clipping |
| `modules/content/` | Background complexity, lighting, object coverage, orientation, aspect ratio |
| `modules/augmentation/adaptive_policy.py` | Decides which augmentations to apply per image |
| `modules/augmentation/augmentations.py` | Applies augmentations using Albumentations |
| `modules/reporting/report_generator.py` | Generates 3-panel PNG reports per image |
| `modules/reporting/dataset_analytics.py` | Dataset-level statistics, distribution plots, duplicate detection |
| `modules/pipeline/pipeline.py` | Orchestrates all 6 pipeline stages |
| `modules/feature_extraction/base_extractor.py` | Abstract base class — all extractors implement this |
| `modules/feature_extraction/preprocessing.py` | Preprocessing transforms (DINOv2 spec) |
| `modules/feature_extraction/dinov2_extractor.py` | DINOv2 via `torch.hub`, CLS-token embeddings, mini-batch inference |
| `modules/feature_extraction/vit_extractor.py` | ViT-B/16 via torchvision, CLS-token embeddings, mini-batch inference |
| `modules/feature_extraction/extractor_factory.py` | `get_extractor()` → concrete extractor instance |
| `modules/feature_extraction/embedding_database.py` | Saves `embeddings.npy`, `filenames.json`, `metadata.json`, `extractor_info.json` |

---

## Embedding Database

After each upload session, embeddings are saved to:

```
data/embeddings/{session_id}/
    embeddings.npy       # float32, shape (N, D) where D is embedding_dim (e.g. 384 or 768)
    filenames.json       # ["img_aug_1.png", ...]
    metadata.json        # quality score, content score, augmentations per image
    extractor_info.json  # model variant, dim, preprocessing config
```

### API

```bash
# Embedding metadata (JSON)
GET /api/embeddings/<session_id>

# Download embeddings.npy
GET /api/embeddings/<session_id>/download
```

### Load in Python

```python
from modules.feature_extraction.embedding_database import load_embeddings

embeddings, filenames, metadata, extractor_info = load_embeddings(session_id)
# embeddings.shape → (N, D) where D is the extractor's embedding dimension
```

---

## Multiple Feature Extractors

FewVision supports multiple independent feature extractors through the extractor factory. The selection is decoupled from the pipeline code and can be set via configuration.

### Available Extractors
1. **DINOv2 (Default)**
   - **Registry Key**: `dinov2`
   - **Variants**: `dinov2_vits14` (384-dim, default), `dinov2_vitb14` (768-dim), `dinov2_vitl14` (1024-dim)
   - **Source**: Loaded via `torch.hub` from `facebookresearch/dinov2`.
2. **ViT (Vision Transformer)**
   - **Registry Key**: `vit`
   - **Variants**: `vit_b_16` (768-dim)
   - **Source**: Loaded via `torchvision.models` (weights: `ViT_B_16_Weights.IMAGENET1K_V1`).
   - **Representation**: Pre-classification CLS token embedding (classification head replaced with `Identity`).
   - **Input Preprocessing**:
     - BGR to RGB color conversion
     - Resize shortest side to `256` using **Bilinear** interpolation
     - Center crop to `224 x 224`
     - Scale values to `[0, 1]` and normalize using ImageNet mean/std statistics

### Extractor Flow Architecture

```
Extractor Selection (FEWVISION_EXTRACTOR)
         │
         ▼
extractor_factory.get_extractor()
         │
         ▼
Selected Extractor Class (DINOv2 / ViT)
         │
         ▼
Model Inference (without gradients, eval mode)
         │
         ▼
Fixed-Dimensional Embedding Vector (D,)
         │
         ▼
Embedding Database (data/embeddings/{session_id}/)
```

### Model Selection Example

To run the pipeline with the **ViT** extractor, set the `FEWVISION_EXTRACTOR` environment variable:

**For PowerShell:**
```powershell
$env:FEWVISION_EXTRACTOR="vit"
python app.py
```

To revert back to the default **DINOv2** extractor:

**For PowerShell:**
```powershell
Remove-Item Env:FEWVISION_EXTRACTOR -ErrorAction SilentlyContinue
python app.py
```

Alternatively, you can set the variable directly in your environment block or configure the default in `config.py`.

---

## Adding a New Extractor

1. Create `modules/feature_extraction/my_extractor.py` implementing `BaseExtractor`
2. Register it in `extractor_factory.py`:
   ```python
   REGISTRY["mymodel"] = lambda: MyExtractor
   ```
3. Set `FEWVISION_EXTRACTOR=mymodel` — no pipeline code changes needed

---

## Tech Stack

- **Python 3.11+**
- **Flask 3.x** — Web framework
- **OpenCV** — Image processing
- **Albumentations** — Augmentation engine
- **Matplotlib** — Report visualisation
- **NumPy / Pandas** — Numeric computing
- **PyTorch 2.x + torchvision** — DINOv2 feature extraction
- **DINOv2 (Meta AI)** — Self-supervised ViT embeddings for anomaly detection

---

## Roadmap

| Stage | Status |
|---|---|
| Image Quality Assessment | ✅ Complete |
| Content Analysis | ✅ Complete |
| Adaptive Augmentation | ✅ Complete |
| DINOv2 Feature Extraction | ✅ Complete |
| Anomaly Detection (PatchCore / PaDiM) | 🔜 Next |
| Product Grading | 🔜 Planned |

---

## License

MIT License — see `LICENSE` for details.
