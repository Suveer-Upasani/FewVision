# FewVision

**Adaptive Quality-Aware Few-Shot Learning for Industrial Part Inspection**

FewVision is a production-grade Flask application that automates the preparation of small image datasets (5–20 images) for few-shot learning. Upload images → analyse quality and content → generate an optimised augmented dataset → download as ZIP.

---

## Workflow

```
Upload Images → Quality Analysis → Content Analysis → Adaptive Policy → Augmented Dataset → Download ZIP
```

---

## Project Structure

```
FewVision/
├── app.py              # Flask routes only — no processing logic
├── config.py           # All configuration constants
├── requirements.txt
├── README.md
│
├── modules/
│   ├── quality/        # Blur, brightness, contrast, noise, resolution
│   ├── content/        # Background, lighting, object coverage, orientation
│   ├── augmentation/   # Adaptive policy + augmentation engine
│   ├── reporting/      # Per-image reports + dataset analytics
│   ├── pipeline/       # Orchestrator — coordinates all modules
│   └── utils/          # Dataclasses, image helpers, file helpers
│
├── models/             # Future: ResNet50, ViT, Prototypical Networks
├── templates/          # index.html, dashboard.html, results.html
├── static/             # CSS design system + JS modules
├── data/               # Runtime data (gitignored)
│   ├── uploads/
│   ├── augmented/
│   ├── reports/
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

---

## Modules

| Module | Responsibility |
|---|---|
| `modules/quality/` | Blur, brightness, contrast, noise, resolution, exposure clipping |
| `modules/content/` | Background complexity, lighting, object coverage, orientation, aspect ratio |
| `modules/augmentation/adaptive_policy.py` | Decides which augmentations to apply (pure policy, no image modification) |
| `modules/augmentation/augmentations.py` | Applies augmentations using Albumentations |
| `modules/reporting/report_generator.py` | Generates 3-panel PNG reports per image |
| `modules/reporting/dataset_analytics.py` | Dataset-level statistics, distribution plots, duplicate detection |
| `modules/pipeline/pipeline.py` | Orchestrates the full workflow |

---

## Future Extensions

The architecture is ready for:

- `models/feature_extraction.py` — ResNet50 / ViT / CLIP embeddings
- `models/few_shot_model.py` — Prototypical Networks / Siamese Networks
- REST API endpoints (`/api/v1/...`)
- Docker deployment (`Dockerfile` + `docker-compose.yml`)
- Background task queue (Celery + Redis) for large datasets

---

## Tech Stack

- **Python 3.11+**
- **Flask 3.x** — Web framework
- **OpenCV** — Image processing
- **Albumentations** — Augmentation engine
- **Matplotlib** — Report visualisation
- **NumPy / Pandas** — Numeric computing

---

## License

MIT License — see `LICENSE` for details.
