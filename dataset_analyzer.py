# dataset_analyzer.py
"""Dataset-level analytics for the FewVision pipeline.

Produces:
  - Summary statistics (mean/std/min/max/percentiles) for all numeric metrics.
  - Distribution plots (histograms) saved as ``reports/dataset_summary.png``.
  - Duplicate detection via perceptual hashing + SSIM.
  - Lightweight embedding preview (PCA of resized images) saved as
    ``reports/embedding_preview.png``.
"""

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import AnalysisResult

# Optional import for SSIM
try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False

REPORTS_DIR = "reports"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _average_hash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute a perceptual average hash for an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    return (resized > mean_val).astype(np.uint8).flatten()


def _hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


def _compute_summary(results: list[AnalysisResult]) -> dict[str, dict]:
    """Return {metric_name: {mean, std, min, max, p20, p80}} for numeric metrics."""
    if not results:
        return {}

    keys = [
        ("blur", lambda r: r.quality.blur),
        ("brightness", lambda r: r.quality.brightness),
        ("contrast", lambda r: r.quality.contrast),
        ("noise", lambda r: r.quality.noise),
        ("underexposed_pct", lambda r: r.quality.underexposed_pct),
        ("overexposed_pct", lambda r: r.quality.overexposed_pct),
        ("quality_score", lambda r: r.quality.quality_score),
        ("object_coverage", lambda r: r.content.object_coverage),
        ("orientation", lambda r: r.content.orientation),
        ("aspect_ratio", lambda r: r.content.aspect_ratio),
        ("center_offset", lambda r: r.content.center_offset),
        ("content_score", lambda r: r.content.content_score),
        ("suitability_score", lambda r: r.suitability_score),
    ]

    summary = {}
    for name, extractor in keys:
        vals = np.array([extractor(r) for r in results], dtype=np.float64)
        summary[name] = {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "p20": round(float(np.percentile(vals, 20)), 2),
            "p80": round(float(np.percentile(vals, 80)), 2),
        }
    return summary


def _plot_distributions(
    results: list[AnalysisResult], output_dir: str = REPORTS_DIR
) -> str:
    """Generate histogram grid and save to PNG."""
    _ensure_dir(output_dir)

    metrics = {
        "Blur": [r.quality.blur for r in results],
        "Brightness": [r.quality.brightness for r in results],
        "Contrast": [r.quality.contrast for r in results],
        "Noise": [r.quality.noise for r in results],
        "Object Coverage (%)": [r.content.object_coverage for r in results],
        "Quality Score": [r.quality.quality_score for r in results],
        "Content Score": [r.content.content_score for r in results],
        "Suitability Score": [r.suitability_score for r in results],
    }

    n = len(metrics)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.patch.set_facecolor("#121212")
    axes = axes.flatten()

    for idx, (title, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.set_facecolor("#1e1e2f")
        ax.hist(values, bins=max(5, len(values) // 2), color="#7c4dff", edgecolor="#b388ff", alpha=0.85)
        ax.set_title(title, fontsize=12, weight="bold", color="#e0e0e0")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Dataset Analytics  —  {len(results)} images",
        fontsize=18,
        weight="bold",
        color="white",
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "dataset_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _detect_duplicates(
    results: list[AnalysisResult],
    hamming_threshold: int = 5,
    output_dir: str = REPORTS_DIR,
) -> str:
    """Find near-duplicate images via perceptual hashing."""
    _ensure_dir(output_dir)

    hashes = []
    for r in results:
        img = cv2.imread(r.image_path)
        if img is None:
            continue
        h = _average_hash(img)
        gray_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 128))
        hashes.append((r.image, h, gray_small))

    pairs = []
    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _hamming_distance(hashes[i][1], hashes[j][1])
            if dist <= hamming_threshold:
                ssim_val = -1.0
                if _HAS_SSIM:
                    ssim_val = float(_ssim(hashes[i][2], hashes[j][2]))
                pairs.append({
                    "image_a": hashes[i][0],
                    "image_b": hashes[j][0],
                    "hamming_distance": dist,
                    "ssim": round(ssim_val, 4) if ssim_val >= 0 else "N/A",
                    "is_duplicate": "Yes" if dist <= 2 else "Likely",
                })

    csv_path = os.path.join(output_dir, "duplicates.csv")
    if pairs:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pairs[0].keys())
            writer.writeheader()
            writer.writerows(pairs)
    else:
        with open(csv_path, "w") as f:
            f.write("No near-duplicate pairs found.\n")

    return csv_path


def _embedding_preview(
    results: list[AnalysisResult], output_dir: str = REPORTS_DIR
) -> str:
    """Resize all images to 64×64, flatten, PCA → 2-D scatter plot."""
    _ensure_dir(output_dir)

    vectors = []
    labels = []
    names = []
    for r in results:
        img = cv2.imread(r.image_path)
        if img is None:
            continue
        resized = cv2.resize(img, (64, 64)).flatten().astype(np.float32)
        resized /= 255.0
        vectors.append(resized)
        labels.append(r.suitability_rating)
        names.append(r.image)

    if len(vectors) < 2:
        out_path = os.path.join(output_dir, "embedding_preview.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Need ≥ 2 images for embedding preview",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    X = np.array(vectors)
    X -= X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]

    colour_map = {"Ready": "#00e676", "Marginal": "#ffea00", "Unsuitable": "#ff1744", "Unknown": "#888"}
    colours = [colour_map.get(l, "#888") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#1e1e2f")

    ax.scatter(coords[:, 0], coords[:, 1], c=colours, s=120, edgecolors="white", linewidths=0.5, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(
            name[:20],
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            color="#ccc",
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title("Image Embedding Preview (PCA)", fontsize=16, weight="bold", color="white")
    ax.set_xlabel("PC 1", color="#aaa")
    ax.set_ylabel("PC 2", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=l)
        for l, c in colour_map.items() if l in labels
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#2a2a40", edgecolor="#444",
              labelcolor="#e0e0e0")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "embedding_preview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def analyze_dataset(results: list[AnalysisResult], output_dir: str = REPORTS_DIR) -> dict:
    """Run all dataset-level analytics."""
    summary = _compute_summary(results)

    dist_path = _plot_distributions(results, output_dir)
    dup_path = _detect_duplicates(results, output_dir=output_dir)
    emb_path = _embedding_preview(results, output_dir)

    print("\n" + "=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print(f"  Images analysed: {len(results)}")
    print()
    header = f"  {'Metric':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P20':>8} {'P80':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric, stats in summary.items():
        print(
            f"  {metric:<22} {stats['mean']:>8.2f} {stats['std']:>8.2f} "
            f"{stats['min']:>8.2f} {stats['max']:>8.2f} "
            f"{stats['p20']:>8.2f} {stats['p80']:>8.2f}"
        )
    print("=" * 70 + "\n")

    return {
        "summary": summary,
        "distribution_plot": dist_path,
        "duplicates_csv": dup_path,
        "embedding_plot": emb_path,
    }
