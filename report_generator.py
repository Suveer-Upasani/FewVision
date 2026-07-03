# report_generator.py
"""Visual report generator for per-image analysis results.

Produces a 3-panel PNG for each image:
  Panel 1: Original image + bounding box + orientation line + centre marker
  Panel 2: Sharpness heatmap (local Laplacian variance)
  Panel 3: Stats card (quality score, content score, suitability, exposure,
           augmentation list, confidence badges)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models import AnalysisResult

REPORTS_DIR = "reports"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _draw_bbox_overlay(ax, img_rgb: np.ndarray, result: AnalysisResult) -> None:
    """Draw the original image with bounding box, orientation line, and centre."""
    ax.imshow(img_rgb)

    brect = result.content.bounding_rect
    if brect is not None:
        box = cv2.boxPoints(brect)
        box = box.astype(int)
        # Draw bounding box
        for i in range(4):
            p1 = tuple(box[i])
            p2 = tuple(box[(i + 1) % 4])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#00e676", linewidth=2)

        # Centre dot
        (cx, cy), _, _ = brect
        ax.plot(cx, cy, "o", color="#ff1744", markersize=8, zorder=5)

        # Orientation line
        angle_rad = np.deg2rad(result.content.orientation)
        length = min(img_rgb.shape[0], img_rgb.shape[1]) * 0.25
        dx = length * np.cos(angle_rad)
        dy = -length * np.sin(angle_rad)  # y-axis inverted in image coords
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="#ffea00", lw=2),
        )

    # Image centre crosshair
    ih, iw = img_rgb.shape[:2]
    ax.axhline(ih / 2, color="white", linewidth=0.5, alpha=0.4)
    ax.axvline(iw / 2, color="white", linewidth=0.5, alpha=0.4)

    ax.set_title("Object Detection", fontsize=13, weight="bold", color="#e0e0e0")
    ax.axis("off")


def _draw_sharpness_heatmap(ax, result: AnalysisResult) -> None:
    """Draw the block-wise sharpness heatmap."""
    smap = result.quality.sharpness_map
    if smap is not None:
        ax.imshow(smap, cmap="inferno", interpolation="nearest", aspect="auto")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20, color="gray",
                transform=ax.transAxes)
    ax.set_title("Sharpness Heatmap", fontsize=13, weight="bold", color="#e0e0e0")
    ax.axis("off")


def _draw_stats_card(ax, result: AnalysisResult) -> None:
    """Draw a text-based stats card with scores and badges."""
    q = result.quality
    c = result.content

    # Colour code the suitability
    suit = result.suitability_score
    if suit >= 75:
        suit_color = "#00e676"
        badge = "✅ Ready"
    elif suit >= 50:
        suit_color = "#ffea00"
        badge = "⚠️ Marginal"
    else:
        suit_color = "#ff1744"
        badge = "❌ Unsuitable"

    lines = [
        f"SUITABILITY:  {suit:.0f}/100  {badge}",
        "",
        f"Quality Score:   {q.quality_score:.0f}/100  [{q.quality_rating}]",
        f"Content Score:   {c.content_score:.0f}/100",
        "",
        f"Blur:        {q.blur:.1f}   (conf {q.blur_confidence:.0%})",
        f"Brightness:  {q.brightness:.1f}",
        f"Contrast:    {q.contrast:.1f}",
        f"Noise:       {q.noise:.4f}  (conf {q.noise_confidence:.0%})",
        "",
        f"Under-exposed: {q.underexposed_pct:.1f}%",
        f"Over-exposed:  {q.overexposed_pct:.1f}%",
        "",
        f"Background:  {c.background}  (conf {c.background_confidence:.0%})",
        f"Lighting:    {c.lighting}",
        f"Coverage:    {c.object_coverage:.1f}%  (conf {c.coverage_confidence:.0%})",
        f"Orientation: {c.orientation:.1f}°",
        f"Aspect:      {c.aspect_ratio:.2f}",
        f"Offset:      {c.center_offset:.1f}%",
        "",
        f"Augmentations ({len(result.augmentations)}):",
    ]
    for aug in result.augmentations:
        lines.append(f"  • {aug}")

    text = "\n".join(lines)

    ax.set_facecolor("#1e1e2f")
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        color="#e0e0e0",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#2a2a40", alpha=0.95, edgecolor="#444"),
    )
    ax.set_title("Analysis Report", fontsize=13, weight="bold", color="#e0e0e0")
    ax.axis("off")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_image_report(result: AnalysisResult, output_dir: str = REPORTS_DIR) -> str:
    """Generate a 3-panel visual report PNG for a single image.

    Returns the path to the saved report.
    """
    _ensure_dir(output_dir)

    img = cv2.imread(result.image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {result.image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor("#121212")

    fig.suptitle(
        result.image,
        fontsize=16,
        weight="bold",
        color="white",
        y=0.98,
    )

    _draw_bbox_overlay(axes[0], img_rgb, result)
    _draw_sharpness_heatmap(axes[1], result)
    _draw_stats_card(axes[2], result)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"report_{os.path.splitext(result.image)[0]}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
