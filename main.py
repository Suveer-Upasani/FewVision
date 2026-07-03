#!/usr/bin/env python3
"""Main entry point for the modular computer‑vision pipeline.

The script walks through an image folder, runs the quality checker and the
content analyzer on each image, and writes the combined results to CSV and JSON
reports under the ``reports`` directory.
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from quality import ImageQualityChecker
from content_analysis import ContentAnalyzer
import cv2
REPORTS_DIR = "reports"


def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def process_image(image_path: str) -> dict:
    """Run both quality and content analysis and merge their results.

    Returns a dictionary that can be written directly to CSV/JSON.
    """
    # Quality metrics
    quality = ImageQualityChecker(image_path)
    q_dict = quality.analyze()

    # Content metrics
    content = ContentAnalyzer(image_path)
    c_dict = content.analyze()

    merged = {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        **q_dict,
        **c_dict,
    }
    # Ensure recommendations exists if it wasn't in q_dict or c_dict
    if "recommendations" not in merged:
        merged["recommendations"] = []
    return merged


def write_reports(results: list[dict]):
    """Generate a visual analyzed image report for each processed image."""
    ensure_reports_dir()

    if not results:
        print("No results to generate report.")
        return

    for r in results:
        img_path = r.get("image_path")
        img_name = r.get("image")
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path} for report generation.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a heatmap-like visual (Gradient magnitude for blur/edges)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        fig = plt.figure(figsize=(14, 7))
        fig.patch.set_facecolor('#f4f4f9')
        
        # Original Image
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img_rgb)
        ax1.set_title("Analyzed Product Image", fontsize=16, weight='bold')
        ax1.axis('off')
        
        # Heatmap
        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(magnitude, cmap='inferno')
        ax2.set_title("Sharpness / Edge Heatmap", fontsize=16, weight='bold')
        ax2.axis('off')
        
        # Add a text box with the stats on the left image
        recs = r.get("recommendations", [])
        status = "DEFECT ❌" if recs else "PASS ✅"
        color = "#ffebee" if recs else "#e8f5e9" # light red or green background
        edge_color = "red" if recs else "green"
        
        # Extract scalar values if they are arrays
        def get_val(key):
            v = r.get(key, 0)
            return float(np.mean(v)) if isinstance(v, (list, tuple, np.ndarray)) else float(v)

        stats_text = (
            f"STATUS: {status}\n"
            f"{'-'*30}\n"
            f"Blur: {get_val('blur'):.1f}\n"
            f"Contrast: {get_val('contrast'):.1f}\n"
            f"Brightness: {get_val('brightness'):.1f}\n"
            f"Noise: {get_val('noise'):.1f}\n"
        )
        if recs:
            notes = ", ".join(recs) if isinstance(recs, list) else str(recs)
            stats_text += f"\nIssues Detected:\n{notes}"
            
        props = dict(boxstyle='round,pad=1', facecolor=color, alpha=0.9, edgecolor=edge_color, linewidth=2)
        ax1.text(0.03, 0.97, stats_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, color='black', weight='bold', family='monospace')

        plt.tight_layout()
        out_name = f"analyzed_{img_name}"
        out_path = os.path.join(REPORTS_DIR, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Analyzed report images written to ./{REPORTS_DIR}/")


def process_folder(folder_path: str):
    """Iterate over supported image files in *folder_path* and generate reports."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if os.path.splitext(fname)[1].lower() in valid_ext:
            img_path = os.path.join(folder_path, fname)
            try:
                results.append(process_image(img_path))
            except Exception as exc:
                print(f"[WARN] Failed processing {fname}: {exc}")
    if results:
        write_reports(results)
        
        # Augmentation: generate 10 synthetic angle/lighting variants per image
        from augmentations import generate_batch
        aug_dir = os.path.join(os.getcwd(), "augmented_dataset")
        for r in results:
            img_path = r.get("image_path")
            try:
                generate_batch(img_path, output_dir=aug_dir, num_images=10)
            except Exception as e:
                print(f"[WARN] Augmentation failed for {img_path}: {e}")
        print(f"Augmented dataset saved to: {aug_dir}")

        # Feature Extraction: extract 2048-dim ResNet50 embeddings
        try:
            from feature_extraction import run as extract_features_run
            extract_features_run(image_dir=aug_dir)
        except ImportError:
            print("[WARN] torch/torchvision not installed. Run: pip install torch torchvision")
        except Exception as e:
            print(f"[WARN] Feature extraction failed: {e}")
        
    else:
        print("⚠️ No supported images found – nothing to report.")

if __name__ == "__main__":
    default = os.path.join(os.getcwd(), "images")
    folder = input(f"Enter dataset folder path (default {default}): ").strip() or default
    if not os.path.isdir(folder):
        print(f"❌ Folder does not exist: {folder}")
    else:
        process_folder(folder)
