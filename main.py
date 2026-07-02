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

    # Recommendations placeholder (can be extended later)
    merged = {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        **q_dict,
        **c_dict,
        "recommendations": [],
    }
    return merged


def write_reports(results: list[dict]):
    """Write CSV and JSON reports from the list of result dictionaries."""
    ensure_reports_dir()
    csv_path = os.path.join(REPORTS_DIR, "report.csv")
    json_path = os.path.join(REPORTS_DIR, "report.json")

    fieldnames = [
        "image",
        "image_path",
        "blur",
        "brightness",
        "contrast",
        "noise",
        "resolution",
        "recommendations",
        "background",
        "lighting",
        "object_coverage",
        "orientation",
        "aspect_ratio",
        "center_offset",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = r.copy()
            for k, v in list(row.items()):
                if isinstance(v, np.generic):
                    row[k] = v.item()
            rec = row.get("recommendations", [])
            if isinstance(rec, list):
                row["recommendations"] = " | ".join(rec)
            writer.writerow(row)

    json_compatible = []
    for r in results:
        row = {}
        for k, v in r.items():
            row[k] = v.item() if isinstance(v, np.generic) else v
        json_compatible.append(row)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_compatible, jf, indent=2)
    print(f"✅ Reports written to ./{REPORTS_DIR}/ (CSV & JSON)")


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
    else:
        print("⚠️ No supported images found – nothing to report.")

if __name__ == "__main__":
    default = os.path.join(os.getcwd(), "images")
    folder = input(f"Enter dataset folder path (default {default}): ").strip() or default
    if not os.path.isdir(folder):
        print(f"❌ Folder does not exist: {folder}")
    else:
        process_folder(folder)
