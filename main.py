# main.py
"""Entry point for the few‑shot computer‑vision pipeline.

The script asks the user for a folder containing images (defaulting to the
`images/` sub‑directory), runs quality analysis and content analysis on each
image, prints a formatted combined report, and writes CSV/JSON reports to
the `reports/` directory.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict

from quality import ImageQualityChecker
from content_analysis import ContentAnalyzer

def merge_results(quality: Dict, content: Dict) -> Dict:
    """Combine quality and content dictionaries into a single report."""
    return {
        "blur": quality.get("blur"),
        "brightness": quality.get("brightness"),
        "contrast": quality.get("contrast"),
        "noise": quality.get("noise"),
        "resolution": quality.get("resolution"),
        "recommendations": quality.get("recommendations", []),
        "background": content.get("background"),
        "lighting": content.get("lighting"),
        "object_coverage": content.get("object_coverage"),
        "orientation": content.get("orientation"),
        "aspect_ratio": content.get("aspect_ratio"),
        "center_offset": content.get("center_offset"),
    }

def pretty_print(image_name: str, data: Dict) -> None:
    print("=" * 60)
    print(image_name)
    print("=" * 60)
    print("QUALITY")
    print(f"Blur : {data['blur']:.2f}")
    print(f"Brightness : {data['brightness']:.2f}")
    print(f"Contrast : {data['contrast']:.2f}")
    print(f"Noise : {data['noise']:.2f}\n")
    print("CONTENT")
    print(f"Background : {data['background']}")
    print(f"Lighting : {data['lighting']}")
    print(f"Object Coverage : {data['object_coverage']:.1f}%")
    print(f"Orientation : {data['orientation']:.1f}°")
    print(f"Aspect Ratio : {data['aspect_ratio']:.2f}")
    print(f"Center Offset : {data['center_offset']:.1f}%\n")
    print("Recommended")
    for r in data.get("recommendations", []):
        print(f"✓ {r}")
    print("=" * 60)

def main() -> None:
    folder = input("Enter dataset folder path (default ./images): ").strip()
    if not folder:
        folder = os.path.join(os.getcwd(), "images")
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return

    results: List[Dict] = []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(valid_ext):
            continue
        path = os.path.join(folder, fname)
        try:
            q = ImageQualityChecker(path)
            c = ContentAnalyzer(path)
            merged = merge_results(q.analyze(), c.analyze())
            merged["image_name"] = fname
            results.append(merged)
            pretty_print(fname, merged)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(reports_dir, "report.csv"), index=False)
    json_path = os.path.join(reports_dir, "report.json")
    with open(json_path, "w") as f:
        # Convert numpy scalar types to native Python types for JSON serialization
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nReports saved to {reports_dir}")

if __name__ == "__main__":
    main()
