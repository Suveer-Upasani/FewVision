import os
import math
import cv2
import numpy as np
from models import QualityMetrics



class ImageQualityChecker:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Cannot open image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape

    # ----------------------------
    # Blur Detection
    # ----------------------------
    def blur_score(self):
        return cv2.Laplacian(self.gray, cv2.CV_64F).var()

    # ----------------------------
    # Brightness
    # ----------------------------
    def brightness(self):
        return np.mean(self.gray)

    # ----------------------------
    # Contrast
    # ----------------------------
    def contrast(self):
        return np.std(self.gray)

    # ----------------------------
    # Resolution
    # ----------------------------
    def resolution(self):
        h, w = self.gray.shape
        return w, h

    # ----------------------------
    # Noise Estimate (MAD-based)
    # ----------------------------
    def noise_score(self):
        """Estimate noise using Median Absolute Deviation of high-frequency
        content.  More stable than simple Gaussian subtraction."""
        lap = cv2.Laplacian(self.gray, cv2.CV_64F)
        # MAD estimator: sigma ≈ MAD / 0.6745
        mad = float(np.median(np.abs(lap - np.median(lap))))
        sigma = mad / 0.6745 if mad > 0 else 0.0
        return sigma

    # ----------------------------
    # Exposure Clipping  (#2)
    # ----------------------------
    def exposure_clipping(self):
        """Return (underexposed_pct, overexposed_pct) — % of pixels
        that are clipped to near-black (<5) or near-white (>250)."""
        total = self.width * self.height
        under = float(np.sum(self.gray < 5) / total * 100)
        over = float(np.sum(self.gray > 250) / total * 100)
        return round(under, 2), round(over, 2)

    # ----------------------------
    # Sharpness Heatmap  (#3)
    # ----------------------------
    def sharpness_heatmap(self, block_size=32):
        """Block-wise Laplacian variance — returns a 2-D numpy array.
        Each cell = sharpness of that image region."""
        rows = math.ceil(self.height / block_size)
        cols = math.ceil(self.width / block_size)
        hmap = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * block_size, min((r + 1) * block_size, self.height)
                x0, x1 = c * block_size, min((c + 1) * block_size, self.width)
                patch = self.gray[y0:y1, x0:x1]
                hmap[r, c] = cv2.Laplacian(patch, cv2.CV_64F).var()
        return hmap

    # ----------------------------
    # Confidence Scores  (#8)
    # ----------------------------
    def _blur_confidence(self, blur):
        """Higher blur variance → higher confidence the image is sharp."""
        return round(1.0 - math.exp(-blur / 200.0), 4)

    def _noise_confidence(self):
        """Stability of noise estimate across image quadrants."""
        h2, w2 = self.height // 2, self.width // 2
        quarters = [
            self.gray[:h2, :w2], self.gray[:h2, w2:],
            self.gray[h2:, :w2], self.gray[h2:, w2:],
        ]
        stds = []
        for q in quarters:
            blurred = cv2.GaussianBlur(q, (3, 3), 0)
            diff = q.astype(np.float32) - blurred.astype(np.float32)
            stds.append(float(np.std(diff)))
        mean_std = np.mean(stds)
        if mean_std == 0:
            return 1.0
        cv = float(np.std(stds) / mean_std)
        return round(max(0.0, min(1.0, 1.0 - cv)), 4)

    # ----------------------------
    # Quality Score  (#1)
    # ----------------------------
    def quality_score(self, blur, bright, contrast, noise, w, h, under_pct, over_pct):
        """Weighted composite score (0–100) and rating string.

        Weights: blur 30%, brightness 20%, contrast 20%, noise 15%, resolution 15%.
        Penalised if >5% pixels are under/over-exposed.
        """

        def sigmoid(val, mid, steep=0.05):
            x = steep * (val - mid)
            return 1.0 / (1.0 + math.exp(-x))

        s_blur = sigmoid(blur, 150, 0.02)                          # higher = sharper
        s_bright = max(0.0, 1.0 - abs(bright - 128) / 128.0)      # optimal ~128
        s_contrast = sigmoid(contrast, 50, 0.04)                   # higher = better
        s_noise = 1.0 - sigmoid(noise, 15, 0.15)                   # lower = better
        s_res = min(1.0, (w * h) / 640_000)                        # ≥800×800 = 1.0

        raw = (0.30 * s_blur + 0.20 * s_bright + 0.20 * s_contrast
               + 0.15 * s_noise + 0.15 * s_res)

        if under_pct > 5.0:
            raw *= 0.9
        if over_pct > 5.0:
            raw *= 0.9

        score = round(raw * 100, 2)
        if score >= 85:
            rating = "Excellent"
        elif score >= 60:
            rating = "Good"
        elif score >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        return score, rating

    # ----------------------------
    # Recommendations
    # ----------------------------
    def recommendations(self):
        rec = []
        avoid = []

        blur = self.blur_score()
        bright = self.brightness()
        contrast = self.contrast()
        under, over = self.exposure_clipping()

        if blur < 100:
            rec.append("Sharpen Image")
            avoid.append("Gaussian Blur")
        else:
            rec.extend(["Small Rotation", "Horizontal Flip"])

        if bright < 70:
            rec.append("Increase Brightness")
        elif bright > 180:
            rec.append("Decrease Brightness")

        if contrast < 40:
            rec.append("Increase Contrast")

        if under > 5.0:
            rec.append("Fix Underexposure")
        if over > 5.0:
            rec.append("Fix Overexposure")

        return rec

    # ----------------------------
    # Analyze (returns enriched dict)
    # ----------------------------
    def analyze(self):
        """Return a dictionary of all quality metrics including new enhancements."""
        blur = self.blur_score()
        bright = self.brightness()
        cont = self.contrast()
        noise = self.noise_score()
        width, height = self.resolution()
        under, over = self.exposure_clipping()
        score, rating = self.quality_score(blur, bright, cont, noise,
                                           width, height, under, over)
        rec = self.recommendations()
        s_map = self.sharpness_heatmap()

        return QualityMetrics(
            blur=round(float(blur), 2),
            brightness=round(float(bright), 2),
            contrast=round(float(cont), 2),
            noise=round(float(noise), 4),
            resolution=f"{width}x{height}",
            underexposed_pct=under,
            overexposed_pct=over,
            quality_score=score,
            quality_rating=rating,
            blur_confidence=self._blur_confidence(blur),
            noise_confidence=self._noise_confidence(),
            recommendations=rec,
            sharpness_map=s_map,
        )

    # ----------------------------
    # Print Report
    # ----------------------------
    def report(self):
        m = self.analyze()
        print("=" * 60)
        print(f"IMAGE : {os.path.basename(self.image_path)}")
        print("=" * 60)
        print(f"Resolution     : {m.resolution}")
        print(f"Blur Score     : {m.blur:.2f}  (conf {m.blur_confidence:.0%})")
        print(f"Brightness     : {m.brightness:.2f}")
        print(f"Contrast       : {m.contrast:.2f}")
        print(f"Noise (MAD)    : {m.noise:.4f}  (conf {m.noise_confidence:.0%})")
        print(f"Under-exposed  : {m.underexposed_pct:.1f}%")
        print(f"Over-exposed   : {m.overexposed_pct:.1f}%")
        print(f"Quality Score  : {m.quality_score:.1f} / 100  [{m.quality_rating}]")
        print("\nRecommended:")
        for r in m.recommendations:
            print(f"  [OK] {r}")
        print()


# -------------------------------------------------
# Process Entire Folder
# -------------------------------------------------

def process_folder(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                checker = ImageQualityChecker(image_path)
                checker.report()
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = input("Enter dataset folder path: ").strip()
    process_folder(folder)