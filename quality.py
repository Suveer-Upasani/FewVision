import os
import cv2
import numpy as np


class ImageQualityChecker:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Cannot open image: {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

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
    # Noise Estimate
    # ----------------------------
    def noise_score(self):
        blur = cv2.GaussianBlur(self.gray, (3, 3), 0)
        noise = self.gray.astype(np.float32) - blur.astype(np.float32)
        return np.std(noise)

    # ----------------------------
    # Recommendations
    # ----------------------------
    def recommendations(self):

        rec = []
        avoid = []

        blur = self.blur_score()
        bright = self.brightness()
        contrast = self.contrast()

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

        return rec, avoid

    # ----------------------------
    # Print Report
    # ----------------------------
    def report(self):

        blur = self.blur_score()
        bright = self.brightness()
        contrast = self.contrast()
        noise = self.noise_score()
        width, height = self.resolution()

        rec, avoid = self.recommendations()

        print("=" * 60)
        print(f"IMAGE : {os.path.basename(self.image_path)}")
        print("=" * 60)

        print(f"Resolution : {width} x {height}")
        print(f"Blur Score : {blur:.2f}")
        print(f"Brightness : {bright:.2f}")
        print(f"Contrast   : {contrast:.2f}")
        print(f"Noise      : {noise:.2f}")

        print("\nRecommended:")
        for r in rec:
            print(f"  ✓ {r}")

        if avoid:
            print("\nAvoid:")
            for a in avoid:
                print(f"  ✗ {a}")

        print("\n")


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

    folder = input("Enter dataset folder path: ").strip()

    process_folder(folder)