# FewVision: Digital Image Processing based Defect Detection

**FewVision** is a comprehensive computer-vision pipeline built as a foundational project for the **Digital Image Processing (DIP)** subject. Designed specifically for MSMEs to detect defective products on rotating machines, this project heavily leverages core DIP algorithms to evaluate image quality, extract meaningful features, and generate visual diagnostic heatmaps before feeding data into a Few-Shot Learning architecture.

## Foundation in Digital Image Processing (DIP)

Digital Image Processing is the absolute core of this system. Rather than relying solely on black-box neural networks, FewVision uses mathematical and spatial image transformations to guarantee data quality and provide explainable visual diagnostics:

* **Edge Detection & Blur Analysis (Laplacian Variance):** Uses the second derivative of the image (Laplacian operator) on the grayscale matrix to calculate the variance of edges, mathematically determining if a product image is in focus.
* **Gradient Heatmaps (Sobel Operators):** Applies Sobel filters in both X and Y directions, calculating the gradient magnitude to generate a false-color heatmap that visually proves edge sharpness to the factory worker.
* **Spatial Filtering & Noise Estimation:** Applies a low-pass Gaussian Blur filter to smooth the image, then subtracts it from the original to isolate and calculate the high-frequency noise standard deviation.
* **Intensity Transformations:** Analyzes global grayscale pixel intensity (mean) and pixel distribution spread (standard deviation) to strictly evaluate lighting, overexposure, and contrast.

## Overview

The core of the system is the **Quality and Content Analysis Pipeline**, which automatically scans product images, computes essential metrics (blur, brightness, contrast, noise, object coverage, etc.), and generates a highly readable visual diagnostic report indicating whether a product passes or fails inspection.

### Key Features
* **Automated Quality Checks:** Instantly flags images that are too blurry, too dark, overexposed, or lack sufficient contrast.
* **Content Analysis:** Analyzes the physical properties of the product in the frame, such as orientation, center offset, and object coverage.
* **MSME-Friendly Visual Reports:** Generates an easy-to-read, side-by-side visual diagnostic for every scanned image. It includes:
  * The original image overlaid with a clear **PASS ✅** or **DEFECT ❌** status.
  * A gradient/edge heatmap (using Sobel operators) to visually prove image sharpness and highlight blurry areas.
  * Exact numerical scores for transparency.

## Pass / Fail Criteria

The system automatically marks an image as a **DEFECT** if it violates any of the following strict thresholds (configured in `quality.py`):
1. **Blur (Laplacian Variance):** Must be `>= 100`. (Below 100 is flagged as blurry).
2. **Brightness (Mean Pixel Intensity):** Must be between `70` and `180`.
3. **Contrast (Standard Deviation):** Must be `>= 40`.

## Installation

1. Clone this repository.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependencies include OpenCV, NumPy, Pandas, Albumentations, and Matplotlib).*

## Usage

1. Place your raw product images (e.g., screenshots from the machine camera) inside the `images/` directory.
2. Run the main pipeline script:
   ```bash
   python main.py
   ```
3. When prompted, press **Enter** to use the default `images` folder (or type a custom path).
4. The system will process the images and generate the visual diagnostics in the `reports/` folder.

## System Architecture

*(See `pipeline.png` for a visual flowchart)*
1. **Small Dataset** (5-20 images per class)
2. **Image Quality Check** (Blur, Brightness, Contrast)
3. **Content Analysis** (Background, object size, orientation)
4. **Adaptive Augmentation** (Meaningful transforms based on content)
5. **Feature Extraction** (Using pre-trained backbones like ResNet50, ViT, or CLIP)
6. **Few-Shot Learning Model** (Prototypical/Siamese/Matching Networks)
7. **Classification** (Pass / Defect)
