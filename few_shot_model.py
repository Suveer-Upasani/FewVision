# few_shot_model.py
"""Member 3 Task: Few-Shot Learning Classifier using Prototypical Networks.

This module takes the feature vectors extracted by Member 2 (feature_extraction.py)
and builds a Prototypical Network classifier. 

How Prototypical Networks work:
  1. For each class (PASS / DEFECT), compute the mean of all its feature vectors.
     This mean vector is called the class "prototype".
  2. To classify a new image, compute its Euclidean distance to each prototype.
  3. The class with the nearest prototype wins — closest = predicted label.

This approach is ideal for Few-Shot Learning because it only requires a small number
of labeled examples to define a reliable class prototype.

Inputs:
  - features.npy   -> Shape [N, 2048] feature vectors from ResNet50 (Member 2).
  - labels.npy     -> Shape [N,] filenames corresponding to each feature row.

Outputs:
  - prototypes.npy       -> The computed class prototype vectors.
  - few_shot_report.png  -> Visual confusion matrix + accuracy report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
FEATURES_FILE   = "features.npy"
LABELS_FILE     = "labels.npy"
PROTOTYPES_FILE = "prototypes.npy"
REPORTS_DIR     = "reports"


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def infer_class_from_filename(filename: str) -> str:
    """Heuristic: derive a class label from the filename.
    
    Convention used:
      - Filenames containing 'blur', 'dark', 'noise', 'defect' -> DEFECT
      - Everything else                                         -> PASS
    
    You can extend this list to match your specific dataset naming convention.
    """
    name_lower = filename.lower()
    defect_keywords = ["blur", "dark", "noise", "defect", "bad", "broken", "damaged"]
    for kw in defect_keywords:
        if kw in name_lower:
            return "DEFECT"
    return "PASS"


# -----------------------------------------------------------------------
# Core: Prototypical Network
# -----------------------------------------------------------------------
class PrototypicalNetwork:
    """Simple Prototypical Network for binary Few-Shot classification.

    The prototype for each class is simply the mean feature vector
    of all support (labeled) examples belonging to that class.
    """

    def __init__(self):
        self.prototypes: dict[str, np.ndarray] = {}
        self.class_names: list[str] = []

    def fit(self, features: np.ndarray, class_labels: list[str]):
        """Compute class prototypes from support set.

        Parameters
        ----------
        features     : np.ndarray, shape [N, D]
        class_labels : list of str, length N
        """
        self.class_names = sorted(list(set(class_labels)))
        for cls in self.class_names:
            indices = [i for i, l in enumerate(class_labels) if l == cls]
            cls_features = features[indices]
            self.prototypes[cls] = cls_features.mean(axis=0)
            print(f"  Prototype for '{cls}': computed from {len(indices)} support images.")

    def predict(self, feature_vector: np.ndarray) -> tuple[str, dict]:
        """Classify a single feature vector by nearest-prototype rule.

        Returns
        -------
        predicted_class : str
        distances       : dict mapping class_name -> distance
        """
        distances = {
            cls: euclidean_distance(feature_vector, proto)
            for cls, proto in self.prototypes.items()
        }
        predicted = min(distances, key=distances.get)
        return predicted, distances

    def predict_batch(self, features: np.ndarray) -> list[str]:
        """Classify a batch of feature vectors."""
        return [self.predict(f)[0] for f in features]

    def save_prototypes(self, path: str = PROTOTYPES_FILE):
        """Persist the computed prototype vectors to disk."""
        np.save(path, self.prototypes)
        print(f"  Prototypes saved to: {path}")


# -----------------------------------------------------------------------
# Evaluation & Reporting
# -----------------------------------------------------------------------
def evaluate(true_labels: list, pred_labels: list, class_names: list) -> dict:
    """Compute accuracy, per-class precision, recall metrics."""
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / len(true_labels) if true_labels else 0.0

    metrics = {"accuracy": accuracy, "per_class": {}}
    for cls in class_names:
        tp = sum(t == cls and p == cls for t, p in zip(true_labels, pred_labels))
        fp = sum(t != cls and p == cls for t, p in zip(true_labels, pred_labels))
        fn = sum(t == cls and p != cls for t, p in zip(true_labels, pred_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics["per_class"][cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return metrics


def generate_report(
    filenames: list,
    true_labels: list,
    pred_labels: list,
    metrics: dict,
    class_names: list,
    output_path: str,
):
    """Generate a clean visual classification report image."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    n = len(filenames)
    # Confusion matrix
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    cls_idx = {c: i for i, c in enumerate(class_names)}
    for t, p in zip(true_labels, pred_labels):
        cm[cls_idx[t]][cls_idx[p]] += 1

    fig = plt.figure(figsize=(16, 8), facecolor="#0f0f1a")
    fig.suptitle("FewVision — Few-Shot Classification Report", fontsize=18, color="white", weight="bold", y=0.98)

    # ---- Left: Confusion Matrix ----
    ax1 = fig.add_axes([0.04, 0.12, 0.38, 0.78])
    im = ax1.imshow(cm, cmap="Blues", vmin=0)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, color="white", fontsize=13)
    ax1.set_yticklabels(class_names, color="white", fontsize=13)
    ax1.set_xlabel("Predicted Label", color="#aaaaaa", fontsize=12)
    ax1.set_ylabel("True Label", color="#aaaaaa", fontsize=12)
    ax1.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333355")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "#222244",
                     fontsize=22, weight="bold")

    # ---- Middle: Per-class metrics bar chart ----
    ax2 = fig.add_axes([0.48, 0.12, 0.24, 0.78])
    bar_colors   = ["#5e81f4", "#e57373"]
    metric_names = ["Precision", "Recall", "F1"]
    x = np.arange(len(metric_names))
    width = 0.28
    for idx, cls in enumerate(class_names):
        vals = [
            metrics["per_class"][cls]["precision"],
            metrics["per_class"][cls]["recall"],
            metrics["per_class"][cls]["f1"],
        ]
        offset = (idx - len(class_names) / 2 + 0.5) * width
        bars = ax2.bar(x + offset, vals, width, label=cls,
                       color=bar_colors[idx % len(bar_colors)], alpha=0.88, edgecolor="#0f0f1a")
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=9)

    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, color="white", fontsize=11)
    ax2.set_title("Per-Class Metrics", color="white", fontsize=14, pad=12)
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#333355")
    ax2.spines["left"].set_color("#333355")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.label.set_color("white")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=10)

    # ---- Right: Summary card ----
    ax3 = fig.add_axes([0.76, 0.12, 0.22, 0.78])
    ax3.set_facecolor("#1a1a2e")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("Summary", color="white", fontsize=14, pad=12)

    acc = metrics["accuracy"]
    acc_color = "#66bb6a" if acc >= 0.8 else "#ffa726" if acc >= 0.6 else "#ef5350"
    ax3.text(0.5, 0.82, f"{acc * 100:.1f}%", ha="center", va="center",
             fontsize=42, color=acc_color, weight="bold")
    ax3.text(0.5, 0.70, "Overall Accuracy", ha="center", va="center",
             fontsize=11, color="#aaaaaa")

    ax3.axhline(0.63, color="#333355", linewidth=1)

    summary_lines = [
        ("Model", "Prototypical Net"),
        ("Backbone", "ResNet50"),
        ("Images", str(n)),
        ("Classes", ", ".join(class_names)),
    ]
    for i, (k, v) in enumerate(summary_lines):
        y = 0.54 - i * 0.10
        ax3.text(0.08, y, k, ha="left", va="center", fontsize=10, color="#aaaaaa")
        ax3.text(0.92, y, v, ha="right", va="center", fontsize=10, color="white", weight="bold")

    for spine in ax3.spines.values():
        spine.set_edgecolor("#333355")

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    print(f"  Report saved to: {output_path}")


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------
def run(
    features_file: str = FEATURES_FILE,
    labels_file: str   = LABELS_FILE,
):
    """Main entry point for Member 3's Few-Shot classification task."""

    # 1. Load features and filenames from Member 2's output
    print("Loading feature vectors from Member 2...")
    features  = np.load(features_file, allow_pickle=True)
    filenames = np.load(labels_file,   allow_pickle=True).tolist()

    print(f"  Loaded {features.shape[0]} images, each with {features.shape[1]} features.")

    # 2. Infer class labels from filenames
    class_labels = [infer_class_from_filename(f) for f in filenames]
    print(f"  Class distribution: { {c: class_labels.count(c) for c in set(class_labels)} }")

    # 3. Train the Prototypical Network (compute prototypes)
    print("Training Prototypical Network...")
    model = PrototypicalNetwork()
    model.fit(features, class_labels)
    model.save_prototypes()

    # 4. Predict on the entire dataset (self-evaluation on support set)
    print("Running classification on all images...")
    predictions = model.predict_batch(features)

    # 5. Evaluate
    metrics = evaluate(class_labels, predictions, model.class_names)
    acc = metrics["accuracy"]
    print(f"\nFew-Shot Classification complete!")
    print(f"  Overall Accuracy : {acc * 100:.1f}%")
    for cls, m in metrics["per_class"].items():
        print(f"  [{cls}] Precision={m['precision']:.2f}  Recall={m['recall']:.2f}  F1={m['f1']:.2f}")

    # 6. Generate visual report
    report_path = os.path.join(REPORTS_DIR, "few_shot_report.png")
    generate_report(filenames, class_labels, predictions, metrics, model.class_names, report_path)

    return model, metrics


if __name__ == "__main__":
    run()
