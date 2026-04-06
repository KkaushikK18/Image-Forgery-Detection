"""
evaluate.py — Model Evaluation for Image Forgery Detection
============================================================

Loads the best saved model and evaluates it on the test set.

Computes
--------
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix (saved as PNG)
    - Full classification report

Usage
-----
    python evaluate.py
    python evaluate.py --backbone resnet50 --batch_size 32

Author : Auto-generated for Deep Learning Based Image Forgery Detection
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import get_dataloaders
from model import ForgeryDetector
from utils import get_device, plot_confusion_matrix


# ────────────────────────────────────────────────────────────
# 1.  Argument Parser
# ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Image Forgery Detection Model"
    )
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "mobilenet_v2"],
                        help="CNN backbone (must match the trained model)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (default: outputs/models/best_model.pth)")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# 2.  Evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    Run the model on the test set and collect all predictions.

    Returns
    -------
    all_labels : np.ndarray
    all_preds  : np.ndarray
    all_probs  : np.ndarray  (softmax probabilities)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(test_loader, desc="Evaluating", leave=True,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


# ────────────────────────────────────────────────────────────
# 3.  Main
# ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "dataset")
    results_dir = os.path.join(project_root, "outputs", "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(project_root, "outputs", "models",
                                  "best_model.pth")

    print("=" * 60)
    print(" Image Forgery Detection — Evaluation")
    print("=" * 60)

    # ── Device ──
    device = get_device()

    # ── Load checkpoint ──
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        print("        Please train the model first using train.py")
        sys.exit(1)

    print(f"\n[STEP 1] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract config from checkpoint (fallback to defaults)
    backbone = checkpoint.get("backbone", args.backbone)
    dropout = checkpoint.get("dropout", 0.5)
    in_channels = checkpoint.get("in_channels", 6)

    model = ForgeryDetector(
        backbone_name=backbone,
        num_classes=2,
        pretrained=False,
        dropout=dropout,
        in_channels=in_channels
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"[INFO] Checkpoint val accuracy: "
          f"{checkpoint.get('val_acc', 'N/A'):.2f}%")

    # ── Load test data ──
    print("\n[STEP 2] Loading test dataset...")
    use_ela = (in_channels == 6)
    _, _, test_loader = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ela=use_ela
    )

    # ── Evaluate ──
    print("\n[STEP 3] Running evaluation on test set...")
    all_labels, all_preds, all_probs = evaluate_model(
        model, test_loader, device
    )

    # ── Compute Metrics ──
    print("\n[STEP 4] Computing metrics...")

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary",
                           pos_label=1)
    rec = recall_score(all_labels, all_preds, average="binary",
                       pos_label=1)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)

    class_names = ["Authentic", "Forged"]

    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f" Accuracy  : {acc * 100:.2f}%")
    print(f" Precision : {prec * 100:.2f}%")
    print(f" Recall    : {rec * 100:.2f}%")
    print(f" F1 Score  : {f1 * 100:.2f}%")
    print("=" * 60)

    print("\n Confusion Matrix:")
    print(f"   {'':>12} {'Pred Auth':>10} {'Pred Forg':>10}")
    print(f"   {'Actual Auth':>12} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"   {'Actual Forg':>12} {cm[1][0]:>10} {cm[1][1]:>10}")

    print("\n Full Classification Report:")
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names)
    print(report)

    # ── Save results ──
    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "total_test_samples": len(all_labels),
        "backbone": backbone
    }

    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {results_path}")

    # ── Plot confusion matrix ──
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels=class_names, save_path=cm_path)

    print("\n[DONE] Evaluation complete!")


if __name__ == "__main__":
    main()
