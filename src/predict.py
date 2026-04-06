"""
predict.py — Single-Image Prediction for Image Forgery Detection
==================================================================

Classifies a single image as **Authentic** or **Forged** and
optionally generates a Grad-CAM heatmap highlighting suspicious
regions.

Usage
-----
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --gradcam
    python predict.py --image path/to/image.jpg --gradcam --save_heatmap

Author : Auto-generated for Deep Learning Based Image Forgery Detection
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ForgeryDetector
from utils import get_device, GradCAM, overlay_gradcam, compute_ela


# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────

CLASS_NAMES = {0: "Authentic", 1: "Forged"}
IMG_SIZE = 224

# ImageNet normalization
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────────────────
# Argument Parser
# ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict if an image is Authentic or Forged"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint "
                             "(default: outputs/models/best_model.pth)")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "mobilenet_v2"],
                        help="Backbone (auto-detected from checkpoint)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM heatmap")
    parser.add_argument("--save_heatmap", action="store_true",
                        help="Save Grad-CAM heatmap to outputs/results/")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# Load Model
# ────────────────────────────────────────────────────────────

def load_model(model_path, device, backbone_fallback="resnet50"):
    """Load the trained model from a checkpoint."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    backbone = checkpoint.get("backbone", backbone_fallback)
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
    model.eval()

    print(f"[INFO] Model loaded: {backbone}")
    print(f"[INFO] Input channels: {in_channels}")
    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"[INFO] Checkpoint val acc: "
          f"{checkpoint.get('val_acc', 'N/A'):.2f}%")

    return model, backbone, in_channels


# ────────────────────────────────────────────────────────────
# Predict
# ────────────────────────────────────────────────────────────

def predict_image(model, image_path, device, in_channels=6):
    """
    Predict the class of a single image.

    Returns
    -------
    pred_class : int (0 or 1)
    confidence : float (0–100%)
    class_name : str ('Authentic' or 'Forged')
    input_tensor : torch.Tensor (1, C, 224, 224) — for Grad-CAM
    """
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open image: {e}")
        sys.exit(1)

    if in_channels == 6:
        # Compute ELA and create 6-channel input
        ela_image = compute_ela(image)
        rgb_tensor = TRANSFORM(image)      # (3, 224, 224)
        ela_tensor = TRANSFORM(ela_image)   # (3, 224, 224)
        input_tensor = torch.cat([rgb_tensor, ela_tensor], dim=0)  # (6, 224, 224)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 6, 224, 224)
    else:
        input_tensor = TRANSFORM(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = probs.max(1)

    pred_class = pred_class.item()
    confidence = confidence.item() * 100.0
    class_name = CLASS_NAMES[pred_class]

    return pred_class, confidence, class_name, input_tensor


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(project_root, "outputs", "models",
                                  "best_model.pth")

    print("=" * 60)
    print(" Image Forgery Detection — Prediction")
    print("=" * 60)
    print(f" Image : {args.image}")
    print(f" Model : {model_path}")
    print("=" * 60)

    # ── Device ──
    device = get_device()

    # ── Load model ──
    model, backbone, in_channels = load_model(model_path, device, args.backbone)

    # ── Predict ──
    pred_class, confidence, class_name, input_tensor = predict_image(
        model, args.image, device, in_channels=in_channels
    )

    # ── Display result ──
    print("\n" + "=" * 60)
    print(" PREDICTION RESULT")
    print("=" * 60)
    print(f"   Image       : {os.path.basename(args.image)}")
    print(f"   Prediction  : {class_name}")
    print(f"   Confidence  : {confidence:.2f}%")

    if pred_class == 1:
        print("   ⚠ This image appears to be FORGED")
    else:
        print("   ✓ This image appears to be AUTHENTIC")
    print("=" * 60)

    # ── Grad-CAM (optional) ──
    if args.gradcam:
        print("\n[INFO] Generating Grad-CAM heatmap...")

        # Re-enable gradients for Grad-CAM
        input_tensor.requires_grad_(True)

        target_layer = model.get_gradcam_target_layer()
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor, target_class=pred_class)

        # Display or save
        if args.save_heatmap:
            results_dir = os.path.join(project_root, "outputs", "results")
            os.makedirs(results_dir, exist_ok=True)

            img_basename = os.path.splitext(
                os.path.basename(args.image))[0]
            heatmap_path = os.path.join(
                results_dir, f"gradcam_{img_basename}.png"
            )

            overlay = overlay_gradcam(args.image, heatmap,
                                      save_path=heatmap_path)
            print(f"[INFO] Heatmap saved to: {heatmap_path}")
        else:
            # Display inline
            overlay = overlay_gradcam(args.image, heatmap)

            # Also create a matplotlib figure
            import cv2
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            orig = cv2.cvtColor(
                cv2.resize(cv2.imread(args.image), (224, 224)),
                cv2.COLOR_BGR2RGB
            )
            axes[0].imshow(orig)
            axes[0].set_title("Original Image", fontsize=13)
            axes[0].axis("off")

            # Heatmap
            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
            axes[1].axis("off")

            # Overlay
            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title(
                f"Overlay — {class_name} ({confidence:.1f}%)",
                fontsize=13
            )
            axes[2].axis("off")

            fig.suptitle("Grad-CAM Visualization",
                         fontsize=15, fontweight="bold")
            fig.tight_layout()

            save_path = os.path.join(
                project_root, "outputs", "results",
                f"gradcam_viz_{os.path.splitext(os.path.basename(args.image))[0]}.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Grad-CAM visualization saved to: {save_path}")

    print("\n[DONE] Prediction complete!")


if __name__ == "__main__":
    main()
