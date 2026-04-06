"""
utils.py — Utility Functions for Image Forgery Detection
=========================================================

This module provides shared utility functions used across the project:
  - Device selection (CPU / GPU)
  - Plotting training curves (loss & accuracy)
  - Grad-CAM heatmap generation for model interpretability
  - Image de-normalization for visualization

Author : Auto-generated for Deep Learning Based Image Forgery Detection
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for servers & scripts
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ExifTags
import cv2


# ────────────────────────────────────────────────────────────
# 1.  Device Selection
# ────────────────────────────────────────────────────────────

def get_device():
    """
    Returns the best available device (CUDA GPU if present, else CPU).
    Prints which device was selected.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU Name : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Mem  : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


# ────────────────────────────────────────────────────────────
# 2.  Training / Validation Curve Plotting
# ────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str = "outputs/plots"):
    """
    Plot and save training vs. validation loss and accuracy curves.

    Parameters
    ----------
    history : dict
        Must contain keys: train_loss, val_loss, train_acc, val_acc
        Each key maps to a list of per-epoch values.
    save_dir : str
        Directory where the PNG plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Set a modern aesthetic
    sns.set_theme(style="darkgrid")

    # ── Loss Curve ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss",
            color="#FF6B6B", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s-", label="Val Loss",
            color="#4ECDC4", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training vs Validation Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss curve saved to {loss_path}")

    # ── Accuracy Curve ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], "o-", label="Train Accuracy",
            color="#FF6B6B", linewidth=2)
    ax.plot(epochs, history["val_acc"], "s-", label="Val Accuracy",
            color="#4ECDC4", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Training vs Validation Accuracy",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Accuracy curve saved to {acc_path}")


# ────────────────────────────────────────────────────────────
# 3.  Confusion-Matrix Plotting
# ────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels=("Authentic", "Forged"),
                          save_path="outputs/results/confusion_matrix.png"):
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray  (2×2)
        Confusion matrix from sklearn.metrics.confusion_matrix.
    labels : tuple
        Class label names.
    save_path : str
        File path to save the generated PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="white")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="gray", ax=ax,
                annot_kws={"size": 16})
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {save_path}")


# ────────────────────────────────────────────────────────────
# 4.  Grad-CAM Implementation
# ────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Generates a heatmap that highlights the image regions most
    responsible for the model's prediction.

    Usage
    -----
    >>> cam = GradCAM(model, target_layer=model.backbone.layer4[-1])
    >>> heatmap = cam.generate(input_tensor)
    """

    def __init__(self, model, target_layer):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The trained CNN model.
        target_layer : torch.nn.Module
            The convolutional layer to hook into for Grad-CAM
            (e.g. model.backbone.layer4[-1] for ResNet50).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook to capture activations
        self.target_layer.register_forward_hook(self._forward_hook)
        # Register backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Stores the feature-map activations from the target layer."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Stores the gradients flowing back through the target layer."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap for the given input.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input image tensor, shape (1, C, H, W).
        target_class : int or None
            Class index to explain. If None, the predicted class is used.

        Returns
        -------
        heatmap : np.ndarray
            Heatmap of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients
        self.model.zero_grad()

        # Backward pass for the target class score
        target_score = output[0, target_class]
        target_score.backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU — keep only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# ────────────────────────────────────────────────────────────
# 5.  Grad-CAM Overlay Visualization
# ────────────────────────────────────────────────────────────

def overlay_gradcam(image_path: str, heatmap: np.ndarray,
                    save_path: str = None, alpha: float = 0.5):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    image_path : str
        Path to the original image file.
    heatmap : np.ndarray
        Grad-CAM heatmap, shape (H, W), values in [0, 1].
    save_path : str or None
        If provided, save the overlay image to this path.
    alpha : float
        Blending factor for the heatmap overlay.

    Returns
    -------
    overlay : np.ndarray  (H, W, 3) BGR
    """
    # Load & resize image to match heatmap intent (224×224 model input)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to colour (JET colourmap)
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # Blend
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)
        print(f"[INFO] Grad-CAM overlay saved to {save_path}")

    return overlay


# ────────────────────────────────────────────────────────────
# 6.  De-normalize Image (for display)
# ────────────────────────────────────────────────────────────

def denormalize(tensor, mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)):
    """
    Reverse ImageNet normalization on a (C, H, W) tensor
    so it can be displayed with matplotlib / PIL.
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


# ────────────────────────────────────────────────────────────
# 7.  Error Level Analysis (ELA)
# ────────────────────────────────────────────────────────────

def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15):
    """
    Compute Error Level Analysis (ELA) of an image.

    ELA works by re-saving the image as JPEG at a known quality
    level and computing the pixel-level difference between the
    original and the re-saved version.

    Authentic images have uniform error levels across the entire
    image. Forged regions show **distinctly different** error
    levels because they were compressed a different number of times.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (RGB).
    quality : int
        JPEG re-compression quality (default: 90).
    scale : int
        Amplification factor for the difference image (default: 15).
        This makes the error patterns more visible.

    Returns
    -------
    ela_image : PIL.Image.Image
        The ELA image (RGB), same size as input.
    """
    import io

    # Re-compress the image as JPEG in memory
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Compute pixel-level difference and amplify
    original_arr = np.array(image, dtype=np.float32)
    recompressed_arr = np.array(recompressed, dtype=np.float32)

    # Absolute difference, amplified by scale factor
    ela_arr = np.abs(original_arr - recompressed_arr) * scale

    # Clip to valid range [0, 255]
    ela_arr = np.clip(ela_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(ela_arr, mode="RGB")


# ────────────────────────────────────────────────────────────
# 8.  Mixup Augmentation
# ────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation: blend pairs of training examples.

    Creates virtual training examples by taking convex combinations
    of pairs of examples and their labels. This regularizes the
    model and prevents overfitting.

    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

    Parameters
    ----------
    x : torch.Tensor  (B, C, H, W)
        Batch of input images.
    y : torch.Tensor  (B,)
        Batch of labels.
    alpha : float
        Mixup interpolation strength. Higher = more blending.
        0.2 is a good default.

    Returns
    -------
    mixed_x : torch.Tensor  (B, C, H, W)
    y_a : torch.Tensor  (B,)   — original labels
    y_b : torch.Tensor  (B,)   — shuffled labels
    lam : float                 — interpolation factor
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    # Random permutation for pairing
    index = torch.randperm(batch_size, device=x.device)

    # Blend images
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the Mixup loss: weighted combination of losses
    for the two blended label sets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ────────────────────────────────────────────────────────────
# 9.  EXIF Metadata Analysis
# ────────────────────────────────────────────────────────────

def analyze_exif(image: Image.Image):
    """
    Extracts EXIF metadata and checks for common forgery signatures.
    Returns:
        dict: Raw EXIF data with human-readable tags
        list: Warnings/anomalies found in the metadata
    """
    exif_data = image.getexif()
    if not exif_data:
        return {}, ["No EXIF metadata found. This is common if the image was downloaded from social media (which strips EXIF), but could also indicate deliberate metadata wiping."]
        
    readable_exif = {}
    warnings = []
    
    # Process standard EXIF tags
    for tag_id, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        # Avoid displaying huge binary chunks
        if isinstance(value, bytes):
            if len(value) > 64:
                value = f"<binary data: {len(value)} bytes>"
            else:
                try:
                    value = value.decode("utf-8", errors="ignore")
                except Exception:
                    value = "<undecodable bytes>"
        readable_exif[tag_name] = value

    # Process IFD (Image File Directory) EXIF tags
    try:
        ifd = image.getexif().get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in ifd.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            if isinstance(value, bytes):
                if len(value) > 64:
                    value = f"<binary data: {len(value)} bytes>"
                else:
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except Exception:
                        value = "<undecodable bytes>"
            readable_exif[f"EXIF_{tag_name}"] = value
    except Exception:
        pass

    # Deep Analysis for Forgery Indicators
    software = str(readable_exif.get("Software", "")).lower()
    suspicious_software = ["photoshop", "gimp", "lightroom", "paint", "canva", "snapseed"]
    if any(keyword in software for keyword in suspicious_software):
        warnings.append(f"Image was edited with a known image manipulation suite: {readable_exif.get('Software')}")

    # Check Date modification vs Origination
    datetime_orig = readable_exif.get("EXIF_DateTimeOriginal", readable_exif.get("DateTimeOriginal", ""))
    datetime_mod = readable_exif.get("DateTime", "")
    
    if datetime_orig and datetime_mod and (datetime_orig != datetime_mod):
        warnings.append(f"Modification date ({datetime_mod}) differs from the original creation date ({datetime_orig}).")
        
    return readable_exif, warnings
