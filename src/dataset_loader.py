"""
dataset_loader.py — Custom PyTorch Dataset for Image Forgery Detection
========================================================================

Combines images from **CASIA 2.0** and **Columbia Image Splicing**
datasets into a single binary classification dataset.

Label Mapping
-------------
    0 → Authentic
    1 → Forged

Folder Layout Expected
----------------------
    dataset/
        casia/
            Au/          ← authentic images  (.jpg, .bmp)
            Tp/          ← tampered images   (.tif, .jpg)
            CASIA 2 Groundtruth/  ← forgery masks (not used here)
        columbia/
            4cam_auth/
                4cam_auth/   ← authentic images (.tif)
            4cam_splc/
                4cam_splc/   ← spliced images   (.tif)

Note: Columbia has a *nested* subfolder structure. This loader
handles that automatically.

Author : Auto-generated for Deep Learning Based Image Forgery Detection
"""

import os
import glob
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ────────────────────────────────────────────────────────────
# Supported image extensions
# ────────────────────────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(filename: str) -> bool:
    """Return True if the file has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS


# ────────────────────────────────────────────────────────────
# 1.  Collect all image paths and labels
# ────────────────────────────────────────────────────────────

def collect_image_paths(dataset_root: str) -> Tuple[List[str], List[int]]:
    """
    Walk through the CASIA and Columbia dataset directories and
    build a combined list of (image_path, label) pairs.

    Parameters
    ----------
    dataset_root : str
        Path to the top-level ``dataset/`` directory.

    Returns
    -------
    image_paths : list[str]
    labels      : list[int]   (0 = Authentic, 1 = Forged)
    """
    image_paths: List[str] = []
    labels: List[int] = []

    # ── CASIA 2.0 ──────────────────────────────────────────
    casia_auth = os.path.join(dataset_root, "casia", "Au")
    casia_tamp = os.path.join(dataset_root, "casia", "Tp")

    if os.path.isdir(casia_auth):
        for f in os.listdir(casia_auth):
            if _is_image(f):
                image_paths.append(os.path.join(casia_auth, f))
                labels.append(0)  # Authentic

    if os.path.isdir(casia_tamp):
        for f in os.listdir(casia_tamp):
            if _is_image(f):
                image_paths.append(os.path.join(casia_tamp, f))
                labels.append(1)  # Forged

    # ── Columbia ───────────────────────────────────────────
    # Handle nested subfolder: 4cam_auth/4cam_auth/ and 4cam_splc/4cam_splc/
    columbia_auth_candidates = [
        os.path.join(dataset_root, "columbia", "4cam_auth", "4cam_auth"),
        os.path.join(dataset_root, "columbia", "4cam_auth"),
    ]
    columbia_splc_candidates = [
        os.path.join(dataset_root, "columbia", "4cam_splc", "4cam_splc"),
        os.path.join(dataset_root, "columbia", "4cam_splc"),
    ]

    columbia_auth = None
    for path in columbia_auth_candidates:
        if os.path.isdir(path):
            # Verify it contains actual image files (not just subdirs)
            if any(_is_image(f) for f in os.listdir(path)):
                columbia_auth = path
                break

    columbia_splc = None
    for path in columbia_splc_candidates:
        if os.path.isdir(path):
            if any(_is_image(f) for f in os.listdir(path)):
                columbia_splc = path
                break

    if columbia_auth:
        for f in os.listdir(columbia_auth):
            if _is_image(f):
                image_paths.append(os.path.join(columbia_auth, f))
                labels.append(0)  # Authentic
    else:
        print("[WARNING] Columbia authentic directory not found!")

    if columbia_splc:
        for f in os.listdir(columbia_splc):
            if _is_image(f):
                image_paths.append(os.path.join(columbia_splc, f))
                labels.append(1)  # Forged
    else:
        print("[WARNING] Columbia spliced directory not found!")

    # ── Summary ────────────────────────────────────────────
    n_auth = labels.count(0)
    n_forg = labels.count(1)
    print(f"[INFO] Dataset loaded: {len(image_paths)} images total")
    print(f"       ├─ Authentic : {n_auth}")
    print(f"       └─ Forged    : {n_forg}")

    return image_paths, labels


# ────────────────────────────────────────────────────────────
# 2.  Transforms (Train / Validation)
# ────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224):
    """
    Training data transforms with STRONG augmentation to prevent overfitting.

    Augmentations
    -------------
    - Random horizontal flip
    - Random vertical flip
    - Random rotation (±20°)
    - Random affine (translate, scale)
    - Random perspective distortion
    - Stronger brightness/contrast/saturation jitter
    - Gaussian blur
    - Random erasing (after ToTensor)
    - Resize → 224×224
    - ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # Resize slightly larger
        transforms.RandomCrop((img_size, img_size)),         # Then random crop to target
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # After ToTensor
    ])


def get_val_transforms(img_size: int = 224):
    """
    Validation / test transforms — no augmentation, just resize + normalize.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ────────────────────────────────────────────────────────────
# 3.  PyTorch Dataset Class (with ELA support)
# ────────────────────────────────────────────────────────────

class ForgeryDataset(Dataset):
    """
    Custom PyTorch Dataset for binary image forgery detection.

    When use_ela=True, each sample returns:
        ((rgb_image, ela_image), label)
    When use_ela=False:
        (rgb_image, label)

    Labels: 0 = Authentic, 1 = Forged
    """

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None, use_ela: bool = True):
        """
        Parameters
        ----------
        image_paths : list[str]
            Absolute paths to all image files.
        labels : list[int]
            Corresponding binary labels (0 or 1).
        transform : torchvision.transforms.Compose or None
            Image transform pipeline.
        use_ela : bool
            If True, compute ELA and return alongside the RGB image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_ela = use_ela

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image — convert to RGB to handle grayscale / RGBA images
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Could not load {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.use_ela:
            # Compute ELA — returns a PIL Image
            from utils import compute_ela
            ela_image = compute_ela(image)
            return (image, ela_image), label
        else:
            return image, label


# ────────────────────────────────────────────────────────────
# 4.  DataLoader Factory
# ────────────────────────────────────────────────────────────

def get_dataloaders(dataset_root: str,
                    batch_size: int = 32,
                    img_size: int = 224,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    num_workers: int = 4,
                    seed: int = 42,
                    use_ela: bool = True):
    """
    Create train / val / test DataLoader objects.

    When use_ela=True, returns 6-channel tensors (RGB + ELA).
    When use_ela=False, returns standard 3-channel RGB tensors.

    Parameters
    ----------
    dataset_root : str
        Path to the ``dataset/`` directory.
    batch_size : int
    img_size : int
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    num_workers : int
    seed : int
    use_ela : bool
        If True, compute ELA and create 6-channel input.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    # Collect paths
    image_paths, labels = collect_image_paths(dataset_root)

    # Full dataset (no transform yet — transforms applied per-split later)
    full_dataset = ForgeryDataset(image_paths, labels, transform=None,
                                  use_ela=use_ela)

    # Compute split sizes
    total = len(full_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # remainder goes to test

    print(f"[INFO] Split sizes: train={train_size}  val={val_size}  "
          f"test={test_size}")
    print(f"[INFO] ELA mode: {'ENABLED (6-ch)' if use_ela else 'DISABLED (3-ch)'}")

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=generator
    )

    # Wrap subsets with per-split transforms
    train_transforms = get_train_transforms(img_size)
    val_transforms = get_val_transforms(img_size)

    train_ds = TransformSubset(train_subset, train_transforms, use_ela=use_ela)
    val_ds = TransformSubset(val_subset, val_transforms, use_ela=use_ela)
    test_ds = TransformSubset(test_subset, val_transforms, use_ela=use_ela)

    # Build DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


# ────────────────────────────────────────────────────────────
# 5.  Helper: Apply transforms to a Subset
# ────────────────────────────────────────────────────────────

class TransformSubset(Dataset):
    """
    Wraps a ``torch.utils.data.Subset`` and applies a transform
    to images returned by the underlying dataset.

    When use_ela=True, applies the same transform to both the RGB
    and ELA images, then concatenates them into a 6-channel tensor.
    """

    def __init__(self, subset, transform, use_ela: bool = True):
        self.subset = subset
        self.transform = transform
        self.use_ela = use_ela

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]

        if self.use_ela:
            # data is a tuple: (rgb_pil_image, ela_pil_image)
            rgb_image, ela_image = data

            if self.transform:
                # Apply same resize + normalize to both
                # Note: augmentation (flip, rotation) is applied independently
                # which is fine — both get resized and normalized consistently
                rgb_tensor = self.transform(rgb_image)
                ela_tensor = self.transform(ela_image)

            # Concatenate along channel dimension: (6, H, W)
            combined = torch.cat([rgb_tensor, ela_tensor], dim=0)
            return combined, label
        else:
            # data is a PIL Image
            if self.transform:
                data = self.transform(data)
            return data, label


# ────────────────────────────────────────────────────────────
# 6.  Quick Sanity Check
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to verify the dataset loads correctly.

        python dataset_loader.py

    Expected output: dataset stats + a sample batch shape.
    """
    import sys

    # Resolve dataset root relative to this file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "dataset")

    print("=" * 60)
    print(" Dataset Loader — Sanity Check")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=8,
        num_workers=0  # 0 for quick debug on Windows
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))
    print(f"\n[OK] Sample batch:")
    print(f"     Images shape : {images.shape}")   # (8, 3, 224, 224)
    print(f"     Labels       : {labels.tolist()}")
    print("=" * 60)
