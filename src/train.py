"""
train.py — Training Pipeline for Image Forgery Detection
==========================================================

End-to-end training script:
    1. Load & split the combined CASIA + Columbia dataset
    2. Initialize the pretrained CNN model
    3. Train with CrossEntropyLoss + Adam optimizer
    4. Track per-epoch train/val loss & accuracy
    5. Save the best model (by val accuracy)
    6. Plot training curves after training completes

Usage
-----
    python train.py

    # Or with custom arguments:
    python train.py --backbone resnet50 --epochs 20 --batch_size 32 --lr 0.0001

Author : Auto-generated for Deep Learning Based Image Forgery Detection
"""

import os
import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import get_dataloaders
from model import ForgeryDetector
from utils import get_device, plot_training_curves, mixup_data, mixup_criterion


# ────────────────────────────────────────────────────────────
# 1.  Argument Parser
# ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Image Forgery Detection Model"
    )
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "mobilenet_v2"],
                        help="CNN backbone to use (default: resnet50)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# 2.  Single Epoch: Train
# ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup: bool = True, mixup_alpha: float = 0.2):
    """
    Train for one epoch with optional Mixup augmentation.

    Returns
    -------
    avg_loss : float
    accuracy : float (percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply Mixup augmentation
        if use_mixup:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels,
                                                      alpha=mixup_alpha)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Stats — for accuracy we use the original (non-mixed) labels
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ────────────────────────────────────────────────────────────
# 3.  Single Epoch: Validate
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validate for one epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float (percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ────────────────────────────────────────────────────────────
# 4.  Main Training Loop
# ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Paths — resolve relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "dataset")
    model_save_dir = os.path.join(project_root, "outputs", "models")
    plots_save_dir = os.path.join(project_root, "outputs", "plots")
    results_dir = os.path.join(project_root, "outputs", "results")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(plots_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print(" Image Forgery Detection — Training")
    print("=" * 60)
    print(f" Backbone     : {args.backbone}")
    print(f" Epochs       : {args.epochs}")
    print(f" Batch Size   : {args.batch_size}")
    print(f" Learning Rate: {args.lr}")
    print(f" Image Size   : {args.img_size}")
    print(f" Dropout      : {args.dropout}")
    print("=" * 60)

    # ── Device ──
    device = get_device()

    # ── Data ──
    print("\n[STEP 1] Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_ela=True  # Enable ELA: 6-channel input (RGB + ELA)
    )

    # ── Model ──
    print("\n[STEP 2] Building model...")
    model = ForgeryDetector(
        backbone_name=args.backbone,
        num_classes=2,
        pretrained=True,
        dropout=args.dropout,
        in_channels=6  # RGB (3) + ELA (3) = 6 channels
    ).to(device)

    # ── Class-Weighted Loss with Label Smoothing ──
    # Handles imbalance: ~7674 authentic vs ~5303 forged
    # Higher weight for the minority class (forged) improves recall
    n_authentic = 7674  # approximate counts from dataset
    n_forged = 5303
    total_samples = n_authentic + n_forged
    weight_authentic = total_samples / (2.0 * n_authentic)  # ~0.845
    weight_forged = total_samples / (2.0 * n_forged)        # ~1.224
    class_weights = torch.tensor([weight_authentic, weight_forged],
                                  dtype=torch.float32).to(device)
    print(f"[INFO] Class weights: Authentic={weight_authentic:.3f}, "
          f"Forged={weight_forged:.3f}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1  # Prevents overconfident predictions → reduces overfitting
    )

    # ── Optimizer with stronger weight decay ──
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4  # Increased L2 regularization (was 1e-4)
    )

    # ── Cosine Annealing LR Scheduler ──
    # Smoother than ReduceLROnPlateau — gradually decays LR following cosine curve
    # T_0=5 means LR restarts every 5 epochs, T_mult=2 doubles the period each restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # ── Training History ──
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_epoch = 0

    # ── Early Stopping ──
    early_stop_patience = 7  # Stop if val loss doesn't improve for 7 epochs
    early_stop_counter = 0
    best_val_loss = float("inf")

    # ── Training Loop ──
    print("\n[STEP 3] Starting training...\n")
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch [{epoch}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Step scheduler (cosine annealing steps per epoch)
        scheduler.step(epoch)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ── Save best model (by val accuracy) ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "backbone": args.backbone,
                "num_classes": 2,
                "dropout": args.dropout,
                "in_channels": 6,  # Save so evaluate/predict know the architecture
            }, best_model_path)
            print(f"  ★ New best model saved! (Val Acc: {val_acc:.2f}%)")

        # ── Early Stopping check (based on val loss) ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  ⚠ No val loss improvement for "
                  f"{early_stop_counter}/{early_stop_patience} epochs")

        if early_stop_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] Validation loss hasn't improved for "
                  f"{early_stop_patience} epochs. Stopping training.")
            break

        print()

    total_time = time.time() - total_start

    # ── Summary ──
    print("=" * 60)
    print(" Training Complete!")
    print("=" * 60)
    print(f" Total Time    : {total_time / 60:.1f} minutes")
    print(f" Best Epoch    : {best_epoch}")
    print(f" Best Val Acc  : {best_val_acc:.2f}%")
    print(f" Model Saved   : {best_model_path}")
    print("=" * 60)

    # ── Save training history to JSON ──
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Training history saved to {history_path}")

    # ── Plot curves ──
    print("\n[STEP 4] Generating training plots...")
    plot_training_curves(history, save_dir=plots_save_dir)

    print("\n[DONE] Training pipeline finished successfully!")


if __name__ == "__main__":
    main()
