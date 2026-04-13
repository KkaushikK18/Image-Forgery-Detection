# 🔍 Deep Learning Based Image Forgery Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-84.39%25-brightgreen)]()
[![Recall](https://img.shields.io/badge/Forged%20Recall-89.26%25-brightgreen)]()

> A deep learning system that detects manipulated images using a dual-stream ResNet-50 CNN with Error Level Analysis (ELA) preprocessing and Grad-CAM visual explanations — achieving **84.39% accuracy** and **89.26% recall** on forged images.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Methodology](#-methodology)
  - [Error Level Analysis](#error-level-analysis-ela)
  - [Dual-Stream Pipeline](#dual-stream-input-pipeline)
  - [Model Architecture](#cnn-model-architecture)
  - [Transfer Learning Strategy](#transfer-learning-strategy)
  - [Training Configuration](#training-configuration)
  - [Grad-CAM Visualization](#grad-cam-visualization)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training-the-model)
  - [Evaluation](#evaluating-the-model)
  - [Single Image Prediction](#single-image-prediction)
  - [Streamlit Web App](#running-the-streamlit-web-app)
- [Dependencies](#-dependencies)
- [References](#-references)

---

## 🧠 Overview

Digital image forgery has become a serious threat to information integrity across journalism, legal proceedings, insurance documentation, and social media. Modern editing tools like Adobe Photoshop and GIMP can produce manipulations that are completely invisible to the human eye. This project addresses that threat with a fully automated, interpretable, end-to-end forgery detection system.

The system classifies any input image as either **Authentic** or **Forged**, and then produces a **Grad-CAM heatmap** that highlights the exact region the model found suspicious — providing forensically meaningful spatial evidence rather than just a binary verdict.

### Types of Forgery Detected

| Forgery Type | Description |
|---|---|
| **Image Splicing** | Combining regions from two or more different images into one composite |
| **Copy-Move Forgery** | Copying a region within an image and pasting it elsewhere in the same image |
| **Image Retouching** | Subtle alterations to features, colors, or blemishes |
| **Image Inpainting** | Removing objects and filling with synthesized background content |

---

## ✨ Key Features

- **Dual-stream 6-channel input** — fuses RGB appearance with ELA forensic artifact maps
- **Modified ResNet-50 backbone** — adapted for 6-channel input with warm initialization for ELA channels
- **Error Level Analysis (ELA) preprocessing** — amplifies compression inconsistencies invisible to the human eye
- **Selective transfer learning** — ImageNet pretrained weights fine-tuned with a layer-freezing strategy
- **Grad-CAM heatmap visualization** — spatially interprets model decisions by highlighting suspicious regions
- **Modern training pipeline** — Mixup augmentation, label smoothing, cosine annealing, class-weighted loss
- **Streamlit web application** — end-to-end deployable real-time inference interface
- **84.39% test accuracy** and **89.26% forged recall** on the combined CASIA 2.0 + Columbia benchmark

---

## 🏗 System Architecture

```
Input Image
    │
    ├──── RGB Stream ────────────────────────────────┐
    │     Resize(256) → RandomCrop(224) →            │
    │     Augment → Normalize → [3-ch tensor]        │
    │                                                 ├──→ [6-ch tensor] → ResNet-50 (modified)
    └──── ELA Stream ────────────────────────────────┘     Conv1(6→64) + Layers 1-4
          compute_ela() → Resize(256) →                          │
          RandomCrop(224) → Augment →                            ▼
          Normalize → [3-ch tensor]                    AdaptiveAvgPool(1×1)
                                                                 │
                                                                 ▼
                                                           Dropout(p=0.5)
                                                                 │
                                                                 ▼
                                                         Linear(2048 → 2)
                                                          /              \
                                                   Authentic           Forged
                                                   (Class 0)          (Class 1)
                                                         │
                                                         ▼
                                                  Grad-CAM (Layer4)
                                                         │
                                                         ▼
                                               Heatmap Visualization
```

---

## 📦 Datasets

This project trains and evaluates on a combined benchmark of two publicly available forensic datasets.

### CASIA 2.0 Image Tampering Dataset

| Property | Details |
|---|---|
| Authentic Images | ~7,491 (folder: `Au/`) |
| Tampered Images | ~5,123 (folder: `Tp/`) |
| Forgery Types | Splicing, Copy-Move |
| Formats | JPEG, BMP, TIFF |
| Ground Truth | Pixel-level masks available in `CASIA 2 Groundtruth/` |

> Download: [CASIA 2.0](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)

### Columbia Image Splicing Dataset

| Property | Details |
|---|---|
| Authentic Images | ~183 (folder: `4cam_auth/4cam_auth/`) |
| Spliced Images | ~180 (folder: `4cam_splc/4cam_splc/`) |
| Format | TIFF (uncompressed) |
| Cameras | 4 different camera models |

> Download: [Columbia Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/)

### Combined Dataset Summary

| Class | Count | Percentage | Label |
|---|---|---|---|
| Authentic | 7,674 | 59.1% | 0 |
| Forged | 5,303 | 40.9% | 1 |
| **Total** | **12,977** | **100%** | — |

**Dataset Split (70/15/15):**
- Training Set: ~9,084 images
- Validation Set: ~1,946 images
- Test Set: ~1,947 images (held out — never seen during training or tuning)

---

## 🔬 Methodology

### Error Level Analysis (ELA)

ELA is the core forensic preprocessing step. When a JPEG image is saved, the entire image undergoes uniform compression via the Discrete Cosine Transform (DCT) applied to 8×8 pixel blocks. If a region was edited or spliced from a different source, that region carries a **different compression history** — it was originally compressed at a different quality level, saved multiple times, or sourced from a lossless format. Upon re-compression at a fixed reference quality, authentic regions converge uniformly (small difference), while manipulated regions respond wildly (large difference).

**ELA Formula:**

```
E(x, y) = α × |I(x, y) − I_recomp(x, y)|
```

Where:
- `I(x, y)` — original image pixel value
- `I_recomp(x, y)` — pixel value after re-saving at JPEG quality Q = 90
- `α = 15` — scale factor to amplify the difference into the visible range

**ELA Computation Steps:**
1. Read original image `I`
2. Re-save `I` to a temporary buffer as JPEG at quality Q = 90, yielding `I_recomp`
3. Compute per-pixel absolute difference scaled by α = 15
4. Convert the difference map to a 3-channel RGB image for concatenation

**Interpretation of the ELA Map:**

| Region | ELA Appearance | Meaning |
|---|---|---|
| Authentic/Unaltered | Dark (near-black / dark blue) | Uniform compression history |
| Forged/Spliced | Bright (yellow / red / white) | Inconsistent compression history — manipulation detected |

**Impact of ELA (Ablation Study):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RGB only (3-channel) | ~79–80% | 0.71 | 0.82 | 0.76 |
| **RGB + ELA (6-channel)** | **84.39%** | **0.76** | **0.89** | **0.82** |

ELA provides a consistent **4–5 percentage point accuracy improvement** and a **7-point recall improvement**, confirming its critical role in the pipeline.

> **Limitation:** ELA is most effective on JPEG images. For lossless formats (PNG, TIFF), the ELA signal is near-zero, which accounts for the majority of the 83 false negative cases in the confusion matrix.

---

### Dual-Stream Input Pipeline

```
Input Image (RGB)
│
├── RGB Stream
│   └── Resize(256×256) → RandomCrop(224×224) → Augment → Normalize → [3-ch tensor]
│
└── ELA Stream
    └── compute_ela() → Resize(256×256) → RandomCrop(224×224) → Augment → Normalize → [3-ch tensor]
                                                                    │
                                                            Concatenate (channel-dim)
                                                                    │
                                                        [6-ch tensor ∈ ℝ^{6×224×224}]
```

> **Important:** The same random spatial transformation parameters (crop coordinates, flip, rotation angle) are applied identically to both streams to preserve spatial alignment between RGB and ELA.

**Training Augmentations:**

| Augmentation | Parameters | Purpose |
|---|---|---|
| Resize + RandomCrop | 256 → 224 | Position invariance |
| Horizontal Flip | p = 0.5 | Orientation invariance |
| Vertical Flip | p = 0.2 | Additional orientation invariance |
| Rotation | ±20° | Rotation invariance |
| Random Affine | translate=10%, scale=90–110% | Scale and translation invariance |
| Random Perspective | distortion=0.2, p=0.3 | Perspective robustness |
| Color Jitter | brightness=0.3, contrast=0.3 | Lighting variation robustness |
| Gaussian Blur | kernel=3, σ ∈ [0.1, 2.0] | Blur robustness |
| Random Erasing | p=0.25, scale=2–15% | Occlusion robustness |

**Validation/Test Preprocessing (no augmentation):**
```
Image → Resize(224×224) → ToTensor → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

### CNN Model Architecture

The backbone is **ResNet-50** (Residual Network, 50 layers) with a custom classification head.

#### Why ResNet-50?

ResNet-50 was selected over other pretrained architectures (VGGNet, AlexNet, Inception, DenseNet, EfficientNet) for three specific reasons:

1. **Excellent Grad-CAM compatibility** — ResNet-50's Layer4 produces clean 7×7 spatial feature maps, and its residual connections maintain strong gradient signal, producing spatially accurate and forensically interpretable heatmaps.

2. **Clean, modifiable first layer** — The initial Conv2d(3→64, 7×7) is easily replaced with Conv2d(6→64, 7×7). Pretrained RGB weights are preserved for channels 1–3 and duplicated (scaled by 0.5) for channels 4–6, enabling a principled warm initialization.

3. **Appropriate parameter scale for the dataset** — At 23.5M parameters, ResNet-50 is well-matched to ~9,000 training images. Larger architectures like VGGNet (138M parameters) would massively overfit without extreme regularization.

#### The Residual Connection — Solving Vanishing Gradients

The defining feature of ResNet is the skip connection:

```
y = F(x, {Wᵢ}) + x
```

Instead of learning a full transformation, each block learns only the *residual* — the difference from identity. Gradients can flow directly backward through the addition operation, bypassing the convolutional layers if needed. This solves the vanishing gradient problem that prevented plain CNNs from training effectively beyond ~20 layers.

#### First Layer Modification

```python
# Standard ResNet-50
nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

# Modified for 6-channel dual-stream input
nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)

# Weight initialization for ELA channels
new_conv.weight.data[:, :3] = pretrained_weights       # RGB channels — keep pretrained
new_conv.weight.data[:, 3:] = pretrained_weights * 0.5  # ELA channels — warm start
```

#### Complete Model Architecture

| Component | Configuration | Parameters |
|---|---|---|
| Input | 6 × 224 × 224 | — |
| Conv1 (modified) | 6→64, 7×7, stride=2, BN, ReLU | 18,816 |
| MaxPool | 3×3, stride=2 | — |
| Layer 1 | 3× Bottleneck[64] | 215,808 |
| Layer 2 | 4× Bottleneck[128] | 1,219,584 |
| Layer 3 | 6× Bottleneck[256] | 7,098,368 |
| Layer 4 | 3× Bottleneck[512] | 14,964,736 |
| AdaptiveAvgPool | 1×1 | — |
| Dropout | p = 0.5 | — |
| Linear (head) | 2048 → 2 | 4,098 |
| **Total** | | **~23.5M** |
| **Trainable (Phase 1)** | | **~21M (89.7%)** |

---

### Transfer Learning Strategy

| Layer | Phase 1 Status | Feature Type Learned |
|---|---|---|
| Conv1 + BN | **Frozen** | Edge and texture detectors |
| Layer 1 | **Frozen** | Low-level features (corners, blobs) |
| Layer 2 | **Frozen** | Mid-level textures and patterns |
| Layer 3 | **Trainable** | High-level forgery-specific features |
| Layer 4 | **Trainable** | Forgery-specific decision features |
| Classifier | **Trainable** | Binary Authentic vs. Forged decision |

**Phase 2 (Optional Fine-tuning):** All layers unfrozen, trained end-to-end at a very low learning rate. Only executed when sufficient data and compute are available.

---

### Training Configuration

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate, effective for transfer learning |
| Base Learning Rate | 1×10⁻⁴ | Conservative rate for fine-tuning pretrained weights |
| Weight Decay (L2) | 5×10⁻⁴ | L2 regularization to prevent overfitting |
| LR Scheduler | CosineAnnealingWarmRestarts | Smooth decay with periodic restarts (T₀=5, Tₘ=2) |
| Loss Function | CrossEntropyLoss | Standard multi-class classification loss |
| Label Smoothing | ε = 0.1 | Prevents overconfident probability predictions |
| Class Weights | Auth=0.845, Forged=1.224 | Compensates for 59:41 class imbalance |
| Batch Size | 32 | Balance of speed and GPU memory |
| Max Epochs | 25 | With early stopping |
| Early Stopping | patience = 7 | Stops when validation loss stagnates |
| Gradient Clipping | max_norm = 1.0 | Prevents gradient explosion |
| Mixup α | 0.2 | Creates virtual training examples; smooths decision boundary |

**Mixup Augmentation:**

```
x̃ = λxᵢ + (1−λ)xⱼ
ỹ = λyᵢ + (1−λ)yⱼ
where λ ~ Beta(α, α), α = 0.2
```

> Note: Because Mixup creates harder training examples, validation accuracy consistently exceeds training accuracy during mid-training epochs. This is expected behavior — not data leakage.

**Loss Function with Label Smoothing:**

```
L = −Σ y_smooth_c × log(p_c)
where y_smooth_c = (1−ε)y_c + ε/C
```

**Cosine Annealing Schedule:**

```
ηₜ = η_min + (1/2)(η_max − η_min)(1 + cos(T_cur/Tᵢ × π))
```

---

### Grad-CAM Visualization

Grad-CAM provides post-hoc spatial explanations by identifying which image regions most influenced the model's prediction.

**Computation Steps:**

```
Step 1 — Feature maps at last conv layer (Layer4):
    Aᵏ ∈ ℝ^{7×7}  for k = 1...2048

Step 2 — Global average pool gradients:
    αᵏ_c = (1/Z) Σᵢ Σⱼ (∂y_c / ∂Aᵏᵢⱼ)

Step 3 — Weighted combination with ReLU:
    L^c_Grad-CAM = ReLU(Σₖ αᵏ_c × Aᵏ)

Step 4 — Upsample 7×7 → 224×224 and overlay with JET colormap
```

**Heatmap Interpretation:**

| Color | Meaning |
|---|---|
| 🔴 Red / 🟡 Yellow | High activation — region strongly contributed to the forgery prediction |
| 🔵 Blue / ⚫ Dark | Low activation — region had minimal influence on the decision |

**For forged images:** Red/yellow concentrates on the manipulated region (spliced area, copy-moved region, inpainting boundary).

**For authentic images:** Activation is diffuse and spread across the image — no single suspicious focal region.

> Grad-CAM serves a dual purpose: during development, it validates that the model attends to forensically meaningful regions rather than spurious correlates; in deployment, it provides human analysts with interpretable spatial evidence.

---

## 📊 Results

### Overall Test Set Metrics (1,948 images, never seen during training)

| Metric | Score |
|---|---|
| **Accuracy** | **84.39%** |
| **Precision** | **75.74%** |
| **Recall** | **89.26%** |
| **F1 Score** | **81.95%** |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Authentic | 0.92 | 0.81 | 0.86 | 1,175 |
| Forged | 0.76 | 0.89 | 0.82 | 773 |
| **Weighted Avg** | **0.86** | **0.84** | **0.85** | **1,948** |

### Confusion Matrix

```
                    Predicted Authentic    Predicted Forged
Actual Authentic         954 (TN)              221 (FP)
Actual Forged             83 (FN)              690 (TP)
```

- **True Positives (690):** Forged images correctly identified ✅
- **True Negatives (954):** Authentic images correctly cleared ✅
- **False Positives (221):** Authentic images incorrectly flagged ⚠️ (typically high-texture authentic images)
- **False Negatives (83):** Forged images missed ❌ (predominantly PNG/TIFF lossless format images)

### Comparison with State-of-the-Art

| Method | Features | Accuracy | Dataset |
|---|---|---|---|
| ELA + SVM | Hand-crafted ELA | ~76% | CASIA |
| Dong et al. (2013) | SIFT + SVM | ~79% | CASIA 2.0 |
| Bayar & Stamm (2016) | Constrained CNN | ~82% | CASIA |
| Rao & Ni (2016) | Deep CNN | ~83% | CASIA |
| **Proposed (RGB+ELA)** | **ResNet-50 + ELA** | **84.39%** | **Combined** |
| ManTra-Net | Multi-task CNN | ~86%† | Various |
| Zhou et al. (2018) | Two-stream FRCNN | ~87%† | CASIA |

† Trained on substantially larger and more diverse datasets.

---

## 📁 Project Structure

```
image-forgery-detection/
│
├── data/
│   ├── CASIA2/
│   │   ├── Au/                          # Authentic images
│   │   ├── Tp/                          # Tampered images
│   │   └── CASIA 2 Groundtruth/         # Pixel-level masks
│   └── Columbia/
│       ├── 4cam_auth/4cam_auth/         # Authentic images
│       └── 4cam_splc/4cam_splc/         # Spliced images
│
├── src/
│   ├── dataset.py                       # Dataset loading and ELA computation
│   ├── model.py                         # ResNet-50 with 6-channel modification
│   ├── train.py                         # Training loop with Mixup, early stopping
│   ├── evaluate.py                      # Test set evaluation and confusion matrix
│   ├── gradcam.py                       # Grad-CAM implementation
│   ├── ela.py                           # Error Level Analysis utility
│   └── predict.py                       # Single image inference
│
├── app/
│   └── streamlit_app.py                 # Streamlit web application
│
├── checkpoints/
│   └── best_model.pth                   # Saved best model weights
│
├── outputs/
│   ├── loss_curves.png                  # Training vs validation loss
│   ├── accuracy_curves.png              # Training vs validation accuracy
│   ├── confusion_matrix.png             # Test set confusion matrix
│   └── gradcam_samples/                 # Example Grad-CAM visualizations
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended; CPU inference supported but slower)
- 8GB+ RAM

### Step 1 — Clone the Repository

```bash
git clone https://github.com/kaushik-kumar/image-forgery-detection.git
cd image-forgery-detection
```

### Step 2 — Create and Activate a Virtual Environment

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download and Organize Datasets

Download the CASIA 2.0 and Columbia datasets (links above) and place them in the `data/` directory following the structure shown in [Project Structure](#-project-structure).

---

## 🚀 Usage

### Training the Model

```bash
python src/train.py \
    --casia_path data/CASIA2 \
    --columbia_path data/Columbia \
    --epochs 25 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 25 | Maximum training epochs |
| `--batch_size` | 32 | Mini-batch size |
| `--lr` | 1e-4 | Base learning rate |
| `--weight_decay` | 5e-4 | L2 regularization coefficient |
| `--mixup_alpha` | 0.2 | Mixup Beta distribution parameter |
| `--label_smoothing` | 0.1 | Label smoothing epsilon |
| `--patience` | 7 | Early stopping patience |
| `--seed` | 42 | Random seed for reproducibility |

---

### Evaluating the Model

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --casia_path data/CASIA2 \
    --columbia_path data/Columbia \
    --output_dir outputs/
```

This generates the confusion matrix, per-class classification report, and saves visualization outputs.

---

### Single Image Prediction

```bash
python src/predict.py \
    --image path/to/your/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --gradcam \
    --output outputs/prediction_result.png
```

**Output:** Prediction label (Authentic / Forged), confidence percentage, and a side-by-side visualization of the original image, ELA map, and Grad-CAM overlay.

**Python API:**

```python
from src.predict import ForgeryDetector

detector = ForgeryDetector(checkpoint_path="checkpoints/best_model.pth")

result = detector.predict("path/to/image.jpg", visualize=True)

print(f"Prediction : {result['label']}")
print(f"Confidence : {result['confidence']:.2f}%")
print(f"Authentic  : {result['probabilities']['authentic']:.4f}")
print(f"Forged     : {result['probabilities']['forged']:.4f}")
# result['gradcam_overlay'] → numpy array of the heatmap overlay
```

---

### Running the Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

**Application workflow:**
1. **Upload** — Drag and drop any JPEG, PNG, BMP, or TIFF image
2. **ELA Preview** — Instantly see the computed ELA forensic map
3. **Inference** — The model runs in real-time and returns prediction + confidence
4. **Grad-CAM** — A heatmap overlay highlights the suspicious region
5. **Output** — Side-by-side display of original, ELA map, and annotated result

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.22.0
opencv-python>=4.7.0
tqdm>=4.65.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

<p align="center">
  Made with ❤️ &nbsp;|&nbsp; 
</p>
