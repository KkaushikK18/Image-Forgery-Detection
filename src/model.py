"""
model.py — CNN Model Architecture for Image Forgery Detection
==============================================================

Implements a transfer-learning-based binary classifier using
pretrained backbones from torchvision.

Supported backbones
-------------------
    - resnet50      (default — best accuracy / Grad-CAM compatibility)
    - efficientnet_b0
    - mobilenet_v2  (lightweight — good for deployment)

The final classification head is replaced with:
    AdaptiveAvgPool → Dropout → Linear(features, 2)

Author : Auto-generated for Deep Learning Based Image Forgery Detection
"""

import torch
import torch.nn as nn
from torchvision import models


class ForgeryDetector(nn.Module):
    """
    Binary image forgery detector using a pretrained CNN backbone.

    Parameters
    ----------
    backbone_name : str
        One of 'resnet50', 'efficientnet_b0', 'mobilenet_v2'.
    num_classes : int
        Number of output classes (default=2: Authentic vs Forged).
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    dropout : float
        Dropout probability before the final linear layer.
    """

    SUPPORTED_BACKBONES = {"resnet50", "efficientnet_b0", "mobilenet_v2"}

    def __init__(self, backbone_name: str = "resnet50",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout: float = 0.5,
                 in_channels: int = 6):

        super().__init__()

        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ── Build backbone ──────────────────────────────────
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features       # 2048

            # Remove original FC layer — we attach our own
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            # After backbone: (B, 2048, 7, 7) for 224×224 input

            # Modify first conv layer for custom input channels
            if in_channels != 3:
                old_conv = self.backbone[0]  # Conv2d(3, 64, 7, stride=2, padding=3)
                new_conv = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                # Initialize: copy pretrained weights for first 3 channels,
                # duplicate them for the remaining ELA channels
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, :3] = old_conv.weight
                        new_conv.weight[:, 3:] = old_conv.weight[:, :in_channels-3]
                        if old_conv.bias is not None:
                            new_conv.bias = old_conv.bias
                self.backbone[0] = new_conv

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features  # 1280

            # Remove classifier
            self.backbone = base.features

            # Modify first conv for custom channels
            if in_channels != 3:
                old_conv = self.backbone[0][0]  # First conv in features
                new_conv = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, :3] = old_conv.weight
                        new_conv.weight[:, 3:] = old_conv.weight[:, :in_channels-3]
                self.backbone[0][0] = new_conv

        elif backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
            in_features = base.classifier[1].in_features  # 1280

            self.backbone = base.features

            # Modify first conv for custom channels
            if in_channels != 3:
                old_conv = self.backbone[0][0]  # First conv in features
                new_conv = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                if pretrained:
                    with torch.no_grad():
                        new_conv.weight[:, :3] = old_conv.weight
                        new_conv.weight[:, 3:] = old_conv.weight[:, :in_channels-3]
                self.backbone[0][0] = new_conv

        # ── Classification head ─────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → (B, C, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        # Freeze early backbone layers for fine-tuning efficiency
        self._freeze_early_layers()

        print(f"[INFO] Model created: {backbone_name}")
        print(f"       ├─ Input channels  : {in_channels}")
        print(f"       ├─ Backbone features : {in_features}")
        print(f"       ├─ Num classes        : {num_classes}")
        print(f"       ├─ Dropout            : {dropout}")
        print(f"       └─ Pretrained         : {pretrained}")

    # ────────────────────────────────────────────────────────
    # Freeze early layers (only fine-tune later layers)
    # ────────────────────────────────────────────────────────

    def _freeze_early_layers(self):
        """
        Freeze the first ~50% of backbone parameters.
        With stronger augmentation, we can safely train more layers
        to learn forgery-specific features deeper in the network.
        """
        params = list(self.backbone.parameters())
        freeze_until = int(len(params) * 0.50)

        for i, param in enumerate(params):
            if i < freeze_until:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Parameters: {trainable:,} trainable / {total:,} total "
              f"({100 * trainable / total:.1f}%)")

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning (optional)."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] All parameters unfrozen: {trainable:,} trainable")

    # ────────────────────────────────────────────────────────
    # Forward pass
    # ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, 3, 224, 224)

        Returns
        -------
        logits : torch.Tensor  (B, num_classes)
        """
        features = self.backbone(x)           # (B, C, H, W)
        pooled = self.pool(features)           # (B, C, 1, 1)
        pooled = torch.flatten(pooled, 1)      # (B, C)
        logits = self.classifier(pooled)       # (B, num_classes)
        return logits

    # ────────────────────────────────────────────────────────
    # Convenience: get target layer for Grad-CAM
    # ────────────────────────────────────────────────────────

    def get_gradcam_target_layer(self):
        """
        Returns the last convolutional layer of the backbone,
        suitable for Grad-CAM visualization.
        """
        if self.backbone_name == "resnet50":
            # backbone is Sequential of ResNet children (minus FC/pool)
            # Layer4 is the last block → children[-1] is Bottleneck
            return self.backbone[-1][-1]  # Last Bottleneck in layer4
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone[-1]      # Last feature block
        elif self.backbone_name == "mobilenet_v2":
            return self.backbone[-1]      # Last InvertedResidual


# ────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Model Architecture — Sanity Check")
    print("=" * 60)

    for name in ["resnet50", "efficientnet_b0", "mobilenet_v2"]:
        print(f"\n{'─' * 40}")
        model = ForgeryDetector(backbone_name=name, pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        print(f"Output shape: {out.shape}")  # (2, 2)
        assert out.shape == (2, 2), "Output shape mismatch!"
        print(f"✓ {name} OK")

    print(f"\n{'=' * 60}")
    print(" All models passed!")
    print("=" * 60)
