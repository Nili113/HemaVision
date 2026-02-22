"""
HemaVision Model Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dual-Stream Late Fusion Network for AML detection.

Architecture:
┌─────────────────────────────────────────────────────┐
│                   INPUT LAYER                       │
│  ┌───────────────────┐  ┌───────────────────────┐   │
│  │  Cell Image       │  │  Morphological Feats  │   │
│  │  (224×224×3)      │  │  (20 handcrafted)     │   │
│  └────────┬──────────┘  └──────────┬────────────┘   │
│           │                        │                │
│  ┌────────▼──────────┐  ┌──────────▼────────────┐   │
│  │  Visual Stream    │  │  Tabular Stream       │   │
│  │  ResNet50         │  │  MLP (3 layers)       │   │
│  │  → 2048-dim       │  │  → 32-dim             │   │
│  └────────┬──────────┘  └──────────┬────────────┘   │
│           │                        │                │
│  ┌────────▼────────────────────────▼────────────┐   │
│  │            FUSION LAYER                      │   │ 
│  │     Concatenate → [2048 + 32] = 2080-dim     │   │
│  └────────────────────┬─────────────────────────┘   │
│                       │                             │
│  ┌────────────────────▼─────────────────────────┐   │
│  │         CLASSIFICATION HEAD                  │   │
│  │  FC(2080→256) → ReLU → Dropout → FC(256→1)   │   │
│  │  → Sigmoid → Probability [0, 1]              │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

Multimodal Design:
  Stream 1 — Deep learned features (ResNet50 backbone)
  Stream 2 — Handcrafted morphological features:
             geometry, nucleus, colour, texture, shape

Mathematical Formulation:
  f(x_img, x_morph; Θ) = σ(W₂ · ReLU(W₁ · [f_CNN(x_img) ⊕ f_MLP(x_morph)] + b₁) + b₂)

Author: Firoj
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from utils.config import ModelConfig

logger = logging.getLogger(__name__)


class TabularStream(nn.Module):
    """
    Multi-Layer Perceptron for encoding morphological features.

    Transforms handcrafted cytological features (geometry, nucleus,
    colour, texture, shape) into a dense representation suitable
    for fusion with deep visual features.

    Architecture:
        Input → [FC → BatchNorm → ReLU → Dropout] × 3 → Output

    Input:  (batch, num_features) — e.g., (32, 20)
    Output: (batch, hidden_dims[-1]) — e.g., (32, 32)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        layers = []
        in_dim = num_features

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VisualStream(nn.Module):
    """
    ResNet50-based visual feature extractor for cell microscopy images.

    Uses ImageNet-pretrained weights with optional backbone freezing
    for transfer learning. Extracts a 2048-dim feature vector from
    the global average pooling layer.

    Input:  (batch, 3, 224, 224)
    Output: (batch, 2048)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        # Load pretrained backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract feature dimension before removing the classifier
        self.output_dim = self.backbone.fc.in_features

        # Remove the original classification head
        self.backbone.fc = nn.Identity()

        # Freeze backbone if requested (transfer learning)
        if freeze:
            self._freeze_backbone()

        logger.info(
            f"Visual stream: {backbone} "
            f"(pretrained={pretrained}, frozen={freeze}, "
            f"output_dim={self.output_dim})"
        )

    def _freeze_backbone(self):
        """Freeze all backbone parameters (for transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only fusion/classifier will train.")

    def unfreeze_backbone(self, unfreeze_from: str = "layer3"):
        """
        Selectively unfreeze backbone layers for fine-tuning.

        Args:
            unfreeze_from: Which layer to start unfreezing from.
                          Options: 'layer1', 'layer2', 'layer3', 'layer4'
        """
        unfreeze_start = False
        unfrozen_count = 0

        for name, param in self.backbone.named_parameters():
            if unfreeze_from in name:
                unfreeze_start = True
            if unfreeze_start:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False

        logger.info(
            f"Unfroze {unfrozen_count} parameters from '{unfreeze_from}' onwards."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DualStreamFusionModel(nn.Module):
    """
    Dual-Stream Late Fusion Network for AML detection.

    Combines deep visual features from microscopic cell images with
    handcrafted morphological features via late fusion
    (concatenation after independent encoding).

    Args:
        num_tabular_features: Number of morphological features (default: 20)
        config: Model configuration object

    Example:
        >>> model = DualStreamFusionModel(num_tabular_features=20)
        >>> image = torch.randn(4, 3, 224, 224)
        >>> morphology = torch.randn(4, 20)
        >>> output = model(image, morphology)
        >>> output.shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        num_tabular_features: int,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.config = config or ModelConfig()

        # Stream 1: Visual (CNN)
        self.visual_stream = VisualStream(
            backbone=self.config.backbone,
            pretrained=self.config.pretrained,
            freeze=self.config.freeze_backbone,
        )

        # Stream 2: Tabular (MLP)
        self.tabular_stream = TabularStream(
            num_features=num_tabular_features,
            hidden_dims=self.config.tabular_hidden_dims,
            dropout=self.config.tabular_dropout,
        )

        # Fusion dimension
        fusion_dim = (
            self.visual_stream.output_dim + self.tabular_stream.output_dim
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.config.fusion_hidden_dim),
            nn.BatchNorm1d(self.config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.fusion_hidden_dim, self.config.num_classes),
        )

        # Store dimensions for reference
        self.fusion_dim = fusion_dim
        self.num_tabular_features = num_tabular_features

        logger.info(
            f"DualStreamFusionModel initialized:\n"
            f"  Visual:  {self.config.backbone} → {self.visual_stream.output_dim}-dim\n"
            f"  Tabular: {num_tabular_features} → {self.tabular_stream.output_dim}-dim\n"
            f"  Fusion:  {fusion_dim}-dim → {self.config.fusion_hidden_dim} → 1"
        )

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through both streams and fusion.

        Args:
            image:   (batch, 3, 224, 224) — cell microscopy image
            tabular: (batch, num_features) — morphological features

        Returns:
            (batch, 1) — raw logits (apply sigmoid for probability)
        """
        # Encode visual features
        visual_features = self.visual_stream(image)  # (B, 2048)

        # Encode tabular features
        tabular_features = self.tabular_stream(tabular)  # (B, 32)

        # Late fusion: concatenate
        fused = torch.cat([visual_features, tabular_features], dim=1)  # (B, 2080)

        # Classify
        logits = self.classifier(fused)  # (B, 1)

        return logits

    def unfreeze_backbone(self, from_layer: str = "layer3"):
        """Unfreeze visual backbone for fine-tuning."""
        self.visual_stream.unfreeze_backbone(from_layer)

    def get_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract visual features only (for Grad-CAM)."""
        return self.visual_stream(image)

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self) -> str:
        """Human-readable model summary."""
        total, trainable = self.count_parameters()
        return (
            f"\n{'━' * 50}\n"
            f"  DualStreamFusionModel Summary\n"
            f"{'━' * 50}\n"
            f"  Backbone:      {self.config.backbone}\n"
            f"  Visual dim:    {self.visual_stream.output_dim}\n"
            f"  Tabular dim:   {self.tabular_stream.output_dim}\n"
            f"  Fusion dim:    {self.fusion_dim}\n"
            f"  Total params:  {total:,}\n"
            f"  Trainable:     {trainable:,}\n"
            f"  Frozen:        {total - trainable:,}\n"
            f"{'━' * 50}\n"
        )
