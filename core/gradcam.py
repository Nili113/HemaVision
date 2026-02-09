"""
HemaVision Grad-CAM Explainability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gradient-weighted Class Activation Mapping for visual explanations.

Grad-CAM highlights the regions of the cell image that the model
considers most important for its prediction. This is critical for
medical AI — clinicians need to understand WHY the model flagged
a cell as potentially malignant.

Algorithm:
  1. Forward pass through the model
  2. Backpropagate the target class gradient to the target layer
  3. Compute importance weights via global average pooling of gradients
  4. Weight the feature maps and apply ReLU
  5. Resize the activation map to the original image size

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization" (ICCV 2017)

Author: Firoj
"""

import io
import base64
import logging
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.model import DualStreamFusionModel

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for the DualStreamFusionModel.

    Targets the ResNet50's `layer4` (final convolutional block)
    to produce class activation maps.

    Usage:
        >>> gradcam = GradCAM(model, target_layer="layer4")
        >>> heatmap, prediction = gradcam.generate(image, tabular)
        >>> overlay = gradcam.create_overlay(original_image, heatmap)

    Architecture hook points:
        ResNet50:
          layer1 → 256 channels, 56×56
          layer2 → 512 channels, 28×28
          layer3 → 1024 channels, 14×14
          layer4 → 2048 channels, 7×7  ← default target
    """

    def __init__(
        self,
        model: DualStreamFusionModel,
        target_layer: str = "layer4",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device

        # Register hooks on the target layer
        self.activations = None
        self.gradients = None

        # Get the target layer from the visual backbone
        backbone = model.visual_stream.backbone
        target = getattr(backbone, target_layer, None)
        if target is None:
            raise ValueError(
                f"Target layer '{target_layer}' not found in backbone. "
                f"Available: layer1, layer2, layer3, layer4"
            )

        # Forward hook — saves activations
        target.register_forward_hook(self._save_activations)
        # Backward hook — saves gradients
        target.register_full_backward_hook(self._save_gradients)

        logger.info(f"Grad-CAM initialized targeting '{target_layer}'")

    def _save_activations(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a Grad-CAM heatmap for a single sample.

        Args:
            image:        (1, 3, 224, 224) or (3, 224, 224) tensor
            tabular:      (1, num_features) or (num_features,) tensor
            target_class: Class to explain (1=AML blast, 0=normal)

        Returns:
            heatmap:    (224, 224) numpy array normalized to [0, 1]
            prediction: Model prediction probability
        """
        self.model.eval()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if tabular.dim() == 1:
            tabular = tabular.unsqueeze(0)

        image = image.to(self.device).requires_grad_(True)
        tabular = tabular.to(self.device)

        # Forward pass
        logits = self.model(image, tabular)
        prob = torch.sigmoid(logits).item()

        # For binary: if target is class 0, negate the logit
        if target_class == 0:
            score = -logits
        else:
            score = logits

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Grad-CAM computation
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured.")
            return np.zeros((224, 224)), prob

        # Global average pooling of gradients → importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU — only keep positive activations
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, prob

    def create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap on the original image.

        Args:
            original_image: (H, W, 3) uint8 numpy array
            heatmap:        (H, W) float array in [0, 1]
            alpha:          Overlay transparency (0=invisible, 1=opaque)
            colormap:       Matplotlib colormap name

        Returns:
            (H, W, 3) uint8 overlay image
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) in [0, 1]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Resize heatmap to match original image
        if original_image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_img = Image.fromarray(heatmap_colored)
            heatmap_img = heatmap_img.resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.BILINEAR,
            )
            heatmap_colored = np.array(heatmap_img)

        # Blend
        overlay = (
            (1 - alpha) * original_image.astype(float)
            + alpha * heatmap_colored.astype(float)
        ).clip(0, 255).astype(np.uint8)

        return overlay

    def visualize(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Complete visualization pipeline.

        Args:
            image:          Preprocessed image tensor
            tabular:        Tabular features tensor
            original_image: Original (non-normalized) image as numpy array
            save_path:      Path to save the visualization

        Returns:
            (original, heatmap_colored, overlay, prediction)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Generate Grad-CAM
        heatmap, prediction = self.generate(image, tabular)

        # If no original image provided, denormalize the tensor
        if original_image is None:
            original_image = self._denormalize(image)

        # Ensure original is the right shape
        if original_image.ndim == 3 and original_image.shape[0] == 3:
            original_image = np.transpose(original_image, (1, 2, 0))
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        # Create colored heatmap
        cmap = cm.get_cmap("jet")
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

        # Create overlay
        overlay = self.create_overlay(original_image, heatmap)

        # Save visualization
        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_image)
            axes[0].set_title("Original Cell Image", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(overlay)
            pred_label = "AML Blast" if prediction > 0.5 else "Normal"
            axes[2].set_title(
                f"Overlay — {pred_label} ({prediction:.1%})", fontsize=12
            )
            axes[2].axis("off")

            plt.suptitle(
                "HemaVision Explainability Visualization",
                fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Grad-CAM visualization saved to {save_path}")

        return original_image, heatmap_colored, overlay, prediction

    @staticmethod
    def _denormalize(
        image_tensor: torch.Tensor,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """Reverse ImageNet normalization."""
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        img = image_tensor.cpu().numpy()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = np.clip(img, 0, 1)
        img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
        return img

    def batch_visualize(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 10,
        save_dir: Optional[str] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Generate Grad-CAM visualizations for multiple samples.

        Args:
            dataloader:  DataLoader to sample from
            num_samples: Number of samples to visualize
            save_dir:    Directory to save individual visualizations

        Returns:
            List of (original, heatmap, overlay, prediction) tuples
        """
        results = []
        count = 0

        for images, tabular, labels in dataloader:
            batch_size = images.shape[0]

            for i in range(batch_size):
                if count >= num_samples:
                    return results

                save_path = None
                if save_dir:
                    from pathlib import Path
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    label_str = "blast" if labels[i].item() > 0.5 else "normal"
                    save_path = f"{save_dir}/gradcam_{count:03d}_{label_str}.png"

                result = self.visualize(
                    image=images[i],
                    tabular=tabular[i],
                    save_path=save_path,
                )
                results.append(result)
                count += 1

        return results


def generate_gradcam_base64(
    model: DualStreamFusionModel,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    target_layer: str = "layer4",
) -> Tuple[str, float]:
    """
    Generate Grad-CAM visualization and return as base64 string.
    Used by the web interfaces (Gradio / FastAPI).

    Returns:
        (base64_string, prediction_probability)
    """
    gradcam = GradCAM(model, target_layer=target_layer)
    original, heatmap_colored, overlay, prediction = gradcam.visualize(
        image_tensor, tabular_tensor
    )

    # Convert overlay to base64
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    buffer.seek(0)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64_str, prediction
