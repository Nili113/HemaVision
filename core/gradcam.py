"""
HemaVision Grad-CAM Explainability (Enhanced)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
High-quality visual explanations for the AML detection model.

Improvements over vanilla Grad-CAM:
  1. Grad-CAM++ — second-order gradient weighting for sharper localization
  2. Multi-layer fusion — combines layer3 (14×14) + layer4 (7×7)
  3. Gaussian smoothing — eliminates blocky upsampling artifacts
  4. Noise thresholding — suppresses low-activation distractors
  5. SmoothGrad option — averages over noisy inputs for stability

Why vanilla Grad-CAM was poor:
  - layer4 is only 7×7 → bilinear upsampling creates a coarse grid
  - First-order gradients are noisy, especially with a frozen backbone
  - No post-processing → random edge/corner activations leak through

References:
  Selvaraju et al., "Grad-CAM" (ICCV 2017)
  Chattopadhay et al., "Grad-CAM++" (WACV 2018)
  Smilkov et al., "SmoothGrad" (ICML Workshop 2017)

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
from scipy.ndimage import gaussian_filter

from core.model import DualStreamFusionModel

logger = logging.getLogger(__name__)


# ── Helper: Gaussian smoothing ──────────────────────────────


def _smooth_heatmap(cam: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Apply Gaussian blur to remove blocky artifacts from low-res CAMs."""
    return gaussian_filter(cam, sigma=sigma)


def _normalize_cam(cam: np.ndarray, percentile_clip: float = 99.0) -> np.ndarray:
    """
    Robust normalization with percentile clipping.

    Clips to the top percentile to avoid a few extreme pixels
    washing out the rest of the heatmap.
    """
    if cam.max() == 0:
        return np.zeros_like(cam)
    top = np.percentile(cam, percentile_clip)
    cam = np.clip(cam, 0, top)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _threshold_cam(cam: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Zero out low-activation noise below threshold."""
    cam[cam < threshold] = 0.0
    return cam


# ── Multi-layer hook manager ────────────────────────────────


class _LayerHook:
    """Captures forward activations and backward gradients for a layer."""

    def __init__(self, layer: torch.nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = layer.register_forward_hook(self._save_act)
        self._bwd = layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


# ── Main Grad-CAM class (enhanced) ──────────────────────────


class GradCAM:
    """
    Enhanced Grad-CAM for the DualStreamFusionModel.

    Key improvements:
    - **Grad-CAM++** weighting (second-order gradients)
    - **Multi-layer fusion** (layer3 14×14 + layer4 7×7) for finer detail
    - **Gaussian smoothing** to remove 7×7 grid artifacts
    - **Percentile normalization** + **noise thresholding**
    - **SmoothGrad** option for more stable heatmaps

    Usage:
        >>> gradcam = GradCAM(model)
        >>> heatmap, prediction = gradcam.generate(image, tabular)
        >>> overlay = gradcam.create_overlay(original_image, heatmap)

    Architecture hook points (ResNet50):
        layer1 → 256 ch, 56×56
        layer2 → 512 ch, 28×28
        layer3 → 1024 ch, 14×14  ← used for detail
        layer4 → 2048 ch, 7×7    ← used for semantics
    """

    def __init__(
        self,
        model: DualStreamFusionModel,
        target_layers: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        use_gradcam_pp: bool = True,
        smooth_sigma: float = 8.0,
        noise_threshold: float = 0.15,
        smoothgrad_samples: int = 0,
        smoothgrad_noise: float = 0.1,
    ):
        """
        Args:
            model:              The trained DualStreamFusionModel
            target_layers:      Layers to hook. Default: ["layer3", "layer4"]
                                for multi-resolution fusion.
            use_gradcam_pp:     Use Grad-CAM++ weighting (recommended)
            smooth_sigma:       Gaussian sigma for heatmap smoothing (0=off)
            noise_threshold:    Zero out activations below this (0=off)
            smoothgrad_samples: Number of noisy forward passes (0=off)
            smoothgrad_noise:   Std of noise for SmoothGrad
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.use_gradcam_pp = use_gradcam_pp
        self.smooth_sigma = smooth_sigma
        self.noise_threshold = noise_threshold
        self.smoothgrad_samples = smoothgrad_samples
        self.smoothgrad_noise = smoothgrad_noise

        # Default: fuse layer3 (14×14) + layer4 (7×7)
        target_layers = target_layers or ["layer3", "layer4"]

        backbone = model.visual_stream.backbone
        self.hooks: List[_LayerHook] = []
        for layer_name in target_layers:
            layer = getattr(backbone, layer_name, None)
            if layer is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: layer1, layer2, layer3, layer4"
                )
            self.hooks.append(_LayerHook(layer))

        layer_str = " + ".join(target_layers)
        mode = "Grad-CAM++" if use_gradcam_pp else "Grad-CAM"
        logger.info(f"{mode} initialized targeting [{layer_str}]")

    # ── Core generation ──────────────────────────────────────

    def _compute_cam_for_hook(
        self, hook: _LayerHook, target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """Compute a single-layer CAM from hook data."""
        if hook.activations is None or hook.gradients is None:
            return None

        activations = hook.activations  # (1, C, H, W)
        gradients = hook.gradients      # (1, C, H, W)

        if self.use_gradcam_pp:
            # ── Grad-CAM++ weighting ──
            # α_kc = grad² / (2·grad² + Σ(A_k · grad³) + ε)
            grad2 = gradients.pow(2)
            grad3 = gradients.pow(3)
            spatial_sum = (activations * grad3).sum(dim=[2, 3], keepdim=True)
            alpha = grad2 / (2.0 * grad2 + spatial_sum + 1e-7)

            # Weight by ReLU(gradient) to focus on positive influence
            weights = (alpha * F.relu(gradients)).sum(dim=[2, 3], keepdim=True)
        else:
            # ── Vanilla Grad-CAM weighting ──
            weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to target size
        cam = F.interpolate(
            cam, size=target_size, mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        return cam

    def _single_forward_backward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        target_class: int,
    ) -> Tuple[float, np.ndarray]:
        """One forward+backward pass, returns (prob, fused_cam)."""
        # Forward
        logits = self.model(image, tabular)
        prob = torch.sigmoid(logits).item()

        # Target score
        score = logits if target_class == 1 else -logits

        # Backward
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Compute per-layer CAMs and fuse
        cams = []
        for hook in self.hooks:
            cam = self._compute_cam_for_hook(hook, target_size=(224, 224))
            if cam is not None:
                cams.append(cam)

        if not cams:
            return prob, np.zeros((224, 224))

        # Fuse by averaging (each layer contributes equally)
        fused = np.mean(cams, axis=0)
        return prob, fused

    def generate(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate a high-quality Grad-CAM heatmap.

        Args:
            image:        (1, 3, 224, 224) or (3, 224, 224) tensor
            tabular:      (1, num_features) or (num_features,) tensor
            target_class: Class to explain (1=AML blast, 0=normal)

        Returns:
            heatmap:    (224, 224) numpy array in [0, 1]
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

        if self.smoothgrad_samples > 0:
            # ── SmoothGrad: average over noisy inputs ──
            accumulated_cam = np.zeros((224, 224))
            prob = 0.0
            for _ in range(self.smoothgrad_samples):
                noise = torch.randn_like(image) * self.smoothgrad_noise
                noisy_image = (image + noise).requires_grad_(True)
                p, cam = self._single_forward_backward(
                    noisy_image, tabular, target_class
                )
                accumulated_cam += cam
                prob += p
            cam = accumulated_cam / self.smoothgrad_samples
            prob = prob / self.smoothgrad_samples
        else:
            prob, cam = self._single_forward_backward(
                image, tabular, target_class
            )

        # ── Post-processing pipeline ──
        # 1. Gaussian smoothing (eliminates 7×7 grid artifacts)
        if self.smooth_sigma > 0:
            cam = _smooth_heatmap(cam, sigma=self.smooth_sigma)

        # 2. Percentile-based normalization
        cam = _normalize_cam(cam, percentile_clip=99.0)

        # 3. Noise thresholding
        if self.noise_threshold > 0:
            cam = _threshold_cam(cam, threshold=self.noise_threshold)
            # Re-normalize after thresholding
            if cam.max() > 0:
                cam = cam / cam.max()

        return cam, prob

    # ── Overlay & visualization ──────────────────────────────

    def create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap on the original image.

        Uses alpha blending weighted by heatmap intensity so that
        regions with LOW activation stay clear (showing the original
        image) while HIGH activation regions get the full colormap.

        Args:
            original_image: (H, W, 3) uint8 numpy array
            heatmap:        (H, W) float array in [0, 1]
            alpha:          Maximum overlay opacity
            colormap:       Matplotlib colormap name

        Returns:
            (H, W, 3) uint8 overlay image
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) in [0, 1]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Resize if needed
        if original_image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_img = Image.fromarray(heatmap_colored)
            heatmap_img = heatmap_img.resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.BILINEAR,
            )
            heatmap_colored = np.array(heatmap_img)

        # Intensity-weighted alpha: transparent where heatmap is low
        intensity = np.clip(heatmap, 0, 1)[:, :, np.newaxis]  # (H, W, 1)
        if intensity.shape[:2] != original_image.shape[:2]:
            intensity = np.array(
                Image.fromarray((intensity.squeeze() * 255).astype(np.uint8)).resize(
                    (original_image.shape[1], original_image.shape[0]),
                    Image.BILINEAR,
                )
            ).astype(float)[:, :, np.newaxis] / 255.0

        pixel_alpha = alpha * intensity  # (H, W, 1)

        overlay = (
            (1 - pixel_alpha) * original_image.astype(float)
            + pixel_alpha * heatmap_colored.astype(float)
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

        # Generate enhanced Grad-CAM
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

            axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title("Grad-CAM++ Heatmap", fontsize=12)
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

        img = image_tensor.cpu().detach().numpy()
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
        threshold: float = 0.5,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Generate Grad-CAM visualizations for multiple samples.

        Args:
            dataloader:  DataLoader to sample from
            num_samples: Number of samples to visualize
            save_dir:    Directory to save individual visualizations
            threshold:   Classification threshold

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
                gt_label = int(labels[i].item() > 0.5)
                if save_dir:
                    from pathlib import Path
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    label_str = "blast" if gt_label == 1 else "normal"
                    save_path = f"{save_dir}/gradcam_{count:03d}_{label_str}.png"

                result = self.visualize(
                    image=images[i],
                    tabular=tabular[i],
                    save_path=save_path,
                    ground_truth_label=gt_label,
                    threshold=threshold,
                )
                results.append(result)
                count += 1

        return results

    def cleanup(self):
        """Remove all hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def generate_gradcam_base64(
    model: DualStreamFusionModel,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    target_layers: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Generate Grad-CAM visualization and return as base64 string.
    Used by the web interfaces (Gradio / FastAPI).

    Returns:
        (base64_string, prediction_probability)
    """
    gradcam = GradCAM(model, target_layers=target_layers)
    original, heatmap_colored, overlay, prediction = gradcam.visualize(
        image_tensor, tabular_tensor
    )
    gradcam.cleanup()

    # Convert overlay to base64
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    buffer.seek(0)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64_str, prediction
