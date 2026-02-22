"""
HemaVision Gradio Interface — Tier 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-quality Gradio demo with Jony Ive-inspired design.

Design Principles:
  • Generous whitespace — no cramped elements
  • Clear typography — large, readable fonts
  • Subtle shadows — depth without distraction
  • Progressive disclosure — results appear after analysis
  • Medical-grade color coding:
      Normal → #34C759 (green)
      AML Blast → #FF3B30 (red)
      Neutral → #8E8E93 (gray)

Author: Firoj
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

from core.model import DualStreamFusionModel
from core.gradcam import GradCAM
from core.dataset import get_eval_transforms
from core.train import AMLTrainer
from core.morphology import (
    extract_single_image_features,
    MORPHOLOGY_FEATURE_NAMES,
    NUM_MORPHOLOGY_FEATURES,
)
from utils.config import get_config, AugmentationConfig

logger = logging.getLogger(__name__)

# ── Global State ─────────────────────────────────────────────
MODEL: Optional[DualStreamFusionModel] = None
GRADCAM: Optional[GradCAM] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transforms()
CONFIG = get_config()
OPTIMAL_THRESHOLD: float = 0.5  # Updated when checkpoint is loaded

# Feature configuration — morphological features extracted from images
TABULAR_FEATURE_NAMES = list(MORPHOLOGY_FEATURE_NAMES)  # 20 features


def load_model(checkpoint_path: Optional[str] = None) -> DualStreamFusionModel:
    """Load the trained model or create a fresh one for demo."""
    global MODEL, GRADCAM, OPTIMAL_THRESHOLD

    if checkpoint_path and Path(checkpoint_path).exists():
        # Peek at checkpoint for optimal threshold
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        OPTIMAL_THRESHOLD = ckpt.get("optimal_threshold", 0.5)

        # If best_model.pt has default threshold, try final_model.pt for the optimized one
        if OPTIMAL_THRESHOLD == 0.5:
            final_ckpt_path = Path(checkpoint_path).parent / "final_model.pt"
            if final_ckpt_path.exists():
                final_ckpt = torch.load(str(final_ckpt_path), map_location=DEVICE, weights_only=False)
                opt_thresh = final_ckpt.get("optimal_threshold", 0.5)
                if opt_thresh != 0.5:
                    OPTIMAL_THRESHOLD = opt_thresh
                    logger.info(f"Loaded optimized threshold from final_model.pt: {OPTIMAL_THRESHOLD:.4f}")

        logger.info(f"Using optimal threshold: {OPTIMAL_THRESHOLD:.4f}")

        # Get num_tabular_features from checkpoint config
        saved_config = ckpt.get("config", {})
        num_tab = saved_config.get("num_tabular_features", len(TABULAR_FEATURE_NAMES))

        model = AMLTrainer.load_checkpoint(
            checkpoint_path,
            num_tabular_features=num_tab,
            device=DEVICE,
        )
    else:
        # Demo mode — create untrained model
        logger.info("No checkpoint found. Running in demo mode.")
        model = DualStreamFusionModel(
            num_tabular_features=len(TABULAR_FEATURE_NAMES),
        )
        model = model.to(DEVICE)
        model.eval()

    MODEL = model
    GRADCAM = GradCAM(model, target_layers=["layer3", "layer4"])
    return model


def predict(
    image: Image.Image,
) -> Tuple[str, Optional[Image.Image]]:
    """
    Run AML prediction on a single cell image.

    Morphological features are automatically extracted from the
    uploaded image and fed into the tabular stream alongside the
    deep visual features from ResNet50.

    Returns:
        (result_text, gradcam_image)
    """
    global MODEL, GRADCAM

    if MODEL is None:
        load_model()

    if image is None:
        return "⚠️ Please upload a cell microscopy image.", None

    # ── Prepare image ────────────────────────────────────────
    image_pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)

    # ── Extract morphological features from the image ───────
    morph_vec = extract_single_image_features(image_pil, normalize=True)
    tabular = torch.tensor(morph_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Pad or truncate to match model's expected feature count
    expected = MODEL.num_tabular_features
    if tabular.shape[1] < expected:
        pad = torch.zeros(1, expected - tabular.shape[1], device=DEVICE)
        tabular = torch.cat([tabular, pad], dim=1)
    elif tabular.shape[1] > expected:
        tabular = tabular[:, :expected]

    # ── Run prediction ───────────────────────────────────────
    MODEL.eval()

    # Grad-CAM + prediction
    heatmap, prob = GRADCAM.generate(image_tensor, tabular)

    # Create overlay visualization
    original_np = np.array(image_pil.resize((224, 224)))
    overlay = GRADCAM.create_overlay(original_np, heatmap, alpha=0.45)
    gradcam_image = Image.fromarray(overlay)

    # ── Format result ────────────────────────────────────────
    is_blast = prob > OPTIMAL_THRESHOLD
    confidence = prob if is_blast else 1 - prob

    if is_blast:
        risk_level = "HIGH RISK" if prob > 0.75 else "MODERATE RISK"
        risk_color = "🔴" if prob > 0.75 else "🟡"
        prediction_text = "AML Blast (Malignant)"
    else:
        risk_level = "LOW RISK"
        risk_color = "🟢"
        prediction_text = "Normal Cell (Benign)"

    result = (
        f"{'━' * 40}\n"
        f"  🎯 DIAGNOSTIC RESULT\n"
        f"{'━' * 40}\n\n"
        f"  Prediction:   {prediction_text}\n"
        f"  Confidence:   {confidence:.1%}\n"
        f"  Probability:  {prob:.4f}\n"
        f"  Risk Level:   {risk_color} {risk_level}\n\n"
        f"{'━' * 40}\n"
        f"  ⚠️  DISCLAIMER\n"
        f"{'━' * 40}\n\n"
        f"  This is a research tool for educational\n"
        f"  purposes only. Always consult qualified\n"
        f"  hematologists for clinical diagnosis.\n"
    )

    return result, gradcam_image


# ── Custom CSS — Jony Ive Aesthetic ──────────────────────────
CUSTOM_CSS = """
/* Global */
.gradio-container {
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
}

/* Header */
.app-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #1C1C1E;
    margin-bottom: 0.25rem;
}

.app-subtitle {
    text-align: center;
    font-size: 1rem;
    color: #8E8E93;
    margin-bottom: 2rem;
}

/* Upload area */
.upload-zone {
    border: 2px dashed #007AFF !important;
    border-radius: 16px !important;
    padding: 40px !important;
    background: #F2F2F7 !important;
    transition: all 0.3s ease !important;
}

.upload-zone:hover {
    border-color: #0056CC !important;
    background: #E8E8ED !important;
}

/* Input sections */
.input-section {
    background: white !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 16px !important;
}

/* Result box */
.result-box textarea {
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    background: #1C1C1E !important;
    color: #F2F2F7 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    border: none !important;
}

/* Grad-CAM visualization */
.gradcam-viz img {
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12) !important;
}

/* Buttons */
.primary-btn {
    background: #007AFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    background: #0056CC !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0, 122, 255, 0.3) !important;
}

/* Slider */
input[type="range"] {
    accent-color: #007AFF !important;
}

/* Radio buttons */
.radio-clean label {
    border-radius: 8px !important;
    padding: 8px 16px !important;
}

/* Footer disclaimer */
.disclaimer {
    text-align: center;
    color: #8E8E93;
    font-size: 0.85rem;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #E5E5EA;
}

/* Smooth animations */
* {
    transition: background-color 0.2s ease, border-color 0.2s ease;
}
"""


def create_gradio_app() -> gr.Blocks:
    """Create the Gradio interface with Apple-inspired design."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="gray",
        spacing_size="lg",
        radius_size="lg",
        font=[
            gr.themes.GoogleFont("Inter"),
            "SF Pro Display",
            "system-ui",
            "sans-serif",
        ],
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="HemaVision — AML Diagnostic Assistant") as demo:
        # ── Header ───────────────────────────────────────────
        gr.HTML("""
            <div style="text-align: center; padding: 2rem 0 1rem;">
                <div style="display: inline-flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #007AFF, #5856D6);
                                border-radius: 14px; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 24px;">🔬</span>
                    </div>
                    <h1 style="font-size: 2rem; font-weight: 700; color: #1C1C1E; margin: 0;">
                        HemaVision
                    </h1>
                </div>
                <p style="color: #8E8E93; font-size: 1.05rem; margin: 0;">
                    Multimodal AML detection — fusing deep visual features with handcrafted morphological analysis
                </p>
            </div>
        """)

        with gr.Row(equal_height=True):
            # ── Left Column: Input ───────────────────────────
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #1C1C1E; margin-bottom: 4px;'>📸 Cell Microscopy Image</h3>")
                gr.HTML("<p style='color: #8E8E93; font-size: 0.9rem; margin-bottom: 12px;'>Upload a blood smear image — morphological features are extracted automatically</p>")
                image_input = gr.Image(
                    type="pil",
                    label="Upload microscopic cell image",
                    elem_classes="upload-zone",
                    height=320,
                )

                analyze_btn = gr.Button(
                    "🔍 Analyze Cell",
                    variant="primary",
                    elem_classes="primary-btn",
                    size="lg",
                )

            # ── Right Column: Results ────────────────────────
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #1C1C1E; margin-bottom: 4px;'>🎯 Diagnostic Result</h3>")
                result_output = gr.Textbox(
                    label="Analysis",
                    lines=16,
                    interactive=False,
                    elem_classes="result-box",
                    show_copy_button=True,
                )

                gr.HTML("<h3 style='color: #1C1C1E; margin-top: 16px; margin-bottom: 4px;'>🔬 Explainability Map</h3>")
                gradcam_output = gr.Image(
                    label="Grad-CAM Visualization",
                    elem_classes="gradcam-viz",
                    height=280,
                )

        # ── Footer ───────────────────────────────────────────
        gr.HTML("""
            <div class="disclaimer">
                <strong>⚠️ Medical Disclaimer:</strong> HemaVision is a research tool for educational and
                demonstration purposes only. It is not intended for clinical diagnosis.
                Always consult qualified hematologists for patient care decisions.
                <br><br>
                <span style="color: #C7C7CC;">
                    Powered by PyTorch • ResNet50 + Morphological MLP Late Fusion • Grad-CAM Explainability
                </span>
            </div>
        """)

        # ── Event Handlers ───────────────────────────────────
        analyze_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[result_output, gradcam_output],
        )

    return demo


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    # Try to load model from default checkpoint
    default_ckpt = CONFIG.paths.checkpoints_dir / "best_model.pt"
    load_model(str(default_ckpt) if default_ckpt.exists() else None)

    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=CONFIG.inference.gradio_port,
        share=False,
        show_error=True,
    )
