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
from utils.config import get_config, AugmentationConfig

logger = logging.getLogger(__name__)

# ── Global State ─────────────────────────────────────────────
MODEL: Optional[DualStreamFusionModel] = None
GRADCAM: Optional[GradCAM] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transforms()
CONFIG = get_config()OPTIMAL_THRESHOLD: float = 0.5  # Updated when checkpoint is loaded
# Feature configuration (must match training)
TABULAR_FEATURE_NAMES = [
    "age_normalized", "sex_encoded",
    "npm1_mutated", "flt3_mutated", "genetic_other",
]


def load_model(checkpoint_path: Optional[str] = None) -> DualStreamFusionModel:
    """Load the trained model or create a fresh one for demo."""
    global MODEL, GRADCAM, OPTIMAL_THRESHOLD

    if checkpoint_path and Path(checkpoint_path).exists():
        # Peek at checkpoint for optimal threshold
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        OPTIMAL_THRESHOLD = ckpt.get("optimal_threshold", 0.5)
        logger.info(f"Using optimal threshold from checkpoint: {OPTIMAL_THRESHOLD:.4f}")

        model = AMLTrainer.load_checkpoint(
            checkpoint_path,
            num_tabular_features=len(TABULAR_FEATURE_NAMES),
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
    age: int,
    sex: str,
    npm1: bool,
    flt3: bool,
    genetic_other: bool,
) -> Tuple[str, Optional[Image.Image]]:
    """
    Run AML prediction on a single cell image.

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

    # ── Prepare tabular features ─────────────────────────────
    # Normalize age (approximate: center around 55, std ~15)
    age_norm = (age - 55.0) / 15.0
    sex_enc = 1.0 if sex == "Male" else 0.0

    tabular = torch.tensor(
        [[age_norm, sex_enc,
          float(npm1), float(flt3), float(genetic_other)]],
        dtype=torch.float32,
    ).to(DEVICE)

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
        f"  📋 PATIENT CONTEXT\n"
        f"{'━' * 40}\n\n"
        f"  Age:          {age} years\n"
        f"  Sex:          {sex}\n"
        f"  NPM1:         {'Positive' if npm1 else 'Negative'}\n"
        f"  FLT3:         {'Positive' if flt3 else 'Negative'}\n"
        f"  Other:        {'Positive' if genetic_other else 'Negative'}\n\n"
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
                    AI-powered Acute Myeloid Leukemia detection combining cell microscopy with clinical data
                </p>
            </div>
        """)

        with gr.Row(equal_height=True):
            # ── Left Column: Inputs ──────────────────────────
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #1C1C1E; margin-bottom: 4px;'>📸 Cell Image</h3>")
                image_input = gr.Image(
                    type="pil",
                    label="Upload microscopic cell image",
                    elem_classes="upload-zone",
                    height=260,
                )

                gr.HTML("<h3 style='color: #1C1C1E; margin-top: 16px; margin-bottom: 4px;'>👤 Patient Information</h3>")
                with gr.Group(elem_classes="input-section"):
                    age_input = gr.Slider(
                        minimum=18, maximum=100, value=60, step=1,
                        label="Age (years)"
                    )
                    sex_input = gr.Radio(
                        choices=["Male", "Female"],
                        value="Male",
                        label="Sex",
                        elem_classes="radio-clean",
                    )

                gr.HTML("<h3 style='color: #1C1C1E; margin-top: 16px; margin-bottom: 4px;'>🧬 Genetic Markers</h3>")
                with gr.Group(elem_classes="input-section"):
                    npm1_input = gr.Checkbox(label="NPM1 Mutation", value=False)
                    flt3_input = gr.Checkbox(label="FLT3 Mutation", value=False)
                    genetic_other_input = gr.Checkbox(label="Other Mutations", value=False)

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
                    lines=20,
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
                    Powered by PyTorch • ResNet50 + MLP Late Fusion • Grad-CAM Explainability
                </span>
            </div>
        """)

        # ── Event Handlers ───────────────────────────────────
        analyze_btn.click(
            fn=predict,
            inputs=[image_input, age_input, sex_input,
                    npm1_input, flt3_input, genetic_other_input],
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
