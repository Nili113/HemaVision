"""
HemaVision Gradio Interface — Tier 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-quality Gradio demo with clean, clinical-grade design.

Features:
  • Auto-segments multi-cell blood smear fields into individual crops
  • Analyzes each cell independently with Grad-CAM overlays
  • Rich HTML result cards — no monospace text boxes
  • Input validation with clear user guidance

Author: Firoj
"""

import io
import base64
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
from core.cell_segmenter import segment_cells, SegmentationResult, CellCrop
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

    # Auto-discover checkpoint if none provided
    if not checkpoint_path:
        default_ckpt = CONFIG.paths.checkpoints_dir / "best_model.pt"
        if default_ckpt.exists():
            checkpoint_path = str(default_ckpt)
            logger.info(f"Auto-discovered checkpoint: {checkpoint_path}")

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


def _predict_single_cell(
    image_pil: Image.Image,
) -> Tuple[float, Image.Image]:
    """
    Run prediction + Grad-CAM on a single cell crop.

    Returns:
        (probability, gradcam_overlay_pil)
    """
    image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)

    morph_vec = extract_single_image_features(image_pil, normalize=True)
    tabular = torch.tensor(morph_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    expected = MODEL.num_tabular_features
    if tabular.shape[1] < expected:
        pad = torch.zeros(1, expected - tabular.shape[1], device=DEVICE)
        tabular = torch.cat([tabular, pad], dim=1)
    elif tabular.shape[1] > expected:
        tabular = tabular[:, :expected]

    MODEL.eval()
    heatmap, prob = GRADCAM.generate(image_tensor, tabular)

    original_np = np.array(image_pil.resize((224, 224)))
    overlay = GRADCAM.create_overlay(original_np, heatmap, alpha=0.45)
    return prob, Image.fromarray(overlay)


def predict(
    image: Image.Image,
) -> Tuple[str, Optional[Image.Image]]:
    """
    Run AML prediction — auto-segments multi-cell images.

    If the uploaded image contains multiple cells (e.g. a whole
    blood smear field), each cell is cropped and analyzed separately.
    Single-cell crops are analyzed directly.

    Returns:
        (result_html, composite_gradcam_image)
    """
    global MODEL, GRADCAM

    if MODEL is None:
        load_model()

    if image is None:
        return "⚠️ Please upload a cell microscopy image.", None

    image_pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")

    # ── Auto-segment ─────────────────────────────────────────
    seg = segment_cells(image_pil, max_cells=20, annotate=True)
    cells = seg.cells
    n_cells = len(cells)

    if n_cells == 0:
        return _empty_state_html(), None

    # ── Run predictions on each cell ─────────────────────────
    cell_results = []  # list of (cell_crop, prob, gradcam_pil)
    for cell in cells:
        prob, gcam = _predict_single_cell(cell.image)
        cell_results.append((cell, prob, gcam))

    # ── Build composite Grad-CAM image ───────────────────────
    if n_cells == 1:
        composite = cell_results[0][2]
    else:
        composite = _build_composite_image(cell_results, seg)

    # ── Build HTML ───────────────────────────────────────────
    expected = MODEL.num_tabular_features
    result_html = _build_result_html(cell_results, seg, expected)

    return result_html, composite


def _build_composite_image(
    cell_results: list,
    seg: SegmentationResult,
) -> Image.Image:
    """Build a grid of Grad-CAM overlays, plus the annotated source image."""
    n = len(cell_results)

    # Decide grid layout: annotated full image on left, cell grid on right
    cell_size = 160
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * cell_size

    # Scale annotated image to fit beside the grid
    ann = seg.annotated_image or Image.new("RGB", (400, 400), (240, 240, 240))
    ann_h = grid_h if grid_h > 0 else 400
    ann_w = int(ann.width * ann_h / max(ann.height, 1))
    ann_w = min(ann_w, 500)
    ann = ann.resize((ann_w, ann_h), Image.LANCZOS)

    total_w = ann_w + 16 + grid_w
    total_h = max(ann_h, grid_h)
    canvas = Image.new("RGB", (total_w, total_h), (15, 25, 35))

    # Paste annotated image on left
    canvas.paste(ann, (0, (total_h - ann_h) // 2))

    # Paste Grad-CAM cells in a grid on the right
    for i, (cell, prob, gcam) in enumerate(cell_results):
        r, c = divmod(i, cols)
        gcam_resized = gcam.resize((cell_size, cell_size), Image.LANCZOS)
        x_off = ann_w + 16 + c * cell_size
        y_off = r * cell_size
        canvas.paste(gcam_resized, (x_off, y_off))

        # Draw a thin colored border (blast=red, normal=green)
        try:
            import cv2
            arr = np.array(canvas)
            color = (220, 38, 38) if prob > OPTIMAL_THRESHOLD else (5, 150, 105)
            cv2.rectangle(arr, (x_off, y_off),
                          (x_off + cell_size - 1, y_off + cell_size - 1),
                          color, 2)
            # Cell number label
            label = f"#{i+1}"
            cv2.putText(arr, label, (x_off + 4, y_off + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(arr, label, (x_off + 4, y_off + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            canvas = Image.fromarray(arr)
        except ImportError:
            pass

    return canvas


def _build_result_html(
    cell_results: list,
    seg: SegmentationResult,
    num_features: int,
) -> str:
    """Build an HTML result card for single or multi-cell analysis."""
    n = len(cell_results)
    blasts = [(c, p, g) for c, p, g in cell_results if p > OPTIMAL_THRESHOLD]
    normals = [(c, p, g) for c, p, g in cell_results if p <= OPTIMAL_THRESHOLD]
    n_blast = len(blasts)
    n_normal = len(normals)
    blast_pct = n_blast / n * 100 if n > 0 else 0

    # Overall assessment
    if n_blast == 0:
        overall_label = "Normal — No Blasts Detected"
        overall_bg = "#071A12"; overall_border = "#0D3B24"; overall_text = "#6EE7B7"
        overall_icon = (
            '<svg width="28" height="28" fill="none" viewBox="0 0 24 24" stroke="#059669" stroke-width="2">'
            '<path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>'
            '</svg>'
        )
        risk_label = "LOW RISK"
        risk_emoji = "🟢"
    elif blast_pct >= 20:
        overall_label = f"AML Suspected — {n_blast}/{n} Blast Cells ({blast_pct:.0f}%)"
        overall_bg = "#1C0A0A"; overall_border = "#3B1515"; overall_text = "#FCA5A5"
        overall_icon = (
            '<svg width="28" height="28" fill="none" viewBox="0 0 24 24" stroke="#DC2626" stroke-width="2">'
            '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856'
            'c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16'
            'c-.77 1.333.192 3 1.732 3z"/></svg>'
        )
        risk_label = "HIGH RISK"
        risk_emoji = "🔴"
    else:
        overall_label = f"Atypical — {n_blast}/{n} Blast Cells ({blast_pct:.0f}%)"
        overall_bg = "#1C1505"; overall_border = "#3B2C0A"; overall_text = "#FCD34D"
        overall_icon = (
            '<svg width="28" height="28" fill="none" viewBox="0 0 24 24" stroke="#D97706" stroke-width="2">'
            '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856'
            'c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16'
            'c-.77 1.333.192 3 1.732 3z"/></svg>'
        )
        risk_label = "MODERATE RISK"
        risk_emoji = "🟡"

    # Multi-cell info banner
    multi_banner = ""
    if seg.is_multi_cell:
        multi_banner = (
            '<div style="background:#0C1929; border:1px solid #1E3A5F; border-radius:12px;'
            ' padding:12px 16px; margin-bottom:16px; font-size:13px; color:#60A5FA; line-height:1.5;">'
            f'<strong>🔍 Auto-Segmentation:</strong> {seg.message}'
            '</div>'
        )

    # Per-cell table rows (only show for multi-cell)
    cell_rows = ""
    if n > 1:
        rows_html = ""
        for cell, prob, gcam in cell_results:
            is_blast = prob > OPTIMAL_THRESHOLD
            conf = prob if is_blast else 1 - prob
            dot = '<span style="color:#DC2626;">●</span>' if is_blast else '<span style="color:#059669;">●</span>'
            cls = "AML Blast" if is_blast else "Normal"
            rows_html += (
                f'<tr style="border-bottom:1px solid #1E2D3D;">'
                f'<td style="padding:8px 12px; font-size:13px; font-weight:600; color:#E2E8F0;">#{cell.index+1}</td>'
                f'<td style="padding:8px 12px; font-size:13px; color:#CBD5E1;">{dot} {cls}</td>'
                f'<td style="padding:8px 12px; font-size:13px; color:#94A3B8; text-align:right;">{prob:.4f}</td>'
                f'<td style="padding:8px 12px; font-size:13px; font-weight:600; color:#E2E8F0; text-align:right;">{conf:.1%}</td>'
                f'</tr>'
            )
        cell_rows = (
            '<div style="margin-top:16px;">'
            '<div style="font-size:11px; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;'
            ' margin-bottom:8px; font-weight:600;">Per-Cell Breakdown</div>'
            '<table style="width:100%; border-collapse:collapse; background:#19232e; border-radius:12px;'
            ' overflow:hidden; border:1px solid #2A3A4A;">'
            '<thead><tr style="background:#141D27; border-bottom:1px solid #2A3A4A;">'
            '<th style="padding:8px 12px; font-size:11px; color:#94A3B8; text-align:left; font-weight:600;">Cell</th>'
            '<th style="padding:8px 12px; font-size:11px; color:#94A3B8; text-align:left; font-weight:600;">Classification</th>'
            '<th style="padding:8px 12px; font-size:11px; color:#94A3B8; text-align:right; font-weight:600;">P(blast)</th>'
            '<th style="padding:8px 12px; font-size:11px; color:#94A3B8; text-align:right; font-weight:600;">Confidence</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    # Average probability for single-cell display
    if n == 1:
        prob = cell_results[0][1]
        conf = prob if prob > OPTIMAL_THRESHOLD else 1 - prob
        bar_color = "#DC2626" if prob > OPTIMAL_THRESHOLD else "#059669"
        single_metrics = f"""
        <div style="background:#19232e; border-radius:12px; padding:16px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:13px; color:#94A3B8;">Confidence</span>
            <span style="font-size:15px; font-weight:600; color:#E2E8F0;">{conf:.1%}</span>
          </div>
          <div style="width:100%; height:8px; background:#334155; border-radius:99px; overflow:hidden;">
            <div style="width:{conf*100:.1f}%; height:100%; background:{bar_color};
                        border-radius:99px; transition:width 0.6s ease;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:14px; font-size:12px; color:#64748B;">
            <span>P(blast) = {prob:.4f}</span>
            <span>threshold = {OPTIMAL_THRESHOLD:.4f}</span>
          </div>
        </div>
        """
    else:
        avg_prob = sum(p for _, p, _ in cell_results) / n
        bar_color = "#DC2626" if blast_pct >= 20 else "#D97706" if n_blast > 0 else "#059669"
        single_metrics = f"""
        <div style="background:#19232e; border-radius:12px; padding:16px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:13px; color:#94A3B8;">Blast Proportion</span>
            <span style="font-size:15px; font-weight:600; color:#E2E8F0;">{n_blast}/{n} ({blast_pct:.0f}%)</span>
          </div>
          <div style="width:100%; height:8px; background:#334155; border-radius:99px; overflow:hidden;">
            <div style="width:{blast_pct:.1f}%; height:100%; background:{bar_color};
                        border-radius:99px; transition:width 0.6s ease;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; margin-top:14px; font-size:12px; color:#64748B;">
            <span>Avg P(blast) = {avg_prob:.4f}</span>
            <span>threshold = {OPTIMAL_THRESHOLD:.4f}</span>
          </div>
        </div>
        """

    html = f"""
    <div style="font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; color:#E2E8F0;">
      {multi_banner}
      <div style="background:{overall_bg}; border:1px solid {overall_border}; border-radius:16px;
                  padding:24px; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:16px;">
          {overall_icon}
          <div>
            <div style="font-size:18px; font-weight:700; color:{overall_text};">{overall_label}</div>
            <div style="font-size:13px; color:{overall_text}; opacity:0.8;">{risk_emoji} {risk_label}</div>
          </div>
        </div>
        {single_metrics}
      </div>
      {cell_rows}
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:16px;">
        <div style="background:#19232e; border:1px solid #2A3A4A; border-radius:12px; padding:14px;">
          <div style="font-size:11px; color:#64748B; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Cells Analyzed</div>
          <div style="font-size:14px; font-weight:500; color:#E2E8F0;">{n} cell{'s' if n != 1 else ''}</div>
        </div>
        <div style="background:#19232e; border:1px solid #2A3A4A; border-radius:12px; padding:14px;">
          <div style="font-size:11px; color:#64748B; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Architecture</div>
          <div style="font-size:14px; font-weight:500; color:#E2E8F0;">ResNet50 + MLP</div>
        </div>
        <div style="background:#19232e; border:1px solid #2A3A4A; border-radius:12px; padding:14px;">
          <div style="font-size:11px; color:#64748B; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Features</div>
          <div style="font-size:14px; font-weight:500; color:#E2E8F0;">{num_features} morphological</div>
        </div>
        <div style="background:#19232e; border:1px solid #2A3A4A; border-radius:12px; padding:14px;">
          <div style="font-size:11px; color:#64748B; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Dataset</div>
          <div style="font-size:14px; font-weight:500; color:#E2E8F0;">Munich AML-Cyto</div>
        </div>
      </div>
    </div>
    """
    return html


# ── Custom CSS — Dark Mode Clinical Grade ───────────────────
CUSTOM_CSS = """
/* ══════════════════════════════════════════════════════════
   HemaVision — Dark Mode · Clinical Grade Styling
   ══════════════════════════════════════════════════════════ */

/* ── Dark mode variables ──────────────────────────── */
html, body,
.dark, .dark body,
.gradio-container, .dark .gradio-container,
.gradio-container.dark {
    background: #0F1923 !important;
    color: #E2E8F0 !important;
    --body-background-fill: #0F1923 !important;
    --body-text-color: #E2E8F0 !important;
    --block-background-fill: #19232e !important;
    --block-border-color: #2A3A4A !important;
    --block-label-text-color: #94A3B8 !important;
    --input-background-fill: #19232e !important;
    --color-accent: #3B82F6 !important;
    --background-fill-primary: #19232e !important;
    --background-fill-secondary: #141D27 !important;
    --border-color-primary: #2A3A4A !important;
    --neutral-50: #0F1923 !important;
    --neutral-100: #141D27 !important;
    --neutral-200: #19232e !important;
    --neutral-300: #2A3A4A !important;
    --neutral-400: #475569 !important;
    --neutral-500: #64748B !important;
    --neutral-600: #94A3B8 !important;
    --neutral-700: #CBD5E1 !important;
    --neutral-800: #E2E8F0 !important;
    --neutral-900: #F1F5F9 !important;
    --neutral-950: #F8FAFC !important;
}

/* ── Base ─────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    background: #0F1923 !important;
}

/* All text should be light on dark bg */
.gradio-container *, .dark .gradio-container * {
    color: inherit;
}
.gradio-container label, .gradio-container .label-wrap,
.dark .gradio-container label, .dark .gradio-container .label-wrap {
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

/* Panels, blocks, groups */
.gradio-container .block,
.dark .gradio-container .block {
    background: #19232e !important;
    border-color: #2A3A4A !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}

/* ── Hero header ──────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 40px 20px 24px;
}
.hero-header .logo-pill {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: #19232e;
    border: 1px solid #2A3A4A;
    border-radius: 9999px;
    padding: 8px 24px 8px 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    margin-bottom: 16px;
}
.hero-header .logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 20px;
    box-shadow: 0 2px 8px rgba(59,130,246,0.3);
}
.hero-header h1 {
    font-size: 1.4rem; font-weight: 700; color: #F1F5F9 !important; margin: 0;
}
.hero-header .subtitle {
    font-size: 0.95rem; color: #94A3B8 !important; max-width: 520px;
    margin: 0 auto; line-height: 1.5;
}

/* ── Section cards ────────────────────────────────── */
.section-title {
    font-size: 0.8rem; font-weight: 600; color: #64748B !important;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 0 0 16px 0;
    display: flex; align-items: center; gap: 8px;
}
.section-title .dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
}
.section-title .dot-blue  { background: #3B82F6; }
.section-title .dot-green { background: #10B981; }
.section-title .dot-amber { background: #F59E0B; }

/* ── Upload zone ──────────────────────────────────── */
.upload-zone,
.dark .upload-zone {
    border: 2px dashed #2A3A4A !important;
    border-radius: 14px !important;
    background: #19232e !important;
    transition: all 0.25s ease !important;
    min-height: 220px !important;
}
.upload-zone:hover,
.dark .upload-zone:hover {
    border-color: #3B82F6 !important;
    background: #1E2D3D !important;
}
/* Upload placeholder text/icons */
.upload-zone .wrap,
.dark .upload-zone .wrap {
    color: #64748B !important;
}
.upload-zone svg,
.dark .upload-zone svg {
    stroke: #475569 !important;
    fill: none !important;
}
/* Ensure uploaded image preview IS visible */
.upload-zone img,
.dark .upload-zone img {
    opacity: 1 !important;
    max-height: 280px !important;
    object-fit: contain !important;
    display: block !important;
    visibility: visible !important;
}
/* Gradio 6 image component internals */
.upload-zone [data-testid="image"],
.dark .upload-zone [data-testid="image"],
.upload-zone .image-container,
.dark .upload-zone .image-container,
.upload-zone .upload-container,
.dark .upload-zone .upload-container,
.upload-zone .image-frame,
.dark .upload-zone .image-frame {
    background: #19232e !important;
    overflow: visible !important;
}
.upload-zone .image-frame img,
.dark .upload-zone .image-frame img,
.upload-zone .image-container img,
.dark .upload-zone .image-container img,
.upload-zone canvas,
.dark .upload-zone canvas {
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    max-width: 100% !important;
    max-height: 280px !important;
}
/* Ensure the preview wrapper doesn't collapse */
.upload-zone .preview,
.dark .upload-zone .preview,
.upload-zone .image-preview,
.dark .upload-zone .image-preview {
    min-height: 100px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* ── Analyze button ───────────────────────────────── */
.analyze-btn,
.dark .analyze-btn {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
    transition: all 0.2s ease !important;
    color: white !important;
}
.analyze-btn:hover,
.dark .analyze-btn:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Result area ──────────────────────────────────── */
.result-area,
.dark .result-area {
    min-height: 200px !important;
    background: #19232e !important;
    border: 1px solid #2A3A4A !important;
    border-radius: 14px !important;
    padding: 8px !important;
}
.result-area .prose,
.dark .result-area .prose {
    font-size: 14px !important;
    color: #E2E8F0 !important;
}

/* ── Grad-CAM image ───────────────────────────────── */
.gradcam-img,
.dark .gradcam-img {
    background: #19232e !important;
    border: 1px solid #2A3A4A !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
.gradcam-img img,
.dark .gradcam-img img {
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}
.gradcam-img .empty-image,
.gradcam-img [data-testid="image"],
.dark .gradcam-img [data-testid="image"] {
    background: #141D27 !important;
    border: 1px solid #2A3A4A !important;
}

/* ── Info chips ───────────────────────────────────── */
.info-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #19232e !important; border: 1px solid #2A3A4A;
    border-radius: 9999px; padding: 6px 14px;
    font-size: 12px; color: #94A3B8 !important; font-weight: 500;
}

/* ── Footer ───────────────────────────────────────── */
.footer-area {
    text-align: center; padding: 28px 20px 16px;
    border-top: 1px solid #2A3A4A; margin-top: 32px;
}
.footer-area .disclaimer {
    font-size: 12px; color: #64748B !important; line-height: 1.6;
    max-width: 600px; margin: 0 auto;
}
.footer-area .powered {
    font-size: 11px; color: #475569 !important; margin-top: 12px;
}

/* ── Misc Gradio overrides ────────────────────────── */
textarea,
.dark textarea {
    font-family: inherit !important;
    background: #19232e !important;
    color: #E2E8F0 !important;
}
input[type="text"],
.dark input[type="text"] {
    background: #19232e !important;
    color: #E2E8F0 !important;
    border-color: #2A3A4A !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0F1923; }
::-webkit-scrollbar-thumb { background: #2A3A4A; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3E5066; }

/* Hide default gradio footer */
footer { display: none !important; }
"""


# JS injected into page to force dark mode and load Inter font
FORCE_DARK_JS = """
() => {
    // Force dark mode
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    const root = document.querySelector('.gradio-container');
    if (root) root.classList.add('dark');

    // Observe and ensure dark class stays
    const darkObserver = new MutationObserver((mutations) => {
        for (const m of mutations) {
            if (m.target.classList && !m.target.classList.contains('dark')) {
                m.target.classList.add('dark');
            }
        }
    });
    darkObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    darkObserver.observe(document.body, { attributes: true, attributeFilter: ['class'] });

    // Ensure uploaded images in upload-zone are visible
    function fixUploadedImages() {
        document.querySelectorAll('.upload-zone img, .upload-zone canvas, .upload-zone video').forEach(el => {
            el.style.opacity = '1';
            el.style.visibility = 'visible';
            el.style.display = 'block';
            el.style.maxHeight = '280px';
            el.style.objectFit = 'contain';
            el.style.margin = '0 auto';
        });
        // Also fix any parent containers that might be collapsed
        document.querySelectorAll('.upload-zone [data-testid] > div, .upload-zone .wrap').forEach(el => {
            if (el.offsetHeight < 10) {
                el.style.minHeight = '200px';
                el.style.display = 'flex';
                el.style.alignItems = 'center';
                el.style.justifyContent = 'center';
            }
        });
    }

    // Observe the upload zone for new child elements (images being added after upload)
    const imgObserver = new MutationObserver(() => { fixUploadedImages(); });
    const watchUploadZones = () => {
        document.querySelectorAll('.upload-zone').forEach(zone => {
            imgObserver.observe(zone, { childList: true, subtree: true, attributes: true });
        });
    };
    // Re-check periodically until upload zones are in DOM
    setTimeout(watchUploadZones, 1000);
    setTimeout(watchUploadZones, 3000);
    // Also fix on any interaction
    document.addEventListener('change', () => setTimeout(fixUploadedImages, 300));
    document.addEventListener('click', () => setTimeout(fixUploadedImages, 500));

    // Load Inter font
    if (!document.querySelector('link[href*="Inter"]')) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap';
        document.head.appendChild(link);
    }
}
"""


def _ensure_model_loaded():
    """Eagerly load the model so predictions work immediately."""
    global MODEL
    if MODEL is None:
        load_model()


def create_gradio_app() -> gr.Blocks:
    """Create the Gradio interface with a clean, clinical-grade design."""

    # Eagerly load model so predictions use the real checkpoint + threshold
    _ensure_model_loaded()

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        spacing_size=gr.themes.sizes.spacing_lg,
        radius_size=gr.themes.sizes.radius_lg,
        font=[
            gr.themes.GoogleFont("Inter"),
            "system-ui",
            "-apple-system",
            "sans-serif",
        ],
    ).set(
        body_background_fill="#0F1923",
        block_background_fill="#19232e",
        block_border_color="#2A3A4A",
        block_border_width="1px",
        block_label_text_color="#94A3B8",
        block_label_text_size="sm",
        block_shadow="0 1px 3px rgba(0,0,0,0.3)",
        button_primary_background_fill="linear-gradient(135deg, #3B82F6, #2563EB)",
        button_primary_text_color="white",
        input_background_fill="#19232e",
    )

    with gr.Blocks(
        title="HemaVision — AML Diagnostic Assistant",
    ) as demo:

        # ── Hero ─────────────────────────────────────────────
        gr.HTML("""
        <div class="hero-header">
            <div class="logo-pill">
                <div class="logo-icon">🔬</div>
                <h1>HemaVision</h1>
            </div>
            <p class="subtitle">
                AI-powered AML detection fusing deep visual features
                with handcrafted morphological analysis
            </p>
        </div>
        """)

        # ── Capability pills ─────────────────────────────────
        gr.HTML("""
        <div style="display:flex; justify-content:center; gap:10px; margin-bottom:28px; flex-wrap:wrap;">
            <span class="info-chip">🧠 ResNet50 Visual Stream</span>
            <span class="info-chip">📐 20 Morphological Features</span>
            <span class="info-chip">🔥 Grad-CAM Explainability</span>
            <span class="info-chip">⚡ &lt;50ms Inference</span>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Left: Upload ─────────────────────────────────
            with gr.Column(scale=5, min_width=360):
                gr.HTML("""
                <div class="section-title">
                    <span class="dot dot-blue"></span> UPLOAD
                </div>
                """)
                image_input = gr.Image(
                    type="pil",
                    label="Upload microscopic cell image",
                    elem_classes="upload-zone",
                    height=300,
                    sources=["upload", "clipboard"],
                )
                gr.HTML("""
                <div style="background:#0C1929; border:1px solid #1E3A5F; border-radius:10px;
                            padding:12px 16px; margin-top:12px; font-size:12.5px; color:#60A5FA; line-height:1.5;">
                    <strong>💡 Accepts both:</strong> Upload a <em>single-cell crop</em>
                    or a <em>whole blood smear field</em> — the system will auto-detect
                    and segment individual cells for per-cell analysis.
                </div>
                """)
                analyze_btn = gr.Button(
                    "Analyze",
                    variant="primary",
                    elem_classes="analyze-btn",
                    size="lg",
                )

            # ── Right: Results ───────────────────────────────
            with gr.Column(scale=6, min_width=380):
                gr.HTML("""
                <div class="section-title">
                    <span class="dot dot-green"></span> DIAGNOSTIC RESULT
                </div>
                """)
                result_output = gr.HTML(
                    value=_empty_state_html(),
                    elem_classes="result-area",
                )

                gr.HTML("""
                <div class="section-title" style="margin-top:20px;">
                    <span class="dot dot-amber"></span> EXPLAINABILITY MAP
                </div>
                """)
                gradcam_output = gr.Image(
                    label="Grad-CAM Visualization",
                    elem_classes="gradcam-img",
                    height=260,
                    show_label=False,
                )

        # ── Footer ───────────────────────────────────────────
        gr.HTML("""
        <div class="footer-area">
            <div class="disclaimer">
                <strong>⚠️ Medical Disclaimer</strong> — HemaVision is a research tool
                for educational and demonstration purposes only. It is <em>not</em>
                intended for clinical diagnosis. Always consult qualified hematologists
                for patient care decisions.
            </div>
            <div class="powered">
                Powered by PyTorch · ResNet50 + Morphological MLP Late Fusion · Grad-CAM++
            </div>
        </div>
        """)

        # ── Events ───────────────────────────────────────────
        analyze_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[result_output, gradcam_output],
        )

    return demo, theme


def _empty_state_html() -> str:
    """Placeholder shown before the user uploads an image."""
    return """
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                min-height:200px; color:#94A3B8; text-align:center; padding:32px;">
        <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="#475569" stroke-width="1.5"
             style="margin-bottom:16px;">
            <path stroke-linecap="round" stroke-linejoin="round"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414
                     a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
        <p style="font-size:15px; font-weight:500; color:#94A3B8; margin:0 0 6px;">
            No analysis yet
        </p>
        <p style="font-size:13px; color:#64748B; margin:0;">
            Upload a cell image and click <strong style='color:#94A3B8;'>Analyze</strong> to begin
        </p>
    </div>
    """


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    # Try to load model from default checkpoint
    default_ckpt = CONFIG.paths.checkpoints_dir / "best_model.pt"
    load_model(str(default_ckpt) if default_ckpt.exists() else None)

    app, theme = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=CONFIG.inference.gradio_port,
        share=False,
        show_error=True,
        theme=theme,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
    )
