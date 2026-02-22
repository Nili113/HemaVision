"""
HemaVision Morphological Feature Extractor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extracts handcrafted cytological features from cell microscopy images.

These features mimic the diagnostic criteria pathologists use:
  • Geometry:  cell area, perimeter, circularity, eccentricity
  • Nucleus:   nuclear area, nucleus-cytoplasm ratio, irregularity
  • Color:     mean RGB/HSV channels, stain variance
  • Texture:   GLCM contrast, homogeneity, energy, correlation
  • Shape:     solidity, convexity, elongation

Architecture (Hybrid Multimodal):
┌─────────────────────────────────────────────────────────┐
│  Stream 1: ResNet50 (deep learned features)  → 2048-dim │
│  Stream 2: Morphological Extractor           → 20-dim   │
│  Fusion:   Concatenate [2048 + 20]           → 2068-dim │
│  Head:     FC → ReLU → Dropout → FC → sigmoid           │
└─────────────────────────────────────────────────────────┘

The second stream encodes domain knowledge — the same features a
haematologist evaluates manually — so the fusion layer can learn
to weight deep visual patterns alongside clinically meaningful
morphological descriptors.

Author: Firoj
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy imports (graceful degradation) ────────────
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV (cv2) not installed. Install: pip install opencv-python-headless")

try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label as sk_label, regionprops
    from skimage.morphology import remove_small_objects
    from skimage.filters import threshold_otsu

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    logger.warning("scikit-image not installed. Install: pip install scikit-image")


# ── Feature names (must be stable across train/inference) ────
MORPHOLOGY_FEATURE_NAMES: List[str] = [
    # Geometry (4)
    "cell_area",
    "cell_perimeter",
    "circularity",
    "eccentricity",
    # Nucleus (3)
    "nuclear_area",
    "nc_ratio",
    "nuclear_irregularity",
    # Color — RGB means (3)
    "mean_r",
    "mean_g",
    "mean_b",
    # Color — HSV means (3)
    "mean_h",
    "mean_s",
    "mean_v",
    # Color — stain statistics (2)
    "stain_intensity",
    "stain_variance",
    # Texture — GLCM (4)
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    # Shape (1)
    "solidity",
]

NUM_MORPHOLOGY_FEATURES = len(MORPHOLOGY_FEATURE_NAMES)  # 20


def _ensure_numpy(image) -> np.ndarray:
    """Convert PIL Image or tensor to uint8 numpy array (H, W, 3)."""
    if hasattr(image, "numpy"):
        # PyTorch tensor
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        return arr
    if hasattr(image, "convert"):
        # PIL Image
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0 and image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


# ═════════════════════════════════════════════════════════════
# PRIMARY EXTRACTION PIPELINE
# ═════════════════════════════════════════════════════════════


def extract_morphology_features(image, normalize: bool = True) -> np.ndarray:
    """
    Extract 20 morphological features from a single cell image.

    This is the main entry point. It converts the image to numpy,
    segments the cell, and computes geometry / colour / texture /
    shape descriptors.

    Args:
        image:     PIL Image, numpy array (H,W,3) uint8, or torch Tensor
        normalize: Whether to apply per-feature z-score normalization
                   using population statistics from AML-Cytomorphology.

    Returns:
        np.ndarray of shape (20,) — one value per feature in
        MORPHOLOGY_FEATURE_NAMES order.
    """
    img = _ensure_numpy(image)  # (H, W, 3) uint8

    features: Dict[str, float] = {}

    # ── 1. Segment the cell ──────────────────────────────────
    mask, nucleus_mask = _segment_cell(img)

    # ── 2. Geometry features ─────────────────────────────────
    features.update(_geometry_features(mask))

    # ── 3. Nucleus features ──────────────────────────────────
    features.update(_nucleus_features(mask, nucleus_mask))

    # ── 4. Colour features ───────────────────────────────────
    features.update(_colour_features(img, mask))

    # ── 5. Texture features (GLCM) ──────────────────────────
    features.update(_texture_features(img, mask))

    # ── 6. Shape features ────────────────────────────────────
    features.update(_shape_features(mask))

    # Assemble into ordered array
    vec = np.array(
        [features.get(name, 0.0) for name in MORPHOLOGY_FEATURE_NAMES],
        dtype=np.float32,
    )

    # Replace NaN / Inf with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        vec = _normalize_features(vec)

    return vec


# ═════════════════════════════════════════════════════════════
# SEGMENTATION
# ═════════════════════════════════════════════════════════════


def _segment_cell(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment the cell and its nucleus from a stained microscopy image.

    Strategy:
    1. Convert to grayscale, apply Otsu threshold → cell mask
    2. Convert to HSV, threshold the dark/blue-purple nucleus

    Returns:
        (cell_mask, nucleus_mask) — boolean arrays of shape (H, W)
    """
    h, w = img.shape[:2]

    if _HAS_CV2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Otsu's threshold for cell body
        _, cell_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cell_mask = cell_bin.astype(bool)

        # Nucleus: segment the darkest / most purple region
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # Dark nuclei have low value and moderate-high saturation
        low_val = hsv[:, :, 2] < 140
        high_sat = hsv[:, :, 1] > 40
        nucleus_mask = low_val & high_sat & cell_mask

    elif _HAS_SKIMAGE:
        from skimage.color import rgb2gray, rgb2hsv

        gray = (rgb2gray(img) * 255).astype(np.uint8)
        try:
            thresh = threshold_otsu(gray)
        except ValueError:
            thresh = 128
        cell_mask = gray < thresh  # Inverted (stained cell is darker)

        hsv = rgb2hsv(img)
        low_val = hsv[:, :, 2] < (140 / 255.0)
        high_sat = hsv[:, :, 1] > (40 / 255.0)
        nucleus_mask = low_val & high_sat & cell_mask

    else:
        # Pure-numpy fallback (no cv2, no skimage)
        gray = np.mean(img, axis=2).astype(np.uint8)
        thresh = int(np.mean(gray))
        cell_mask = gray < thresh

        # Rough nucleus: darkest 30% of cell pixels
        cell_pixels = gray[cell_mask]
        if len(cell_pixels) > 0:
            nuc_thresh = np.percentile(cell_pixels, 30)
            nucleus_mask = (gray < nuc_thresh) & cell_mask
        else:
            nucleus_mask = np.zeros_like(cell_mask)

    # Clean up small noise regions
    min_size = max(50, int(h * w * 0.002))
    if _HAS_SKIMAGE:
        try:
            cell_mask = remove_small_objects(cell_mask, max_size=min_size)
            nucleus_mask = remove_small_objects(nucleus_mask, max_size=min_size // 4)
        except Exception:
            pass

    # If segmentation failed (empty mask), fall back to center circle
    if cell_mask.sum() < min_size:
        yy, xx = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        cell_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) < radius ** 2
        nucleus_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) < (radius // 2) ** 2

    return cell_mask, nucleus_mask


# ═════════════════════════════════════════════════════════════
# FEATURE GROUPS
# ═════════════════════════════════════════════════════════════


def _geometry_features(mask: np.ndarray) -> Dict[str, float]:
    """Cell area, perimeter, circularity, eccentricity."""
    area = float(mask.sum())
    h, w = mask.shape

    # Normalise area by image size so features are resolution-invariant
    total_pixels = h * w
    norm_area = area / total_pixels if total_pixels > 0 else 0.0

    # Perimeter (count edge pixels)
    if _HAS_CV2:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
    else:
        # Approximate: count border transitions
        diff_h = np.abs(np.diff(mask.astype(np.int8), axis=0))
        diff_w = np.abs(np.diff(mask.astype(np.int8), axis=1))
        perimeter = float(diff_h.sum() + diff_w.sum())

    norm_perimeter = perimeter / max(h + w, 1)

    # Circularity: 4π × area / perimeter²
    circularity = 0.0
    if perimeter > 0:
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)

    # Eccentricity via region props
    eccentricity = 0.0
    if _HAS_SKIMAGE:
        try:
            labelled = sk_label(mask)
            props = regionprops(labelled)
            if props:
                eccentricity = float(props[0].eccentricity)
        except Exception:
            pass
    else:
        # Rough estimate from bounding box aspect ratio
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            bbox_h = r_max - r_min + 1
            bbox_w = c_max - c_min + 1
            if max(bbox_h, bbox_w) > 0:
                eccentricity = 1.0 - min(bbox_h, bbox_w) / max(bbox_h, bbox_w)

    return {
        "cell_area": norm_area,
        "cell_perimeter": norm_perimeter,
        "circularity": circularity,
        "eccentricity": eccentricity,
    }


def _nucleus_features(
    cell_mask: np.ndarray, nucleus_mask: np.ndarray
) -> Dict[str, float]:
    """Nuclear area, nucleus-cytoplasm ratio, nuclear irregularity."""
    cell_area = float(cell_mask.sum())
    nuc_area = float(nucleus_mask.sum())
    h, w = cell_mask.shape
    total_pixels = h * w

    norm_nuc_area = nuc_area / total_pixels if total_pixels > 0 else 0.0

    # Nucleus-Cytoplasm ratio — high in blasts
    nc_ratio = nuc_area / cell_area if cell_area > 0 else 0.0

    # Nuclear irregularity — deviation from a perfect circle
    nuc_irreg = 0.0
    if nuc_area > 50:
        if _HAS_CV2:
            contours, _ = cv2.findContours(
                nucleus_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                nuc_peri = cv2.arcLength(contours[0], True)
                if nuc_peri > 0:
                    nuc_circ = (4.0 * np.pi * nuc_area) / (nuc_peri ** 2)
                    nuc_irreg = 1.0 - min(nuc_circ, 1.0)
        else:
            # Rough: 1 − circularity estimate
            diff_h = np.abs(np.diff(nucleus_mask.astype(np.int8), axis=0))
            diff_w = np.abs(np.diff(nucleus_mask.astype(np.int8), axis=1))
            nuc_peri = float(diff_h.sum() + diff_w.sum())
            if nuc_peri > 0:
                nuc_circ = (4.0 * np.pi * nuc_area) / (nuc_peri ** 2)
                nuc_irreg = 1.0 - min(nuc_circ, 1.0)

    return {
        "nuclear_area": norm_nuc_area,
        "nc_ratio": nc_ratio,
        "nuclear_irregularity": nuc_irreg,
    }


def _colour_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Mean RGB, mean HSV, stain intensity and variance."""
    # Extract cell pixels only
    cell_pixels = img[mask] if mask.any() else img.reshape(-1, 3)

    # RGB means (0–1 scale)
    mean_rgb = cell_pixels.mean(axis=0).astype(np.float64) / 255.0
    mean_r, mean_g, mean_b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])

    # HSV means
    if _HAS_CV2:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv[mask] if mask.any() else hsv.reshape(-1, 3)
        mean_hsv = hsv_pixels.mean(axis=0).astype(np.float64)
        mean_h = float(mean_hsv[0]) / 180.0  # OpenCV H range: [0, 180]
        mean_s = float(mean_hsv[1]) / 255.0
        mean_v = float(mean_hsv[2]) / 255.0
    else:
        # Manual RGB → HSV
        r, g, b = mean_r, mean_g, mean_b
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        diff = cmax - cmin
        if diff == 0:
            mean_h = 0.0
        elif cmax == r:
            mean_h = (60 * ((g - b) / diff) % 360) / 360.0
        elif cmax == g:
            mean_h = (60 * ((b - r) / diff) + 120) / 360.0
        else:
            mean_h = (60 * ((r - g) / diff) + 240) / 360.0
        mean_s = diff / cmax if cmax > 0 else 0.0
        mean_v = cmax

    # Stain intensity: mean luminance of cell (lower = more stain)
    gray_pixels = cell_pixels.mean(axis=1) / 255.0
    stain_intensity = 1.0 - float(gray_pixels.mean())  # Inverted: higher = more stained

    # Stain variance: how uneven the staining is
    stain_variance = float(gray_pixels.var())

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "stain_intensity": stain_intensity,
        "stain_variance": stain_variance,
    }


def _texture_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Gray-Level Co-occurrence Matrix (GLCM) features.

    GLCM captures spatial relationships between pixel intensities,
    encoding texture patterns critical for discriminating blast cells
    (coarse chromatin) from normal cells (smooth chromatin).
    """
    # Convert to grayscale
    if img.ndim == 3:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img.astype(np.uint8)

    # Crop to cell bounding box for efficiency
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        gray_crop = gray[r_min : r_max + 1, c_min : c_max + 1]
    else:
        gray_crop = gray

    # Ensure minimum size
    if gray_crop.shape[0] < 4 or gray_crop.shape[1] < 4:
        gray_crop = gray

    # Quantize to 32 levels for a manageable GLCM
    gray_q = (gray_crop // 8).astype(np.uint8)  # 256 / 8 = 32 levels

    contrast, homogeneity, energy, correlation = 0.0, 0.0, 0.0, 0.0

    if _HAS_SKIMAGE:
        try:
            glcm = graycomatrix(
                gray_q,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=32,
                symmetric=True,
                normed=True,
            )
            contrast = float(graycoprops(glcm, "contrast").mean())
            homogeneity = float(graycoprops(glcm, "homogeneity").mean())
            energy = float(graycoprops(glcm, "energy").mean())
            correlation = float(graycoprops(glcm, "correlation").mean())
        except Exception:
            pass
    else:
        # Simplified texture approximation without skimage
        # Variance ≈ contrast, smoothness ≈ homogeneity
        contrast = float(gray_crop.astype(np.float64).var()) / 1000.0
        homogeneity = 1.0 / (1.0 + contrast)
        energy = float((gray_crop.astype(np.float64) ** 2).mean()) / 65025.0
        correlation = 0.5  # Neutral default

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
    }


def _shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Solidity: ratio of cell area to convex hull area."""
    solidity = 0.0
    cell_area = float(mask.sum())

    if cell_area < 10:
        return {"solidity": 0.0}

    if _HAS_CV2:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = cell_area / hull_area
                solidity = min(solidity, 1.0)
    elif _HAS_SKIMAGE:
        try:
            labelled = sk_label(mask)
            props = regionprops(labelled)
            if props:
                solidity = float(props[0].solidity)
        except Exception:
            pass
    else:
        # Approximate solidity as ratio of filled pixels to bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            bbox_area = (r_max - r_min + 1) * (c_max - c_min + 1)
            if bbox_area > 0:
                solidity = cell_area / bbox_area

    return {"solidity": solidity}


# ═════════════════════════════════════════════════════════════
# NORMALIZATION (population statistics from AML-Cytomorphology)
# ═════════════════════════════════════════════════════════════

# Mean / std computed from a representative sample of the dataset.
# These are approximate — they'll be refined after the first full
# training run, but provide stable z-score centering from the start.
_FEATURE_STATS: Dict[str, Tuple[float, float]] = {
    # (mean, std) — std is clamped to min 1e-6 to prevent division by zero
    "cell_area":            (0.25,  0.12),
    "cell_perimeter":       (0.35,  0.15),
    "circularity":          (0.65,  0.15),
    "eccentricity":         (0.45,  0.20),
    "nuclear_area":         (0.10,  0.06),
    "nc_ratio":             (0.40,  0.15),
    "nuclear_irregularity": (0.35,  0.15),
    "mean_r":               (0.60,  0.10),
    "mean_g":               (0.45,  0.10),
    "mean_b":               (0.55,  0.10),
    "mean_h":               (0.50,  0.15),
    "mean_s":               (0.35,  0.12),
    "mean_v":               (0.55,  0.10),
    "stain_intensity":       (0.45,  0.10),
    "stain_variance":        (0.02,  0.01),
    "glcm_contrast":         (5.0,   4.0),
    "glcm_homogeneity":      (0.60,  0.15),
    "glcm_energy":           (0.05,  0.03),
    "glcm_correlation":      (0.85,  0.10),
    "solidity":              (0.85,  0.10),
}


def _normalize_features(vec: np.ndarray) -> np.ndarray:
    """Z-score normalize using population statistics."""
    normed = np.zeros_like(vec)
    for i, name in enumerate(MORPHOLOGY_FEATURE_NAMES):
        mu, sigma = _FEATURE_STATS.get(name, (0.0, 1.0))
        sigma = max(sigma, 1e-6)
        normed[i] = (vec[i] - mu) / sigma
    return normed


# ═════════════════════════════════════════════════════════════
# BATCH EXTRACTION (for DataFrame-level preprocessing)
# ═════════════════════════════════════════════════════════════


def extract_features_for_dataframe(
    image_paths: List[str],
    normalize: bool = True,
    max_workers: int = 4,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract morphological features for a list of image paths.

    Used during data preprocessing to compute the tabular stream
    inputs for the entire dataset.

    Args:
        image_paths:  List of absolute paths to cell images
        normalize:    Apply z-score normalization
        max_workers:  Number of parallel workers (0 = sequential)
        show_progress: Print progress bar

    Returns:
        np.ndarray of shape (N, 20)
    """
    from PIL import Image as PILImage

    n = len(image_paths)
    features = np.zeros((n, NUM_MORPHOLOGY_FEATURES), dtype=np.float32)

    log_interval = max(1, n // 20)  # Log every 5%

    for i, path in enumerate(image_paths):
        try:
            img = PILImage.open(path).convert("RGB")
            features[i] = extract_morphology_features(img, normalize=normalize)
        except Exception as e:
            logger.warning(f"Feature extraction failed for {path}: {e}")
            # Row stays all-zeros (neutral after normalization)

        if show_progress and (i + 1) % log_interval == 0:
            pct = (i + 1) / n * 100
            logger.info(f"  Morphology extraction: {i + 1}/{n} ({pct:.0f}%)")

    logger.info(
        f"Extracted {NUM_MORPHOLOGY_FEATURES} morphological features "
        f"from {n} images (normalize={normalize})"
    )
    return features


def extract_single_image_features(image, normalize: bool = True) -> np.ndarray:
    """
    Convenience wrapper for inference on a single image.

    Accepts PIL Image, numpy array, or file path string.
    Returns (20,) feature vector.
    """
    if isinstance(image, (str, Path)):
        from PIL import Image as PILImage
        image = PILImage.open(str(image)).convert("RGB")

    return extract_morphology_features(image, normalize=normalize)
