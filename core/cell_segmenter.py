"""
HemaVision Cell Segmenter
━━━━━━━━━━━━━━━━━━━━━━━━━
Detects and crops individual cells from whole blood smear field images.

When users upload full-field microscopy images (many cells visible),
this module segments them into individual cell crops so the model
can analyze each one independently.

Pipeline:
  1. Resize large images to a workable resolution
  2. Convert to grayscale → Otsu threshold → binary mask
  3. Morphological cleanup (open/close to remove noise)
  4. Find contours → filter by area → extract bounding boxes
  5. Expand bounding boxes to square cells with padding
  6. Crop each cell from the original full-resolution image

Author: Firoj
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not available — cell segmentation disabled")


# ── Configuration ────────────────────────────────────────────

# Minimum / maximum relative area of a detected cell region.
# Expressed as a fraction of the total image area.
MIN_CELL_AREA_FRAC = 0.002   # Cell must be > 0.2% of image
MAX_CELL_AREA_FRAC = 0.25    # Cell must be < 25% of image
MAX_CELLS = 30               # Don't return more than this
CELL_CROP_PAD = 0.15         # 15% padding around each cell crop
MIN_CELL_SIZE_PX = 32        # Minimum crop dimension in pixels

# Images below this size are already single-cell crops
SINGLE_CELL_THRESHOLD = 600  # w or h ≤ 600 → single cell


@dataclass
class CellCrop:
    """A single detected cell from a larger image."""
    image: Image.Image       # Cropped cell as PIL Image
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original coords
    area: float              # Contour area in pixels
    center: Tuple[int, int]  # Center (cx, cy) in original coords
    index: int               # Cell number (0-based)


@dataclass
class SegmentationResult:
    """Result of cell segmentation."""
    cells: List[CellCrop] = field(default_factory=list)
    is_multi_cell: bool = False
    original_size: Tuple[int, int] = (0, 0)   # (w, h)
    annotated_image: Optional[Image.Image] = None  # Original with bboxes drawn
    message: str = ""


def is_multi_cell_image(image: Image.Image) -> bool:
    """Quick check: is this image likely a multi-cell field?"""
    w, h = image.size
    if w <= SINGLE_CELL_THRESHOLD and h <= SINGLE_CELL_THRESHOLD:
        return False
    if not _HAS_CV2:
        return w > 1000 or h > 1000  # Fallback heuristic
    # Do a quick contour check
    result = segment_cells(image, max_cells=5, annotate=False)
    return len(result.cells) > 1


def segment_cells(
    image: Image.Image,
    max_cells: int = MAX_CELLS,
    annotate: bool = True,
) -> SegmentationResult:
    """
    Segment individual cells from a microscopy image.

    Args:
        image:     PIL Image (RGB)
        max_cells: Maximum number of cells to return
        annotate:  Whether to draw bounding boxes on the original

    Returns:
        SegmentationResult with list of CellCrop objects
    """
    image_rgb = image.convert("RGB")
    w, h = image_rgb.size
    result = SegmentationResult(original_size=(w, h))

    # If the image is small, treat it as a single cell
    if w <= SINGLE_CELL_THRESHOLD and h <= SINGLE_CELL_THRESHOLD:
        result.cells = [CellCrop(
            image=image_rgb,
            bbox=(0, 0, w, h),
            area=float(w * h),
            center=(w // 2, h // 2),
            index=0,
        )]
        result.is_multi_cell = False
        result.message = "Single-cell image detected — analyzing directly."
        return result

    if not _HAS_CV2:
        # Fallback: return the whole image as one crop
        result.cells = [CellCrop(
            image=image_rgb,
            bbox=(0, 0, w, h),
            area=float(w * h),
            center=(w // 2, h // 2),
            index=0,
        )]
        result.message = "OpenCV not available — analyzing whole image as single cell."
        return result

    # ── Convert and threshold ────────────────────────────────
    img_np = np.array(image_rgb)
    # Work at reduced resolution for speed, keep original for cropping
    scale = min(1.0, 1024.0 / max(w, h))
    if scale < 1.0:
        small = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_np.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold works better than Otsu for stained smears
    # that have uneven illumination
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Morphological cleanup ────────────────────────────────
    kernel_size = max(3, int(7 * scale))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # ── Find contours ────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sh, sw = small.shape[:2]
    total_area = sh * sw
    min_area = total_area * MIN_CELL_AREA_FRAC
    max_area = total_area * MAX_CELL_AREA_FRAC

    # Filter and sort by area (largest first)
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            valid_contours.append((c, area))
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    valid_contours = valid_contours[:max_cells]

    if len(valid_contours) == 0:
        # No cells found — return whole image
        result.cells = [CellCrop(
            image=image_rgb,
            bbox=(0, 0, w, h),
            area=float(w * h),
            center=(w // 2, h // 2),
            index=0,
        )]
        result.message = "No individual cells detected — analyzing whole image."
        return result

    # ── Extract cell crops ───────────────────────────────────
    annotated = img_np.copy() if annotate else None
    cells: List[CellCrop] = []

    for idx, (contour, area) in enumerate(valid_contours):
        x, y, cw, ch = cv2.boundingRect(contour)

        # Scale bounding box back to original resolution
        inv_scale = 1.0 / scale
        ox = int(x * inv_scale)
        oy = int(y * inv_scale)
        ocw = int(cw * inv_scale)
        och = int(ch * inv_scale)

        # Make square with padding
        side = max(ocw, och)
        pad = int(side * CELL_CROP_PAD)
        side += 2 * pad

        cx = ox + ocw // 2
        cy = oy + och // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        # Ensure minimum size
        if (x2 - x1) < MIN_CELL_SIZE_PX or (y2 - y1) < MIN_CELL_SIZE_PX:
            continue

        crop = image_rgb.crop((x1, y1, x2, y2))
        cells.append(CellCrop(
            image=crop,
            bbox=(x1, y1, x2, y2),
            area=float(area * inv_scale * inv_scale),
            center=(cx, cy),
            index=idx,
        ))

        # Draw on annotated image
        if annotated is not None:
            color = (59, 130, 246)  # Blue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"#{idx + 1}"
            font_scale = max(0.6, min(1.5, side / 300))
            thickness = max(1, int(font_scale * 2))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    result.cells = cells
    result.is_multi_cell = len(cells) > 1
    result.annotated_image = Image.fromarray(annotated) if annotated is not None else None

    n = len(cells)
    if result.is_multi_cell:
        result.message = f"Detected {n} cells in the blood smear — analyzing each individually."
    else:
        result.message = f"Single cell detected — analyzing directly."

    logger.info(f"Segmented {n} cells from {w}×{h} image")
    return result
