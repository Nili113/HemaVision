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
MAX_CELLS = 80               # Upper safety cap for very dense fields
CELL_CROP_PAD = 0.15         # 15% padding around each cell crop
MIN_CELL_SIZE_PX = 32        # Minimum crop dimension in pixels
MAX_BOX_IOU = 0.35           # Suppress heavily overlapping boxes
MIN_NUCLEUS_FILL_RATIO = 0.30  # Reject thin/ring artifacts (RBC edges)
DOMINANT_CELL_MIN_AREA_FRAC = 0.06  # Largest nucleus should occupy >= 6% of frame
DOMINANT_CELL_RATIO_MIN = 2.2        # Largest nucleus should be much bigger than #2
DOMINANT_CELL_CENTER_MAX_DIST = 0.42 # Relative center distance from image center
DOMINANT_CELL_MAX_COMPONENTS = 5      # Too many nucleus blobs => likely multi-cell field
DOMINANT_CELL_MIN_SHARE = 0.55        # Largest nucleus should dominate total nucleus area

# Very tiny images are usually already single-cell crops.
# Keep this conservative so mid-size smear snapshots are still segmented.
SINGLE_CELL_THRESHOLD = 256  # w or h ≤ 256 → likely single cell


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute IoU for two boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_thr: float = MAX_BOX_IOU,
) -> List[int]:
    """Simple NMS returning kept indices in descending score order."""
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []

    for idx in order:
        candidate = boxes[idx]
        if all(_box_iou(candidate, boxes[k]) < iou_thr for k in keep):
            keep.append(idx)
    return keep


def _color_for_index(i: int) -> Tuple[int, int, int]:
    """Stable, high-contrast color palette for overlay labels/contours."""
    palette = [
        (59, 130, 246),   # blue
        (16, 185, 129),   # emerald
        (245, 158, 11),   # amber
        (236, 72, 153),   # pink
        (139, 92, 246),   # violet
        (14, 165, 233),   # sky
        (234, 88, 12),    # orange
        (132, 204, 22),   # lime
    ]
    return palette[i % len(palette)]


def _build_nucleus_mask(small_rgb: np.ndarray) -> np.ndarray:
    """Build a robust nucleus-like mask in HSV space."""
    hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, (115, 45, 35), (145, 255, 255))
    mask2 = cv2.inRange(hsv, (145, 35, 25), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    sh, sw = small_rgb.shape[:2]
    k = max(3, int(round(min(sh, sw) / 140)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _detect_single_dominant_cell(
    small_rgb: np.ndarray,
) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Detect whether image is a single-cell close-up with one dominant nucleus.

    Returns:
        ((x, y, w, h), contour) in resized image coordinates if detected,
        otherwise None.
    """
    mask = _build_nucleus_mask(small_rgb)
    sh, sw = small_rgb.shape[:2]
    total_area = float(sh * sw)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None

    comps = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < total_area * 0.003:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comps.append((i, area, x, y, w, h, centroids[i]))

    if not comps:
        return None

    comps.sort(key=lambda t: t[1], reverse=True)
    i0, a0, x, y, w, h, c0 = comps[0]
    a1 = comps[1][1] if len(comps) > 1 else 0.0
    total_nucleus_area = float(sum(c[1] for c in comps))

    area_frac = a0 / total_area
    ratio = a0 / max(a1, 1.0)
    share = a0 / max(total_nucleus_area, 1.0)
    cx, cy = float(c0[0]), float(c0[1])
    center_dist = np.hypot(cx - (sw / 2), cy - (sh / 2)) / max(1.0, np.hypot(sw / 2, sh / 2))

    if len(comps) > DOMINANT_CELL_MAX_COMPONENTS:
        return None
    if area_frac < DOMINANT_CELL_MIN_AREA_FRAC:
        return None
    if ratio < DOMINANT_CELL_RATIO_MIN:
        return None
    if share < DOMINANT_CELL_MIN_SHARE:
        return None
    if center_dist > DOMINANT_CELL_CENTER_MAX_DIST:
        return None

    component_mask = np.zeros((sh, sw), dtype=np.uint8)
    component_mask[labels == i0] = 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Expand bbox around the nucleus so crop includes cytoplasm context.
    side = max(w, h)
    pad = int(side * 0.45)
    bx = max(0, x - pad)
    by = max(0, y - pad)
    bw = min(sw - bx, w + 2 * pad)
    bh = min(sh - by, h + 2 * pad)

    return (bx, by, bw, bh), contour


def _extract_cell_regions_watershed(
    small_rgb: np.ndarray,
) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    """
    Segment likely WBC nuclei with HSV color mask + watershed.

    Returns:
        List of ((x, y, w, h), area, contour) in resized image coordinates.
    """
    mask = _build_nucleus_mask(small_rgb)

    sh, sw = small_rgb.shape[:2]
    total_area = sh * sw

    # Clean tiny noise while preserving nucleus cores.
    k = max(3, int(round(min(sh, sw) / 140)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sure foreground from distance peaks to split touching nuclei.
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 0.0:
        return []

    # Use local maxima as watershed seeds to better separate touching cells.
    # This is more robust than a single global threshold for clustered nuclei.
    dist_f = dist.astype(np.float32)
    local_max = (dist_f == cv2.dilate(dist_f, np.ones((3, 3), np.uint8)))
    seed_mask = local_max & (dist_f > (0.40 * max_dist))
    sure_fg = seed_mask.astype(np.uint8) * 255

    # Fallback if local-max seeds are too sparse.
    if np.count_nonzero(sure_fg) < 8:
        sure_fg = (dist > (0.45 * max_dist)).astype(np.uint8) * 255

    sure_bg = cv2.dilate(clean, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 1:
        return []

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR), markers)

    # Keep minima stricter to avoid tiny magenta specks/rings.
    min_area = total_area * MIN_CELL_AREA_FRAC * 0.7
    max_area = total_area * MAX_CELL_AREA_FRAC * 0.7

    regions: List[Tuple[Tuple[int, int, int, int], float, np.ndarray]] = []
    for lbl in np.unique(markers):
        if lbl <= 1:
            continue
        ys, xs = np.where(markers == lbl)
        if xs.size == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = max(1, x2 - x1 + 1)
        h = max(1, y2 - y1 + 1)
        area = float(xs.size)
        fill_ratio = area / float(w * h)

        label_mask = np.zeros((sh, sw), dtype=np.uint8)
        label_mask[markers == lbl] = 255
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        # Reject tiny specks and very large merged regions.
        if not (min_area <= area <= max_area):
            continue
        # Reject elongated debris and edge artifacts.
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 2.3:
            continue
        # Reject hollow/thin ring-like regions common in RBC artifacts.
        if fill_ratio < MIN_NUCLEUS_FILL_RATIO:
            continue
        if w < 12 or h < 12:
            continue

        regions.append(((x1, y1, w, h), area, contour))

    return regions


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
    estimated_total_cells: int = 0
    segmentation_mode_used: str = "auto"
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
    segmentation_mode: str = "auto",
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
    if segmentation_mode not in {"auto", "single", "multi"}:
        segmentation_mode = "auto"

    image_rgb = image.convert("RGB")
    w, h = image_rgb.size
    result = SegmentationResult(original_size=(w, h), segmentation_mode_used=segmentation_mode)

    # Only very tiny images bypass contour segmentation.
    # Mid-size snapshots (e.g. ~400x400) can still contain many cells.
    if w <= SINGLE_CELL_THRESHOLD and h <= SINGLE_CELL_THRESHOLD:
        result.cells = [CellCrop(
            image=image_rgb,
            bbox=(0, 0, w, h),
            area=float(w * h),
            center=(w // 2, h // 2),
            index=0,
        )]
        result.is_multi_cell = False
        result.estimated_total_cells = 1
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
        result.estimated_total_cells = 1
        result.message = "OpenCV not available — analyzing whole image as single cell."
        return result

    # ── Build candidate regions ───────────────────────────────
    img_np = np.array(image_rgb)
    # Work at reduced resolution for speed, keep original for cropping
    scale = min(1.0, 1024.0 / max(w, h))
    if scale < 1.0:
        small = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_np.copy()

    # Special case: single-cell close-up with one dominant nucleus.
    dominant = _detect_single_dominant_cell(small) if segmentation_mode != "multi" else None
    if dominant is not None:
        (x, y, cw, ch), contour = dominant
        candidate_regions = [((x, y, cw, ch), float(cw * ch), contour)]
    else:
        # 1) Preferred path: watershed split on nucleus-like color mask.
        candidate_regions = _extract_cell_regions_watershed(small)

    # 2) Fallback path: contouring if watershed yields nothing.
    if len(candidate_regions) == 0:
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_size = max(3, int(7 * scale))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sh, sw = small.shape[:2]
        total_area = sh * sw
        min_area = total_area * MIN_CELL_AREA_FRAC
        max_area = total_area * MAX_CELL_AREA_FRAC

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(c)
                box_area = max(1, cw * ch)
                fill_ratio = float(area) / float(box_area)
                if fill_ratio < 0.22:
                    continue
                candidate_regions.append(((x, y, cw, ch), float(area), c))

    if len(candidate_regions) == 0:
        # No cells found — return whole image
        result.cells = [CellCrop(
            image=image_rgb,
            bbox=(0, 0, w, h),
            area=float(w * h),
            center=(w // 2, h // 2),
            index=0,
        )]
        result.estimated_total_cells = 1
        result.message = "No individual cells detected — analyzing whole image."
        return result

    # ── Extract cell crops ───────────────────────────────────
    annotated = img_np.copy() if annotate else None
    cells: List[CellCrop] = []

    pre_boxes: List[Tuple[int, int, int, int]] = []
    pre_scores: List[float] = []
    pre_areas: List[float] = []
    pre_contours: List[np.ndarray] = []

    for (x, y, cw, ch), area, contour in candidate_regions:

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

        pre_boxes.append((x1, y1, x2, y2))
        pre_scores.append(float(area))
        pre_areas.append(float(area * inv_scale * inv_scale))

        contour_scaled = contour.astype(np.float32).copy()
        contour_scaled[:, 0, 0] *= inv_scale
        contour_scaled[:, 0, 1] *= inv_scale
        contour_scaled[:, 0, 0] = np.clip(contour_scaled[:, 0, 0], 0, w - 1)
        contour_scaled[:, 0, 1] = np.clip(contour_scaled[:, 0, 1], 0, h - 1)
        pre_contours.append(contour_scaled.astype(np.int32))

    keep_idx = _nms_boxes(pre_boxes, pre_scores, iou_thr=MAX_BOX_IOU)

    # Close-up safety guard: if exactly one substantial nucleus exists and
    # other detections are tiny fragments, force single-cell mode.
    if keep_idx and segmentation_mode == "auto":
        sh, sw = small.shape[:2]
        img_area_small = float(sh * sw)
        areas = [pre_scores[i] for i in keep_idx]
        max_pos = int(np.argmax(areas))
        dominant_idx = keep_idx[max_pos]
        dominant_area = areas[max_pos]
        dominant_frac = dominant_area / max(1.0, img_area_small)

        substantial_count = sum(1 for a in areas if a >= 0.28 * dominant_area)
        cx1, cy1, cx2, cy2 = pre_boxes[dominant_idx]
        dcx = (cx1 + cx2) / 2.0
        dcy = (cy1 + cy2) / 2.0
        center_dist = np.hypot(dcx - (w / 2.0), dcy - (h / 2.0)) / max(1.0, np.hypot(w / 2.0, h / 2.0))

        if substantial_count == 1 and dominant_frac >= 0.02 and center_dist <= 0.55:
            keep_idx = [dominant_idx]

    if keep_idx and segmentation_mode == "single":
        dominant_idx = keep_idx[int(np.argmax([pre_scores[i] for i in keep_idx]))]
        keep_idx = [dominant_idx]

    num_kept_before_cap = len(keep_idx)
    keep_idx = keep_idx[:max_cells]
    capped = num_kept_before_cap > max_cells
    result.estimated_total_cells = max(1, num_kept_before_cap)

    # Sort retained boxes by top-left position for stable display order.
    keep_idx.sort(key=lambda i: (pre_boxes[i][1], pre_boxes[i][0]))

    for out_idx, i in enumerate(keep_idx):
        x1, y1, x2, y2 = pre_boxes[i]
        area_scaled = pre_areas[i]

        crop = image_rgb.crop((x1, y1, x2, y2))
        cells.append(CellCrop(
            image=crop,
            bbox=(x1, y1, x2, y2),
            area=area_scaled,
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            index=out_idx,
        ))

        # Draw on annotated image
    if annotated is not None and len(keep_idx) > 0:
        overlay = annotated.copy()

        # Fill segmentation regions with light transparency.
        for out_idx, i in enumerate(keep_idx):
            contour = pre_contours[i]
            color = _color_for_index(out_idx)
            cv2.drawContours(overlay, [contour], -1, color, thickness=-1)

        annotated = cv2.addWeighted(overlay, 0.24, annotated, 0.76, 0)

        # Draw crisp contour edges and compact labels.
        for out_idx, i in enumerate(keep_idx):
            contour = pre_contours[i]
            color = _color_for_index(out_idx)
            cv2.drawContours(annotated, [contour], -1, color, thickness=2)

            m = cv2.moments(contour)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                x1, y1, x2, y2 = pre_boxes[i]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            label = f"{out_idx + 1}"
            font_scale = 0.65
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            lx1 = max(0, cx - tw // 2 - 6)
            ly1 = max(0, cy - th - 10)
            lx2 = min(w - 1, lx1 + tw + 12)
            ly2 = min(h - 1, ly1 + th + 10)
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(
                annotated,
                label,
                (lx1 + 6, ly2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

    result.cells = cells
    result.is_multi_cell = len(cells) > 1
    result.annotated_image = Image.fromarray(annotated) if annotated is not None else None

    n = len(cells)
    if dominant is not None and n == 1:
        mode_prefix = "[single] " if segmentation_mode == "single" else ""
        result.message = f"{mode_prefix}Dominant single cell detected — analyzing one cell crop."
        result.is_multi_cell = False
        result.annotated_image = Image.fromarray(annotated) if annotated is not None else None
        logger.info(f"Segmented {n} cells from {w}×{h} image (dominant-cell mode)")
        return result

    if result.is_multi_cell:
        mode_prefix = "[multi] " if segmentation_mode == "multi" else ""
        result.message = f"{mode_prefix}Detected {n} cells in the blood smear — analyzing each individually."
        if capped:
            result.message += f" (showing top {max_cells} detections; more were found)"
    else:
        mode_prefix = "[single] " if segmentation_mode == "single" else ""
        result.message = f"{mode_prefix}Single cell detected — analyzing directly."

    logger.info(f"Segmented {n} cells from {w}×{h} image")
    return result
