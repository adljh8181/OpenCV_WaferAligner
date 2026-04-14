"""
================================================================================
LINE-2D SHAPE-BASED MATCHER
================================================================================
Python port of the SBM LINE-2D algorithm (shape-based matching using
quantized gradient orientations).

Reference: SeanYan604/SBM_line2D (C++ implementation)

Core Algorithm:
  1. Sobel gradients → angle → quantize into 8 orientation bins
  2. Spread quantized orientations over T×T cells (tolerance)
  3. Compute response maps: for each label, lookup table [0..255] → score
  4. Linearize response maps into T×T grids for fast access
  5. Score = sum of response values at sparse template feature locations
  Images are always processed at their original full resolution.

Usage:
    from linemod_matcher import LinemodMatcher, LinemodConfig

    config = LinemodConfig()
    matcher = LinemodMatcher(config)
    matcher.load_template(template_img)
    matcher.generate_templates()
    results = matcher.match(search_img, return_all=True)

Author: Wafer Alignment System (ported from SBM_line2D C++)
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# GPU / CUDA detection  (one-shot at import time)
# ======================================================================
try:
    _CUDA_DEVICE_COUNT = cv2.cuda.getCudaEnabledDeviceCount()
except Exception:
    _CUDA_DEVICE_COUNT = 0
_CUDA_AVAILABLE = _CUDA_DEVICE_COUNT > 0

if _CUDA_AVAILABLE:
    print(f"[LINE-2D] CUDA enabled — {_CUDA_DEVICE_COUNT} device(s) detected.")
else:
    print("[LINE-2D] CUDA not available — using optimised CPU path.")


# ======================================================================
# Configuration
# ======================================================================
class LinemodConfig:
    """Configuration for the LINE-2D matcher."""

    def __init__(self):
        # --- Template generation ---
        self.ANGLE_STEP = 5           # Degrees between templates
        self.SCALE_MIN = 1.0
        self.SCALE_MAX = 1.0
        self.SCALE_STEP = 0.1

        # --- Gradient / Feature ---
        # Negative value = percentile-based adaptive threshold (recommended).
        # e.g. -70 = "use 70th percentile of image gradients as cutoff".
        # This makes detection resolution-independent (works for 919px and 5120px alike).
        # Use a positive absolute value only if you want a fixed threshold.
        self.WEAK_THRESHOLD = -70.0   # 70th-percentile adaptive (default)
        self.NUM_FEATURES = 128       # Features per template
        # Hysteresis kernel for majority-vote edge thickening.
        # 0 = auto-detect from the image being processed (use for uniform
        #     resolution pipelines). Set explicitly (e.g. 9) when template
        #     is a small crop of a large sensor image so both template and
        #     search use the same kernel size.
        self.HYSTERESIS_KERNEL = 0    # 0=auto, or odd int 3..13

        # --- Pyramid ---
        self.T_PYRAMID = [4, 8, 16]       # Spreading T per level

        # --- Matching ---
        self.MATCH_THRESHOLD = 50.0   # Similarity 0–100
        self.NMS_DISTANCE = 30

        # --- Performance / GPU ---
        self.USE_GPU = _CUDA_AVAILABLE          # Auto-detect, set False to force CPU
        self.FAST_SEARCH_QUANTIZE = True        # Skip hysteresis on search images
        self.COARSE_NUM_FEATURES  = 128          # Fewer features for coarse scanning
        self.MAX_COARSE_CANDIDATES = 2           # Number of top candidates to keep per template at coarse level

    @property
    def PYRAMID_LEVELS(self):
        return len(self.T_PYRAMID)


# ======================================================================
# Structures
# ======================================================================
class Feature:
    """A template feature: position + quantized label (0-7)."""
    __slots__ = ['x', 'y', 'label']
    def __init__(self, x=0, y=0, label=0):
        self.x = x; self.y = y; self.label = label


class TemplatePyr:
    """Features for one pyramid level."""
    __slots__ = ['width', 'height', 'tl_x', 'tl_y', 'pyramid_level', 'features']
    def __init__(self):
        self.width = 0; self.height = 0
        self.tl_x = 0; self.tl_y = 0
        self.pyramid_level = 0; self.features = []


# ======================================================================
# Core Algorithm  (mirrors SBM line2Dup.cpp)
# ======================================================================

def _quantize_gradients(src, weak_threshold=30.0, fast_mode=False, kernel_size=0,
                        use_gpu=False):
    """
    Compute gradient and quantize angle into 8 orientation bins.
    Each pixel gets a label 0-7 encoded as a power of 2 (1,2,...,128) or 0.

    Args:
        src: Input image (grayscale or BGR).
        weak_threshold: Minimum gradient magnitude.
        fast_mode: If True, skip hysteresis for speed.
        kernel_size: Hysteresis majority-vote kernel (odd int, e.g. 3/5/7/9).
                     0 = auto-compute from image resolution. Explicit values
                     ensure the template crop and search image use the same
                     kernel even when they differ in spatial size.
        use_gpu: If True AND CUDA is available, offload Blur/Sobel to GPU.

    Mirrors: hysteresisGradient() in line2Dup.cpp
    """
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = src.astype(np.float32)

    h, w = gray.shape
    max_dim = max(h, w)

    # ---- GPU path for Blur + Sobel + magnitude --------------------------
    _did_gpu = False
    if use_gpu and _CUDA_AVAILABLE:
        try:
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)

            # Gaussian blur
            if not fast_mode:
                if kernel_size > 0 and kernel_size >= 3:
                    ks_blur = max(3, kernel_size | 1)  # enforce odd
                    blur_filter = cv2.cuda.createGaussianFilter(
                        cv2.CV_32F, cv2.CV_32F,
                        (ks_blur, ks_blur), 0)
                    gpu_gray = blur_filter.apply(gpu_gray)
                elif max_dim > 1500:
                    sigma = max(1.0, (max_dim - 1000) / 1500.0)
                    ksize = int(sigma * 2) * 2 + 1
                    blur_filter = cv2.cuda.createGaussianFilter(
                        cv2.CV_32F, cv2.CV_32F,
                        (ksize, ksize), sigma)
                    gpu_gray = blur_filter.apply(gpu_gray)

            # Sobel dx, dy
            sobel_x = cv2.cuda.createSobelFilter(
                cv2.CV_32F, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.cuda.createSobelFilter(
                cv2.CV_32F, cv2.CV_32F, 0, 1, ksize=3)
            gpu_dx = sobel_x.apply(gpu_gray)
            gpu_dy = sobel_y.apply(gpu_gray)

            # Magnitude and Phase on GPU
            gpu_mag = cv2.cuda.magnitude(gpu_dx, gpu_dy, cv2.cuda_GpuMat())
            gpu_angle = cv2.cuda.phase(gpu_dx, gpu_dy, cv2.cuda_GpuMat(), angleInDegrees=True)

            # Download results (skip dx, dy, gray as they are unused, saves large PCIe transfer latency)
            magnitude = gpu_mag.download()
            angle = gpu_angle.download() % 360.0
            _did_gpu = True
        except Exception as e:
            print(f"[LINE-2D CUDA ERROR] Fallback to CPU: {e}")
            # Silently fall back to CPU
            _did_gpu = False

    # ---- CPU path for Blur + Sobel + magnitude --------------------------
    if not _did_gpu:
        if not fast_mode:
            if kernel_size > 0:
                ks_blur = max(3, kernel_size | 1)  # enforce odd, min 3
                gray = cv2.GaussianBlur(gray, (ks_blur, ks_blur), 0)
            elif max_dim > 1500:
                sigma = max(1.0, (max_dim - 1000) / 1500.0)
                ksize = int(sigma * 2) * 2 + 1
                gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        angle = cv2.phase(dx, dy, angleInDegrees=True) % 360.0

    # Quantize to 16 buckets then fold to 8 (& 7)
    quant_16 = (angle * (16.0 / 360.0)).astype(np.uint8)
    quant_8 = quant_16 & 7

    # Zero borders
    quant_8[0, :] = 0; quant_8[-1, :] = 0
    quant_8[:, 0] = 0; quant_8[:, -1] = 0

    h, w = gray.shape

    # --- Adaptive / relative weak threshold ---
    if weak_threshold < 0:
        nonzero_mags = magnitude[magnitude > 0]
        if len(nonzero_mags) > 0:
            pct = min(99.0, max(0.0, -weak_threshold))
            weak_threshold = float(np.percentile(nonzero_mags, pct))
        else:
            weak_threshold = 1.0

    if fast_mode:
        # Fast path: skip hysteresis, just use magnitude threshold
        accept = magnitude > weak_threshold
        quantized = np.zeros((h, w), dtype=np.uint8)
        quantized[accept] = (1 << quant_8[accept]).astype(np.uint8)
    else:
        # Hysteresis: adaptive majority-vote + magnitude threshold
        if kernel_size <= 0:
            raw_ks = max(3, int(3 * max_dim / 1500))
            ks = min(raw_ks | 1, 9)
        else:
            ks = max(3, kernel_size | 1)
        min_votes = (ks * ks) // 2 + 1
        kernel = np.ones((ks, ks), dtype=np.uint8)

        best_label = np.zeros((h, w), dtype=np.uint8)
        best_count = np.zeros((h, w), dtype=np.uint16)

        for i in range(8):
            plane = (quant_8 == i).astype(np.uint8)
            votes = cv2.filter2D(plane, cv2.CV_16U, kernel,
                                 borderType=cv2.BORDER_CONSTANT)
            better = votes > best_count
            best_count[better] = votes[better]
            best_label[better] = i

        accept = (magnitude > weak_threshold) & (best_count >= min_votes)

        quantized = np.zeros((h, w), dtype=np.uint8)
        quantized[accept] = (1 << best_label[accept]).astype(np.uint8)

    # Zero borders
    quantized[0, :] = 0; quantized[-1, :] = 0
    quantized[:, 0] = 0; quantized[:, -1] = 0

    return quantized, magnitude


def _spread(quantized, T, use_gpu=False):
    """
    OR-spread quantized orientations over a (2T-1)×(2T-1) neighbourhood.

    Equivalent to the original loop-based spread but implemented via
    cv2.dilate on each of the 8 bit-planes. This is 3-8× faster natively on CPU
    because cv2.dilate implements highly optimized vector SIMD instructions.
    GPU acceleration is DISABLED here because PCIe transfer overhead of 8 bit-planes 
    is natively slower than the vectorized CPU.
    """
    if T <= 1:
        return quantized.copy()

    ks = 2 * T - 1
    kernel = np.ones((ks, ks), dtype=np.uint8)

    # ---- CPU path (Optimized natively in openCV over PCIe overheads) ----
    result = np.zeros_like(quantized)
    for bit in range(8):
        plane = ((quantized >> bit) & 1).astype(np.uint8)
        if not plane.any():
            continue
        dilated = cv2.dilate(plane, kernel)
        result |= (dilated.astype(np.uint8) << bit)
    return result


# ---- Response map LUT (mirrors computeResponseMaps in C++) ----
# For each of the 8 template labels, build a 256-entry lookup table:
#   LUT[spread_byte] = max similarity score for that label vs the spread byte.
# Similarity between label `ori` and `spread_val`:
#   If the spread_val contains the same bit as ori → score = 4
#   If it contains a neighbour bit (±1) → score = 1
#   Else → score = 0
#   (The C++ uses pre-built LUTs; we compute them once.)

def _build_response_luts():
    """Build 8 LUTs of size 256. LUT[label][spread_byte] = response score."""
    luts = np.zeros((8, 256), dtype=np.uint8)
    for label in range(8):
        bit = 1 << label
        # Neighbours wrap around (0↔7)
        left_bit = 1 << ((label - 1) % 8)
        right_bit = 1 << ((label + 1) % 8)
        for val in range(256):
            if val & bit:
                luts[label, val] = 4   # Exact match
            elif val & (left_bit | right_bit):
                luts[label, val] = 1   # Neighbour match
            # else 0
    return luts

_RESPONSE_LUTS = _build_response_luts()


def _compute_response_maps(spread_img):
    """
    Compute 8 response maps using the LUT approach.
    response_map[label][r,c] = LUT[label][ spread_img[r,c] ]

    Mirrors: computeResponseMaps() in line2Dup.cpp
    """
    response_maps = []
    for label in range(8):
        rmap = cv2.LUT(spread_img, _RESPONSE_LUTS[label])
        response_maps.append(rmap)
    return response_maps


def _extract_scattered_features(quantized, magnitude, num_features, mask=None):
    """
    Select N spatially scattered feature points from the quantized image.
    Sorted by gradient magnitude, with minimum distance constraint.

    Uses a spatial grid hash so the distance check per candidate is O(9 cells)
    instead of O(n_selected), giving ~5-10x speedup over the naive approach.

    Mirrors: selectScatteredFeatures() in line2Dup.cpp
    """
    if mask is not None:
        valid = (quantized > 0) & (mask > 0)
    else:
        valid = quantized > 0

    ys, xs = np.where(valid)
    if len(xs) < 2:
        return []

    mags = magnitude[ys, xs]
    quants = quantized[ys, xs]
    labels = np.log2(quants.astype(np.float32)).astype(np.int32)

    order = np.argsort(-mags)

    span_x = int(np.max(xs) - np.min(xs))
    span_y = int(np.max(ys) - np.min(ys))
    perimeter_approx = 2.0 * (span_x + span_y)
    distance = max(5.0, perimeter_approx / (num_features * 0.8))

    features = []
    used = np.zeros(len(order), dtype=bool)

    while distance >= 1.0 and len(features) < num_features:
        dist_sq = distance * distance
        cell = max(1, int(distance))

        # Rebuild spatial grid from already-accepted features for this cell size
        grid = {}  # (grid_y, grid_x) -> list of (x, y)
        for feat in features:
            gx, gy = feat.x // cell, feat.y // cell
            key = (gy, gx)
            if key not in grid:
                grid[key] = []
            grid[key].append((feat.x, feat.y))

        for i, idx in enumerate(order):
            if used[i]:
                continue

            x, y, label = int(xs[idx]), int(ys[idx]), int(labels[idx])
            gx, gy = x // cell, y // cell

            # Check only the 3×3 neighbourhood of grid cells (O(9×k) vs O(n_selected))
            close = False
            for dgy in range(-1, 2):
                ngy = gy + dgy
                for dgx in range(-1, 2):
                    bucket = grid.get((ngy, gx + dgx))
                    if bucket:
                        for fx, fy in bucket:
                            if (x - fx) * (x - fx) + (y - fy) * (y - fy) < dist_sq:
                                close = True
                                break
                    if close:
                        break
                if close:
                    break

            if not close:
                key = (gy, gx)
                if key not in grid:
                    grid[key] = []
                grid[key].append((x, y))
                features.append(Feature(x, y, label))
                used[i] = True
                if len(features) >= num_features:
                    break

        step = max(1.0, distance * 0.15)
        distance -= step

    return features


def _crop_templates(templates):
    """
    Compute overall bounding box and make feature positions relative to it.
    Mirrors: cropTemplates() in line2Dup.cpp
    """
    if not templates:
        return

    min_x = min_y = 999999
    max_x = max_y = -999999

    for t in templates:
        for f in t.features:
            px = f.x << t.pyramid_level
            py = f.y << t.pyramid_level
            min_x = min(min_x, px); min_y = min(min_y, py)
            max_x = max(max_x, px); max_y = max(max_y, py)

    if min_x % 2 == 1: min_x -= 1
    if min_y % 2 == 1: min_y -= 1

    for t in templates:
        t.tl_x = min_x >> t.pyramid_level
        t.tl_y = min_y >> t.pyramid_level
        t.width = (max_x - min_x) >> t.pyramid_level
        t.height = (max_y - min_y) >> t.pyramid_level
        for f in t.features:
            f.x -= t.tl_x
            f.y -= t.tl_y


def _subpixel_refine(score_map, r, c):
    """
    Parabolic (3-point quadratic) sub-pixel interpolation.

    Given the integer-peak position (r, c) in *score_map*, fits a 1-D
    parabola independently along each axis through the three neighbouring
    score values and returns the sub-pixel fractional offset (dx, dy).

    Formula (standard Harris / SIFT derivation):
        delta = (s_left - s_right) / (2 * (s_left - 2*s_centre + s_right))

    Both offsets are clamped to [-0.5, +0.5] so the refined position
    never moves to a different integer grid cell.
    Cost: 5 scalar lookups + 4 arithmetic ops — negligible vs match time.
    """
    h, w = score_map.shape

    if 0 < c < w - 1:
        sl = float(score_map[r, c - 1])
        sc = float(score_map[r, c])
        sr = float(score_map[r, c + 1])
        denom_x = sl - 2.0 * sc + sr
        dx = 0.5 * (sl - sr) / denom_x if abs(denom_x) > 1e-6 else 0.0
    else:
        dx = 0.0

    if 0 < r < h - 1:
        su = float(score_map[r - 1, c])
        sc_y = float(score_map[r, c])
        sd = float(score_map[r + 1, c])
        denom_y = su - 2.0 * sc_y + sd
        dy = 0.5 * (su - sd) / denom_y if abs(denom_y) > 1e-6 else 0.0
    else:
        dy = 0.0

    dx = max(-0.5, min(0.5, dx))
    dy = max(-0.5, min(0.5, dy))
    return dx, dy


# ======================================================================
# High-Level Matcher Class
# ======================================================================
class LinemodMatcher:
    """Shape-based template matcher — Python port of SBM LINE-2D."""

    def __init__(self, config=None):
        self.config = config or LinemodConfig()
        self.template_image = None
        self.detection_mask = None    # User-drawn ROI mask (same size as template, 255=detect)
        self.template_pyramids = []   # List of dicts: angle, scale, templates, image, size

    def load_template(self, template_img, detection_mask=None):
        """Load the base template image (grayscale) and optional detection mask.

        Args:
            template_img: Template image (grayscale or BGR).
            detection_mask: Optional binary mask (same size as template,
                           255=detect region, 0=ignore). When provided,
                           features will only be extracted from masked area.
        """
        if len(template_img.shape) == 3:
            self.template_image = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            self.template_image = template_img.copy()
        self.detection_mask = detection_mask.copy() if detection_mask is not None else None

    def generate_templates(self):
        """Generate rotated+scaled templates (mirrors shapeInfo.produce_infos).

        Parallelised with ThreadPoolExecutor across the angle × scale set.
        OpenCV/NumPy release the GIL so threads run truly in parallel on
        multi-core CPUs.

        GPU (CUDA) is intentionally disabled inside the worker threads because
        template crops are small — PCIe upload/download overhead exceeds the
        GPU compute saving for images < ~500 px.  GPU is still used in match().
        """
        import os as _os
        from concurrent.futures import ThreadPoolExecutor

        if self.template_image is None:
            raise ValueError("Call load_template() first.")

        cfg = self.config
        self.template_pyramids = []

        angles = list(np.arange(0, 360, cfg.ANGLE_STEP)) if cfg.ANGLE_STEP < 360 else [0]
        if cfg.SCALE_MIN == cfg.SCALE_MAX:
            scales = [cfg.SCALE_MIN]
        else:
            scales = list(np.arange(cfg.SCALE_MIN, cfg.SCALE_MAX + cfg.SCALE_STEP / 2, cfg.SCALE_STEP))

        total = len(angles) * len(scales)

        # Snapshot read-only state once so worker threads don't race on self.*
        _tmpl_img   = self.template_image
        _det_mask   = self.detection_mask
        _cfg        = cfg

        def _process_one(angle, scale):
            src = LinemodMatcher._transform(_tmpl_img, angle, scale)
            rotation_mask = LinemodMatcher._transform(
                np.ones_like(_tmpl_img) * 255, angle, scale)
            rotation_mask = (rotation_mask > 128).astype(np.uint8) * 255

            if _det_mask is not None:
                user_mask_xformed = LinemodMatcher._transform(_det_mask, angle, scale)
                user_mask_xformed = (user_mask_xformed > 128).astype(np.uint8) * 255
                mask_img = cv2.bitwise_and(rotation_mask, user_mask_xformed)
            else:
                mask_img = rotation_mask

            pyr_src  = src.copy()
            pyr_mask = mask_img.copy()
            templates = []

            for level in range(_cfg.PYRAMID_LEVELS):
                if level > 0:
                    pyr_src  = cv2.pyrDown(pyr_src)
                    pyr_mask = cv2.pyrDown(pyr_mask)
                    pyr_mask = (pyr_mask > 128).astype(np.uint8) * 255

                # use_gpu=False: template crops are small; GPU PCIe overhead > CPU gain.
                # GPU is still used during match() on the full search image.
                quantized, mag = _quantize_gradients(
                    pyr_src, _cfg.WEAK_THRESHOLD,
                    kernel_size=_cfg.HYSTERESIS_KERNEL,
                    use_gpu=False)
                features = _extract_scattered_features(
                    quantized, mag, _cfg.NUM_FEATURES, pyr_mask)

                if len(features) < 4:
                    return None

                t = TemplatePyr()
                t.pyramid_level = level
                t.features = features
                templates.append(t)

            _crop_templates(templates)
            return {
                'angle': angle, 'scale': scale,
                'templates': templates,
                # 'image' is NOT stored here to avoid keeping 72 rotated numpy
                # arrays in RAM permanently (~200 MB+ for large templates).
                # visualize_templates() regenerates it on demand.
                'size': (src.shape[1], src.shape[0]),
            }

        combos = [(angle, scale) for scale in scales for angle in angles]
        # Cap workers: no benefit beyond cpu_count; keep at least 1
        max_workers = max(1, min(len(combos), (_os.cpu_count() or 4), 8))

        results = []
        if max_workers > 1 and len(combos) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_one, a, s) for a, s in combos]
                for fut in futures:
                    r = fut.result()
                    if r is not None:
                        results.append(r)
        else:
            for angle, scale in combos:
                r = _process_one(angle, scale)
                if r is not None:
                    results.append(r)

        # Preserve deterministic ordering (scale asc, angle asc)
        results.sort(key=lambda r: (r['scale'], r['angle']))
        self.template_pyramids = results
        added = len(self.template_pyramids)

        print(f"[LINE-2D] Generated {added}/{total} templates "
              f"({len(angles)} angles × {len(scales)} scales, "
              f"{max_workers} worker{'s' if max_workers > 1 else ''})")

    def match(self, search_img, threshold=None, return_all=False, search_roi=None):
        """Find template matches in the search image.

        Automatically routes large images (> 2000px) through
        _match_pyramid() (coarse half-res scan → fine ROI) for a 5-10×
        speedup over the naive full-resolution scan.

        Args:
            search_img: Grayscale or BGR search image.
            threshold: Minimum similarity score (0-100). Default: config value.
            return_all: If True, return list of all matches. Otherwise best match.
            search_roi: Optional (x, y, w, h) tuple to restrict search area.

        Side-effect:
            Sets self._last_timing dict with per-phase millisecond breakdowns
            so the ViewModel can display a timing chart without changing the
            public return-type signature.
        """
        import time as _time
        if not self.template_pyramids:
            raise ValueError("Call generate_templates() first.")

        cfg = self.config
        threshold = threshold or cfg.MATCH_THRESHOLD

        t0_prep = _time.perf_counter()
        if len(search_img.shape) == 3:
            search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_img.copy()

        # Apply ROI if specified
        roi_offset = (0, 0)
        if search_roi is not None:
            rx, ry, rw, rh = search_roi
            search_gray = search_gray[ry:ry+rh, rx:rx+rw].copy()
            roi_offset = (rx, ry)
        t_prep = (_time.perf_counter() - t0_prep) * 1000

        # Auto pyramid: for high-res images use coarse-to-fine search.
        # Requires PYRAMID_LEVELS >= 2 (templates must have a level-1 half-res
        # feature set). This is always true because generate_templates() builds
        # level-0 AND level-1 templates when PYRAMID_LEVELS == 2.
        sh, sw = search_gray.shape[:2]
        use_pyramid = (
            max(sh, sw) > 2000
            and cfg.PYRAMID_LEVELS >= 2
            and all(len(tp['templates']) >= 2 for tp in self.template_pyramids)
        )

        if use_pyramid:
            matches, inner_timing = self._match_pyramid(search_gray, threshold)
            # Robust fallback: if pyramid found nothing (coarse was off or ROI
            # missed), retry with the reliable full-image single-level scan.
            if not matches:
                print("  [Pyramid] 0 matches — falling back to full-image scan")
                matches, inner_timing = self._match_single_level(search_gray, threshold)
                inner_timing['mode'] = 'Single-Level (fallback)'
            else:
                inner_timing['mode'] = 'Pyramid'
        else:
            matches, inner_timing = self._match_single_level(search_gray, threshold)
            inner_timing['mode'] = 'Single-Level'

        # Offset positions back to original image coordinates
        t0_nms = _time.perf_counter()
        if roi_offset != (0, 0):
            ox, oy = roi_offset
            for m in matches:
                m['x'] += ox
                m['y'] += oy
                bx, by, bw, bh = m['bbox']
                m['bbox'] = (bx + ox, by + oy, bw, bh)

        matches = self._nms(matches)
        matches.sort(key=lambda m: m['score'], reverse=True)
        t_nms = (_time.perf_counter() - t0_nms) * 1000

        # Persist timing for ViewModel to harvest
        self._last_timing = {
            **inner_timing,
            'prep_ms': t_prep,
            'nms_ms':  t_nms,
        }

        if return_all:
            return matches
        return matches[0] if matches else None

    def _match_single_level(self, search_gray, threshold):
        """
        Single-level scoring via vectorized response map slicing.
        Returns (matches, timing_dict).
        """
        import time as _time
        cfg = self.config
        all_matches = []
        timing = {}

        T = cfg.T_PYRAMID[0]
        # Auto-scale spread for high-res images: thin edges need wider spread
        sh, sw = search_gray.shape[:2]
        max_dim = max(sh, sw)
        if max_dim > 1500:
            # Scale T proportionally but cap at 8 to preserve scoring accuracy.
            # T>8 creates spread windows so wide that response maps lose
            # discriminating power (wrong positions score as high as correct).
            T = min(max(T, int(T * max_dim / 1500)), 8)
            print(f"  [Auto] Spread T scaled to {T} for {sw}×{sh} image")

        t0 = _time.perf_counter()
        quantized, _ = _quantize_gradients(
            search_gray, cfg.WEAK_THRESHOLD,
            fast_mode=cfg.FAST_SEARCH_QUANTIZE,
            kernel_size=cfg.HYSTERESIS_KERNEL,
            use_gpu=cfg.USE_GPU)
        timing['quantize_ms'] = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        spread_q = _spread(quantized, T, use_gpu=cfg.USE_GPU)
        timing['spread_ms'] = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        rmaps = _compute_response_maps(spread_q)
        # Pre-cast response maps to int32 once to avoid massive memory allocations in the inner loop
        rmaps_int32 = [r.astype(np.int32) for r in rmaps]
        timing['response_maps_ms'] = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        sh, sw = search_gray.shape[:2]

        for tp_idx, tp in enumerate(self.template_pyramids):
            templ = tp['templates'][0]  # Level 0 template
            n_feats = len(templ.features)
            if n_feats == 0:
                continue

            tw = templ.width
            th = templ.height

            # Valid scan area
            vy = sh - th
            vx = sw - tw
            if vy <= 0 or vx <= 0:
                continue

            # Vectorized scoring: sum response slices
            score_map = np.zeros((vy, vx), dtype=np.int32)
            valid_feats = 0

            for feat in templ.features:
                fx, fy = feat.x, feat.y
                y_end = fy + vy
                x_end = fx + vx
                if y_end <= sh and x_end <= sw and fy >= 0 and fx >= 0:
                    score_map += rmaps_int32[feat.label][fy:y_end, fx:x_end]
                    valid_feats += 1

            if valid_feats == 0:
                continue

            # Normalize: max score per feature = 4
            max_possible = 4 * valid_feats
            sim_map = (score_map * 100.0) / max_possible

            # Find candidates with NMS
            ys, xs = np.where(sim_map > threshold)
            if len(xs) == 0:
                continue

            scores = sim_map[ys, xs]
            order = np.argsort(-scores)

            nms_dist_sq = cfg.NMS_DISTANCE ** 2
            kept = []
            img_w, img_h = tp['size']

            for idx in order:
                r_i, c_i = int(ys[idx]), int(xs[idx])
                dx, dy = _subpixel_refine(sim_map, r_i, c_i)
                cx = c_i + dx - templ.tl_x + img_w // 2
                cy = r_i + dy - templ.tl_y + img_h // 2
                s = float(scores[idx])

                dup = False
                for k in kept:
                    if (cx - k['x'])**2 + (cy - k['y'])**2 < nms_dist_sq:
                        dup = True; break
                if not dup:
                    icx, icy = int(round(cx)), int(round(cy))
                    kept.append({
                        'x': cx, 'y': cy,
                        'angle': tp['angle'], 'scale': tp['scale'],
                        'score': s, 'template_id': tp_idx,
                        'bbox': (icx - img_w//2, icy - img_h//2, img_w, img_h),
                    })
                    if len(kept) >= 20:
                        break

            all_matches.extend(kept)

        timing['scoring_ms'] = (_time.perf_counter() - t0) * 1000
        return all_matches, timing

    def _match_pyramid(self, search_gray, threshold):
        """
        Coarse-to-fine pyramid matching for large images.

        Level 1 (coarse): Downsample search image by 2×, fast scan with
            level-1 features and lower threshold to find candidate regions.
        Level 0 (fine):   Search only within ROI windows around each coarse
            candidate at full resolution for precise positioning.

        Returns (matches, timing_dict).
        """
        import time
        import math
        cfg = self.config
        sh, sw = search_gray.shape[:2]
        timing = {}

        # Determine optimal coarse level (1/2, 1/4, 1/8 size)
        max_dim = max(sh, sw)
        ideal_level = int(max(1, round(math.log2(max_dim / 1200.0))))
        coarse_level = min(cfg.PYRAMID_LEVELS - 1, ideal_level)
        coarse_level = max(1, coarse_level) # Ensure at least 1

        if getattr(cfg, "FORCE_COARSE_LEVEL", None) is not None:
            coarse_level = min(cfg.PYRAMID_LEVELS - 1, max(0, cfg.FORCE_COARSE_LEVEL))

        # ---- Level N: Coarse search at reduced resolution ----
        t0_coarse_total = time.perf_counter()

        t0 = time.perf_counter()
        search_coarse = search_gray
        for _ in range(coarse_level):
            search_coarse = cv2.pyrDown(search_coarse)
        sh_c, sw_c = search_coarse.shape[:2]
        timing['downsample_ms'] = (time.perf_counter() - t0) * 1000

        T_c = cfg.T_PYRAMID[coarse_level] if len(cfg.T_PYRAMID) > coarse_level else cfg.T_PYRAMID[-1]

        t0 = time.perf_counter()
        q_c, _ = _quantize_gradients(search_coarse, cfg.WEAK_THRESHOLD,
                                    fast_mode=cfg.FAST_SEARCH_QUANTIZE,
                                    kernel_size=cfg.HYSTERESIS_KERNEL,
                                    use_gpu=False) # Force CPU for Coarse downsampled image (GPU PCIe is 4x slower here)
        timing['coarse_quantize_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        s_c = _spread(q_c, T_c, use_gpu=cfg.USE_GPU)
        timing['coarse_spread_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rmaps_c = _compute_response_maps(s_c)
        rmaps_c_int32 = [r.astype(np.int32) for r in rmaps_c]
        timing['coarse_response_maps_ms'] = (time.perf_counter() - t0) * 1000

        # Lower threshold for coarse pass — be generous to not miss candidates.
        coarse_threshold = max(threshold * 0.3, 15.0)

        coarse_candidates = []  # (tp_idx, cx_full, cy_full, coarse_score)

        t0 = time.perf_counter()
        for tp_idx, tp in enumerate(self.template_pyramids):
            if len(tp['templates']) <= coarse_level:
                continue

            templ_c = tp['templates'][coarse_level]
            n_feats = len(templ_c.features)
            if n_feats == 0:
                continue

            tw_c = templ_c.width
            th_c = templ_c.height
            vy_c = sh_c - th_c
            vx_c = sw_c - tw_c
            if vy_c <= 0 or vx_c <= 0:
                continue

            # Score at coarse level using 48 evenly-spaced features.
            # Coarse only needs to find approximate position (±4px at L2);
            # the fine pass uses ALL features for precise scoring.
            score_map = np.zeros((vy_c, vx_c), dtype=np.int32)
            valid = 0

            n_feats_c = len(templ_c.features)
            max_coarse_feats = cfg.COARSE_NUM_FEATURES
            if n_feats_c > max_coarse_feats:
                step = max(1, n_feats_c // max_coarse_feats)
                coarse_feats = templ_c.features[::step][:max_coarse_feats]
            else:
                coarse_feats = templ_c.features

            for feat in coarse_feats:
                fx, fy = feat.x, feat.y
                if fy + vy_c <= sh_c and fx + vx_c <= sw_c and fy >= 0 and fx >= 0:
                    score_map += rmaps_c_int32[feat.label][fy:fy+vy_c, fx:fx+vx_c]
                    valid += 1

            if valid == 0:
                continue

            sim_map = (score_map * 100.0) / (4 * valid)
            ys, xs = np.where(sim_map > coarse_threshold)
            if len(xs) == 0:
                continue

            scores = sim_map[ys, xs]
            order = np.argsort(-scores)

            # Keep top candidates at coarse level (NMS with loose distance)
            nms_sq = (cfg.NMS_DISTANCE // 2) ** 2
            kept_coarse = []
            img_w, img_h = tp['size']

            scale_factor = 2 ** coarse_level

            for idx in order:
                # Convert coarse position → full-resolution coordinates
                cx_c = int(xs[idx]) - templ_c.tl_x + img_w // (2 * scale_factor)
                cy_c = int(ys[idx]) - templ_c.tl_y + img_h // (2 * scale_factor)
                cx_full = cx_c * scale_factor
                cy_full = cy_c * scale_factor

                dup = False
                for kc in kept_coarse:
                    if (cx_full - kc[1])**2 + (cy_full - kc[2])**2 < nms_sq * 4:
                        dup = True; break
                if not dup:
                    kept_coarse.append((tp_idx, cx_full, cy_full, float(scores[idx])))
                    if len(kept_coarse) >= 1:   # best candidate only per template
                        break

            coarse_candidates.extend(kept_coarse)

            # Early exit only for single-template (Simple/Fast) mode.
            # Multi-template modes (With Rotation / Full Search) MUST evaluate
            # all templates so the correct angle AND scale can win — stopping
            # early on score > 80 causes the wrong scale/angle to be selected.
            if len(kept_coarse) > 0 and len(self.template_pyramids) == 1:
                peak_score = max(c[3] for c in kept_coarse)
                if peak_score > threshold * 0.7:
                    break

        timing['coarse_scoring_ms'] = (time.perf_counter() - t0) * 1000

        # Limit fine-search work: keep only the top-N coarse candidates by
        # score so we don't run the full fine pass on hundreds of duplicates.
        # 30 is generous — clusters at the same position are merged so the
        # actual number of fine ROI evaluations is much smaller.
        if len(coarse_candidates) > 30:
            coarse_candidates.sort(key=lambda c: c[3], reverse=True)
            coarse_candidates = coarse_candidates[:30]

        t_coarse = (time.perf_counter() - t0_coarse_total) * 1000
        print(f"  [Pyramid] Coarse (L{coarse_level} {sw_c}×{sh_c}): {len(coarse_candidates)} candidates in {t_coarse:.0f}ms")

        if not coarse_candidates:
            return [], timing

        # ---- Level 0: Fine search in ROI windows ----
        t0_fine_total = time.perf_counter()
        T0 = cfg.T_PYRAMID[0]
        # Auto-scale spread for fine search, capped at 8 to preserve
        # scoring accuracy on the small fine ROI (~500px)
        if max_dim > 1500:
            T0 = min(max(T0, int(T0 * max_dim / 1500)), 8)

        # Margin scaling based on maximum expected coarse spatial drift 
        # (e.g., 8-12 pixels of drift in coarse space scaled upward)
        search_margin = 12 * scale_factor 

        all_matches = []
        t_fine_quantize = t_fine_spread = t_fine_rmap = t_fine_score = 0.0

        # Collect and cluster candidates by geometrical overlap
        clusters = []
        for tp_idx, cx_full, cy_full, coarse_score in coarse_candidates:
            tp = self.template_pyramids[tp_idx]
            templ0 = tp['templates'][0]
            if len(templ0.features) == 0:
                continue

            img_w, img_h = tp['size']
            tw0, th0 = templ0.width, templ0.height

            tl_feat_x = cx_full - img_w // 2 + templ0.tl_x
            tl_feat_y = cy_full - img_h // 2 + templ0.tl_y

            rx  = max(0,  tl_feat_x - search_margin)
            ry  = max(0,  tl_feat_y - search_margin)
            rx2 = min(sw, tl_feat_x + tw0 + search_margin + T0 + 4)
            ry2 = min(sh, tl_feat_y + th0 + search_margin + T0 + 4)

            if rx2 - rx < tw0 + 10 or ry2 - ry < th0 + 10:
                continue

            cand = {
                'tp_idx': tp_idx, 'cx_full': cx_full, 'cy_full': cy_full,
                'coarse_score': coarse_score, 'tp': tp, 'templ0': templ0,
                'rx': rx, 'ry': ry, 'rx2': rx2, 'ry2': ry2
            }

            merged = False
            for cluster in clusters:
                crx, cry, crx2, cry2 = cluster['bounds']
                # Check intersection
                if not (rx2 < crx or rx > crx2 or ry2 < cry or ry > cry2):
                    # Merge bounds
                    cluster['bounds'] = [min(rx, crx), min(ry, cry), max(rx2, crx2), max(ry2, cry2)]
                    cluster['candidates'].append(cand)
                    merged = True
                    break
            
            if not merged:
                clusters.append({'bounds': [rx, ry, rx2, ry2], 'candidates': [cand]})

        # Evaluate each super-ROI cluster exactly once
        for cluster in clusters:
            R_X, R_Y, R_X2, R_Y2 = cluster['bounds']
            super_roi = search_gray[R_Y:R_Y2, R_X:R_X2]
            
            _t = time.perf_counter()
            # kernel_size is missing in fine quantize because fast_mode=True means hysteresis is skipped, so kernel_size is irrelevant
            q_super, _ = _quantize_gradients(super_roi, cfg.WEAK_THRESHOLD,
                                           fast_mode=True, # Always skip blur/hysteresis for fine search speed
                                           use_gpu=False)
            t_fine_quantize += (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            s_super = _spread(q_super, T0, use_gpu=False)
            t_fine_spread += (time.perf_counter() - _t) * 1000

            _t = time.perf_counter()
            rmaps_super = _compute_response_maps(s_super)
            rmaps_super_int32 = [r.astype(np.int32) for r in rmaps_super]
            t_fine_rmap += (time.perf_counter() - _t) * 1000

            for cand in cluster['candidates']:
                rx, ry, rx2, ry2 = cand['rx'], cand['ry'], cand['rx2'], cand['ry2']
                templ0, tp, tp_idx = cand['templ0'], cand['tp'], cand['tp_idx']
                img_w, img_h = tp['size']
                tw0, th0 = templ0.width, templ0.height

                # Extract proper subset views for this candidate
                lx, ly = rx - R_X, ry - R_Y
                lx2, ly2 = rx2 - R_X, ry2 - R_Y
                
                _t = time.perf_counter()
                rh, rw = ly2 - ly, lx2 - lx

                vy, vx = rh - th0, rw - tw0
                if vy <= 0 or vx <= 0:
                    t_fine_score += (time.perf_counter() - _t) * 1000
                    continue

                score_map = np.zeros((vy, vx), dtype=np.int32)
                valid = 0
                for feat in templ0.features:
                    fx, fy = feat.x, feat.y
                    if fy + vy <= rh and fx + vx <= rw and fy >= 0 and fx >= 0:
                        score_map += rmaps_super_int32[feat.label][ly+fy:ly+fy+vy, lx+fx:lx+fx+vx]
                        valid += 1

                if valid == 0:
                    t_fine_score += (time.perf_counter() - _t) * 1000
                    continue

                sim_map = (score_map * 100.0) / (4 * valid)
                ys, xs = np.where(sim_map > threshold)
                
                if len(xs) == 0:
                    t_fine_score += (time.perf_counter() - _t) * 1000
                    continue

                scores = sim_map[ys, xs]
                order = np.argsort(-scores)

                nms_dist_sq = cfg.NMS_DISTANCE ** 2
                kept = []
                for idx in order:
                    r_i, c_i = int(ys[idx]), int(xs[idx])
                    dx, dy = _subpixel_refine(sim_map, r_i, c_i)
                    cx = c_i + dx + rx - templ0.tl_x + img_w // 2
                    cy = r_i + dy + ry - templ0.tl_y + img_h // 2
                    s = float(scores[idx])

                    dup = False
                    for k in kept:
                        if (cx - k['x'])**2 + (cy - k['y'])**2 < nms_dist_sq:
                            dup = True; break
                    for k in all_matches:
                        if (cx - k['x'])**2 + (cy - k['y'])**2 < nms_dist_sq:
                            if s <= k['score']:
                                dup = True; break

                    if not dup:
                        icx, icy = int(round(cx)), int(round(cy))
                        kept.append({
                            'x': cx, 'y': cy,
                            'angle': tp['angle'], 'scale': tp['scale'],
                            'score': s, 'template_id': tp_idx,
                            'bbox': (icx - img_w//2, icy - img_h//2, img_w, img_h),
                        })
                        if len(kept) >= 5:
                            break

                t_fine_score += (time.perf_counter() - _t) * 1000
                all_matches.extend(kept)

        t_fine = (time.perf_counter() - t0_fine_total) * 1000
        print(f"  [Pyramid] Fine   (L0 {sw}×{sh}): {len(all_matches)} matches in {t_fine:.0f}ms")
        print(f"  [Pyramid] Total: {t_coarse + t_fine:.0f}ms")

        timing.update({
            'fine_quantize_ms':      t_fine_quantize,
            'fine_spread_ms':        t_fine_spread,
            'fine_response_maps_ms': t_fine_rmap,
            'fine_scoring_ms':       t_fine_score,
            # Alias uniform keys used by the chart builder
            'quantize_ms':      timing.get('coarse_quantize_ms', 0) + t_fine_quantize,
            'spread_ms':        timing.get('coarse_spread_ms', 0)   + t_fine_spread,
            'response_maps_ms': timing.get('coarse_response_maps_ms', 0) + t_fine_rmap,
            'scoring_ms':       timing.get('coarse_scoring_ms', 0)  + t_fine_score,
        })
        return all_matches, timing

    def _nms(self, matches):
        """Non-maximum suppression by distance."""
        if len(matches) <= 1:
            return matches
        matches.sort(key=lambda m: m['score'], reverse=True)
        keep = []
        dist = self.config.NMS_DISTANCE
        for m in matches:
            dup = False
            for k in keep:
                if (m['x'] - k['x'])**2 + (m['y'] - k['y'])**2 < dist * dist:
                    dup = True; break
            if not dup:
                keep.append(m)
        return keep

    def visualize_match(self, image, match_result, show=True):
        """Draw detection result on image."""
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        if match_result is None:
            return vis

        # Scale drawing sizes proportional to image size
        h, w = vis.shape[:2]
        scale = max(1, max(w, h) // 1000)  # 1px per 1000px of image
        thick = max(2, scale * 2)
        marker_sz = max(20, scale * 20)
        dot_r = max(2, scale * 2)
        font_scale = max(0.6, scale * 0.6)

        bx, by, bw, bh = match_result['bbox']
        cx, cy = int(match_result['x']), int(match_result['y'])

        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), thick)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, marker_sz, thick)

        # Draw template features color-coded by orientation
        tp_idx = match_result.get('template_id', -1)
        if 0 <= tp_idx < len(self.template_pyramids):
            templ = self.template_pyramids[tp_idx]['templates'][0]
            tp_w, tp_h = self.template_pyramids[tp_idx]['size']
            colors = [(0,0,255),(0,170,255),(0,255,170),(0,255,0),
                      (170,255,0),(255,170,0),(255,0,0),(255,0,170)]
            for feat in templ.features:
                fx = int(cx - tp_w // 2 + feat.x + templ.tl_x)
                fy = int(cy - tp_h // 2 + feat.y + templ.tl_y)
                cv2.circle(vis, (fx, fy), dot_r, colors[feat.label % 8], -1)

        info = (f"Score:{match_result['score']:.1f}%  "
                f"Angle:{match_result['angle']:.1f}  "
                f"Scale:{match_result['scale']:.2f}  "
                f"Pos:({cx},{cy})")
        cv2.putText(vis, info, (10, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 255, 0), thick)

        if show:
            fig = plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title("LINE-2D Match Result")
            plt.axis('off')
            plt.show()
            plt.close(fig)  # release figure memory

        return vis

    def visualize_templates(self, max_show=12):
        """Show a grid of templates with their features."""
        n = min(max_show, len(self.template_pyramids))
        cols = min(4, n)
        rows = max(1, (n + cols - 1) // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        colors = [(0,0,255),(0,170,255),(0,255,170),(0,255,0),
                  (170,255,0),(255,170,0),(255,0,0),(255,0,170)]
        for i in range(n):
            r, c = divmod(i, cols)
            tp = self.template_pyramids[i]
            # Regenerate the rotated image on-demand (not stored to save RAM)
            src_rebuilt = LinemodMatcher._transform(
                self.template_image, tp['angle'], tp['scale'])
            img = cv2.cvtColor(src_rebuilt, cv2.COLOR_GRAY2BGR)
            templ = tp['templates'][0]
            for feat in templ.features:
                cv2.circle(img, (templ.tl_x + feat.x, templ.tl_y + feat.y),
                          2, colors[feat.label % 8], -1)
            axes[r,c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[r,c].set_title(f"{tp['angle']:.0f}° s={tp['scale']:.1f}", fontsize=8)
            axes[r,c].axis('off')
        for i in range(n, rows*cols):
            r, c = divmod(i, cols)
            axes[r,c].axis('off')
        fig.suptitle('LINE-2D Templates', fontsize=12)
        plt.tight_layout(); plt.show()

    @staticmethod
    def _transform(src, angle, scale):
        """Rotate+scale around center (same canvas size)."""
        h, w = src.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        return cv2.warpAffine(src, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
