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
        self.PYRAMID_LEVELS = 3       # Number of levels

        # --- Matching ---
        self.MATCH_THRESHOLD = 50.0   # Similarity 0–100
        self.NMS_DISTANCE = 30


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

def _quantize_gradients(src, weak_threshold=30.0, fast_mode=False, kernel_size=0):
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

    Mirrors: hysteresisGradient() in line2Dup.cpp
    """
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = src.astype(np.float32)

    # Resolution-adaptive pre-blur: thicken thin edges on high-res images
    # If a strict kernel size is enforced (e.g. from config), use it for blur as well
    # to maintain consistency between small templates and large search images.
    h, w = gray.shape
    max_dim = max(h, w)
    
    if kernel_size > 0:
        if kernel_size >= 3:
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    elif max_dim > 1500:
        # Scale blur dynamically ONLY if auto-detect (kernel_size=0) is requested
        sigma = max(1.0, (max_dim - 1000) / 1500.0)
        ksize = int(sigma * 2) * 2 + 1  # Ensure odd kernel size
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
    # When weak_threshold < 0, treat its absolute value as a PERCENTILE
    # of the non-zero gradient magnitudes in this image. E.g. -70 means
    # "use the 70th percentile as the cutoff".
    # This makes feature selection resolution-independent: a template
    # crop of 200x200 and a 5120x5120 search image will both consistently
    # select the strongest features relative to their own content.
    if weak_threshold < 0:
        nonzero_mags = magnitude[magnitude > 0]
        if len(nonzero_mags) > 0:
            pct = min(99.0, max(0.0, -weak_threshold))  # e.g. -70 → 70th %ile
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
        # kernel_size=0 means auto-detect from this image's resolution.
        # When set explicitly (e.g. from config.HYSTERESIS_KERNEL) the same
        # kernel is used for both the template crop and the full search image,
        # which is essential when the template is a small crop of a large
        # sensor image (same physical pixel density, different spatial extent).
        if kernel_size <= 0:
            # Auto: scale kernel with image resolution
            #   919px  → ks=3  (3×3,  min_votes=5/9)
            #   2048px → ks=5  (5×5,  min_votes=13/25)
            #   5120px → ks=9  (9×9,  min_votes=41/81)
            raw_ks = max(3, int(3 * max_dim / 1500))
            ks = min(raw_ks | 1, 9)   # odd, capped at 9
        else:
            ks = max(3, kernel_size | 1)   # force odd, minimum 3
        min_votes = (ks * ks) // 2 + 1   # strict majority (>50%)
        kernel = np.ones((ks, ks), dtype=np.uint8)

        # Process planes one at a time to reduce peak memory
        best_label = np.zeros((h, w), dtype=np.uint8)
        best_count = np.zeros((h, w), dtype=np.uint16)

        for i in range(8):
            plane = (quant_8 == i).astype(np.uint8)
            votes = cv2.filter2D(plane, cv2.CV_16U, kernel,
                                 borderType=cv2.BORDER_CONSTANT)
            better = votes > best_count
            best_count[better] = votes[better]
            best_label[better] = i

        # Accept only where: magnitude > threshold AND majority vote passed
        accept = (magnitude > weak_threshold) & (best_count >= min_votes)

        # Encode as bit: 1 << label
        quantized = np.zeros((h, w), dtype=np.uint8)
        quantized[accept] = (1 << best_label[accept]).astype(np.uint8)

    # Zero borders
    quantized[0, :] = 0; quantized[-1, :] = 0
    quantized[:, 0] = 0; quantized[:, -1] = 0

    return quantized, magnitude


def _spread(quantized, T):
    """
    OR-spread quantized orientations over a (2T-1)×(2T-1) neighbourhood.

    Equivalent to the original loop-based spread but implemented via
    cv2.dilate on each of the 8 bit-planes, which is 3-8× faster because
    cv2.dilate separates the 2-D box into two 1-D passes in optimised C++.
    The mathematical result is identical to the original implementation.
    """
    if T <= 1:
        return quantized.copy()

    # Box structuring element: (2T-1) in each direction covers T-1 neighbours
    ks = 2 * T - 1
    kernel = np.ones((ks, ks), dtype=np.uint8)

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
        rmap = _RESPONSE_LUTS[label][spread_img]
        response_maps.append(rmap)
    return response_maps


def _linearize(response_map, T):
    """
    Linearize a response map into a (T*T, H_dec * W_dec) array.
    Each of the T*T sub-grids is a decimated view of the response map.

    The sub-grid at (gy, gx) contains response_map[gy::T, gx::T].
    Rows of the output are indexed by grid_y*T + grid_x.

    Mirrors: linearize() in line2Dup.cpp
    """
    h, w = response_map.shape
    W = w // T
    H = h // T
    memories = np.zeros((T * T, H * W), dtype=np.uint8)
    for gy in range(T):
        for gx in range(T):
            decimated = response_map[gy:gy + H * T:T, gx:gx + W * T:T]
            row_idx = gy * T + gx
            memories[row_idx, :decimated.size] = decimated.ravel()[:W * H]
    return memories, W, H


def _similarity(linear_memories, templ_features, T, W, H):
    """
    Compute similarity map by summing response values at each feature.

    For each template position (lm_x, lm_y) in the decimated grid,
    accumulate response_maps[feat.label] at the feature's location
    offset by that position.

    Returns similarity map of shape (H, W) as uint16.
    Mirrors: similarity() in line2Dup.cpp
    """
    # Dimensions of the output similarity map
    dst = np.zeros(H * W, dtype=np.int32)

    for feat in templ_features:
        # Which sub-grid does this feature fall in?
        grid_x = feat.x % T
        grid_y = feat.y % T
        grid_index = grid_y * T + grid_x

        # Where in the sub-grid?
        lm_x = feat.x // T
        lm_y = feat.y // T
        lm_offset = lm_y * W + lm_x

        # Get the linearized memory row for this feature's label
        memory = linear_memories[feat.label]  # shape (T*T, H*W)
        lm_row = memory[grid_index]           # shape (H*W,)

        # Compute span: how many positions we can slide to
        wf = (max(f.x for f in templ_features) - min(f.x for f in templ_features)) // T + 1
        hf = (max(f.y for f in templ_features) - min(f.y for f in templ_features)) // T + 1
        span_x = W - wf
        span_y = H - hf
        n_positions = span_y * W + span_x + 1

        # Accumulate: dst[j] += lm_row[lm_offset + j] for each valid j
        end = min(lm_offset + n_positions, len(lm_row))
        start = lm_offset
        if start < len(lm_row) and end > start:
            length = end - start
            dst[:length] += lm_row[start:end].astype(np.int32)

    return dst


def _extract_scattered_features(quantized, magnitude, num_features, mask=None):
    """
    Select N spatially scattered feature points from the quantized image.
    Sorted by gradient magnitude, with minimum distance constraint.

    Mirrors: selectScatteredFeatures() in line2Dup.cpp
    """
    # Vectorized candidate extraction
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

    # Sort by magnitude descending
    order = np.argsort(-mags)

    # Greedy scattered selection
    features = []
    
    # Dynamically scale initial distance based on the object's estimated perimeter
    span_x = np.max(xs) - np.min(xs)
    span_y = np.max(ys) - np.min(ys)
    perimeter_approx = 2.0 * (span_x + span_y)
    distance = max(5.0, perimeter_approx / (num_features * 0.8))  # Slightly optimistic start

    while distance >= 1.0:
        features = []  # CRITICAL: Restart collection for the new distance threshold
        dist_sq = distance * distance
        
        for idx in order:
            x, y, label = int(xs[idx]), int(ys[idx]), int(labels[idx])

            # Check minimum distance to existing features
            keep = True
            for f in features:
                if (x - f.x)**2 + (y - f.y)**2 < dist_sq:
                    keep = False; break
                    
            if keep:
                features.append(Feature(x, y, label))
                if len(features) >= num_features:
                    break

        if len(features) >= num_features:
            break
            
        # Fast relaxation for large starting distances
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


# ======================================================================
# High-Level Matcher Class
# ======================================================================
class LinemodMatcher:
    """Shape-based template matcher — Python port of SBM LINE-2D."""

    def __init__(self, config=None):
        self.config = config or LinemodConfig()
        self.template_image = None
        self.template_pyramids = []   # List of dicts: angle, scale, templates, image, size

    def load_template(self, template_img):
        """Load the base template image (grayscale)."""
        if len(template_img.shape) == 3:
            self.template_image = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            self.template_image = template_img.copy()

    def generate_templates(self):
        """Generate rotated+scaled templates (mirrors shapeInfo.produce_infos)."""
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
        added = 0

        for scale in scales:
            for angle in angles:
                src = self._transform(self.template_image, angle, scale)
                mask_img = self._transform(
                    np.ones_like(self.template_image) * 255, angle, scale)
                mask_img = (mask_img > 128).astype(np.uint8) * 255

                # Extract features at each pyramid level
                pyr_src = src.copy()
                pyr_mask = mask_img.copy()
                templates = []
                ok = True

                for level in range(cfg.PYRAMID_LEVELS):
                    if level > 0:
                        pyr_src = cv2.pyrDown(pyr_src)
                        pyr_mask = cv2.pyrDown(pyr_mask)
                        pyr_mask = (pyr_mask > 128).astype(np.uint8) * 255

                    quantized, mag = _quantize_gradients(
                        pyr_src, cfg.WEAK_THRESHOLD,
                        kernel_size=cfg.HYSTERESIS_KERNEL)
                    features = _extract_scattered_features(
                        quantized, mag, cfg.NUM_FEATURES, pyr_mask)

                    if len(features) < 4:
                        ok = False; break

                    t = TemplatePyr()
                    t.pyramid_level = level
                    t.features = features
                    templates.append(t)

                if not ok:
                    continue

                _crop_templates(templates)

                self.template_pyramids.append({
                    'angle': angle, 'scale': scale,
                    'templates': templates, 'image': src,
                    'size': (src.shape[1], src.shape[0]),
                })
                added += 1

        print(f"[LINE-2D] Generated {added}/{total} templates "
              f"({len(angles)} angles × {len(scales)} scales)")

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
        """
        if not self.template_pyramids:
            raise ValueError("Call generate_templates() first.")

        cfg = self.config
        threshold = threshold or cfg.MATCH_THRESHOLD

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
            matches = self._match_pyramid(search_gray, threshold)
            # Robust fallback: if pyramid found nothing (coarse was off or ROI
            # missed), retry with the reliable full-image single-level scan.
            if not matches:
                print("  [Pyramid] 0 matches — falling back to full-image scan")
                matches = self._match_single_level(search_gray, threshold)
        else:
            matches = self._match_single_level(search_gray, threshold)

        # Offset positions back to original image coordinates
        if roi_offset != (0, 0):
            ox, oy = roi_offset
            for m in matches:
                m['x'] += ox
                m['y'] += oy
                bx, by, bw, bh = m['bbox']
                m['bbox'] = (bx + ox, by + oy, bw, bh)

        matches = self._nms(matches)
        matches.sort(key=lambda m: m['score'], reverse=True)

        if return_all:
            return matches
        return matches[0] if matches else None

    def _match_single_level(self, search_gray, threshold):
        """
        Single-level scoring via vectorized response map slicing.
        """
        cfg = self.config
        all_matches = []

        T = cfg.T_PYRAMID[0]
        # Auto-scale spread for high-res images: thin edges need wider spread
        sh, sw = search_gray.shape[:2]
        max_dim = max(sh, sw)
        if max_dim > 1500:
            # Scale T proportionally: T=4 at 1500px → T≈14 at 5120px
            T = max(T, int(T * max_dim / 1500))
            print(f"  [Auto] Spread T scaled to {T} for {sw}×{sh} image")
        quantized, _ = _quantize_gradients(
            search_gray, cfg.WEAK_THRESHOLD,
            kernel_size=cfg.HYSTERESIS_KERNEL)
        spread_q = _spread(quantized, T)
        rmaps = _compute_response_maps(spread_q)
        # Pre-cast response maps to int32 once to avoid massive memory allocations in the inner loop
        rmaps_int32 = [r.astype(np.int32) for r in rmaps]
        
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
                cx = int(xs[idx]) - templ.tl_x + img_w // 2
                cy = int(ys[idx]) - templ.tl_y + img_h // 2
                s = float(scores[idx])

                dup = False
                for k in kept:
                    if (cx - k['x'])**2 + (cy - k['y'])**2 < nms_dist_sq:
                        dup = True; break
                if not dup:
                    kept.append({
                        'x': cx, 'y': cy,
                        'angle': tp['angle'], 'scale': tp['scale'],
                        'score': s, 'template_id': tp_idx,
                        'bbox': (cx - img_w//2, cy - img_h//2, img_w, img_h),
                    })
                    if len(kept) >= 20:
                        break

            all_matches.extend(kept)

        return all_matches

    def _match_pyramid(self, search_gray, threshold):
        """
        Coarse-to-fine pyramid matching for large images.

        Level 1 (coarse): Downsample search image by 2×, fast scan with
            level-1 features and lower threshold to find candidate regions.
        Level 0 (fine):   Search only within ROI windows around each coarse
            candidate at full resolution for precise positioning.
        """
        import time
        import math
        cfg = self.config
        sh, sw = search_gray.shape[:2]

        # Determine optimal coarse level (1/2, 1/4, 1/8 size)
        max_dim = max(sh, sw)
        ideal_level = int(max(1, round(math.log2(max_dim / 1200.0))))
        coarse_level = min(cfg.PYRAMID_LEVELS - 1, ideal_level)
        coarse_level = max(1, coarse_level) # Ensure at least 1

        if getattr(cfg, "FORCE_COARSE_LEVEL", None) is not None:
            coarse_level = min(cfg.PYRAMID_LEVELS - 1, max(0, cfg.FORCE_COARSE_LEVEL))

        # ---- Level N: Coarse search at reduced resolution ----
        t0 = time.time()
        search_coarse = search_gray
        for _ in range(coarse_level):
            search_coarse = cv2.pyrDown(search_coarse)
        sh_c, sw_c = search_coarse.shape[:2]

        T_c = cfg.T_PYRAMID[coarse_level] if len(cfg.T_PYRAMID) > coarse_level else cfg.T_PYRAMID[-1]
        
        q_c, _ = _quantize_gradients(search_coarse, cfg.WEAK_THRESHOLD,
                                    kernel_size=cfg.HYSTERESIS_KERNEL)
        s_c = _spread(q_c, T_c)
        rmaps_c = _compute_response_maps(s_c)
        rmaps_c_int32 = [r.astype(np.int32) for r in rmaps_c]

        # Lower threshold for coarse pass — be generous to not miss candidates.
        coarse_threshold = max(threshold * 0.3, 15.0)

        coarse_candidates = []  # (tp_idx, cx_full, cy_full, coarse_score)

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

            # Score at coarse level
            score_map = np.zeros((vy_c, vx_c), dtype=np.int32)
            valid = 0
            for feat in templ_c.features:
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
                    if len(kept_coarse) >= 1:   # best candidate only per template mode
                        break

            coarse_candidates.extend(kept_coarse)

        t_coarse = (time.time() - t0) * 1000
        print(f"  [Pyramid] Coarse (L{coarse_level} {sw_c}×{sh_c}): {len(coarse_candidates)} candidates in {t_coarse:.0f}ms")

        if not coarse_candidates:
            return []

        # ---- Level 0: Fine search in ROI windows ----
        t0 = time.time()
        T0 = cfg.T_PYRAMID[0]
        # Auto-scale spread for high-res fine search regions
        if max_dim > 1500:
            T0 = max(T0, int(T0 * max_dim / 1500))

        # Margin scaling based on maximum expected coarse spatial drift 
        # (e.g., 8-12 pixels of drift in coarse space scaled upward)
        search_margin = 12 * scale_factor 

        all_matches = []

        for tp_idx, cx_full, cy_full, coarse_score in coarse_candidates:
            tp = self.template_pyramids[tp_idx]
            templ0 = tp['templates'][0]  # Level 0 (full-res) template
            n_feats = len(templ0.features)
            if n_feats == 0:
                continue

            img_w, img_h = tp['size']
            tw0 = templ0.width
            th0 = templ0.height

            # Expected feature-bbox top-left in image coords.
            # cx_full/cy_full is template IMAGE centre:
            #   feat_tl = centre - img_half + tl_offset_inside_image
            tl_feat_x = cx_full - img_w // 2 + templ0.tl_x
            tl_feat_y = cy_full - img_h // 2 + templ0.tl_y

            # ROI: tight window around expected TL ± search_margin.
            # Score-map slack = 2*search_margin + T0 absorbs coarse inaccuracy
            # without scanning the whole image.
            rx  = max(0,  tl_feat_x - search_margin)
            ry  = max(0,  tl_feat_y - search_margin)
            rx2 = min(sw, tl_feat_x + tw0 + search_margin + T0 + 4)
            ry2 = min(sh, tl_feat_y + th0 + search_margin + T0 + 4)

            roi_crop = search_gray[ry:ry2, rx:rx2]
            rh, rw = roi_crop.shape[:2]

            if rw < tw0 + 10 or rh < th0 + 10:
                continue

            # Quantize + spread + response maps on ROI ONLY
            q_roi, _ = _quantize_gradients(roi_crop, cfg.WEAK_THRESHOLD)
            s_roi = _spread(q_roi, T0)
            rmaps_roi = _compute_response_maps(s_roi)
            rmaps_roi_int32 = [r.astype(np.int32) for r in rmaps_roi]

            # Valid scan area within ROI
            vy = rh - th0
            vx = rw - tw0
            if vy <= 0 or vx <= 0:
                continue

            # Score within ROI
            score_map = np.zeros((vy, vx), dtype=np.int32)
            valid = 0

            for feat in templ0.features:
                fx, fy = feat.x, feat.y
                if fy + vy <= rh and fx + vx <= rw and fy >= 0 and fx >= 0:
                    score_map += rmaps_roi_int32[feat.label][fy:fy+vy, fx:fx+vx]
                    valid += 1

            if valid == 0:
                continue

            sim_map = (score_map * 100.0) / (4 * valid)
            ys, xs = np.where(sim_map > threshold)
            if len(xs) == 0:
                continue

            scores = sim_map[ys, xs]
            order = np.argsort(-scores)

            nms_dist_sq = cfg.NMS_DISTANCE ** 2
            kept = []
            for idx in order:
                # Convert ROI-local position to image coordinates
                cx = int(xs[idx]) + rx - templ0.tl_x + img_w // 2
                cy = int(ys[idx]) + ry - templ0.tl_y + img_h // 2
                s = float(scores[idx])

                dup = False
                for k in kept:
                    if (cx - k['x'])**2 + (cy - k['y'])**2 < nms_dist_sq:
                        dup = True; break
                # Also check against already-found matches
                for k in all_matches:
                    if (cx - k['x'])**2 + (cy - k['y'])**2 < nms_dist_sq:
                        if s <= k['score']:
                            dup = True; break

                if not dup:
                    kept.append({
                        'x': cx, 'y': cy,
                        'angle': tp['angle'], 'scale': tp['scale'],
                        'score': s, 'template_id': tp_idx,
                        'bbox': (cx - img_w//2, cy - img_h//2, img_w, img_h),
                    })
                    if len(kept) >= 5:
                        break

            all_matches.extend(kept)

        t_fine = (time.time() - t0) * 1000
        print(f"  [Pyramid] Fine   (L0 {sw}×{sh}): {len(all_matches)} matches in {t_fine:.0f}ms")
        print(f"  [Pyramid] Total: {t_coarse + t_fine:.0f}ms")

        return all_matches

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
        tp_idx = match_result['template_id']
        if tp_idx < len(self.template_pyramids):
            templ = self.template_pyramids[tp_idx]['templates'][0]
            tp_w, tp_h = self.template_pyramids[tp_idx]['size']
            # When downsample was used, feat coords are in downsampled space
            # but bbox is in original space. Scale features to match.
            feat_scale_x = bw / tp_w if tp_w > 0 else 1.0
            feat_scale_y = bh / tp_h if tp_h > 0 else 1.0
            colors = [(0,0,255),(0,170,255),(0,255,170),(0,255,0),
                      (170,255,0),(255,170,0),(255,0,0),(255,0,170)]
            for feat in templ.features:
                fx = bx + int((feat.x + templ.tl_x) * feat_scale_x)
                fy = by + int((feat.y + templ.tl_y) * feat_scale_y)
                cv2.circle(vis, (fx, fy), dot_r, colors[feat.label % 8], -1)

        info = (f"Score:{match_result['score']:.1f}%  "
                f"Angle:{match_result['angle']:.1f}  "
                f"Scale:{match_result['scale']:.2f}  "
                f"Pos:({cx},{cy})")
        cv2.putText(vis, info, (10, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 255, 0), thick)

        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title("LINE-2D Match Result")
            plt.axis('off'); plt.show()

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
            img = cv2.cvtColor(tp['image'], cv2.COLOR_GRAY2BGR)
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
