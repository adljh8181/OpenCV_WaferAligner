"""
================================================================================
FOV CLASSIFIER v1.0
================================================================================
Classifies the Field of View (FOV) type:
- EDGE_FOV: Wafer edge is visible
- DIE_FOV: Die patterns are visible
- WAFER_FOV: Plain wafer surface

Author: Auto-generated for Wafer Alignment System
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import os


# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

class ClassificationConfig:
    """Configuration for FOV classification parameters"""

    # --- FILE SETTINGS ---
    IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'Images', 'LEFT_EDGE_PNG')
    DEFAULT_IMAGE = 'Copy of sample_wafer.jpg'

    # --- IMAGE PROCESSING ---
    TARGET_PROCESS_DIM = 1000   # Resize longest dimension to this
    CLAHE_CLIP_LIMIT = 2.0      # CLAHE contrast normalization (0=off)
    CLAHE_GRID_SIZE = 8         # CLAHE tile grid size

    # --- GRADIENT KERNEL ---
    KERNEL_SIZE = 7

    # --- ADAPTIVE THRESHOLDS (percentile-based, 0-1 range) ---
    RELATIVE_CHANGE_WEAK = 0.15     # 15% of image dynamic range
    RELATIVE_CHANGE_STRONG = 0.25   # 25% of image dynamic range
    RELATIVE_QUARTER_DIFF = 0.10    # 10% of range for quarter diff

    # --- ABSOLUTE THRESHOLDS (fallback) ---
    MIN_INTENSITY_CHANGE_STRONG = 50
    MIN_INTENSITY_CHANGE_WEAK = 30
    MIN_LENGTH_RATIO_LONG = 0.15
    MIN_LENGTH_RATIO_SHORT = 0.05
    MIN_LEFT_RIGHT_DIFF = 40

    # --- 2D EDGE DETECTION ---
    SOBEL_STRENGTH_THRESHOLD = 0.20  # Min 2D gradient strength (relative)

    # --- CONFIDENCE ---
    CONFIDENCE_THRESHOLD = 0.40     # Below this → UNCERTAIN

    # --- DIE OVERRIDE ---
    DIE_OVERRIDE_PCT = 100          # % of regions that must be DIE to override edge
                                    # 100% = only pure die (6/6 regions), most conservative
                                    # 67% = 4/6 regions, more aggressive
                                    # Applies only when edge detection has NO real edge signal

    # --- EDGE DETECTION SETTINGS ---
    EDGE_THRESHOLD = 25


# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

def create_gradient_kernel(size):
    """
    Creates a gradient kernel with specified window size.
    Uses multiple -1s and +1s for better noise resistance.

    Examples:
        size=3: [-1, 0, 1]
        size=5: [-1, -1, 0, 1, 1]
        size=7: [-1, -1, -1, 0, 1, 1, 1]
        size=9: [-1, -1, -1, -1, 0, 1, 1, 1, 1]

    Args:
        size: Total window size (must be odd, >= 3)

    Returns:
        numpy array representing the kernel
    """
    # Ensure size is odd and at least 3
    size = max(3, size)
    if size % 2 == 0:
        size += 1
    
    # Calculate number of -1s and +1s on each side
    half_width = size // 2
    
    kernel = np.concatenate([
        -np.ones(half_width),
        np.zeros(1),
        np.ones(half_width)
    ])
    
    return kernel


def preprocess_image(img_or_path, target_dim=1000, clahe_clip=2.0, clahe_grid=8):
    """
    Load and preprocess image for classification.

    Args:
        img_or_path: Either a numpy array (grayscale image) or path to image file
        target_dim: Target dimension for longest side
        clahe_clip: CLAHE clip limit (0 to disable)
        clahe_grid: CLAHE tile grid size

    Returns:
        Tuple of (processed_img, original_img, scale)
    """
    if isinstance(img_or_path, str):
        if not os.path.exists(img_or_path):
            raise FileNotFoundError(f'File not found: {img_or_path}')
        original_img = cv2.imread(img_or_path, 0)
    else:
        original_img = img_or_path.copy()

    if original_img is None:
        raise ValueError("Could not load image")

    # Resize
    h_orig, w_orig = original_img.shape
    scale = target_dim / max(h_orig, w_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    processed_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Apply CLAHE for lighting normalization
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        processed_img = clahe.apply(processed_img)

    # processed_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    return processed_img, original_img, scale


# ==============================================================================
#                    FOV CLASSIFIER CLASS
# ==============================================================================

class FOVClassifier:
    """
    Determines if FOV shows Edge, Die, or plain Wafer surface.

    Usage:
        classifier = FOVClassifier()
        result = classifier.classify(image_path)
        classifier.visualize(result)
    """

    def __init__(self, config=None):
        self.config = config or ClassificationConfig()
        self.kernel = create_gradient_kernel(self.config.KERNEL_SIZE)

    def classify(self, img_or_path):
        """
        Main classification method.

        Args:
            img_or_path: Grayscale image (numpy array) or path to image file

        Returns:
            dict with classification results including confidence score
        """
        # Preprocess image (with CLAHE normalization)
        img, original_img, scale = preprocess_image(
            img_or_path, 
            self.config.TARGET_PROCESS_DIM,
            self.config.CLAHE_CLIP_LIMIT,
            self.config.CLAHE_GRID_SIZE
        )

        # 1D edge detection (horizontal + vertical)
        edge_result = self._detect_edge(img)

        # 2D Sobel edge detection (catches diagonal edges)
        edge_2d = self._detect_edge_2d(img)

        # NEW: Global peak-counting classifier
        # Collapses the full image into 1D projections and counts prominent
        # peaks.  Wafer edge → 1-2 peaks; die grid → 5+ peaks.
        peak_result = self._peak_based_classify(img)

        # Analyze regions
        region_result = self._analyze_regions(img)

        # Final classification — region analysis cross-validates edge detection
        h_classifications = [region_result[k]['classification'] 
                            for k in ['left', 'center', 'right'] if k in region_result]
        v_classifications = [region_result[k]['classification'] 
                            for k in ['top', 'center_v', 'bottom'] if k in region_result]
        all_classifications = h_classifications + v_classifications
        die_count = all_classifications.count("DIE")

        # Die override: configurable threshold for when to classify as DIE_FOV
        # even if edge detection triggered. Default 100% = all regions must be DIE.
        total_regions = max(len(all_classifications), 1)
        die_pct = die_count / total_regions
        die_threshold = self.config.DIE_OVERRIDE_PCT / 100.0

        if die_pct >= die_threshold and not edge_result['has_edge']:
            fov_type = "DIE_FOV"
        elif edge_result['has_edge']:
            fov_type = "EDGE_FOV"
        elif edge_2d['has_2d_edge'] and die_pct < die_threshold:
            fov_type = "EDGE_FOV"
        else:
            if die_count >= 3:
                fov_type = "DIE_FOV"
            elif all_classifications.count("WAFER") >= 4:
                fov_type = "WAFER_FOV"
            else:
                fov_type = "DIE_FOV" if die_count >= 2 else "WAFER_FOV"

        # ── Peak-count override ─────────────────────────────────────────
        # High peak count = strong DIE evidence, even if the edge detector
        # saw a faint transition.  Only override EDGE→DIE when the edge
        # signal itself was weak (not has_edge from criterion 1-4).
        if peak_result['label'] == "DIE_LIKELY":
            if fov_type == "EDGE_FOV" and not edge_result['has_edge']:
                # Edge was only from 2D Sobel or quarter-diff — override to DIE
                fov_type = "DIE_FOV"
            elif fov_type != "EDGE_FOV":
                fov_type = "DIE_FOV"
        elif peak_result['label'] == "EDGE_LIKELY" and fov_type == "DIE_FOV":
            # Very few peaks despite region saying DIE — trust region + peak
            # but downgrade to UNCERTAIN (don't flip to EDGE without other evidence)
            pass   # let confidence handle it

        # Compute confidence
        confidence = self._compute_confidence(fov_type, edge_result, edge_2d,
                                              region_result, peak_result)

        # Apply UNCERTAIN fallback
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            fov_type = "UNCERTAIN"

        return {
            'success': True,
            'fov_type': fov_type,
            'confidence': confidence,
            'edge': edge_result,
            'edge_2d': edge_2d,
            'peaks': peak_result,           # new
            'regions': region_result,
            'image': img,
            'original_image': original_img,
            'scale': scale
        }

    def _detect_edge(self, img):
        """Detect edge transition in image (both horizontal and vertical)"""
        h, w = img.shape
        cfg = self.config

        # === HORIZONTAL SCAN (left-right) for LEFT/RIGHT edges ===
        h_result = self._detect_edge_1d(img, axis='horizontal')

        # === VERTICAL SCAN (top-bottom) for TOP/BOTTOM edges ===
        v_result = self._detect_edge_1d(img, axis='vertical')

        # Pick the stronger detection
        if h_result['has_edge'] and v_result['has_edge']:
            # Both detected - pick the one with stronger signal
            h_strength = max(h_result['rising_change'], h_result['falling_change'])
            v_strength = max(v_result['rising_change'], v_result['falling_change'])
            if v_strength > h_strength:
                return v_result
            else:
                return h_result
        elif v_result['has_edge']:
            return v_result
        elif h_result['has_edge']:
            return h_result
        else:
            # Neither detected - return horizontal result (default)
            return h_result

    def _detect_edge_1d(self, img, axis='horizontal'):
        """
        Detect edge transition along one axis.
        
        Args:
            img: Grayscale image
            axis: 'horizontal' (scan left-right) or 'vertical' (scan top-bottom)
        
        Returns:
            dict with edge detection results
        """
        h, w = img.shape
        cfg = self.config

        if axis == 'horizontal':
            # Extract horizontal intensity profile (average across multiple horizontal bands)
            bands = [
                img[int(h*0.2):int(h*0.35), :],
                img[int(h*0.35):int(h*0.5), :],
                img[int(h*0.5):int(h*0.65), :],
                img[int(h*0.65):int(h*0.8), :]
            ]
            profiles = [np.mean(band, axis=0) for band in bands]
            profile_length = w
        else:
            # Extract vertical intensity profile (average across multiple vertical bands)
            bands = [
                img[:, int(w*0.2):int(w*0.35)],
                img[:, int(w*0.35):int(w*0.5)],
                img[:, int(w*0.5):int(w*0.65)],
                img[:, int(w*0.65):int(w*0.8)]
            ]
            profiles = [np.mean(band, axis=1) for band in bands]
            profile_length = h

        avg_profile = np.mean(profiles, axis=0)

        # Apply smoothing
        smoothed_heavy = gaussian_filter1d(avg_profile, sigma=30)
        smoothed_light = gaussian_filter1d(avg_profile, sigma=10)

        # Calculate statistics
        min_intensity = np.min(smoothed_heavy)
        max_intensity = np.max(smoothed_heavy)
        intensity_range = max_intensity - min_intensity

        first_quarter = np.mean(smoothed_heavy[:profile_length//4])
        last_quarter = np.mean(smoothed_heavy[3*profile_length//4:])
        quarter_diff = abs(last_quarter - first_quarter)

        # Calculate gradients
        gradient_smooth = np.gradient(smoothed_heavy)
        gradient_sharp = np.gradient(smoothed_light)

        # Check for periodic gradient pattern (indicates die grid, not edge)
        gradient_abs = np.abs(gradient_smooth)
        grad_peak_height = np.max(gradient_abs) * 0.3 if np.max(gradient_abs) > 0 else 0
        peaks_1d, _ = find_peaks(gradient_abs, height=grad_peak_height, distance=20)
        is_periodic = len(peaks_1d) >= 4  # Die images have many repeating peaks

        # Find monotonic regions
        rising_region, falling_region = self._find_monotonic_regions(gradient_smooth)

        # Calculate intensity changes
        rising_change = (smoothed_heavy[min(rising_region[1]-1, len(smoothed_heavy)-1)] -
                        smoothed_heavy[rising_region[0]]) if rising_region[2] > 0 else 0
        falling_change = (smoothed_heavy[falling_region[0]] -
                         smoothed_heavy[min(falling_region[1]-1, len(smoothed_heavy)-1)]) if falling_region[2] > 0 else 0

        rising_ratio = rising_region[2] / profile_length
        falling_ratio = falling_region[2] / profile_length

        # Adaptive thresholds: use the image's own dynamic range
        dynamic_range = max(intensity_range, 1)  # avoid div by zero
        relative_rising = rising_change / dynamic_range
        relative_falling = falling_change / dynamic_range
        relative_quarter = quarter_diff / dynamic_range

        # Decision logic — ADAPTIVE (relative) OR ABSOLUTE thresholds
        edge_detected = False
        edge_type = "NO_EDGE"
        wafer_side = None
        edge_x = None
        transition_start = None
        transition_end = None
        detection_method = None

        if axis == 'horizontal':
            rising_label, rising_side = "RIGHT_EDGE", "LEFT"
            falling_label, falling_side = "LEFT_EDGE", "RIGHT"
        else:
            rising_label, rising_side = "BOTTOM_EDGE", "TOP"
            falling_label, falling_side = "TOP_EDGE", "BOTTOM"

        # Helper: check if change meets threshold (adaptive OR absolute)
        def meets_weak(change, relative):
            return change >= cfg.MIN_INTENSITY_CHANGE_WEAK or relative >= cfg.RELATIVE_CHANGE_WEAK

        def meets_strong(change, relative):
            return change >= cfg.MIN_INTENSITY_CHANGE_STRONG or relative >= cfg.RELATIVE_CHANGE_STRONG

        # Criterion 1: Curved rising edge (skip if periodic die pattern)
        if not is_periodic and rising_ratio >= cfg.MIN_LENGTH_RATIO_LONG and meets_weak(rising_change, relative_rising):
            edge_detected = True
            edge_type = rising_label
            wafer_side = rising_side
            edge_x = rising_region[0] + rising_region[2] // 2
            transition_start = rising_region[0]
            transition_end = rising_region[1]
            detection_method = "CURVED_RISING"

        # Criterion 2: Curved falling edge
        elif not is_periodic and falling_ratio >= cfg.MIN_LENGTH_RATIO_LONG and meets_weak(falling_change, relative_falling):
            edge_detected = True
            edge_type = falling_label
            wafer_side = falling_side
            edge_x = falling_region[0] + falling_region[2] // 2
            transition_start = falling_region[0]
            transition_end = falling_region[1]
            detection_method = "CURVED_FALLING"

        # Criterion 3: Sharp rising edge (strong signal can override periodicity)
        elif rising_ratio >= cfg.MIN_LENGTH_RATIO_SHORT and meets_strong(rising_change, relative_rising) and not is_periodic:
            edge_detected = True
            edge_type = rising_label
            wafer_side = rising_side
            edge_x = rising_region[0] + rising_region[2] // 2
            transition_start = rising_region[0]
            transition_end = rising_region[1]
            detection_method = "SHARP_RISING"

        # Criterion 4: Sharp falling edge
        elif falling_ratio >= cfg.MIN_LENGTH_RATIO_SHORT and meets_strong(falling_change, relative_falling) and not is_periodic:
            edge_detected = True
            edge_type = falling_label
            wafer_side = falling_side
            edge_x = falling_region[0] + falling_region[2] // 2
            transition_start = falling_region[0]
            transition_end = falling_region[1]
            detection_method = "SHARP_FALLING"

        # Criterion 5: Quarter difference (adaptive OR absolute) — skip if periodic
        elif not is_periodic and (quarter_diff >= cfg.MIN_LEFT_RIGHT_DIFF or 
              relative_quarter >= cfg.RELATIVE_QUARTER_DIFF):
            edge_detected = True
            peak_idx = np.argmax(np.abs(gradient_sharp))
            edge_x = peak_idx
            if first_quarter < last_quarter:
                edge_type = rising_label
                wafer_side = rising_side
            else:
                edge_type = falling_label
                wafer_side = falling_side
            transition_start = max(0, peak_idx - profile_length//10)
            transition_end = min(profile_length, peak_idx + profile_length//10)
            detection_method = "QUARTER_DIFF"

        return {
            'has_edge': edge_detected,
            'edge_type': edge_type,
            'wafer_side': wafer_side,
            'edge_x': edge_x,
            'transition_start': transition_start,
            'transition_end': transition_end,
            'detection_method': detection_method,
            'scan_axis': axis,
            'intensity_profile': smoothed_heavy,
            'gradient': gradient_smooth,
            'intensity_range': intensity_range,
            'left_right_diff': quarter_diff,
            'rising_ratio': rising_ratio,
            'falling_ratio': falling_ratio,
            'rising_change': rising_change,
            'falling_change': falling_change,
            'relative_rising': relative_rising,
            'relative_falling': relative_falling,
            'relative_quarter': relative_quarter
        }

    def _find_monotonic_regions(self, gradient, threshold=0.05):
        """Find longest rising and falling regions"""
        rising_mask = gradient > threshold
        falling_mask = gradient < -threshold

        def find_longest_run(mask):
            if not np.any(mask):
                return 0, 0, 0
            runs = []
            start = None
            for i, val in enumerate(mask):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    runs.append((start, i, i - start))
                    start = None
            if start is not None:
                runs.append((start, len(mask), len(mask) - start))
            if not runs:
                return 0, 0, 0
            return max(runs, key=lambda x: x[2])

        return find_longest_run(rising_mask), find_longest_run(falling_mask)

    def _detect_edge_2d(self, img):
        """
        Fast 2D Sobel edge detection for catching diagonal edges.
        Uses downsampled image for speed (~2ms on 1000px image).
        
        Returns:
            dict with has_2d_edge, edge_strength, edge_angle
        """
        cfg = self.config
        
        # Downsample 4x for speed
        small = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), 
                          interpolation=cv2.INTER_AREA)
        small_f = small.astype(np.float32)
        
        # Sobel gradients
        gx = cv2.Sobel(small_f, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(small_f, cv2.CV_32F, 0, 1, ksize=5)
        mag = np.sqrt(gx**2 + gy**2)
        
        # Normalize magnitude relative to image range
        img_range = max(float(np.percentile(small, 95) - np.percentile(small, 5)), 1)
        mag_normalized = mag / img_range
        
        # Check if there's a strong concentrated gradient region
        # Use top 5% of gradient magnitudes
        threshold = np.percentile(mag_normalized, 95)
        strong_mask = mag_normalized > threshold
        strong_count = np.sum(strong_mask)
        total_pixels = mag.size
        
        # Edge strength = mean of top 5% normalized gradient values
        if strong_count > 0:
            edge_strength = float(np.mean(mag_normalized[strong_mask]))
        else:
            edge_strength = 0.0
        
        # Dominant angle (degrees from horizontal)
        if strong_count > 0:
            angles = np.arctan2(gy[strong_mask], gx[strong_mask])
            # Use circular mean
            mean_angle = float(np.degrees(np.arctan2(
                np.mean(np.sin(angles)), np.mean(np.cos(angles)))))
        else:
            mean_angle = 0.0
        
        # Check if strong gradients are concentrated (edge) vs distributed (die)
        # A real wafer edge has gradients along a line; die patterns spread everywhere
        if strong_count > 0:
            coords = np.argwhere(strong_mask)
            spatial_std = np.std(coords, axis=0)
            # A real edge should be concentrated along one axis
            min_spread = float(np.min(spatial_std))
            is_concentrated = min_spread < (small.shape[0] * 0.25)
        else:
            is_concentrated = False
        
        has_2d_edge = edge_strength >= cfg.SOBEL_STRENGTH_THRESHOLD and is_concentrated
        
        return {
            'has_2d_edge': has_2d_edge,
            'edge_strength': edge_strength,
            'edge_angle': mean_angle
        }

    # ------------------------------------------------------------------
    # Peak-based global classifier  (new)
    # ------------------------------------------------------------------

    def _peak_based_classify(self, img):
        """
        Count prominent peaks in the image's 1D intensity projection.

        Method
        ------
        1. Collapse the whole image into a horizontal profile (mean per column)
           and a vertical profile (mean per row) — use the central 60% to
           avoid border brightness gradients.
        2. Apply a light Gaussian smooth (sigma=5) to remove pixel noise.
        3. Take the absolute gradient of the smoothed profile.
        4. Call find_peaks(..., prominence=5% of dynamic range, distance=len/20)
           so only STRUCTURAL peaks survive — not noise wiggles.

        Interpretation
        --------------
        <=2 peaks  → EDGE_LIKELY   (one clean intensity step)
        3-4 peaks  → UNCERTAIN     (ambiguous)
        >=5 peaks  → DIE_LIKELY    (regular repeating grid)
        """
        h, w = img.shape

        # Use central 60% of the image (avoid bright/dark borders)
        roi = img[int(h * 0.2):int(h * 0.8),
                  int(w * 0.2):int(w * 0.8)]

        # 1-D projections
        h_profile = np.mean(roi, axis=0).astype(np.float64)  # per column
        v_profile = np.mean(roi, axis=1).astype(np.float64)  # per row

        def _count_peaks(profile):
            smoothed = gaussian_filter1d(profile, sigma=5)
            grad     = np.abs(np.gradient(smoothed))
            dyn      = max(float(np.max(smoothed) - np.min(smoothed)), 1.0)
            min_prom = dyn * 0.05           # 5 % of dynamic range
            min_dist = max(len(profile) // 20, 5)
            peaks, _ = find_peaks(grad, prominence=min_prom, distance=min_dist)
            return len(peaks), peaks, grad, smoothed

        h_count, h_peaks, h_grad, h_smooth = _count_peaks(h_profile)
        v_count, v_peaks, v_grad, v_smooth = _count_peaks(v_profile)

        max_count = max(h_count, v_count)

        if max_count <= 2:
            label = "EDGE_LIKELY"
        elif max_count >= 5:
            label = "DIE_LIKELY"
        else:
            label = "UNCERTAIN"

        return {
            'label':       label,
            'h_count':     h_count,
            'v_count':     v_count,
            'max_count':   max_count,
            'h_peaks':     h_peaks,
            'v_peaks':     v_peaks,
            'h_profile':   h_smooth,
            'v_profile':   v_smooth,
            'h_grad':      h_grad,
            'v_grad':      v_grad,
        }


    def _compute_confidence(self, fov_type, edge_result, edge_2d,
                             region_result, peak_result=None):
        """
        Compute a 0-1 confidence score for the classification.
        Higher = more certain. Fast computation (~0ms overhead).
        """
        peak_agreement = 0.0     # bonus when peak classifier agrees
        if peak_result:
            if fov_type == "EDGE_FOV"  and peak_result['label'] == "EDGE_LIKELY":
                peak_agreement = 0.15
            elif fov_type == "DIE_FOV" and peak_result['label'] == "DIE_LIKELY":
                peak_agreement = 0.15
            elif peak_result['label'] == "UNCERTAIN":
                peak_agreement = 0.0
            else:               # disagreement
                peak_agreement = -0.10

        if fov_type == "EDGE_FOV":
            # Edge confidence based on signal strength
            max_change = max(edge_result['rising_change'], edge_result['falling_change'])
            intensity_range = max(edge_result['intensity_range'], 1)
            
            # Signal-to-range ratio (how much of the dynamic range the edge covers)
            signal_ratio = min(max_change / intensity_range, 1.0)
            
            # 2D edge strength adds confidence
            sobel_boost = min(edge_2d['edge_strength'] / 0.5, 1.0) * 0.2
            
            # Region consistency: one side should be different from the other
            region_classes = [region_result[k]['classification'] 
                            for k in region_result if k in region_result]
            has_mixed = len(set(region_classes)) > 1
            region_boost = 0.15 if has_mixed else 0.0
            
            confidence = min(signal_ratio * 0.65 + sobel_boost + region_boost
                             + peak_agreement + 0.1, 1.0)
            
        elif fov_type == "DIE_FOV":
            # Confidence = how consistently regions say DIE
            all_classes = [region_result[k]['classification'] for k in region_result]
            die_count = all_classes.count("DIE")
            total = max(len(all_classes), 1)
            
            # Die texture should be high
            avg_texture = np.mean([region_result[k]['texture_score'] for k in region_result])
            texture_factor = min(avg_texture / 100.0, 1.0) * 0.3
            
            # No edge signal = more confident it's DIE
            no_edge_boost = 0.2 if not edge_result['has_edge'] else 0.0
            
            confidence = min((die_count / total) * 0.5 + texture_factor
                             + no_edge_boost + peak_agreement, 1.0)
            
        elif fov_type == "WAFER_FOV":
            # Confidence = how consistently regions say WAFER
            all_classes = [region_result[k]['classification'] for k in region_result]
            wafer_count = all_classes.count("WAFER")
            total = max(len(all_classes), 1)
            
            # Low texture = more confident it's wafer
            avg_texture = np.mean([region_result[k]['texture_score'] for k in region_result])
            low_texture_boost = 0.3 if avg_texture < 30 else 0.1
            
            # No edge signal = more confident
            no_edge_boost = 0.2 if not edge_result['has_edge'] else 0.0
            
            confidence = min((wafer_count / total) * 0.5 + low_texture_boost + no_edge_boost, 1.0)
        else:
            confidence = 0.3  # Unknown type
        
        return round(confidence, 3)

    def _analyze_regions(self, img):
        """Analyze regions in both horizontal and vertical directions"""
        h, w = img.shape

        # Horizontal split (left/center/right) for LEFT/RIGHT edges
        region_w = w // 3
        h_regions = {
            'left': img[:, :region_w],
            'center': img[:, region_w:2*region_w],
            'right': img[:, 2*region_w:]
        }

        # Vertical split (top/center_v/bottom) for TOP/BOTTOM edges
        region_h = h // 3
        v_regions = {
            'top': img[:region_h, :],
            'center_v': img[region_h:2*region_h, :],
            'bottom': img[2*region_h:, :]
        }

        results = {}
        for name, region in h_regions.items():
            results[name] = self._analyze_single_region(region, name)
        for name, region in v_regions.items():
            results[name] = self._analyze_single_region(region, name)

        return results

    def _analyze_single_region(self, img_region, name):
        """Analyze a single region"""
        h, w = img_region.shape
        cfg = self.config

        # Count peaks across multiple scan lines
        scan_positions = np.linspace(h * 0.2, h * 0.8, 5).astype(int)
        all_peaks = []

        for y in scan_positions:
            y_start = max(0, y - 10)
            y_end = min(h, y + 10)
            profile = np.median(img_region[y_start:y_end, :], axis=0)
            gradient = np.abs(np.convolve(profile, self.kernel, mode='same'))
            peaks, _ = find_peaks(gradient, height=cfg.EDGE_THRESHOLD, distance=15)
            all_peaks.append(len(peaks))

        avg_peaks = np.mean(all_peaks)
        mean_intensity = np.mean(img_region)
        std_intensity = np.std(img_region)

        # Texture analysis
        laplacian = cv2.Laplacian(img_region.astype(np.uint8), cv2.CV_64F)
        texture_score = laplacian.var()

        # Edge density
        edges = cv2.Canny(img_region.astype(np.uint8), 30, 100)
        edge_density = np.sum(edges > 0) / edges.size

        # Classification
        if avg_peaks >= 4 and (edge_density > 0.03 or texture_score > 100):
            classification = "DIE"
        elif avg_peaks <= 2 and texture_score < 50 and std_intensity < 30:
            classification = "WAFER"
        elif std_intensity > 40 or edge_density > 0.02:
            classification = "TRANSITION"
        else:
            classification = "WAFER"

        return {
            'name': name,
            'classification': classification,
            'avg_peaks': avg_peaks,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'texture_score': texture_score,
            'edge_density': edge_density
        }

    def visualize(self, result, save_path=None):
        """
        Visualize the classification result.

        Args:
            result: dict returned by classify()
            save_path: Optional path to save the figure
        """
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return

        img = result['image']
        h, w = img.shape
        edge_info = result['edge']
        regions = result['regions']

        # Create visualization
        fig = plt.figure(figsize=(14, 10))

        # === Panel 1: Result Image ===
        ax1 = fig.add_subplot(2, 2, 1)
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scan_axis = edge_info.get('scan_axis', 'horizontal')

        if result['fov_type'] == 'EDGE_FOV':
            # Draw transition zone
            if edge_info['transition_start'] is not None:
                if scan_axis == 'horizontal':
                    cv2.rectangle(
                        result_img,
                        (edge_info['transition_start'], 0),
                        (edge_info['transition_end'], h),
                        (0, 100, 255), 2
                    )
                    cv2.line(
                        result_img,
                        (edge_info['edge_x'], 0),
                        (edge_info['edge_x'], h),
                        (0, 255, 0), 2
                    )
                else:
                    cv2.rectangle(
                        result_img,
                        (0, edge_info['transition_start']),
                        (w, edge_info['transition_end']),
                        (0, 100, 255), 2
                    )
                    cv2.line(
                        result_img,
                        (0, edge_info['edge_x']),
                        (w, edge_info['edge_x']),
                        (0, 255, 0), 2
                    )

        ax1.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"FOV Classification: {result['fov_type']}", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # === Panel 2: Intensity Profile ===
        ax2 = fig.add_subplot(2, 2, 2)
        profile = edge_info['intensity_profile']
        axis_label = "Y Position" if scan_axis == 'vertical' else "X Position"

        ax2.plot(profile, color='blue', linewidth=2, label='Intensity')
        ax2.fill_between(range(len(profile)), profile, alpha=0.3)

        if edge_info['transition_start'] is not None:
            ax2.axvspan(edge_info['transition_start'], edge_info['transition_end'],
                       alpha=0.3, color='orange', label='Transition Zone')
            ax2.axvline(edge_info['edge_x'], color='red', linewidth=2, linestyle='--',
                       label=f"Edge={edge_info['edge_x']}")

        ax2.set_title(f"Intensity Profile — {scan_axis.upper()} (Diff={edge_info['left_right_diff']:.0f})",
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel(axis_label)
        ax2.set_ylabel("Intensity")
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # === Panel 3: Gradient Profile ===
        ax3 = fig.add_subplot(2, 2, 3)
        gradient = edge_info['gradient']

        ax3.plot(gradient, color='green', linewidth=2)
        ax3.fill_between(range(len(gradient)), gradient,
                        where=np.array(gradient) > 0, alpha=0.3, color='green')
        ax3.fill_between(range(len(gradient)), gradient,
                        where=np.array(gradient) < 0, alpha=0.3, color='red')

        ax3.set_title(f"Gradient Profile — {scan_axis.upper()}", fontsize=12, fontweight='bold')
        ax3.set_xlabel(axis_label)
        ax3.set_ylabel("Gradient")
        ax3.grid(True, alpha=0.3)

        # === Panel 4: Summary ===
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        method_str = edge_info['detection_method'] if edge_info['detection_method'] else "N/A"
        edge_pos_str = f"{edge_info['edge_x']} px" if edge_info['edge_x'] else "N/A"

        # Build region lines
        region_text = ""
        for key in ['left', 'center', 'right']:
            if key in regions:
                r = regions[key]
                region_text += f"║    {key.upper():<8} {r['classification']:<10} (peaks={r['avg_peaks']:.1f})    ║\n"
        for key in ['top', 'center_v', 'bottom']:
            if key in regions:
                r = regions[key]
                label = key.upper().replace('CENTER_V', 'CENTER')
                region_text += f"║    {label:<8} {r['classification']:<10} (peaks={r['avg_peaks']:.1f})    ║\n"

        summary = f"""
╔═══════════════════════════════════════════════════════════╗
║              FOV CLASSIFICATION SUMMARY                   ║
╠═══════════════════════════════════════════════════════════╣
║  FOV Type:       {result['fov_type']:<25}        ║
║  Edge Type:      {edge_info['edge_type']:<25}        ║
║  Wafer Side:     {str(edge_info['wafer_side']):<25}        ║
║  Scan Axis:      {scan_axis.upper():<25}        ║
║  Edge Position:  {edge_pos_str:<25}        ║
║  Method:         {method_str:<25}        ║
║                                                           ║
║  Region Classifications:                                  ║
{region_text}╚═══════════════════════════════════════════════════════════╝
        """

        color = 'lightgreen' if result['fov_type'] == 'EDGE_FOV' else 'lightyellow'

        ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

        plt.suptitle(f"FOV Classifier | {result['fov_type']}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        plt.show()

    def print_results(self, result):
        """Print results to console"""
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return

        edge_info = result['edge']
        edge_2d = result.get('edge_2d', {})
        regions = result['regions']
        scan_axis = edge_info.get('scan_axis', 'horizontal')
        confidence = result.get('confidence', 0)

        print("=" * 60)
        print("          FOV CLASSIFICATION RESULTS")
        print("=" * 60)

        print(f"\n📊 FOV Type:    {result['fov_type']}")
        print(f"   Confidence:  {confidence:.0%}")
        print(f"   Edge Type:   {edge_info['edge_type']}")
        print(f"   Wafer Side:  {edge_info['wafer_side']}")
        print(f"   Scan Axis:   {scan_axis.upper()}")
        print(f"   Method:      {edge_info['detection_method']}")

        if edge_2d:
            print(f"   2D Sobel:    strength={edge_2d.get('edge_strength', 0):.2f}, "
                  f"angle={edge_2d.get('edge_angle', 0):.0f}°")

        print(f"\n   Intensity Analysis:")
        print(f"     Range: {edge_info['intensity_range']:.0f}")
        print(f"     Diff:  {edge_info['left_right_diff']:.0f}")
        print(f"     Rise:  {edge_info.get('rising_change', 0):.0f} ({edge_info.get('relative_rising', 0):.0%})")
        print(f"     Fall:  {edge_info.get('falling_change', 0):.0f} ({edge_info.get('relative_falling', 0):.0%})")

        print(f"\n   Region Classifications:")
        for name in ['left', 'center', 'right', 'top', 'center_v', 'bottom']:
            if name in regions:
                data = regions[name]
                label = name.upper().replace('CENTER_V', 'CENTER_V')
                print(f"     [{label:<9}]: {data['classification']} (peaks={data['avg_peaks']:.1f}, tex={data['texture_score']:.0f})")

        print("\n" + "=" * 60)


# ==============================================================================
#                    INTERACTIVE CLASSIFIER VIEWER
# ==============================================================================

class InteractiveClassifierViewer:
    """
    Interactive GUI for FOV classification with Load Image button
    and parameter tuning sliders for real-time threshold adjustment.
    """

    def __init__(self, image_path=None):
        import tkinter as tk
        from tkinter import filedialog
        from matplotlib.widgets import Button, Slider

        self.tk = tk
        self.filedialog = filedialog
        self.Button = Button
        self.Slider = Slider

        # Default config
        self.config = ClassificationConfig()
        self.classifier = FOVClassifier(self.config)

        # Get image path
        if image_path is None:
            image_path = os.path.join(self.config.IMAGE_FOLDER, self.config.DEFAULT_IMAGE)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path = image_path

        # Setup figure
        self.setup_ui()

        # Initial classification
        self.classify_and_display()

    def setup_ui(self):
        """Setup the matplotlib figure with panels, sliders, and buttons"""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('FOV Classifier - Interactive Tuner (Production)', fontsize=14, fontweight='bold')

        # ---- Main display panels (upper portion) ----
        self.ax_img = self.fig.add_axes([0.04, 0.48, 0.40, 0.42])
        self.ax_img.axis('off')
        self.ax_intensity = self.fig.add_axes([0.52, 0.48, 0.44, 0.42])
        self.ax_gradient = self.fig.add_axes([0.04, 0.22, 0.40, 0.24])
        self.ax_summary = self.fig.add_axes([0.52, 0.22, 0.44, 0.24])
        self.ax_summary.axis('off')

        # ---- Sliders (bottom strip) ----
        slider_color = '#e0e0e0'
        sl = 0.18  # left
        sw = 0.28  # width
        sh = 0.013  # height
        sr = sl + sw + 0.10  # right column left

        # Left column: Edge sensitivity
        ax_weak = self.fig.add_axes([sl, 0.175, sw, sh])
        self.sl_weak = self.Slider(ax_weak, 'Weak Edge ΔI',
                                   5, 80, valinit=self.config.MIN_INTENSITY_CHANGE_WEAK,
                                   valstep=1, color=slider_color)
        self.sl_weak.on_changed(self.on_slider_change)

        ax_strong = self.fig.add_axes([sl, 0.155, sw, sh])
        self.sl_strong = self.Slider(ax_strong, 'Strong Edge ΔI',
                                     10, 100, valinit=self.config.MIN_INTENSITY_CHANGE_STRONG,
                                     valstep=1, color=slider_color)
        self.sl_strong.on_changed(self.on_slider_change)

        ax_diff = self.fig.add_axes([sl, 0.135, sw, sh])
        self.sl_diff = self.Slider(ax_diff, 'Quarter Diff',
                                   5, 80, valinit=self.config.MIN_LEFT_RIGHT_DIFF,
                                   valstep=1, color=slider_color)
        self.sl_diff.on_changed(self.on_slider_change)

        # Right column: Length and new params
        ax_long = self.fig.add_axes([sr, 0.175, sw, sh])
        self.sl_long_ratio = self.Slider(ax_long, 'Curved Len %',
                                         1, 40, valinit=int(self.config.MIN_LENGTH_RATIO_LONG * 100),
                                         valstep=1, color=slider_color)
        self.sl_long_ratio.on_changed(self.on_slider_change)

        ax_short = self.fig.add_axes([sr, 0.155, sw, sh])
        self.sl_short_ratio = self.Slider(ax_short, 'Sharp Len %',
                                          1, 20, valinit=int(self.config.MIN_LENGTH_RATIO_SHORT * 100),
                                          valstep=1, color=slider_color)
        self.sl_short_ratio.on_changed(self.on_slider_change)

        ax_clahe = self.fig.add_axes([sr, 0.135, sw, sh])
        self.sl_clahe = self.Slider(ax_clahe, 'CLAHE Clip',
                                    0, 5.0, valinit=self.config.CLAHE_CLIP_LIMIT,
                                    valstep=0.5, color='#d0e8ff')
        self.sl_clahe.on_changed(self.on_slider_change)

        ax_conf = self.fig.add_axes([sl, 0.115, sw, sh])
        self.sl_conf = self.Slider(ax_conf, 'Conf. Thresh',
                                   0.0, 0.8, valinit=self.config.CONFIDENCE_THRESHOLD,
                                   valstep=0.05, color='#ffe0d0')
        self.sl_conf.on_changed(self.on_slider_change)

        ax_die_ovr = self.fig.add_axes([sr, 0.115, sw, sh])
        self.sl_die_override = self.Slider(ax_die_ovr, 'Die Override %',
                                           50, 100, valinit=self.config.DIE_OVERRIDE_PCT,
                                           valstep=1, color='#ffd0d0')
        self.sl_die_override.on_changed(self.on_slider_change)

        # ---- Buttons ----
        load_ax = self.fig.add_axes([0.20, 0.02, 0.13, 0.04])
        self.load_button = self.Button(load_ax, '📁 Load Image', color='lightblue', hovercolor='skyblue')
        self.load_button.on_clicked(self.load_image)

        reset_ax = self.fig.add_axes([0.36, 0.02, 0.13, 0.04])
        self.reset_button = self.Button(reset_ax, '↺ Reset', color='#ffe0e0', hovercolor='#ffb0b0')
        self.reset_button.on_clicked(self.reset_defaults)

        print_ax = self.fig.add_axes([0.52, 0.02, 0.13, 0.04])
        self.print_button = self.Button(print_ax, '📋 Print Config', color='#e0ffe0', hovercolor='#b0ffb0')
        self.print_button.on_clicked(self.print_config)

    def on_slider_change(self, val):
        """Called when any slider changes — update config and re-classify"""
        self.config.MIN_INTENSITY_CHANGE_WEAK = int(self.sl_weak.val)
        self.config.MIN_INTENSITY_CHANGE_STRONG = int(self.sl_strong.val)
        self.config.MIN_LEFT_RIGHT_DIFF = int(self.sl_diff.val)
        self.config.MIN_LENGTH_RATIO_LONG = self.sl_long_ratio.val / 100.0
        self.config.MIN_LENGTH_RATIO_SHORT = self.sl_short_ratio.val / 100.0
        self.config.CLAHE_CLIP_LIMIT = self.sl_clahe.val
        self.config.CONFIDENCE_THRESHOLD = self.sl_conf.val
        self.config.DIE_OVERRIDE_PCT = int(self.sl_die_override.val)

        self.classifier = FOVClassifier(self.config)
        self.classify_and_display()

    def reset_defaults(self, event):
        """Reset all sliders to default values"""
        defaults = ClassificationConfig()
        self.sl_weak.set_val(defaults.MIN_INTENSITY_CHANGE_WEAK)
        self.sl_strong.set_val(defaults.MIN_INTENSITY_CHANGE_STRONG)
        self.sl_diff.set_val(defaults.MIN_LEFT_RIGHT_DIFF)
        self.sl_long_ratio.set_val(int(defaults.MIN_LENGTH_RATIO_LONG * 100))
        self.sl_short_ratio.set_val(int(defaults.MIN_LENGTH_RATIO_SHORT * 100))
        self.sl_clahe.set_val(defaults.CLAHE_CLIP_LIMIT)
        self.sl_conf.set_val(defaults.CONFIDENCE_THRESHOLD)
        self.sl_die_override.set_val(defaults.DIE_OVERRIDE_PCT)

    def print_config(self, event):
        """Print current config values to console"""
        print("\n" + "=" * 55)
        print("  CURRENT CLASSIFICATION PARAMETERS")
        print("=" * 55)
        print(f"  MIN_INTENSITY_CHANGE_WEAK   = {self.config.MIN_INTENSITY_CHANGE_WEAK}")
        print(f"  MIN_INTENSITY_CHANGE_STRONG = {self.config.MIN_INTENSITY_CHANGE_STRONG}")
        print(f"  MIN_LEFT_RIGHT_DIFF         = {self.config.MIN_LEFT_RIGHT_DIFF}")
        print(f"  MIN_LENGTH_RATIO_LONG       = {self.config.MIN_LENGTH_RATIO_LONG:.2f}")
        print(f"  MIN_LENGTH_RATIO_SHORT      = {self.config.MIN_LENGTH_RATIO_SHORT:.2f}")
        print(f"  CLAHE_CLIP_LIMIT            = {self.config.CLAHE_CLIP_LIMIT:.1f}")
        print(f"  CONFIDENCE_THRESHOLD        = {self.config.CONFIDENCE_THRESHOLD:.2f}")
        print(f"  DIE_OVERRIDE_PCT            = {self.config.DIE_OVERRIDE_PCT}")
        print("=" * 55)
        print("  Copy these values to ClassificationConfig to make permanent.")
        print("=" * 55 + "\n")

    def classify_and_display(self):
        """Run classification and update all panels"""
        result = self.classifier.classify(self.image_path)

        img = result['image']
        h, w = img.shape
        edge_info = result['edge']
        edge_2d = result.get('edge_2d', {})
        regions = result['regions']
        confidence = result.get('confidence', 0)

        # Clear all axes
        self.ax_img.clear()
        self.ax_intensity.clear()
        self.ax_gradient.clear()
        self.ax_summary.clear()
        self.ax_summary.axis('off')

        # === Panel 1: Result Image ===
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scan_axis = edge_info.get('scan_axis', 'horizontal')

        if result['fov_type'] == 'EDGE_FOV':
            if edge_info['transition_start'] is not None:
                if scan_axis == 'horizontal':
                    cv2.rectangle(result_img,
                        (edge_info['transition_start'], 0),
                        (edge_info['transition_end'], h),
                        (0, 100, 255), 2)
                    cv2.line(result_img,
                        (edge_info['edge_x'], 0),
                        (edge_info['edge_x'], h),
                        (0, 255, 0), 2)
                else:
                    cv2.rectangle(result_img,
                        (0, edge_info['transition_start']),
                        (w, edge_info['transition_end']),
                        (0, 100, 255), 2)
                    cv2.line(result_img,
                        (0, edge_info['edge_x']),
                        (w, edge_info['edge_x']),
                        (0, 255, 0), 2)
        elif result['fov_type'] == 'DIE_FOV':
            cv2.rectangle(result_img, (3, 3), (w - 3, h - 3), (0, 0, 255), 3)

        self.ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        # Color-coded title with confidence
        fov_type = result['fov_type']
        color_map = {
            'EDGE_FOV': 'green',
            'DIE_FOV': 'red',
            'WAFER_FOV': 'orange',
            'UNCERTAIN': 'purple'
        }
        title_color = color_map.get(fov_type, 'gray')
        conf_pct = f"{confidence*100:.0f}%"

        filename = os.path.basename(self.image_path)
        self.ax_img.set_title(
            f"{fov_type} ({conf_pct}) — {filename}",
            fontsize=10, fontweight='bold', color=title_color)
        self.ax_img.axis('off')

        # === Panel 2: Intensity Profile ===
        profile = edge_info['intensity_profile']
        axis_label = "Y Position" if scan_axis == 'vertical' else "X Position"
        self.ax_intensity.plot(profile, color='blue', linewidth=2, label='Intensity')
        self.ax_intensity.fill_between(range(len(profile)), profile, alpha=0.3)

        if edge_info['transition_start'] is not None:
            self.ax_intensity.axvspan(edge_info['transition_start'], edge_info['transition_end'],
                                     alpha=0.3, color='orange', label='Transition Zone')
            self.ax_intensity.axvline(edge_info['edge_x'], color='red', linewidth=2, linestyle='--',
                                     label=f"Edge={edge_info['edge_x']}")

        r_chg = edge_info.get('rising_change', 0)
        f_chg = edge_info.get('falling_change', 0)
        rel_r = edge_info.get('relative_rising', 0)
        rel_f = edge_info.get('relative_falling', 0)
        self.ax_intensity.set_title(
            f"{scan_axis.upper()} | Diff={edge_info['left_right_diff']:.0f} | "
            f"Rise={r_chg:.0f}({rel_r:.0%}) Fall={f_chg:.0f}({rel_f:.0%})",
            fontsize=9, fontweight='bold')
        self.ax_intensity.set_xlabel(axis_label)
        self.ax_intensity.set_ylabel("Intensity")
        self.ax_intensity.legend(loc='best', fontsize=8)
        self.ax_intensity.grid(True, alpha=0.3)

        # === Panel 3: Gradient Profile ===
        gradient = edge_info['gradient']
        self.ax_gradient.plot(gradient, color='green', linewidth=2)
        self.ax_gradient.fill_between(range(len(gradient)), gradient,
                                      where=np.array(gradient) > 0, alpha=0.3, color='green')
        self.ax_gradient.fill_between(range(len(gradient)), gradient,
                                      where=np.array(gradient) < 0, alpha=0.3, color='red')

        r_ratio = edge_info.get('rising_ratio', 0)
        f_ratio = edge_info.get('falling_ratio', 0)
        self.ax_gradient.set_title(
            f"Gradient | RiseLen={r_ratio:.0%} FallLen={f_ratio:.0%}",
            fontsize=10, fontweight='bold')
        self.ax_gradient.set_xlabel(axis_label)
        self.ax_gradient.set_ylabel("Gradient")
        self.ax_gradient.grid(True, alpha=0.3)

        # === Panel 4: Summary ===
        method_str = edge_info['detection_method'] if edge_info['detection_method'] else "N/A"
        edge_pos_str = f"{edge_info['edge_x']} px" if edge_info['edge_x'] else "N/A"

        # Confidence bar
        conf_bar_len = int(confidence * 20)
        conf_bar = '█' * conf_bar_len + '░' * (20 - conf_bar_len)
        if confidence >= 0.7:
            conf_label = 'HIGH'
        elif confidence >= 0.4:
            conf_label = 'MED'
        else:
            conf_label = 'LOW'

        # Build region summary
        region_lines = "H-Regions:\n"
        for key in ['left', 'center', 'right']:
            if key in regions:
                r = regions[key]
                region_lines += f"  {key.upper():<7} {r['classification']:<10} pk={r['avg_peaks']:.1f}\n"
        region_lines += "V-Regions:\n"
        for key in ['top', 'center_v', 'bottom']:
            if key in regions:
                r = regions[key]
                label = key.upper().replace('CENTER_V', 'MID')
                region_lines += f"  {label:<7} {r['classification']:<10} pk={r['avg_peaks']:.1f}\n"

        # 2D edge info
        edge_2d_str = ""
        if edge_2d:
            e2d_s = edge_2d.get('edge_strength', 0)
            e2d_a = edge_2d.get('edge_angle', 0)
            edge_2d_str = f"2D Sobel:   str={e2d_s:.2f} ang={e2d_a:.0f}°\n"

        summary = (
            f"FOV Type:   {fov_type}\n"
            f"Confidence: {conf_bar} {confidence:.0%} ({conf_label})\n"
            f"Edge Type:  {edge_info['edge_type']}\n"
            f"Scan Axis:  {scan_axis.upper()}\n"
            f"Edge Pos:   {edge_pos_str}\n"
            f"Method:     {method_str}\n"
            f"{edge_2d_str}"
            f"\n{region_lines}"
        )

        box_colors = {
            'EDGE_FOV': 'lightgreen',
            'DIE_FOV': '#ffcccc',
            'WAFER_FOV': 'lightyellow',
            'UNCERTAIN': '#e0d0ff'
        }
        box_color = box_colors.get(fov_type, '#f0f0f0')

        self.ax_summary.text(0.5, 0.5, summary, transform=self.ax_summary.transAxes,
                            fontsize=9, fontfamily='monospace',
                            verticalalignment='center', horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9, pad=0.5))

        self.fig.canvas.draw_idle()

    def load_image(self, event):
        """Open file dialog and load a new image"""
        root = self.tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = self.filedialog.askopenfilename(
            title="Select Wafer Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )

        root.destroy()

        if file_path:
            try:
                self.image_path = file_path
                self.classify_and_display()
                print(f"Loaded image: {file_path}")
            except Exception as e:
                print(f"Error loading image: {e}")

    def show(self):
        """Display the interactive viewer"""
        plt.show()


# ==============================================================================
#                           MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point for the classifier"""

    # Check for command line argument
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = None

    print("=" * 60)
    print("  FOV CLASSIFIER - Interactive Viewer")
    print("=" * 60)
    print("\n  Click 'Load Image' to classify different images")
    print("=" * 60)

    viewer = InteractiveClassifierViewer(image_path)
    viewer.show()

    return viewer


if __name__ == "__main__":
    result = main()

