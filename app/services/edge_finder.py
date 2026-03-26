"""
================================================================================
EDGE LINE FINDER v1.0
================================================================================
Finds the precise edge line in wafer images with sub-pixel precision
using gradient-based detection and RANSAC line fitting.

Author: Auto-generated for Wafer Alignment System
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import random
import os

# Import from classifier module for shared functions
from app.services.fov_classifier import (
    create_gradient_kernel, 
    preprocess_image, 
    FOVClassifier,
    ClassificationConfig
)


# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

class EdgeFinderConfig(ClassificationConfig):
    """Configuration for edge finding parameters (extends classification config)"""

    # --- SCAN DIRECTION ---
    # Options: "LEFT", "RIGHT", "TOP", "BOTTOM"
    # LEFT/RIGHT: Wafer edge is on left/right side (vertical edge line)
    # TOP/BOTTOM: Wafer edge is on top/bottom side (horizontal edge line)
    SCAN_DIRECTION = "LEFT"

    # --- FIND EDGE SETTINGS ---
    NUM_REGIONS = 40            # Number of scan regions
    EDGE_THRESHOLD = 25         # Minimum gradient to count as edge point
    MAX_CLUSTER_GAP = 5         # Max gap between points in same cluster
    BORDER_IGNORE_PCT = 0.02    # Percentage of image dimension to ignore at borders (2%)

    # --- RANSAC LINE FITTING ---
    RANSAC_ITERATIONS = 2000
    RANSAC_THRESHOLD = 5.0      # Inlier distance threshold


# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

def compute_perpendicular_delta(A, B, C, img_h, img_w, scale):
    """
    Computes Delta X and Delta Y using the perpendicular-intersection method,
    matching EmguVision.cs (WEPLEFT_REQ lines 776-787).

    Drops a perpendicular from the image centre to the detected edge line
    (Ax + By + C = 0), finds the exact foot of the perpendicular, then
    returns the offset relative to the image centre IN ORIGINAL-IMAGE PIXELS.

    Args:
        A, B, C : line equation coefficients  (Ax + By + C = 0)
                  Note: (A, B) need not be a unit vector here.
        img_h   : height of the *downsampled* image (pixels)
        img_w   : width  of the *downsampled* image (pixels)
        scale   : resize factor applied by preprocess_image()  (< 1 for large images)

    Returns:
        (delta_x_orig, delta_y_orig, intercept_x_orig, intercept_y_orig)
        All values are in original-resolution pixel coordinates.
    """
    # Image centre in downsampled space
    cx = img_w / 2.0
    cy = img_h / 2.0

    # Signed distance from centre to line (may be zero if centre is on line)
    denom = A * A + B * B
    if denom < 1e-10:
        return 0.0, 0.0, cx / scale, cy / scale

    # Foot of perpendicular  (foundInterceptPoint in EmguVision.cs)
    t = (A * cx + B * cy + C) / denom
    px = cx - A * t   # intercept X in downsampled coords
    py = cy - B * t   # intercept Y in downsampled coords

    # Convert to original-image pixels
    px_orig = px / scale
    py_orig = py / scale
    cx_orig = cx / scale
    cy_orig = cy / scale

    delta_x = px_orig - cx_orig   # dFndXOffset in EmguVision.cs
    delta_y = py_orig - cy_orig   # dFndYOffset in EmguVision.cs

    return delta_x, delta_y, px_orig, py_orig


def refine_peak_subpixel(y_data, peak_idx):
    """
    Refines integer peak position to sub-pixel using parabola fitting.

    Fits a parabola to 3 points (peak-1, peak, peak+1) and finds the
    true vertex position for sub-pixel accuracy.

    Args:
        y_data: 1D array of values
        peak_idx: Integer index of detected peak

    Returns:
        Float sub-pixel peak position
    """
    if peak_idx <= 0 or peak_idx >= len(y_data) - 1:
        return float(peak_idx)

    left = float(y_data[peak_idx - 1])
    center = float(y_data[peak_idx])
    right = float(y_data[peak_idx + 1])

    # Parabola vertex formula
    denom = left - 2 * center + right

    if abs(denom) < 1e-6:
        return float(peak_idx)

    offset = (left - right) / (2 * denom)

    # Constrain offset to reasonable bounds
    if offset < -0.5 or offset > 0.5:
        return float(peak_idx)

    return peak_idx + offset


def fit_line_ransac(points, iterations=1000, threshold=5.0):
    """
    Fits a line to points using RANSAC algorithm.

    Args:
        points: List of (x, y) tuples
        iterations: Number of RANSAC iterations
        threshold: Inlier distance threshold

    Returns:
        Tuple of (line_params, inlier_points)
        line_params: (A, B, C) for line equation Ax + By + C = 0
        inlier_points: numpy array of inlier coordinates
    """
    best_line = None
    best_inliers = []
    max_count = 0
    count = len(points)

    if count < 2:
        return None, []

    points_array = np.array(points)

    for _ in range(iterations):
        # Random sample 2 points
        idx1, idx2 = random.sample(range(count), 2)
        p1, p2 = points_array[idx1], points_array[idx2]

        # Line equation: Ax + By + C = 0
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = -A * p1[0] - B * p1[1]

        norm = np.sqrt(A*A + B*B)
        if norm == 0:
            continue

        # Calculate distances to line
        distances = np.abs(A * points_array[:, 0] + B * points_array[:, 1] + C) / norm

        # Count inliers
        inlier_mask = distances < threshold
        current_inliers = points_array[inlier_mask]

        if len(current_inliers) > max_count:
            max_count = len(current_inliers)
            best_inliers = current_inliers
            best_line = (A, B, C)

    return best_line, best_inliers


# ==============================================================================
#                    EDGE LINE FINDER CLASS
# ==============================================================================

class EdgeLineFinder:
    """
    Finds the precise edge line using gradient-based detection
    and RANSAC line fitting.

    Usage:
        finder = EdgeLineFinder()
        result = finder.find_edge(image_path)
        finder.visualize(result)
    """

    def __init__(self, config=None):
        self.config = config or EdgeFinderConfig()
        self.kernel = create_gradient_kernel(self.config.KERNEL_SIZE)
        self.classifier = FOVClassifier(self.config)

    def find_edge(self, img_or_path, edge_info=None, skip_classification=False):
        """
        Find the edge line in the image.

        Args:
            img_or_path: Either a numpy array (grayscale image) or path to image file
            edge_info: Optional dict from FOVClassifier with edge hints
            skip_classification: If True, skip FOV classification (used by param_tuner)

        Returns:
            dict with edge line parameters and detected points
        """
        # Preprocess image
        img, original_img, scale = preprocess_image(
            img_or_path,
            self.config.TARGET_PROCESS_DIM
        )

        h, w = img.shape
        cfg = self.config
        direction = cfg.SCAN_DIRECTION.upper()

        # Determine if we're scanning for vertical or horizontal edge
        is_vertical_edge = direction in ["LEFT", "RIGHT"]

        # Skip classification if requested (param_tuner mode)
        if skip_classification:
            fov_type = "EDGE_FOV"
            edge_info = {'has_edge': True, 'wafer_side': direction}
        else:
            # If no edge_info provided, classify first
            if edge_info is None:
                classification = self.classifier.classify(img)
                edge_info = classification['edge']
                fov_type = classification['fov_type']
            else:
                fov_type = "EDGE_FOV" if edge_info.get('has_edge') else "UNKNOWN"

            # Check if this is an edge FOV
            if fov_type != "EDGE_FOV":
                return {
                    'success': False,
                    'reason': f'Not an edge FOV (detected: {fov_type})',
                    'fov_type': fov_type,
                    'image': img,
                    'original_image': original_img,
                    'scale': scale,
                    'scan_direction': direction
                }

        detected_points = []
        region_data = []

        if is_vertical_edge:
            # LEFT/RIGHT: Horizontal regions, scan along X-axis.
            # border_ignore is driven purely by BORDER_IGNORE_PCT.
            # Kernel size NO LONGER shrinks the search region — mode='same'
            # convolution is valid at every position.
            border_ignore = int(w * cfg.BORDER_IGNORE_PCT)
            region_size = h // cfg.NUM_REGIONS

            for i in range(cfg.NUM_REGIONS):
                region_start = i * region_size
                region_end = (i + 1) * region_size if i < cfg.NUM_REGIONS - 1 else h

                # Get region profile (horizontal slice, median along Y)
                region = img[region_start:region_end, :]
                profile = np.median(region, axis=0)

                # Calculate gradient
                gradient = np.convolve(profile, self.kernel, mode='same')
                abs_gradient = np.abs(gradient)

                # Clean borders
                abs_gradient[:border_ignore] = 0
                abs_gradient[-border_ignore:] = 0

                # Detect edge point
                result = self._detect_edge_point_horizontal(
                    abs_gradient, profile, region_start, region_size, edge_info
                )

                region_data.append({
                    'region_start': region_start,
                    'region_end': region_end,
                    'profile': profile,
                    'gradient': abs_gradient,
                    'edge_point': result
                })

                if result['found']:
                    detected_points.append((result['x'], result['y']))

            # Fit vertical line
            line_result = self._fit_edge_line_vertical(detected_points, h, w)

        else:
            # TOP/BOTTOM: Vertical regions, scan along Y-axis.
            border_ignore = int(h * cfg.BORDER_IGNORE_PCT)
            region_size = w // cfg.NUM_REGIONS

            for i in range(cfg.NUM_REGIONS):
                region_start = i * region_size
                region_end = (i + 1) * region_size if i < cfg.NUM_REGIONS - 1 else w

                # Get region profile (vertical slice, median along X)
                region = img[:, region_start:region_end]
                profile = np.median(region, axis=1)

                # Calculate gradient
                gradient = np.convolve(profile, self.kernel, mode='same')
                abs_gradient = np.abs(gradient)

                # Clean borders
                abs_gradient[:border_ignore] = 0
                abs_gradient[-border_ignore:] = 0

                # Detect edge point
                result = self._detect_edge_point_vertical(
                    abs_gradient, profile, region_start, region_size, edge_info
                )

                region_data.append({
                    'region_start': region_start,
                    'region_end': region_end,
                    'profile': profile,
                    'gradient': abs_gradient,
                    'edge_point': result
                })

                if result['found']:
                    detected_points.append((result['x'], result['y']))

            # Fit horizontal line
            line_result = self._fit_edge_line_horizontal(detected_points, h, w)

        # --- Perpendicular-intersection delta (EmguVision.cs method) ---
        delta_x, delta_y, intercept_x, intercept_y = 0.0, 0.0, 0.0, 0.0
        if line_result.get('success'):
            lp = line_result['line_params']
            delta_x, delta_y, intercept_x, intercept_y = compute_perpendicular_delta(
                lp['a'], lp['b'], lp['c'], h, w, scale
            )

        return {
            'success': line_result['success'],
            'line_params': line_result.get('line_params'),
            'line_endpoints': line_result.get('endpoints'),
            'slope': line_result.get('slope'),
            'intercept_c': line_result.get('intercept_c'),
            'delta_x': delta_x,
            'delta_y': delta_y,
            'intercept_point': {'x': intercept_x, 'y': intercept_y},
            'detected_points': detected_points,
            'inliers': line_result.get('inliers', []),
            'num_points': len(detected_points),
            'num_inliers': len(line_result.get('inliers', [])),
            'region_data': region_data,
            'edge_info': edge_info,
            'fov_type': fov_type,
            'image': img,
            'original_image': original_img,
            'scale': scale,
            'scan_direction': direction,
            'is_vertical_edge': is_vertical_edge,
            'reason': line_result.get('reason')
        }

    def _detect_edge_point_horizontal(self, abs_gradient, profile, y_start, region_h, edge_info=None):
        """Detect edge point in a horizontal region (for LEFT/RIGHT edge detection)"""
        cfg = self.config

        # Find potential edge indices
        potential_indices = np.where(abs_gradient > cfg.EDGE_THRESHOLD)[0]

        if len(potential_indices) == 0:
            return {'found': False, 'x': None, 'y': None}

        # Find first cluster (edge vs noise)
        jumps = np.where(np.diff(potential_indices) > cfg.MAX_CLUSTER_GAP)[0]

        if len(jumps) > 0:
            # Use edge_info to determine which cluster to use
            if edge_info and edge_info.get('wafer_side') == 'LEFT':
                # Wafer is on RIGHT, edge is on LEFT, so we want FIRST cluster from left
                edge_cluster = potential_indices[:jumps[0]+1]
            elif edge_info and edge_info.get('wafer_side') == 'RIGHT':
                # Wafer is on LEFT, edge is on RIGHT, so we want LAST cluster from left
                edge_cluster = potential_indices[jumps[-1]+1:]
            else:
                # Default: first cluster
                edge_cluster = potential_indices[:jumps[0]+1]
        else:
            edge_cluster = potential_indices

        if len(edge_cluster) == 0:
            return {'found': False, 'x': None, 'y': None}

        # Find peak within cluster
        relative_idx = np.argmax(abs_gradient[edge_cluster])
        peak_int_x = edge_cluster[relative_idx]

        # Sub-pixel refinement
        x_subpixel = refine_peak_subpixel(abs_gradient, peak_int_x)
        y = y_start + (region_h // 2)

        return {
            'found': True,
            'x': x_subpixel,
            'y': y,
            'x_int': peak_int_x,
            'gradient_value': abs_gradient[peak_int_x]
        }

    def _detect_edge_point_vertical(self, abs_gradient, profile, x_start, region_w, edge_info=None):
        """Detect edge point in a vertical region (for TOP/BOTTOM edge detection)"""
        cfg = self.config

        # Find potential edge indices
        potential_indices = np.where(abs_gradient > cfg.EDGE_THRESHOLD)[0]

        if len(potential_indices) == 0:
            return {'found': False, 'x': None, 'y': None}

        # Find cluster based on direction (edge vs noise)
        jumps = np.where(np.diff(potential_indices) > cfg.MAX_CLUSTER_GAP)[0]

        if len(jumps) > 0:
            if edge_info and edge_info.get('wafer_side') == 'BOTTOM':
                # Wafer is on TOP, edge is on BOTTOM, so we want LAST cluster
                edge_cluster = potential_indices[jumps[-1]+1:]
            else:
                # TOP direction or default: edge is on TOP, so first cluster
                edge_cluster = potential_indices[:jumps[0]+1]
        else:
            edge_cluster = potential_indices

        if len(edge_cluster) == 0:
            return {'found': False, 'x': None, 'y': None}

        # Find peak within cluster
        relative_idx = np.argmax(abs_gradient[edge_cluster])
        peak_int_y = edge_cluster[relative_idx]

        # Sub-pixel refinement
        y_subpixel = refine_peak_subpixel(abs_gradient, peak_int_y)
        x = x_start + (region_w // 2)

        return {
            'found': True,
            'x': x,
            'y': y_subpixel,
            'y_int': peak_int_y,
            'gradient_value': abs_gradient[peak_int_y]
        }

    def _fit_edge_line_vertical(self, points, img_height, img_width):
        """Fit a vertical line to detected points (for LEFT/RIGHT edge detection)"""
        cfg = self.config

        if len(points) < 3:
            return {'success': False, 'reason': 'Not enough points', "delta_x": 0, "delta_y": 0}

        _, inliers = fit_line_ransac(
            points,
            cfg.RANSAC_ITERATIONS,
            cfg.RANSAC_THRESHOLD
        )

        if len(inliers) < 3:
            return {'success': False, 'reason': 'RANSAC rejected points'}

        # Fit final line with cv2.fitLine
        inliers_reshaped = np.array(inliers).reshape((-1, 1, 2)).astype(np.float32)
        line_result = cv2.fitLine(inliers_reshaped, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extract scalar values
        vx = float(line_result[0].item())
        vy = float(line_result[1].item())
        x0 = float(line_result[2].item())
        y0 = float(line_result[3].item())

        if abs(vy) < 1e-5:
            # Nearly horizontal line - just use mean x
            x_mean = int(np.mean([p[0] for p in inliers]))
            x_top = x_mean
            x_bot = x_mean
        else:
            # Calculate line endpoints at top (y=0) and bottom (y=height)
            x_top = int(x0 + (0 - y0) * (vx/vy))
            x_bot = int(x0 + (img_height - y0) * (vx/vy))

        # Calculate line equation: ax + by + c = 0
        a = vy
        b = -vx
        c = -vy * x0 + vx * y0

        # --- Slope and Y-intercept of the edge line ---
        # For a vertical-ish edge: slope = vx/vy  (run/rise),  intercept_c = x0 - slope*y0
        slope = (vx / vy) if abs(vy) > 1e-5 else float('inf')
        intercept_c = x0 - slope * y0 if abs(vy) > 1e-5 else x0

        return {
            'success': True,
            'line_params': {'a': a, 'b': b, 'c': c, 'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0},
            'endpoints': {'x_top': x_top, 'x_bot': x_bot},
            'slope': slope,
            'intercept_c': intercept_c,
            'inliers': inliers
        }

    def _fit_edge_line_horizontal(self, points, img_height, img_width):
        """Fit a horizontal line to detected points (for TOP/BOTTOM edge detection)"""
        cfg = self.config

        if len(points) < 3:
            return {'success': False, 'reason': 'Not enough points', "delta_x": 0, "delta_y": 0}

        _, inliers = fit_line_ransac(
            points,
            cfg.RANSAC_ITERATIONS,
            cfg.RANSAC_THRESHOLD
        )

        if len(inliers) < 3:
            return {'success': False, 'reason': 'RANSAC rejected points'}

        # Fit final line with cv2.fitLine
        inliers_reshaped = np.array(inliers).reshape((-1, 1, 2)).astype(np.float32)
        line_result = cv2.fitLine(inliers_reshaped, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extract scalar values
        vx = float(line_result[0].item())
        vy = float(line_result[1].item())
        x0 = float(line_result[2].item())
        y0 = float(line_result[3].item())

        if abs(vx) < 1e-5:
            # Nearly vertical line - just use mean y
            y_mean = int(np.mean([p[1] for p in inliers]))
            y_left = y_mean
            y_right = y_mean
        else:
            # Calculate line endpoints at left (x=0) and right (x=width)
            y_left = int(y0 + (0 - x0) * (vy/vx))
            y_right = int(y0 + (img_width - x0) * (vy/vx))

        # Calculate line equation: ax + by + c = 0
        a = vy
        b = -vx
        c = -vy * x0 + vx * y0

        # --- Slope and Y-intercept of the edge line ---
        # For a horizontal-ish edge: slope = vy/vx,  intercept_c = y0 - slope*x0
        slope = (vy / vx) if abs(vx) > 1e-5 else float('inf')
        intercept_c = y0 - slope * x0 if abs(vx) > 1e-5 else y0

        return {
            'success': True,
            'line_params': {'a': a, 'b': b, 'c': c, 'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0},
            'endpoints': {'y_left': y_left, 'y_right': y_right},
            'slope': slope,
            'intercept_c': intercept_c,
            'inliers': inliers
        }

    def visualize(self, result, save_path=None):
        """
        Visualize the edge finding result.

        Args:
            result: dict returned by find_edge()
            save_path: Optional path to save the figure
        """
        if not result.get('image') is not None:
            print("No image data available")
            return

        img = result['image']
        h, w = img.shape

        # Create visualization
        fig = plt.figure(figsize=(14, 10))

        # === Panel 1: Result Image ===
        ax1 = fig.add_subplot(2, 2, 1)
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if result['success']:
            endpoints = result['line_endpoints']
            
            # Draw edge line
            cv2.line(
                result_img,
                (endpoints['x_top'], 0),
                (endpoints['x_bot'], h),
                (0, 255, 0),  # Green
                3
            )

            # Draw detected points
            for point in result['detected_points']:
                cv2.circle(result_img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

            # Draw inliers
            for point in result['inliers']:
                cv2.circle(result_img, (int(point[0]), int(point[1])), 5, (255, 255, 0), 2)

        ax1.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        status = "Edge Found" if result['success'] else f"Failed: {result.get('reason', 'Unknown')}"
        if result['success']:
            status += f" ({result['num_inliers']} inliers)"

        ax1.set_title(f"Edge Detection: {status}", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # === Panel 2: Gradient from middle region ===
        ax2 = fig.add_subplot(2, 2, 2)

        if result.get('region_data'):
            mid_idx = len(result['region_data']) // 2
            mid_region = result['region_data'][mid_idx]
            gradient = mid_region['gradient']

            ax2.plot(gradient, color='green', linewidth=2)
            ax2.fill_between(range(len(gradient)), gradient, alpha=0.3, color='green')
            ax2.axhline(self.config.EDGE_THRESHOLD, color='orange', linestyle='--',
                       label=f'Threshold ({self.config.EDGE_THRESHOLD})')

            if mid_region['edge_point']['found']:
                x = mid_region['edge_point']['x_int']
                ax2.axvline(x, color='red', linewidth=2, linestyle='--',
                           label=f"Peak @ x={mid_region['edge_point']['x']:.2f}")

            ax2.legend(loc='best', fontsize=9)

        ax2.set_title("Gradient Profile (Middle Region)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Gradient")
        ax2.grid(True, alpha=0.3)

        # === Panel 3: All detected points ===
        ax3 = fig.add_subplot(2, 2, 3)

        if result.get('detected_points'):
            points = np.array(result['detected_points'])
            ax3.scatter(points[:, 0], points[:, 1], c='red', s=20, label='All Points', alpha=0.6)

            if len(result.get('inliers', [])) > 0:
                inliers = np.array(result['inliers'])
                ax3.scatter(inliers[:, 0], inliers[:, 1], c='green', s=40, 
                           label='Inliers', marker='o', edgecolors='black')

            if result['success']:
                endpoints = result['line_endpoints']
                ax3.plot([endpoints['x_top'], endpoints['x_bot']], [0, h], 
                        'b-', linewidth=2, label='Fitted Line')

        ax3.set_xlim(0, w)
        ax3.set_ylim(h, 0)  # Invert Y axis to match image coordinates
        ax3.set_title(f"Detected Points ({result['num_points']} total, {result['num_inliers']} inliers)",
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel("X Position")
        ax3.set_ylabel("Y Position")
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # === Panel 4: Summary ===
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        if result['success']:
            endpoints = result['line_endpoints']
            line_params = result['line_params']
            line_str = f"X: {endpoints['x_top']} (top) â†’ {endpoints['x_bot']} (bot)"
            angle = np.degrees(np.arctan2(line_params['vx'], line_params['vy']))
            angle_str = f"{angle:.2f}Â°"
        else:
            line_str = "N/A"
            angle_str = "N/A"

        summary = f"""
        EDGE LINE FINDER SUMMARY                     
        Status:         {"âœ“ SUCCESS" if result['success'] else " FAILED":<30}  
        FOV Type:       {result.get('fov_type', 'Unknown'):<30} 
        Detection:                                              
        Points Found:  {result['num_points']:<28}   
        Inliers:       {result['num_inliers']:<28}  
        Line:                                                  
        Endpoints:     {line_str:<28}  
        Angle:         {angle_str:<28}   
        """

        color = 'lightgreen' if result['success'] else 'lightyellow'

        ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

        plt.suptitle("Edge Line Finder Results", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        plt.show()

    def print_results(self, result):
        """Print results to console"""
        print("=" * 60)
        print("          EDGE LINE FINDER RESULTS")
        print("=" * 60)

        if result['success']:
            print(f"\nâœ“ Edge line found successfully!")
            print(f"   Points Detected: {result['num_points']}")
            print(f"   Inliers: {result['num_inliers']}")
            
            endpoints = result['line_endpoints']
            print(f"   Line X: {endpoints['x_top']} (top) â†’ {endpoints['x_bot']} (bottom)")
            
            line_params = result['line_params']
            angle = np.degrees(np.arctan2(line_params['vx'], line_params['vy']))
            print(f"   Angle: {angle:.2f}Â°")
        else:
            print(f"\nâœ— Edge line finding failed")
            print(f"   Reason: {result.get('reason', 'Unknown')}")
            print(f"   FOV Type: {result.get('fov_type', 'Unknown')}")

        print("\n" + "=" * 60)


# ==============================================================================
#                           MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point for the edge finder"""

    # Create configuration
    config = EdgeFinderConfig()

    # Build image path
    image_path = os.path.join(config.IMAGE_FOLDER, config.DEFAULT_IMAGE)

    print(f"Processing: {image_path}")

    # Create edge finder
    finder = EdgeLineFinder(config)

    # Find edge
    result = finder.find_edge(image_path)

    # Print results
    finder.print_results(result)

    # Visualize
    finder.visualize(result)

    return result


if __name__ == "__main__":
    result = main()

