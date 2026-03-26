"""
================================================================================
FEATURE-BASED PATTERN MATCHER
================================================================================
Structure and Texture Aware template matching for wafer fiducial alignment.
Uses keypoint detection, contour shape analysis, and homography for robust
rotation/scale invariant pattern matching.

Key Features:
- Keypoint detection (ORB, AKAZE, SIFT, or LINE-MOD)
- Shape-aware contour matching (Hu Moments + cv2.matchShapes)
- HYBRID mode: fuses shape + keypoint results for maximum robustness
- True rotation invariance (no pre-generated templates needed)
- True scale invariance
- Returns exact rotation angle, scale factor, and position
- Works with partial visibility
- LINE-MOD gradient-based matching for small/simple templates

Usage:
    python feature_matcher.py

Author: Adrain Lim
================================================================================
"""

import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import tkinter as tk
from tkinter import filedialog
from linemod_matcher import LinemodMatcher, LinemodConfig


class FeatureMatcherConfig:
    """Configuration for feature-based pattern matching"""
    
    # Feature detector settings
    DETECTOR_TYPE = "ORB"       # Options: "ORB", "AKAZE", "SIFT", "LINEMOD"
    MAX_FEATURES = 500          # Maximum number of features to detect
    
    # Matching settings
    MATCH_RATIO = 0.75          # Lowe's ratio test threshold
    MIN_MATCHES = 10            # Minimum good matches required
    
    # RANSAC settings
    RANSAC_REPROJ_THRESH = 5.0  # RANSAC reprojection threshold
    
    # Visualization
    DRAW_MATCHES = True         # Draw keypoint matches


class FeatureBasedMatcher:
    """
    Feature-based pattern matcher using keypoint detection.
    
    Provides true rotation and scale invariance through:
    - Edge-based descriptors (ORB/AKAZE/SIFT)
    - Geometric hashing via feature matching
    - Affine transformation estimation via homography
    """
    
    def __init__(self, config=None):
        """
        Initialize the feature matcher.
        
        Args:
            config: FeatureMatcherConfig object
        """
        self.config = config or FeatureMatcherConfig()
        self.template_img = None
        self.template_keypoints = None
        self.template_descriptors = None
        
        # Initialize feature detector
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the feature detector based on config"""
        detector_type = self.config.DETECTOR_TYPE.upper()
        
        if detector_type == "ORB":
            self.detector = cv2.ORB_create(
                nfeatures=self.config.MAX_FEATURES,
                scaleFactor=1.1,        # Finer scale pyramid (default: 1.2)
                nlevels=16,             # More pyramid levels (default: 8)
                edgeThreshold=10,       # Allow keypoints closer to edges (default: 31)
                patchSize=15,           # Smaller patch for small templates (default: 31)
                fastThreshold=10        # Less strict corner detection (default: 20)
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            print(f"[INFO] Using ORB detector ({self.config.MAX_FEATURES} features, tuned for small templates)")
            
        elif detector_type == "AKAZE":
            self.detector = cv2.AKAZE_create(
                threshold=0.0001,       # Detect weaker features (default: 0.001)
                nOctaves=8,             # More scale octaves (default: 4)
                nOctaveLayers=8         # More sub-levels per octave (default: 4)
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            print(f"[INFO] Using AKAZE detector (tuned for more keypoints)")
            
        elif detector_type == "SIFT":
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.config.MAX_FEATURES)
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                print(f"[INFO] Using SIFT detector ({self.config.MAX_FEATURES} features)")
            except AttributeError:
                print("[WARN] SIFT not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.config.MAX_FEATURES)
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def load_template(self, template_path_or_image):
        """
        Load and process a template image.
        
        Args:
            template_path_or_image: Path to template image or numpy array
            
        Returns:
            Number of keypoints detected
        """
        # Load image
        if isinstance(template_path_or_image, str):
            if not os.path.exists(template_path_or_image):
                raise FileNotFoundError(f"Template not found: {template_path_or_image}")
            self.template_img = cv2.imread(template_path_or_image, cv2.IMREAD_GRAYSCALE)
        else:
            if len(template_path_or_image.shape) == 3:
                self.template_img = cv2.cvtColor(template_path_or_image, cv2.COLOR_BGR2GRAY)
            else:
                self.template_img = template_path_or_image.copy()
        
        # Store original size
        orig_h, orig_w = self.template_img.shape
        
        # Auto-upscale small templates for better keypoint detection
        MIN_SIZE = 100  # Minimum dimension for reliable detection
        self.template_scale = 1.0
        
        if orig_w < MIN_SIZE or orig_h < MIN_SIZE:
            # Calculate scale factor to reach minimum size
            scale_factor = max(MIN_SIZE / orig_w, MIN_SIZE / orig_h)
            scale_factor = min(scale_factor, 3.0)  # Cap at 3x to avoid over-scaling
            self.template_scale = scale_factor
            
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            self.template_img = cv2.resize(self.template_img, (new_w, new_h), 
                                          interpolation=cv2.INTER_CUBIC)
            print(f"[INFO] Template upscaled {scale_factor:.1f}x ({orig_w}x{orig_h} → {new_w}x{new_h})")
        
        # Apply CLAHE to enhance contrast for better keypoint detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(self.template_img)
        
        # Detect keypoints and compute descriptors on enhanced image
        self.template_keypoints, self.template_descriptors = self.detector.detectAndCompute(
            enhanced_img, None
        )
        
        num_kp = len(self.template_keypoints) if self.template_keypoints else 0
        print(f"[INFO] Template: {self.template_img.shape[1]}x{self.template_img.shape[0]}, "
              f"{num_kp} keypoints detected")
        
        if num_kp == 0:
            print("[WARN] No keypoints found! Try:")
            print("       - Select a larger region with more texture/edges")
            print("       - Try AKAZE or SIFT detector")
        
        return num_kp
    
    def match(self, search_img, find_multiple=False):
        """
        Find template matches in the search image.
        
        Args:
            search_img: Image to search in (grayscale or BGR)
            find_multiple: If True, attempt to find multiple instances
            
        Returns:
            Dictionary with match results:
                - success: True if match found
                - position: (x, y) center of matched region
                - angle: Rotation angle in degrees
                - scale: Scale factor
                - corners: 4 corner points of matched region
                - homography: 3x3 transformation matrix
                - num_matches: Number of good matches
                - inliers: Number of inlier matches
        """
        if self.template_descriptors is None or len(self.template_descriptors) == 0:
            return {
                'success': False,
                'reason': "No keypoints in template - select different region",
                'num_matches': 0
            }
        
        # Convert to grayscale if needed
        if len(search_img.shape) == 3:
            search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_img
        
        # Apply CLAHE to enhance contrast (same as template)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        search_enhanced = clahe.apply(search_gray)
        
        # Detect keypoints in enhanced search image
        search_kp, search_desc = self.detector.detectAndCompute(search_enhanced, None)
        
        if search_desc is None or len(search_kp) < self.config.MIN_MATCHES:
            return {
                'success': False,
                'reason': f"Not enough keypoints in search image ({len(search_kp) if search_kp else 0})"
            }
        
        # Match descriptors using KNN
        try:
            matches = self.matcher.knnMatch(self.template_descriptors, search_desc, k=2)
        except cv2.error as e:
            return {'success': False, 'reason': f"Matching failed: {e}"}
        
        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.config.MATCH_RATIO * n.distance:
                    good_matches.append(m)
        
        print(f"[INFO] Found {len(good_matches)} good matches (need {self.config.MIN_MATCHES})")
        
        if len(good_matches) < self.config.MIN_MATCHES:
            return {
                'success': False,
                'reason': f"Not enough matches ({len(good_matches)}/{self.config.MIN_MATCHES})",
                'num_matches': len(good_matches)
            }
        
        # Extract matched keypoint locations
        src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([search_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.config.RANSAC_REPROJ_THRESH)
        
        if H is None:
            return {
                'success': False,
                'reason': "Homography estimation failed",
                'num_matches': len(good_matches)
            }
        
        # Count inliers
        inliers = np.sum(mask)
        
        # Get template corners
        h, w = self.template_img.shape
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to search image
        matched_corners = cv2.perspectiveTransform(template_corners, H)
        
        # Calculate center position
        center = np.mean(matched_corners, axis=0)[0]
        
        # Extract rotation, scale from homography
        angle, scale = self._decompose_homography(H)
        
        return {
            'success': True,
            'position': (int(center[0]), int(center[1])),
            'angle': angle,
            'scale': scale,
            'corners': matched_corners.reshape(-1, 2).astype(int),
            'homography': H,
            'num_matches': len(good_matches),
            'inliers': int(inliers),
            'good_matches': good_matches,
            'template_kp': self.template_keypoints,
            'search_kp': search_kp,
            'mask': mask
        }
    
    def _decompose_homography(self, H):
        """
        Extract rotation angle and scale from homography matrix.
        
        Args:
            H: 3x3 homography matrix
            
        Returns:
            (angle_degrees, scale)
        """
        # Extract the upper-left 2x2 (rotation + scale)
        a = H[0, 0]
        b = H[0, 1]
        c = H[1, 0]
        d = H[1, 1]
        
        # Calculate scale (average of x and y scales)
        scale_x = math.sqrt(a**2 + c**2)
        scale_y = math.sqrt(b**2 + d**2)
        scale = (scale_x + scale_y) / 2
        
        # Calculate rotation angle
        angle_rad = math.atan2(c, a)
        angle_deg = math.degrees(angle_rad)
        
        return round(angle_deg, 2), round(scale, 3)
    
    def visualize_match(self, search_img, result, show=True):
        """
        Visualize the match result.
        
        Args:
            search_img: Original search image
            result: Match result dictionary
            show: If True, display the image
            
        Returns:
            Annotated image
        """
        if len(search_img.shape) == 2:
            result_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        else:
            result_img = search_img.copy()
        
        if not result['success']:
            cv2.putText(result_img, f"No match: {result.get('reason', 'Unknown')}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Draw bounding polygon
            corners = result['corners']
            cv2.polylines(result_img, [corners], True, (0, 255, 0), 3)
            
            # Draw center cross
            cx, cy = result['position']
            cv2.drawMarker(result_img, (cx, cy), (0, 255, 255), 
                          cv2.MARKER_CROSS, 30, 3)
            
            # Draw rotation indicator
            angle_rad = math.radians(result['angle'])
            dx = int(50 * math.cos(angle_rad))
            dy = int(50 * math.sin(angle_rad))
            cv2.arrowedLine(result_img, (cx, cy), (cx + dx, cy + dy),
                           (255, 0, 255), 3, tipLength=0.3)
            
            # Add text info
            info1 = f"Position: ({cx}, {cy})"
            info2 = f"Angle: {result['angle']:.1f} deg"
            info3 = f"Scale: {result['scale']:.3f}"
            info4 = f"Matches: {result['inliers']}/{result['num_matches']}"
            
            y = 30
            for info in [info1, info2, info3, info4]:
                cv2.putText(result_img, info, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25
        
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title("Feature-Based Match Result")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return result_img
    
    def visualize_keypoints(self, show=True):
        """Visualize template keypoints"""
        if self.template_img is None:
            print("No template loaded.")
            return None
        
        img = cv2.drawKeypoints(self.template_img, self.template_keypoints, None,
                               color=(0, 255, 0), 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Template Keypoints ({len(self.template_keypoints)})")
            plt.axis('off')
            plt.show()
        
        return img


class ShapeMatcher:
    """
    Shape-based pattern matcher using contour analysis.
    
    Provides shape-aware matching through:
    - Canny edge detection + contour extraction
    - Hu Moments for rotation/scale invariant shape comparison
    - cv2.matchShapes for contour similarity scoring
    - Multi-angle contour matching for angle estimation
    """
    
    def __init__(self, canny_low=50, canny_high=150, min_contour_area=100,
                 angle_step=5, match_method=cv2.CONTOURS_MATCH_I1):
        """
        Initialize the shape matcher.
        
        Args:
            canny_low: Canny edge lower threshold
            canny_high: Canny edge upper threshold
            min_contour_area: Minimum contour area to consider
            angle_step: Angle step for multi-angle matching (degrees)
            match_method: cv2.matchShapes comparison method
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.angle_step = angle_step
        self.match_method = match_method
        
        self.template_contour = None
        self.template_hu = None
        self.template_area = 0
        self.template_img = None
    
    def load_template(self, template_path_or_image):
        """
        Load template and extract primary contour.
        
        Args:
            template_path_or_image: Path to image or numpy array
            
        Returns:
            Number of contour points extracted
        """
        if isinstance(template_path_or_image, str):
            img = cv2.imread(template_path_or_image, cv2.IMREAD_GRAYSCALE)
        elif len(template_path_or_image.shape) == 3:
            img = cv2.cvtColor(template_path_or_image, cv2.COLOR_BGR2GRAY)
        else:
            img = template_path_or_image.copy()
        
        self.template_img = img
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        # Edge detection
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)
        
        # Dilate to close small gaps in contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("[SHAPE] No contours found in template!")
            return 0
        
        # Select largest contour by area
        self.template_contour = max(contours, key=cv2.contourArea)
        self.template_area = cv2.contourArea(self.template_contour)
        self.template_hu = cv2.HuMoments(cv2.moments(self.template_contour)).flatten()
        
        num_pts = len(self.template_contour)
        print(f"[SHAPE] Template contour: {num_pts} points, area={self.template_area:.0f}")
        return num_pts
    
    def match(self, search_img):
        """
        Find the best shape match in the search image.
        
        Extracts contours from the search image, compares each to the
        template contour using Hu Moments, and returns the best match.
        Multi-angle matching rotates the template contour to estimate angle.
        
        Args:
            search_img: Image to search in (grayscale or BGR)
            
        Returns:
            Dictionary with match results:
                - success: True if match found
                - position: (x, y) center of matched contour
                - angle: Estimated rotation angle in degrees
                - scale: Scale factor from area ratio
                - shape_score: Raw similarity score (lower = better)
                - normalized_score: 0-1 score (higher = better)
                - contour: Matched contour points
        """
        if self.template_contour is None:
            return {'success': False, 'reason': 'No template contour loaded'}
        
        # Convert to grayscale
        if len(search_img.shape) == 3:
            gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = search_img
        
        # Enhance + edge detection (same pipeline as template)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find candidate contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'success': False, 'reason': 'No contours in search image'}
        
        # Filter by minimum area
        candidates = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        if not candidates:
            return {'success': False, 'reason': f'No contours above min area ({self.min_contour_area})'}
        
        print(f"[SHAPE] {len(candidates)} candidate contours found")
        
        # Multi-angle matching: rotate template contour and compare
        best_score = float('inf')
        best_contour = None
        best_angle = 0
        
        # Get template centroid for rotation
        M_t = cv2.moments(self.template_contour)
        if M_t['m00'] == 0:
            return {'success': False, 'reason': 'Template contour has zero area'}
        cx_t = M_t['m10'] / M_t['m00']
        cy_t = M_t['m01'] / M_t['m00']
        
        for angle in range(0, 360, self.angle_step):
            # Rotate template contour around its centroid
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            rotated_contour = self.template_contour.copy().astype(np.float64)
            for i in range(len(rotated_contour)):
                px = rotated_contour[i][0][0] - cx_t
                py = rotated_contour[i][0][1] - cy_t
                rotated_contour[i][0][0] = px * cos_a - py * sin_a + cx_t
                rotated_contour[i][0][1] = px * sin_a + py * cos_a + cy_t
            rotated_contour = rotated_contour.astype(np.int32)
            
            # Compare against each candidate contour
            for contour in candidates:
                score = cv2.matchShapes(rotated_contour, contour, self.match_method, 0)
                if score < best_score:
                    best_score = score
                    best_contour = contour
                    best_angle = angle
        
        # Compute position from matched contour centroid
        M_match = cv2.moments(best_contour)
        if M_match['m00'] == 0:
            return {'success': False, 'reason': 'Matched contour has zero area'}
        
        cx = int(M_match['m10'] / M_match['m00'])
        cy = int(M_match['m01'] / M_match['m00'])
        
        # Estimate scale from area ratio
        matched_area = cv2.contourArea(best_contour)
        scale = math.sqrt(matched_area / self.template_area) if self.template_area > 0 else 1.0
        
        # Compute bounding corners from min-area rotated rect
        rect = cv2.minAreaRect(best_contour)
        corners = cv2.boxPoints(rect).astype(int)
        
        # Normalize score: matchShapes returns 0 (perfect) to ~1+ (poor)
        normalized_score = max(0.0, 1.0 - best_score)
        
        print(f"[SHAPE] Best match: score={best_score:.4f} (similarity={normalized_score:.1%}), "
              f"angle={best_angle}°, scale={scale:.3f}")
        
        return {
            'success': True,
            'position': (cx, cy),
            'angle': float(best_angle),
            'scale': round(scale, 3),
            'shape_score': best_score,
            'normalized_score': normalized_score,
            'contour': best_contour,
            'corners': corners,
            'method': 'shape'
        }


class HybridMatcher:
    """
    Hybrid matcher that fuses shape contour matching with keypoint-based
    feature matching for robust pattern detection.
    
    Strategy:
    - Run both ShapeMatcher and FeatureBasedMatcher
    - Fuse results using weighted scoring
    - Fallback: if one method fails, use the other
    """
    
    def __init__(self, feature_config=None, shape_weight=0.4):
        """
        Initialize the hybrid matcher.
        
        Args:
            feature_config: FeatureMatcherConfig for keypoint matching
            shape_weight: Weight for shape score (0-1). Feature weight = 1 - shape_weight
        """
        self.feature_config = feature_config or FeatureMatcherConfig()
        self.shape_weight = shape_weight
        
        self.shape_matcher = ShapeMatcher()
        self.feature_matcher = FeatureBasedMatcher(self.feature_config)
    
    def load_template(self, template_path_or_image):
        """Load template into both matchers.
        
        Returns:
            (shape_contour_points, feature_keypoints)
        """
        shape_pts = self.shape_matcher.load_template(template_path_or_image)
        feature_kps = self.feature_matcher.load_template(template_path_or_image)
        return shape_pts, feature_kps
    
    def match(self, search_img):
        """
        Run hybrid matching: shape + keypoint fusion.
        
        Args:
            search_img: Image to search in
            
        Returns:
            Dictionary with combined match results including:
                - success, position, angle, scale (from best/fused result)
                - shape_result: Full shape matcher output
                - feature_result: Full feature matcher output
                - combined_score: Weighted fusion score
                - method_used: 'hybrid', 'shape_only', 'feature_only', or 'none'
        """
        # Run both matchers
        print("\n[HYBRID] Running shape matching...")
        shape_result = self.shape_matcher.match(search_img)
        
        print("[HYBRID] Running feature matching...")
        feature_result = self.feature_matcher.match(search_img)
        
        shape_ok = shape_result.get('success', False)
        feature_ok = feature_result.get('success', False)
        
        # ---- Case 1: Both succeed → weighted fusion ----
        if shape_ok and feature_ok:
            # Normalize feature score: inlier ratio (0 to 1)
            feature_score = feature_result['inliers'] / max(feature_result['num_matches'], 1)
            shape_score = shape_result['normalized_score']
            
            combined = (self.shape_weight * shape_score +
                        (1 - self.shape_weight) * feature_score)
            
            # Use position/angle from the higher-confidence method
            if shape_score >= feature_score:
                primary = shape_result
                print(f"[HYBRID] Shape confidence higher ({shape_score:.2f} vs {feature_score:.2f})")
            else:
                primary = feature_result
                print(f"[HYBRID] Feature confidence higher ({feature_score:.2f} vs {shape_score:.2f})")
            
            print(f"[HYBRID] Combined score: {combined:.3f} "
                  f"(shape={shape_score:.2f}×{self.shape_weight} + "
                  f"feature={feature_score:.2f}×{1-self.shape_weight:.1f})")
            
            return {
                'success': True,
                'position': primary['position'],
                'angle': primary['angle'],
                'scale': primary['scale'],
                'corners': primary['corners'],
                'combined_score': combined,
                'shape_result': shape_result,
                'feature_result': feature_result,
                'method_used': 'hybrid',
                # Pass through feature data for visualization
                'homography': feature_result.get('homography'),
                'num_matches': feature_result.get('num_matches', 0),
                'inliers': feature_result.get('inliers', 0),
                'good_matches': feature_result.get('good_matches'),
                'template_kp': feature_result.get('template_kp'),
                'search_kp': feature_result.get('search_kp'),
                'mask': feature_result.get('mask'),
            }
        
        # ---- Case 2: Only shape succeeds → fallback ----
        if shape_ok and not feature_ok:
            print("[HYBRID] Fallback → shape only (keypoints failed)")
            result = shape_result.copy()
            result['method_used'] = 'shape_only'
            result['combined_score'] = shape_result['normalized_score'] * self.shape_weight
            result['shape_result'] = shape_result
            result['feature_result'] = feature_result
            return result
        
        # ---- Case 3: Only features succeed → fallback ----
        if feature_ok and not shape_ok:
            print("[HYBRID] Fallback → features only (shape failed)")
            result = feature_result.copy()
            result['method_used'] = 'feature_only'
            feature_score = feature_result['inliers'] / max(feature_result['num_matches'], 1)
            result['combined_score'] = feature_score * (1 - self.shape_weight)
            result['shape_result'] = shape_result
            result['feature_result'] = feature_result
            return result
        
        # ---- Case 4: Both fail ----
        print("[HYBRID] Both methods failed!")
        return {
            'success': False,
            'reason': (f"Shape: {shape_result.get('reason', '?')} | "
                       f"Feature: {feature_result.get('reason', '?')}"),
            'method_used': 'none',
            'shape_result': shape_result,
            'feature_result': feature_result,
        }


class InteractiveFeatureTuner:
    """
    Interactive GUI for feature-based pattern matching.
    """
    
    def __init__(self):
        """Initialize the interactive tuner"""
        self.config = FeatureMatcherConfig()
        self.matcher = None
        
        self.template_img = None
        self.search_img = None
        self.original_search_img = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the matplotlib UI"""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('Feature-Based Pattern Matcher (Structure + Texture Aware)', 
                         fontsize=14, fontweight='bold')
        
        # Image display areas
        self.ax_template = self.fig.add_axes([0.05, 0.55, 0.25, 0.35])
        self.ax_template.set_title("Template + Keypoints")
        self.ax_template.axis('off')
        
        self.ax_result = self.fig.add_axes([0.35, 0.30, 0.60, 0.62])
        self.ax_result.set_title("Detection Result")
        self.ax_result.axis('off')
        
        # Status panel
        self.ax_status = self.fig.add_axes([0.05, 0.35, 0.25, 0.15])
        self.ax_status.axis('off')
        
        # Detector selection
        self.fig.text(0.08, 0.29, 'DETECTOR', fontsize=10, fontweight='bold')
        detector_ax = self.fig.add_axes([0.05, 0.10, 0.10, 0.18])
        detector_ax.set_facecolor('lightgray')
        self.detector_radio = RadioButtons(detector_ax, ('ORB', 'AKAZE', 'SIFT', 'LINEMOD', 'HYBRID'), active=0)
        
        # Sliders
        slider_left = 0.18
        slider_width = 0.12
        slider_height = 0.025
        
        sliders_config = [
            ('MAX_FEATURES', 100, 2000, 500, 100, 'Max Features'),
            ('MATCH_RATIO', 0.5, 0.95, 0.75, 0.05, 'Match Ratio'),
            ('MIN_MATCHES', 4, 50, 10, 2, 'Min Matches'),
            ('SHAPE_WEIGHT', 0.0, 1.0, 0.4, 0.05, 'Shape Weight'),
        ]
        
        self.sliders = {}
        for i, (name, vmin, vmax, vinit, vstep, label) in enumerate(sliders_config):
            y_pos = 0.26 - i * 0.05
            ax = self.fig.add_axes([slider_left, y_pos, slider_width, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=vstep, color='lightblue')
            self.sliders[name] = slider
        
        # Buttons
        load_template_ax = self.fig.add_axes([0.05, 0.05, 0.09, 0.035])
        self.load_template_btn = Button(load_template_ax, 'Load Template', 
                                        color='lightgreen', hovercolor='palegreen')
        self.load_template_btn.on_clicked(self.load_template)
        
        load_search_ax = self.fig.add_axes([0.15, 0.05, 0.09, 0.035])
        self.load_search_btn = Button(load_search_ax, 'Load Search', 
                                      color='lightyellow', hovercolor='khaki')
        self.load_search_btn.on_clicked(self.load_search)
        
        crop_ax = self.fig.add_axes([0.05, 0.01, 0.09, 0.035])
        self.crop_btn = Button(crop_ax, 'Crop Template', 
                              color='lightcyan', hovercolor='cyan')
        self.crop_btn.on_clicked(self.crop_template)
        
        detect_ax = self.fig.add_axes([0.15, 0.01, 0.09, 0.035])
        self.detect_btn = Button(detect_ax, 'Detect!', 
                                color='lightcoral', hovercolor='salmon')
        self.detect_btn.on_clicked(self.run_detection)
        
        # Initial display
        self.update_display()
    
    def load_template(self, event):
        """Load template image via file dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        root.destroy()
        
        if file_path:
            self.template_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded template: {file_path}")
            self.update_display()
    
    def load_search(self, event):
        """Load search image via file dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select Search Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        root.destroy()
        
        if file_path:
            self.search_img = cv2.imread(file_path)
            self.original_search_img = self.search_img.copy()
            print(f"Loaded search image: {file_path}")
            self.update_display()
    
    def crop_template(self, event):
        """Crop template from search image"""
        if self.search_img is None:
            print("Please load a search image first!")
            return
        
        print("\n" + "=" * 50)
        print("CROP TEMPLATE FROM SEARCH IMAGE")
        print("  1. Draw rectangle around the pattern")
        print("  2. Press ENTER to confirm, C to cancel")
        print("=" * 50)
        
        window_name = "Draw rectangle - ENTER to confirm, C to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 800)
        
        roi = cv2.selectROI(window_name, self.search_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        
        x, y, w, h = roi
        
        if w > 0 and h > 0:
            if len(self.search_img.shape) == 3:
                self.template_img = cv2.cvtColor(
                    self.search_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY
                )
            else:
                self.template_img = self.search_img[y:y+h, x:x+w].copy()
            
            print(f"\n✓ Template cropped: {w}x{h} pixels")
            self.update_display()
        else:
            print("Crop cancelled.")
    
    def run_detection(self, event):
        """Run feature-based or LINE-MOD matching"""
        if self.template_img is None:
            print("Please load or crop a template first!")
            return
        if self.search_img is None:
            print("Please load a search image first!")
            return
        
        # Update config from UI
        detector_type = self.detector_radio.value_selected
        self.config.DETECTOR_TYPE = detector_type
        self.config.MAX_FEATURES = int(self.sliders['MAX_FEATURES'].val)
        self.config.MATCH_RATIO = float(self.sliders['MATCH_RATIO'].val)
        self.config.MIN_MATCHES = int(self.sliders['MIN_MATCHES'].val)
        
        if detector_type == 'HYBRID':
            # Use hybrid shape + feature matching
            self._run_hybrid_detection()
        elif detector_type == 'LINEMOD':
            # Use LINE-MOD gradient-based matching
            self._run_linemod_detection()
        else:
            # Use keypoint-based matching (ORB/AKAZE/SIFT)
            self._run_feature_detection()
    
    def _run_feature_detection(self):
        """Run keypoint-based feature matching (ORB/AKAZE/SIFT)"""
        # Create matcher and load template
        self.matcher = FeatureBasedMatcher(self.config)
        num_kp = self.matcher.load_template(self.template_img)
        
        if num_kp < self.config.MIN_MATCHES:
            print(f"[WARN] Only {num_kp} keypoints in template. Try lowering Min Matches.")
        
        # Run detection
        print("Running feature matching...")
        result = self.matcher.match(self.search_img)
        
        # Update template display with keypoints
        self.ax_template.clear()
        template_with_kp = self.matcher.visualize_keypoints(show=False)
        if template_with_kp is not None:
            self.ax_template.imshow(cv2.cvtColor(template_with_kp, cv2.COLOR_BGR2RGB))
            self.ax_template.set_title(f"Template ({num_kp} keypoints)")
        self.ax_template.axis('off')
        
        # Update result display
        self.ax_result.clear()
        
        if result['success']:
            result_img = self.matcher.visualize_match(self.search_img, result, show=False)
            self.ax_result.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title("✓ Match Found!")
            
            # Update status
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"MATCH FOUND\n"
            status += f"━━━━━━━━━━━━━━━━━\n"
            status += f"Position: {result['position']}\n"
            status += f"Angle: {result['angle']:.1f}°\n"
            status += f"Scale: {result['scale']:.3f}\n"
            status += f"Inliers: {result['inliers']}/{result['num_matches']}"
            self.ax_status.text(0.1, 0.9, status, transform=self.ax_status.transAxes,
                               fontsize=9, fontfamily='monospace', verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        else:
            if len(self.search_img.shape) == 2:
                self.ax_result.imshow(self.search_img, cmap='gray')
            else:
                self.ax_result.imshow(cv2.cvtColor(self.search_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title(f"✗ {result.get('reason', 'No match')}", color='red')
            
            # Update status
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"NO MATCH\n"
            status += f"━━━━━━━━━━━━━━━━━\n"
            status += f"Reason: {result.get('reason', 'Unknown')}\n"
            status += f"Matches: {result.get('num_matches', 0)}"
            self.ax_status.text(0.1, 0.9, status, transform=self.ax_status.transAxes,
                               fontsize=9, fontfamily='monospace', verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax_result.axis('off')
        self.fig.canvas.draw_idle()
    
    def _run_hybrid_detection(self):
        """Run hybrid shape + feature matching"""
        # Config for the feature part - use ORB by default
        feature_config = FeatureMatcherConfig()
        feature_config.DETECTOR_TYPE = "ORB"
        feature_config.MAX_FEATURES = int(self.sliders['MAX_FEATURES'].val)
        feature_config.MATCH_RATIO = float(self.sliders['MATCH_RATIO'].val)
        feature_config.MIN_MATCHES = int(self.sliders['MIN_MATCHES'].val)
        
        shape_weight = float(self.sliders['SHAPE_WEIGHT'].val)
        
        # Create hybrid matcher
        hybrid = HybridMatcher(feature_config=feature_config, shape_weight=shape_weight)
        
        # Load template into both matchers
        shape_pts, num_kp = hybrid.load_template(self.template_img)
        
        if shape_pts == 0 and num_kp < feature_config.MIN_MATCHES:
            print("[WARN] Template has no contours and insufficient keypoints!")
        
        # Run hybrid matching
        print("Running hybrid matching...")
        result = hybrid.match(self.search_img)
        
        # Update template display with contour + keypoints overlay
        self.ax_template.clear()
        template_vis = (cv2.cvtColor(self.template_img, cv2.COLOR_GRAY2BGR)
                        if len(self.template_img.shape) == 2
                        else self.template_img.copy())
        if hybrid.shape_matcher.template_contour is not None:
            cv2.drawContours(template_vis, [hybrid.shape_matcher.template_contour], -1,
                             (0, 255, 255), 2)  # cyan contour
        if hybrid.feature_matcher.template_keypoints:
            template_vis = cv2.drawKeypoints(template_vis,
                                             hybrid.feature_matcher.template_keypoints,
                                             template_vis, color=(0, 255, 0),
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.ax_template.imshow(cv2.cvtColor(template_vis, cv2.COLOR_BGR2RGB))
        self.ax_template.set_title(f"Template ({shape_pts} contour, {num_kp} kp)")
        self.ax_template.axis('off')
        
        # Update result display
        self.ax_result.clear()
        
        if result['success']:
            # Draw result on search image
            if len(self.search_img.shape) == 2:
                result_img = cv2.cvtColor(self.search_img, cv2.COLOR_GRAY2BGR)
            else:
                result_img = self.search_img.copy()
            
            # Draw shape contour in cyan if available
            shape_res = result.get('shape_result', {})
            if shape_res.get('success') and shape_res.get('contour') is not None:
                cv2.drawContours(result_img, [shape_res['contour']], -1, (255, 255, 0), 2)
            
            # Draw bounding polygon in green
            if 'corners' in result:
                corners = result['corners']
                cv2.polylines(result_img, [corners], True, (0, 255, 0), 3)
            
            # Draw center cross
            cx, cy = result['position']
            cv2.drawMarker(result_img, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 30, 3)
            
            # Draw rotation indicator
            angle_rad = math.radians(result['angle'])
            dx = int(50 * math.cos(angle_rad))
            dy = int(50 * math.sin(angle_rad))
            cv2.arrowedLine(result_img, (cx, cy), (cx + dx, cy + dy),
                           (255, 0, 255), 3, tipLength=0.3)
            
            # Add text info
            method = result.get('method_used', 'unknown')
            info_lines = [
                f"Method: {method}",
                f"Position: ({cx}, {cy})",
                f"Angle: {result['angle']:.1f} deg",
                f"Scale: {result['scale']:.3f}",
                f"Combined: {result.get('combined_score', 0):.3f}",
            ]
            y = 30
            for info in info_lines:
                cv2.putText(result_img, info, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25
            
            self.ax_result.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title(f"\u2713 Hybrid Match ({method})!")
            
            # Update status panel
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"HYBRID MATCH\n"
            status += f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            status += f"Method: {method}\n"
            status += f"Position: ({cx}, {cy})\n"
            status += f"Angle: {result['angle']:.1f}\u00b0\n"
            status += f"Scale: {result['scale']:.3f}\n"
            status += f"Combined: {result.get('combined_score', 0):.3f}\n"
            shape_r = result.get('shape_result', {})
            feat_r = result.get('feature_result', {})
            if shape_r.get('success'):
                status += f"Shape: {shape_r.get('normalized_score', 0):.1%}\n"
            else:
                status += f"Shape: FAILED\n"
            if feat_r.get('success'):
                status += f"Features: {feat_r.get('inliers', 0)}/{feat_r.get('num_matches', 0)}"
            else:
                status += f"Features: FAILED"
            self.ax_status.text(0.1, 0.9, status, transform=self.ax_status.transAxes,
                               fontsize=8, fontfamily='monospace', verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        else:
            if len(self.search_img.shape) == 2:
                self.ax_result.imshow(self.search_img, cmap='gray')
            else:
                self.ax_result.imshow(cv2.cvtColor(self.search_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title(f"\u2717 {result.get('reason', 'No match')}", color='red')
            
            # Update status
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"NO MATCH\n"
            status += f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            status += f"Method: HYBRID\n"
            status += f"Reason: {result.get('reason', 'Unknown')}"
            self.ax_status.text(0.1, 0.9, status, transform=self.ax_status.transAxes,
                               fontsize=9, fontfamily='monospace', verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax_result.axis('off')
        self.fig.canvas.draw_idle()
    
    def _run_linemod_detection(self):
        """Run LINE-MOD gradient-based matching with coarse-to-fine pyramid speedup"""
        import time
        start_time = time.time()
        
        threshold_pct = float(self.sliders['MATCH_RATIO'].val) * 100
        
        # ---- STAGE 1: Coarse search at reduced resolution ----
        print("\n[LINE-MOD] Stage 1: Coarse search (downscaled)...")
        
        # Downscale search image for fast coarse matching
        search_for_matching = self.search_img.copy()
        scale_down = 0.5  # Match at half resolution
        h, w = search_for_matching.shape[:2]
        small_search = cv2.resize(search_for_matching, (int(w * scale_down), int(h * scale_down)))
        small_template = cv2.resize(self.template_img, 
                                     (int(self.template_img.shape[1] * scale_down), 
                                      int(self.template_img.shape[0] * scale_down)))
        
        # Coarse config: large angle steps, no scale variation
        coarse_config = LinemodConfig()
        coarse_config.ANGLE_STEP = 45           # 8 angles only
        coarse_config.SCALE_MIN = 1.0
        coarse_config.SCALE_MAX = 1.0
        coarse_config.MATCH_THRESHOLD = max(threshold_pct - 15, 30)  # Lower threshold for coarse
        
        coarse_matcher = LinemodMatcher(coarse_config)
        coarse_matcher.load_template(small_template)
        num_coarse = coarse_matcher.generate_templates(verbose=False)
        print(f"  {num_coarse} coarse templates generated")
        
        coarse_match = coarse_matcher.match(small_search)
        
        best_angle = 0
        if coarse_match is not None:
            best_angle = coarse_match['angle']
            print(f"  Coarse match: angle={best_angle}°, score={coarse_match['score']:.1f}%")
        else:
            print("  No coarse match, trying full resolution...")
        
        # ---- STAGE 2: Fine search at full resolution ----
        print("[LINE-MOD] Stage 2: Fine search (full resolution)...")
        
        fine_config = LinemodConfig()
        # Search ±20° around the coarse angle with fine 5° steps
        fine_angles = list(range(int(best_angle - 20), int(best_angle + 25), 5))
        fine_config.ANGLE_STEP = 5
        fine_config.SCALE_MIN = 0.9
        fine_config.SCALE_MAX = 1.1
        fine_config.SCALE_STEP = 0.1
        fine_config.MATCH_THRESHOLD = threshold_pct
        
        fine_matcher = LinemodMatcher(fine_config)
        fine_matcher.load_template(self.template_img)
        
        # Generate templates only around the best coarse angle
        fine_matcher.templates = []
        fine_matcher.template_metadata = []
        template_img = fine_matcher.template_image
        h_t, w_t = template_img.shape[:2]
        center = (w_t // 2, h_t // 2)
        
        for scale in np.arange(fine_config.SCALE_MIN, fine_config.SCALE_MAX + 0.01, fine_config.SCALE_STEP):
            for angle in fine_angles:
                angle = angle % 360
                M = cv2.getRotationMatrix2D(center, angle, scale)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h_t * sin + w_t * cos)
                new_h = int(h_t * cos + w_t * sin)
                M[0, 2] += (new_w - w_t) / 2
                M[1, 2] += (new_h - h_t) / 2
                
                rotated = cv2.warpAffine(template_img, M, (new_w, new_h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = np.ones_like(template_img) * 255
                rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                gradient_template = fine_matcher._compute_gradient_features(rotated)
                
                fine_matcher.templates.append({
                    'image': rotated,
                    'gradient': gradient_template,
                    'mask': rotated_mask,
                    'angle': angle,
                    'scale': scale,
                    'size': (new_w, new_h)
                })
                fine_matcher.template_metadata.append({'angle': angle, 'scale': scale})
        
        num_fine = len(fine_matcher.templates)
        print(f"  {num_fine} fine templates generated (±20° around {best_angle}°)")
        
        # Run fine matching
        match = fine_matcher.match(self.search_img)
        
        elapsed = time.time() - start_time
        print(f"[LINE-MOD] Total time: {elapsed:.1f}s ({num_coarse + num_fine} templates)")
        
        # Update template display
        self.ax_template.clear()
        self.ax_template.imshow(self.template_img, cmap='gray')
        self.ax_template.set_title(f"Template ({num_coarse + num_fine} LINE-MOD templates)")
        self.ax_template.axis('off')
        
        # Update result display
        self.ax_result.clear()
        
        if match is not None:
            result_img = fine_matcher.visualize_match(self.search_img, match, show=False)
            self.ax_result.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title("✓ LINE-MOD Match Found!")
            
            # Update status
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"LINE-MOD MATCH\n"
            status += f"━━━━━━━━━━━━━━━━━\n"
            status += f"Position: ({match['x']}, {match['y']})\n"
            status += f"Angle: {match['angle']:.1f}°\n"
            status += f"Scale: {match['scale']:.3f}\n"
            status += f"Score: {match['score']:.1f}%"
            self.ax_status.text(0.1, 0.9, status, transform=self.ax_status.transAxes,
                               fontsize=9, fontfamily='monospace', verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        else:
            if len(self.search_img.shape) == 2:
                self.ax_result.imshow(self.search_img, cmap='gray')
            else:
                self.ax_result.imshow(cv2.cvtColor(self.search_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title("✗ LINE-MOD: No match found", color='red')
            
            # Update status
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"NO MATCH\n"
            status += f"━━━━━━━━━━━━━━━━━\n"
            status += f"Method: LINE-MOD\n"
            status += f"Templates: {num_templates}"
            self.ax_status.text(0.1, 0.9, status, transform=self.ax_status.transAxes,
                               fontsize=9, fontfamily='monospace', verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax_result.axis('off')
        self.fig.canvas.draw_idle()
    
    def update_display(self):
        """Update the display with current images"""
        self.ax_template.clear()
        if self.template_img is not None:
            self.ax_template.imshow(self.template_img, cmap='gray')
            self.ax_template.set_title(f"Template ({self.template_img.shape[1]}x{self.template_img.shape[0]})")
        else:
            self.ax_template.set_title("Template (not loaded)")
        self.ax_template.axis('off')
        
        self.ax_result.clear()
        if self.search_img is not None:
            if len(self.search_img.shape) == 2:
                self.ax_result.imshow(self.search_img, cmap='gray')
            else:
                self.ax_result.imshow(cv2.cvtColor(self.search_img, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title(f"Search Image ({self.search_img.shape[1]}x{self.search_img.shape[0]})")
        else:
            self.ax_result.set_title("Search Image (not loaded)")
        self.ax_result.axis('off')
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive tuner"""
        plt.show()


def main():
    """Main entry point"""
    print("=" * 60)
    print("  FEATURE-BASED PATTERN MATCHER")
    print("  Structure + Texture Aware Matching")
    print("=" * 60)
    print()
    print("  Advantages over template matching:")
    print("  ✓ True rotation invariance")
    print("  ✓ True scale invariance")
    print("  ✓ Works with partial visibility")
    print("  ✓ Returns exact angle and scale")
    print()
    print("  Detectors available:")
    print("  • ORB - Fast, good for real-time")
    print("  • AKAZE - Better accuracy, slightly slower")
    print("  • SIFT - Best accuracy (if available)")
    print("  • LINEMOD - Gradient-based, best for small templates")
    print("  • HYBRID - Shape + keypoint fusion (most robust)")
    print()
    print("=" * 60)
    
    tuner = InteractiveFeatureTuner()
    tuner.show()


if __name__ == "__main__":
    main()
