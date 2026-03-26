"""
================================================================================
EMGU VISION TEMPLATE MATCHING
================================================================================
Python port of the C# EmguCV WC_REQ template matching pipeline.

Provides cv2.TM_CCORR_NORMED / TM_CCOEFF_NORMED based template matching
with optional mask support, rotation retry, and pixel-resolution offset
calculation — identical logic to the original EmguVision.cs WC_REQ method.

Usage:
    from emgu_matcher import EmguVisionTemplateMatching

    vision = EmguVisionTemplateMatching()
    vision.teach_feature_offset(image, roi=(100, 100, 150, 150))
    offset_x, offset_y, angle, score, plot = vision.inspect_feature_offset(image)

    # Interactive tuner: python emgu_tuner.py

Author: Wafer Alignment System
================================================================================
"""

import cv2
import numpy as np
import os


# --- Configuration ---
class RoiConfig:
    """Stores the taught fiducial ROI location."""
    def __init__(self):
        self.wafer_fiducial_start_x = 0
        self.wafer_fiducial_start_y = 0
        self.wafer_fiducial_width = 0
        self.wafer_fiducial_height = 0


class RecipeParam:
    """Recipe parameters mirroring QvsRecipeParam."""
    def __init__(self):
        self.fiducial_template_match_score_threshold = 0.7
        self.roi_config = RoiConfig()
        self.max_rotation_retries = 5       # Max rotation attempts
        self.rotation_step = 1.0            # Degrees per retry
        self.match_method = cv2.TM_CCORR_NORMED  # Default match method


# --- Core Matcher ---
class EmguVisionTemplateMatching:
    """
    Template matching engine — Python port of EmguVision.cs WC_REQ.

    Pipeline:
      1. Convert input + template + mask to grayscale
      2. cv2.matchTemplate with selected method + optional mask
      3. cv2.minMaxLoc to find best match
      4. Score check vs threshold; if fail, rotate template +1 deg (up to N retries)
      5. Calculate X/Y offset from found center vs taught ROI center
      6. Apply pixel resolution scaling
    """

    def __init__(self):
        self.fiducial_tmp_image = None       # BGR numpy array (taught template)
        self.fiducial_tmp_mask_image = None   # BGR numpy array (taught mask)
        self.recipe_param = RecipeParam()
        self.pixel_resolution = 1.0
        self.str_error_msg = ""

    def set_pixel_resolution(self, pixel_resolution: float):
        """Set the pixel-to-real-world conversion factor."""
        self.pixel_resolution = pixel_resolution

    # ------------------------------------------------------------------
    # Teach
    # ------------------------------------------------------------------
    def teach_feature_offset(self, image: np.ndarray, roi: tuple,
                              debug: bool = False) -> bool:
        """
        Teach/save the template from a given ROI.
        Mirrors TeachFeatureOffset() in EmguVision.cs.

        Args:
            image: BGR or grayscale numpy array (full image)
            roi: (x, y, width, height) tuple
            debug: enable debug output
        Returns:
            True if successful
        """
        try:
            x, y, w, h = roi

            self.recipe_param.roi_config.wafer_fiducial_start_x = x
            self.recipe_param.roi_config.wafer_fiducial_start_y = y
            self.recipe_param.roi_config.wafer_fiducial_width = w
            self.recipe_param.roi_config.wafer_fiducial_height = h

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            self.fiducial_tmp_image = image[y:y + h, x:x + w].copy()

            if debug:
                cv2.imwrite("TeachFeatureOffset_FiducialTmpImage.bmp",
                            self.fiducial_tmp_image)
                print(f"[Teach] Template saved. ROI=({x},{y},{w},{h})")

            return True
        except Exception as ex:
            print(f"[teach_feature_offset] Error: {ex}")
            return False

    def teach_mask(self, mask_image: np.ndarray, debug: bool = False) -> bool:
        """
        Teach/save the mask image.
        Mirrors TeachMask() in EmguVision.cs.

        Args:
            mask_image: BGR or grayscale numpy array
            debug: enable debug output
        Returns:
            True if successful
        """
        try:
            if len(mask_image.shape) == 2:
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

            self.fiducial_tmp_mask_image = mask_image.copy()

            if debug:
                cv2.imwrite("TeachMask_FiducialTmpMaskImage.bmp",
                            self.fiducial_tmp_mask_image)
                print("[Teach] Mask saved.")

            return True
        except Exception as ex:
            print(f"[teach_mask] Error: {ex}")
            return False

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------
    def inspect_feature_offset(self, image: np.ndarray,
                                debug: bool = False):
        """
        Find the feature offset using template matching.
        Mirrors InspectFeatureOffset() -> WC_REQ() in EmguVision.cs.

        Args:
            image: BGR or grayscale numpy array (full input image)
            debug: enable debug output

        Returns:
            (offset_x, offset_y, angle, score, plot_image)
            offset_x, offset_y are in real-world units (pixels * pixel_resolution)
        """
        try:
            if self.fiducial_tmp_image is None:
                print("[WC_REQ] No Template Found.")
                return 0.0, 0.0, 0.0, 0.0, image.copy()

            rp = self.recipe_param

            # --- Grayscale conversion ---
            if len(image.shape) == 3:
                input_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                input_gray = image.copy()

            if len(self.fiducial_tmp_image.shape) == 3:
                gray_fiducial = cv2.cvtColor(self.fiducial_tmp_image,
                                              cv2.COLOR_BGR2GRAY)
            else:
                gray_fiducial = self.fiducial_tmp_image.copy()

            gray_mask = None
            if self.fiducial_tmp_mask_image is not None:
                if len(self.fiducial_tmp_mask_image.shape) == 3:
                    gray_mask = cv2.cvtColor(self.fiducial_tmp_mask_image,
                                              cv2.COLOR_BGR2GRAY)
                else:
                    gray_mask = self.fiducial_tmp_mask_image.copy()

            # --- Matching with rotation retry loop ---
            rotation_degree = 0.0
            rotation_count = 0
            max_retries = rp.max_rotation_retries
            score = 0.0
            max_val = 0.0
            max_loc = (0, 0)
            output = None

            while True:
                rotated_fiducial = gray_fiducial
                rotated_mask = gray_mask

                if rotation_degree != 0:
                    rotated_fiducial = self._rotate_image(gray_fiducial,
                                                          rotation_degree)
                    if gray_mask is not None:
                        rotated_mask = self._rotate_image(gray_mask,
                                                          rotation_degree)

                if debug:
                    cv2.imwrite("Feature_InputImage.bmp", input_gray)
                    cv2.imwrite("Feature_FiducialImage.bmp", rotated_fiducial)
                    print(f"[WC_REQ] Trying rotation={rotation_degree}\u00b0")

                # Template matching
                if rotated_mask is not None:
                    output = cv2.matchTemplate(
                        input_gray, rotated_fiducial,
                        rp.match_method,
                        mask=rotated_mask
                    )
                else:
                    output = cv2.matchTemplate(
                        input_gray, rotated_fiducial,
                        rp.match_method
                    )

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(output)
                score = max_val

                # Score threshold check
                if max_val < rp.fiducial_template_match_score_threshold:
                    if rotation_count <= max_retries:
                        rotation_count += 1
                        rotation_degree += rp.rotation_step
                        continue
                    else:
                        self.str_error_msg = (
                            f"Feature Template found Less than Score Threshold. "
                            f"Score:[{max_val:.2f}]"
                        )
                        print(f"[WC_REQ] {self.str_error_msg}")
                        return 0.0, 0.0, rotation_degree, score, image.copy()
                else:
                    break

            # --- Offset calculation ---
            tmpl_h, tmpl_w = rotated_fiducial.shape[:2]
            found_rect_x = max_loc[0]
            found_rect_y = max_loc[1]
            found_center_x = found_rect_x + tmpl_w / 2.0
            found_center_y = found_rect_y + tmpl_h / 2.0

            rc = rp.roi_config
            fiducial_center_offset_x = rc.wafer_fiducial_start_x + rc.wafer_fiducial_width / 2.0
            fiducial_center_offset_y = rc.wafer_fiducial_start_y + rc.wafer_fiducial_height / 2.0

            d_fnd_x_offset = found_center_x - fiducial_center_offset_x
            d_fnd_y_offset = found_center_y - fiducial_center_offset_y

            # Apply pixel resolution
            offset_x = d_fnd_x_offset * self.pixel_resolution
            offset_y = d_fnd_y_offset * self.pixel_resolution
            angle = rotation_degree

            self.str_error_msg = (
                f"Feature Template found. Score:[{max_val:.4f}] "
                f"OffsetX:{offset_x:.3f} OffsetY:{offset_y:.3f} "
                f"Angle:{angle:.3f}"
            )
            print(f"[WC_REQ] {self.str_error_msg}")

            # --- Draw result plot ---
            plot = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(
                plot,
                (found_rect_x, found_rect_y),
                (found_rect_x + tmpl_w, found_rect_y + tmpl_h),
                (0, 0, 255), 2
            )
            # Draw center cross
            cx, cy = int(found_center_x), int(found_center_y)
            cv2.drawMarker(plot, (cx, cy), (0, 255, 255),
                          cv2.MARKER_CROSS, 20, 2)

            # Draw taught ROI center for reference
            tcx = int(fiducial_center_offset_x)
            tcy = int(fiducial_center_offset_y)
            cv2.drawMarker(plot, (tcx, tcy), (255, 0, 255),
                          cv2.MARKER_TILTED_CROSS, 15, 1)

            # Info text
            info = (f"Score:{max_val:.4f}  Offset:({offset_x:.2f}, {offset_y:.2f})  "
                    f"Angle:{angle:.1f}")
            cv2.putText(plot, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return offset_x, offset_y, angle, score, plot

        except Exception as ex:
            print(f"[inspect_feature_offset] Error: {ex}")
            return 0.0, 0.0, 0.0, 0.0, image.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rotate_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate image around its center (same-size canvas).
        Mirrors EmguCV: grayFiducial.Rotate(rotationDegree, new Gray(0))
        EmguCV Rotate() is counter-clockwise; cv2 is clockwise — negate to match.
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        return cv2.warpAffine(
            image, rot_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
