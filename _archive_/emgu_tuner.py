"""
================================================================================
EMGU VISION TEMPLATE MATCHING — INTERACTIVE TUNER
================================================================================
Interactive GUI for the EmguCV-style template matching pipeline.

Provides:
- Load template / search image via file dialog
- Crop template from search image (with ROI teaching)
- Optional mask teaching
- Match method selection (CCORR_NORMED, CCOEFF_NORMED, SQDIFF_NORMED)
- Adjustable score threshold, rotation retries, and pixel resolution
- Simulated search image rotation
- Pipeline visualization (score heatmap)

Usage:
    python emgu_tuner.py

Author: Wafer Alignment System
================================================================================
"""

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import tkinter as tk
from tkinter import filedialog

from emgu_matcher import EmguVisionTemplateMatching, RecipeParam


class InteractiveEmguTuner:
    """
    Interactive GUI for tuning EmguCV template matching parameters.
    UI layout follows the same pattern as linemod_tuner.py.
    """

    def __init__(self, template_path=None, search_path=None):
        """
        Initialize the interactive tuner.

        Args:
            template_path: Path to template image
            search_path: Path to search image
        """
        self.vision = EmguVisionTemplateMatching()

        # Images
        self.template_img = None
        self.search_img = None
        self.mask_img = None
        self.taught_roi = None          # (x, y, w, h) from crop/teach

        if template_path and os.path.exists(template_path):
            self.template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if search_path and os.path.exists(search_path):
            self.search_img = cv2.imread(search_path)

        # Setup UI
        self.setup_ui()

    # ==================================================================
    # UI Setup
    # ==================================================================
    def setup_ui(self):
        """Setup the matplotlib UI"""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('EmguCV Template Matching - Interactive Tuner',
                         fontsize=14, fontweight='bold')

        # --- Image display areas ---
        self.ax_template = self.fig.add_axes([0.05, 0.55, 0.25, 0.35])
        self.ax_template.set_title("Template")
        self.ax_template.axis('off')

        self.ax_result = self.fig.add_axes([0.35, 0.30, 0.60, 0.62])
        self.ax_result.set_title("Detection Result")
        self.ax_result.axis('off')

        # --- Status panel ---
        self.ax_status = self.fig.add_axes([0.05, 0.35, 0.25, 0.15])
        self.ax_status.axis('off')

        # --- Match method selection ---
        self.fig.text(0.08, 0.28, 'MATCH METHOD', fontsize=10, fontweight='bold')
        method_ax = self.fig.add_axes([0.05, 0.12, 0.12, 0.15])
        method_ax.set_facecolor('lightgray')
        self.method_radio = RadioButtons(method_ax,
            ('CCORR Normed', 'CCOEFF Normed', 'SQDIFF Normed'),
            active=0)

        # --- Sliders ---
        slider_left = 0.20
        slider_width = 0.12
        slider_height = 0.025

        sliders_config = [
            ('ROTATION',    0, 360, 0,    1,    'Rotation \u00b0'),
            ('THRESHOLD',   0.1, 1.0, 0.7, 0.05, 'Score Thr'),
            ('MAX_RETRIES', 0, 10, 5,     1,    'Rot Retries'),
            ('ROT_STEP',    0.5, 5.0, 1.0, 0.5,  'Rot Step \u00b0'),
        ]

        self.sliders = {}
        for i, (name, vmin, vmax, vinit, vstep, label) in enumerate(sliders_config):
            y_pos = 0.26 - i * 0.04
            ax = self.fig.add_axes([slider_left, y_pos, slider_width, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit,
                          valstep=vstep, color='lightblue')
            if name == 'ROTATION':
                slider.on_changed(self.update_rotation)
            else:
                slider.on_changed(self.update)
            self.sliders[name] = slider

        # --- Checkbox: Show heatmap ---
        vis_ax = self.fig.add_axes([0.20, 0.12, 0.12, 0.04])
        self.vis_check = CheckButtons(vis_ax, ['Show Heatmap'], [False])

        # Store original search image for rotation
        self.original_search_img = None

        # --- Buttons row 1 ---
        load_template_ax = self.fig.add_axes([0.05, 0.05, 0.09, 0.035])
        self.load_template_btn = Button(load_template_ax, 'Load Template',
                                        color='lightgreen', hovercolor='palegreen')
        self.load_template_btn.on_clicked(self.load_template)

        load_search_ax = self.fig.add_axes([0.15, 0.05, 0.09, 0.035])
        self.load_search_btn = Button(load_search_ax, 'Load Search',
                                      color='lightyellow', hovercolor='khaki')
        self.load_search_btn.on_clicked(self.load_search)

        # --- Buttons row 2 ---
        crop_ax = self.fig.add_axes([0.05, 0.01, 0.09, 0.035])
        self.crop_btn = Button(crop_ax, 'Crop Template',
                              color='lightcyan', hovercolor='cyan')
        self.crop_btn.on_clicked(self.crop_template_from_search)

        detect_ax = self.fig.add_axes([0.15, 0.01, 0.09, 0.035])
        self.detect_btn = Button(detect_ax, 'Detect!',
                                color='lightcoral', hovercolor='salmon')
        self.detect_btn.on_clicked(self.run_detection)

        # --- Extra: Load Mask button ---
        mask_ax = self.fig.add_axes([0.25, 0.01, 0.09, 0.035])
        self.mask_btn = Button(mask_ax, 'Load Mask',
                              color='plum', hovercolor='violet')
        self.mask_btn.on_clicked(self.load_mask)

        # Initial display
        self.update_display()

    # ==================================================================
    # Image Loading
    # ==================================================================
    def load_template(self, event):
        """Load template image via file dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")]
        )
        root.destroy()

        if file_path:
            self.template_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.taught_roi = None
            print(f"Loaded template: {file_path}")
            self.update_display()

    def load_search(self, event):
        """Load search image via file dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select Search Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")]
        )
        root.destroy()

        if file_path:
            self.search_img = cv2.imread(file_path)
            self.original_search_img = self.search_img.copy()
            self.sliders['ROTATION'].set_val(0)
            print(f"Loaded search image: {file_path}")
            self.update_display()

    def load_mask(self, event):
        """Load mask image via file dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select Mask Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")]
        )
        root.destroy()

        if file_path:
            self.mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded mask: {file_path}")

    def crop_template_from_search(self, event):
        """
        Interactively crop a region from the search image to use as template.
        This also 'teaches' the ROI (stores the crop position for offset
        calculation), mirroring teach_feature_offset().
        """
        if self.search_img is None:
            print("Please load a search image first!")
            return

        print("\n" + "=" * 50)
        print("CROP TEMPLATE FROM SEARCH IMAGE")
        print("=" * 50)
        print("Instructions:")
        print("  1. Draw a rectangle around the fiducial mark")
        print("  2. Press ENTER or SPACE to confirm")
        print("  3. Press 'C' to cancel")
        print("=" * 50)

        display_img = self.search_img.copy()
        window_name = "Draw rectangle around template - ENTER to confirm, C to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 800)

        roi = cv2.selectROI(window_name, display_img,
                           fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        x, y, w, h = roi

        if w > 0 and h > 0:
            # Store ROI for offset calculation (mirrors teach_feature_offset)
            self.taught_roi = (x, y, w, h)

            if len(self.search_img.shape) == 3:
                self.template_img = cv2.cvtColor(
                    self.search_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY
                )
            else:
                self.template_img = self.search_img[y:y+h, x:x+w].copy()

            print(f"\nTemplate cropped: {w}x{h} pixels from ({x}, {y})")
            print(f"  ROI taught at ({x}, {y}) — offset will be relative to this center")
            self.update_display()
        else:
            print("Crop cancelled or invalid selection.")

    # ==================================================================
    # Detection
    # ==================================================================
    def run_detection(self, event):
        """Run template matching using the EmguCV pipeline"""
        if self.template_img is None:
            print("Please load or crop a template image first!")
            return
        if self.search_img is None:
            print("Please load a search image first!")
            return

        # --- Get match method from radio ---
        method_name = self.method_radio.value_selected
        method_map = {
            'CCORR Normed':  cv2.TM_CCORR_NORMED,
            'CCOEFF Normed': cv2.TM_CCOEFF_NORMED,
            'SQDIFF Normed': cv2.TM_SQDIFF_NORMED,
        }
        match_method = method_map.get(method_name, cv2.TM_CCORR_NORMED)

        # --- Configure the matcher ---
        self.vision = EmguVisionTemplateMatching()
        self.vision.recipe_param.fiducial_template_match_score_threshold = \
            float(self.sliders['THRESHOLD'].val)
        self.vision.recipe_param.max_rotation_retries = \
            int(self.sliders['MAX_RETRIES'].val)
        self.vision.recipe_param.rotation_step = \
            float(self.sliders['ROT_STEP'].val)
        self.vision.recipe_param.match_method = match_method

        # --- Teach ---
        if self.taught_roi is not None:
            x, y, w, h = self.taught_roi
        else:
            # If template was loaded directly (not cropped), assume ROI at (0,0)
            h_t, w_t = self.template_img.shape[:2]
            x, y, w, h = 0, 0, w_t, h_t

        # Teach with the search image and ROI
        self.vision.recipe_param.roi_config.wafer_fiducial_start_x = x
        self.vision.recipe_param.roi_config.wafer_fiducial_start_y = y
        self.vision.recipe_param.roi_config.wafer_fiducial_width = w
        self.vision.recipe_param.roi_config.wafer_fiducial_height = h

        # Store template directly (already cropped)
        if len(self.template_img.shape) == 2:
            self.vision.fiducial_tmp_image = cv2.cvtColor(
                self.template_img, cv2.COLOR_GRAY2BGR)
        else:
            self.vision.fiducial_tmp_image = self.template_img.copy()

        # Teach mask if available
        if self.mask_img is not None:
            self.vision.teach_mask(self.mask_img)

        # --- Run inspection ---
        print(f"\n[{method_name}] Running detection...")
        t_start = time.time()
        offset_x, offset_y, angle, score, plot = \
            self.vision.inspect_feature_offset(self.search_img, debug=False)
        elapsed = (time.time() - t_start) * 1000
        print(f"  Elapsed: {elapsed:.1f} ms")

        # --- Show heatmap if enabled ---
        if self.vis_check.get_status()[0]:
            self._show_heatmap(match_method)

        # --- Visualize results ---
        self.ax_result.clear()

        is_sqdiff = (match_method == cv2.TM_SQDIFF_NORMED)
        found = (not is_sqdiff and score >= self.vision.recipe_param.fiducial_template_match_score_threshold) or \
                (is_sqdiff and score <= (1.0 - self.vision.recipe_param.fiducial_template_match_score_threshold))

        if found:
            self.ax_result.imshow(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title("Match Found")

            # Update status panel
            self.ax_status.clear()
            self.ax_status.axis('off')
            status = f"Match Result:\n"
            status += f"  Score: {score:.4f}\n"
            status += f"  Offset X: {offset_x:.3f}\n"
            status += f"  Offset Y: {offset_y:.3f}\n"
            status += f"  Angle: {angle:.1f}\u00b0\n"
            status += f"  Time: {elapsed:.1f} ms\n"
            status += f"  Method: {method_name}"
            if self.mask_img is not None:
                status += "\n  [Mask applied]"
            self.ax_status.text(0.1, 0.95, status,
                               transform=self.ax_status.transAxes,
                               fontsize=8, fontfamily='monospace',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round',
                                        facecolor='lightgreen', alpha=0.8))
        else:
            if len(self.search_img.shape) == 2:
                self.ax_result.imshow(self.search_img, cmap='gray')
            else:
                self.ax_result.imshow(cv2.cvtColor(self.search_img,
                                                    cv2.COLOR_BGR2RGB))
            self.ax_result.set_title("No match found", color='red')

            self.ax_status.clear()
            self.ax_status.axis('off')
            fail_status = f"No Match:\n"
            fail_status += f"  Best Score: {score:.4f}\n"
            fail_status += f"  Threshold: {self.vision.recipe_param.fiducial_template_match_score_threshold:.2f}\n"
            fail_status += f"  Rot Tried: {angle:.1f}\u00b0\n"
            fail_status += f"  Method: {method_name}"
            self.ax_status.text(0.1, 0.95, fail_status,
                               transform=self.ax_status.transAxes,
                               fontsize=8, fontfamily='monospace',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round',
                                        facecolor='lightyellow', alpha=0.8))

        self.ax_result.axis('off')
        self.fig.canvas.draw_idle()

    def _show_heatmap(self, match_method):
        """Show the template matching score heatmap"""
        if self.template_img is None or self.search_img is None:
            return

        print("\n--- Generating score heatmap ---")

        if len(self.search_img.shape) == 3:
            gray_search = cv2.cvtColor(self.search_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_search = self.search_img.copy()

        gray_tmpl = self.template_img
        if len(gray_tmpl.shape) == 3:
            gray_tmpl = cv2.cvtColor(gray_tmpl, cv2.COLOR_BGR2GRAY)

        # Run matching for heatmap
        if self.mask_img is not None:
            output = cv2.matchTemplate(gray_search, gray_tmpl,
                                        match_method, mask=self.mask_img)
        else:
            output = cv2.matchTemplate(gray_search, gray_tmpl, match_method)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Template Matching Heatmap (Close to continue)',
                    fontsize=14)

        axes[0].imshow(gray_tmpl, cmap='gray')
        axes[0].set_title('Template')
        axes[0].axis('off')

        axes[1].imshow(gray_search, cmap='gray')
        axes[1].set_title('Search Image')
        axes[1].axis('off')

        im = axes[2].imshow(output, cmap='hot')
        axes[2].set_title('Score Heatmap')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plt.show()

    # ==================================================================
    # UI Callbacks
    # ==================================================================
    def update(self, val):
        """Slider update callback"""
        pass  # Detection runs on button click

    def update_rotation(self, val):
        """Rotate the search image based on slider value"""
        if self.original_search_img is None:
            return

        angle = int(self.sliders['ROTATION'].val)

        if angle == 0:
            self.search_img = self.original_search_img.copy()
        else:
            h, w = self.original_search_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)

            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            self.search_img = cv2.warpAffine(
                self.original_search_img, M, (new_w, new_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        self.ax_result.clear()
        if len(self.search_img.shape) == 2:
            self.ax_result.imshow(self.search_img, cmap='gray')
        else:
            self.ax_result.imshow(cv2.cvtColor(self.search_img,
                                                cv2.COLOR_BGR2RGB))
        self.ax_result.set_title(f"Search Image (Rotated {angle}\u00b0)")
        self.ax_result.axis('off')
        self.fig.canvas.draw_idle()

    def update_display(self):
        """Update the display with current images"""
        self.ax_template.clear()
        if self.template_img is not None:
            self.ax_template.imshow(self.template_img, cmap='gray')
            title = f"Template ({self.template_img.shape[1]}x{self.template_img.shape[0]})"
            if self.taught_roi is not None:
                title += f"\nROI: {self.taught_roi}"
            self.ax_template.set_title(title, fontsize=9)
        else:
            self.ax_template.set_title("Template (not loaded)")
        self.ax_template.axis('off')

        self.ax_result.clear()
        if self.search_img is not None:
            if len(self.search_img.shape) == 2:
                self.ax_result.imshow(self.search_img, cmap='gray')
            else:
                self.ax_result.imshow(cv2.cvtColor(self.search_img,
                                                    cv2.COLOR_BGR2RGB))
            self.ax_result.set_title(
                f"Search Image ({self.search_img.shape[1]}x{self.search_img.shape[0]})")
        else:
            self.ax_result.set_title("Search Image (not loaded)")
        self.ax_result.axis('off')

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive tuner"""
        plt.show()


# ==================================================================
# Entry Point
# ==================================================================
def main():
    """Main entry point"""
    print("=" * 60)
    print("  EmguCV Template Matching - Interactive Tuner")
    print("=" * 60)
    print()
    print("  Pipeline (mirrors C# WC_REQ):")
    print("  - cv2.matchTemplate with mask support")
    print("  - Rotation retry loop (configurable)")
    print("  - Offset from taught ROI center")
    print()
    print("  Controls:")
    print("  - Load Template  - Select template image")
    print("  - Load Search    - Select search image")
    print("  - Crop Template  - Crop + teach ROI from search image")
    print("  - Load Mask      - Optional mask for matching")
    print("  - Detect!        - Run WC_REQ pipeline")
    print()
    print("=" * 60)

    tuner = InteractiveEmguTuner()
    tuner.show()


if __name__ == "__main__":
    main()
