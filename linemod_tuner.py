"""
================================================================================
LINE-2D INTERACTIVE TUNER
================================================================================
Interactive GUI for tuning LINE-2D shape-based matcher parameters.

Provides:
- Load template / search image via file dialog
- Crop template from search image
- Search mode selection (Simple, With Rotation, Full Search)
- Adjustable threshold, num features, weak threshold
- Quantized orientation visualization
- Pipeline visualization

Usage:
    python linemod_tuner.py

Author: Wafer Alignment System
================================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

from app.services.linemod_matcher import (
    LinemodMatcher, LinemodConfig,
    _quantize_gradients, _spread, _compute_response_maps,
    _extract_scattered_features,
)


class InteractiveLinemodTuner:
    """
    Interactive GUI for tuning LINE-2D parameters with live preview.
    """

    def __init__(self, template_path=None, search_path=None):
        self.config = LinemodConfig()
        self.config.PYRAMID_LEVELS = 1    # Single-level for tuner
        self.matcher = LinemodMatcher(self.config)

        # Load images
        self.template_img = None
        self.search_img = None

        if template_path and os.path.exists(template_path):
            self.template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if search_path and os.path.exists(search_path):
            self.search_img = cv2.imread(search_path)

        self.setup_ui()

    def setup_ui(self):
        """Setup the matplotlib UI"""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('LINE-2D Shape-Based Matcher - Interactive Tuner',
                         fontsize=14, fontweight='bold')

        # Image display areas
        self.ax_template = self.fig.add_axes([0.05, 0.55, 0.25, 0.35])
        self.ax_template.set_title("Template")
        self.ax_template.axis('off')

        self.ax_result = self.fig.add_axes([0.35, 0.30, 0.60, 0.62])
        self.ax_result.set_title("Detection Result")
        self.ax_result.axis('off')

        # Status panel
        self.ax_status = self.fig.add_axes([0.05, 0.35, 0.25, 0.15])
        self.ax_status.axis('off')

        # Mode selection - radio buttons
        self.fig.text(0.08, 0.28, 'SEARCH MODE', fontsize=10, fontweight='bold')
        mode_ax = self.fig.add_axes([0.05, 0.12, 0.12, 0.15])
        mode_ax.set_facecolor('lightgray')
        self.mode_radio = RadioButtons(mode_ax,
            ('Simple (Fast)', 'With Rotation', 'Full Search'),
            active=0)

        # Sliders
        slider_left = 0.20
        slider_width = 0.12
        slider_height = 0.025

        sliders_config = [
            ('ROTATION',        0, 360, 0,    1,    u'Rotation \u00b0'),
            ('MATCH_THRESHOLD', 20, 95, 50,   5,    'Threshold'),
            ('NUM_FEATURES',    16, 256, 128,  16,   'Num Feats'),
            ('WEAK_THRESHOLD',  5, 100, 30,   5,    'Grad Thr'),
        ]

        self.sliders = {}
        for i, (name, vmin, vmax, vinit, vstep, label) in enumerate(sliders_config):
            y_pos = 0.30 - i * 0.045
            ax = self.fig.add_axes([slider_left, y_pos, slider_width, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit,
                          valstep=vstep, color='lightblue')
            if name == 'ROTATION':
                slider.on_changed(self.update_rotation)
            else:
                slider.on_changed(self.update)
            self.sliders[name] = slider

        # Vis options — below the last slider
        vis_ax = self.fig.add_axes([0.20, 0.10, 0.12, 0.04])
        self.vis_check = CheckButtons(vis_ax, ['Show Orientations'], [False])

        # Store original search image for rotation
        self.original_search_img = None

        # Buttons
        btn_specs = [
            ([0.05, 0.05, 0.09, 0.035], 'Load Template', 'lightgreen', self.load_template),
            ([0.15, 0.05, 0.09, 0.035], 'Load Search',   'lightyellow', self.load_search),
            ([0.05, 0.01, 0.09, 0.035], 'Crop Template', 'lightcyan',   self.crop_template),
            ([0.15, 0.01, 0.09, 0.035], 'Detect!',       'lightcoral',  self.run_detection),
            ([0.25, 0.01, 0.09, 0.035], 'Pipeline',      'plum',        self.show_pipeline),
        ]

        self.buttons = []
        for pos, label, color, callback in btn_specs:
            ax = self.fig.add_axes(pos)
            btn = Button(ax, label, color=color)
            btn.on_clicked(callback)
            self.buttons.append(btn)

        self.update_display()

    def load_template(self, event):
        """Load template image via file dialog"""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        root.destroy()
        if path:
            self.template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            print("Loaded template:", path)
            self.update_display()

    def load_search(self, event):
        """Load search image via file dialog"""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Search Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        root.destroy()
        if path:
            self.search_img = cv2.imread(path)
            self.original_search_img = self.search_img.copy()
            self.sliders['ROTATION'].set_val(0)
            print("Loaded search:", path)
            self.update_display()

    def crop_template(self, event):
        """Crop template from search image interactively."""
        if self.search_img is None:
            print("Load a search image first!"); return

        print("\nDraw rectangle on pattern → ENTER to confirm, C to cancel")
        win = "Select template region"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 800)

        # For large images, resize for selection so the blue box is visible
        src = self.search_img
        sh, sw = src.shape[:2]
        max_dim = max(sh, sw)
        roi_scale = 1.0
        if max_dim > 1500:
            roi_scale = 1500.0 / max_dim
            display_img = cv2.resize(src, (int(sw * roi_scale), int(sh * roi_scale)),
                                     interpolation=cv2.INTER_AREA)
        else:
            display_img = src

        roi = cv2.selectROI(win, display_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)

        x, y, w, h = roi
        if w > 0 and h > 0:
            # Scale ROI back to original resolution
            if roi_scale != 1.0:
                x = int(x / roi_scale)
                y = int(y / roi_scale)
                w = int(w / roi_scale)
                h = int(h / roi_scale)

            self.template_img = cv2.cvtColor(src[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY) \
                if len(src.shape) == 3 else src[y:y+h, x:x+w].copy()
            print("Template cropped: %dx%d from (%d,%d)" % (w, h, x, y))
            self.update_display()

    def run_detection(self, event):
        """Run LINE-2D template matching."""
        if self.template_img is None or self.search_img is None:
            print("Load both template and search images!"); return

        # Configure from sliders
        mode = self.mode_radio.value_selected
        self.config.MATCH_THRESHOLD = float(self.sliders['MATCH_THRESHOLD'].val)
        self.config.NUM_FEATURES = int(self.sliders['NUM_FEATURES'].val)
        self.config.WEAK_THRESHOLD = float(self.sliders['WEAK_THRESHOLD'].val)
        self.config.PYRAMID_LEVELS = 2

        if mode == 'Simple (Fast)':
            self.config.ANGLE_STEP = 360
            self.config.SCALE_MIN = self.config.SCALE_MAX = 1.0
        elif mode == 'With Rotation':
            self.config.ANGLE_STEP = 5
            self.config.SCALE_MIN = self.config.SCALE_MAX = 1.0
        else:
            self.config.ANGLE_STEP = 5
            self.config.SCALE_MIN = 0.8; self.config.SCALE_MAX = 1.2

        self.matcher = LinemodMatcher(self.config)
        self.matcher.load_template(self.template_img)

        import time
        print("\nGenerating templates...")
        self.matcher.generate_templates()

        print("Matching...")
        t0 = time.time()
        matches = self.matcher.match(self.search_img, return_all=True)
        elapsed = (time.time() - t0) * 1000
        print("Done in %.1f ms, found %d match(es)" % (elapsed, len(matches)))

        # Show orientation vis if checked
        if self.vis_check.get_status()[0]:
            self._show_orientations()

        # Draw results
        self.ax_result.clear()
        if matches:
            vis = self.matcher.visualize_match(self.search_img, matches[0], show=False)
            self.ax_result.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            self.ax_result.set_title("Found %d match(es) in %.0fms" % (len(matches), elapsed))

            m = matches[0]
            self.ax_status.clear(); self.ax_status.axis('off')
            info = ("Best Match:\n"
                    "  Pos: (%d, %d)\n"
                    "  Angle: %.1f\u00b0\n"
                    "  Scale: %.2f\n"
                    "  Score: %.1f%%\n"
                    "  Features: %d\n"
                    "  Time: %.1f ms" % (
                        m['x'], m['y'], m['angle'], m['scale'],
                        m['score'], self.config.NUM_FEATURES, elapsed))
            self.ax_status.text(0.1, 0.9, info, transform=self.ax_status.transAxes,
                               fontsize=9, fontfamily='monospace',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            self._show_search()
            self.ax_result.set_title("No matches found", color='red')

        self.ax_result.axis('off')
        self.fig.canvas.draw_idle()

    def _show_orientations(self):
        """Show quantized orientation visualization."""
        if self.template_img is None or self.search_img is None:
            return

        gray_s = cv2.cvtColor(self.search_img, cv2.COLOR_BGR2GRAY) \
            if len(self.search_img.shape) == 3 else self.search_img
        weak = self.config.WEAK_THRESHOLD

        q_t, _ = _quantize_gradients(self.template_img, weak)
        q_s, _ = _quantize_gradients(gray_s, weak)
        T = self.config.T_PYRAMID[0]
        # Auto-scale T for large images (same logic as _match_single_level)
        sh, sw = gray_s.shape[:2]
        max_dim = max(sh, sw)
        if max_dim > 1500:
            T = max(T, int(T * max_dim / 1500))
        s_s = _spread(q_s, T)

        colors = np.array([
            [255,0,0],[255,170,0],[170,255,0],[0,255,0],
            [0,255,170],[0,170,255],[0,0,255],[170,0,255]], dtype=np.uint8)

        def colorize(q):
            vis = np.zeros((*q.shape, 3), dtype=np.uint8)
            for i in range(8):
                vis[(q & (1 << i)) > 0] = colors[i]
            return vis

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Quantized Orientations (8 bins)', fontsize=14)
        axes[0].imshow(cv2.cvtColor(colorize(q_t), cv2.COLOR_BGR2RGB))
        axes[0].set_title('Template'); axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(colorize(q_s), cv2.COLOR_BGR2RGB))
        axes[1].set_title('Search'); axes[1].axis('off')
        axes[2].imshow(cv2.cvtColor(colorize(s_s), cv2.COLOR_BGR2RGB))
        axes[2].set_title('Spread (T=%d)' % T); axes[2].axis('off')
        plt.tight_layout(); plt.show()

    def show_pipeline(self, event):
        """Show every processing phase as images in a 2x4 grid."""
        if self.template_img is None:
            print("Load a template first!"); return

        weak = float(self.sliders['WEAK_THRESHOLD'].val)
        T = self.config.T_PYRAMID[0]
        # Auto-scale T for large images (same logic as _match_single_level)
        if self.search_img is not None:
            sh, sw = self.search_img.shape[:2]
            max_dim = max(sh, sw)
            if max_dim > 1500:
                T = max(T, int(T * max_dim / 1500))
        num_feats = int(self.sliders['NUM_FEATURES'].val)

        # Determine which image to show pipeline for
        if self.search_img is not None:
            gray = cv2.cvtColor(self.search_img, cv2.COLOR_BGR2GRAY) \
                if len(self.search_img.shape) == 3 else self.search_img.copy()
            title_prefix = "Search Image"
        else:
            gray = self.template_img.copy()
            title_prefix = "Template"

        # Downsample large images — same logic as linemod_matcher.match()
        ds_scale = 1.0
        orig_shape = gray.shape
        sh0, sw0 = gray.shape[:2]
        max_dim = max(sh0, sw0)
        if max_dim > 1500:
            ds_scale = 1500.0 / max_dim
            new_w = int(sw0 * ds_scale)
            new_h = int(sh0 * ds_scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            title_prefix += f" (downsampled {sw0}×{sh0} → {new_w}×{new_h}, scale={ds_scale:.3f})"
            print(f"  [Pipeline] Downsampled {sw0}×{sh0} → {new_w}×{new_h} (scale={ds_scale:.3f})")
            # Reset T to base value — no need to inflate, we work on the smaller image
            T = self.config.T_PYRAMID[0]

        # Also downsample template to match ds_scale
        tmpl_gray = self.template_img.copy()
        if ds_scale < 1.0:
            th0, tw0 = tmpl_gray.shape[:2]
            tmpl_gray = cv2.resize(tmpl_gray,
                                   (int(tw0 * ds_scale), int(th0 * ds_scale)),
                                   interpolation=cv2.INTER_AREA)

        # ---- Phase 1: Original ----
        original = gray.copy()

        # ---- Phase 2: Sobel Gradients ----
        dx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx*dx + dy*dy)
        angle = np.degrees(np.arctan2(dy, dx)) % 360.0

        # ---- Phase 3: Quantization ----
        quantized, mag = _quantize_gradients(gray, weak)

        # ---- Phase 4: Spread ----
        spread = _spread(quantized, T)

        # ---- Phase 5: Response Maps ----
        rmaps = _compute_response_maps(spread)
        # Combine: take max response across all 8 labels
        response_combined = np.max(np.stack(rmaps, axis=0), axis=0)

        # ---- Phase 6: Feature Extraction (on template) ----
        q_tmpl, mag_tmpl = _quantize_gradients(self.template_img, weak)
        features = _extract_scattered_features(q_tmpl, mag_tmpl, num_feats)
        actual_feats = len(features)
        # Count how many edge pixels survived quantization
        edge_pixels = np.count_nonzero(q_tmpl)
        print(f"  [Pipeline] Template edge pixels: {edge_pixels}, "
              f"extracted features: {actual_feats}/{num_feats}")

        # ---- Phase 7: Score Map (if search image available) ----
        score_map = None
        if self.search_img is not None and len(features) > 0:
            # Crop template features for scoring
            from app.services.linemod_matcher import _crop_templates, TemplatePyr
            templ = TemplatePyr()
            templ.pyramid_level = 0
            templ.features = [type(features[0])(f.x, f.y, f.label) for f in features]
            _crop_templates([templ])

            tw, th = templ.width, templ.height
            sh, sw = gray.shape[:2]
            vy, vx = sh - th, sw - tw
            if vy > 0 and vx > 0:
                score_map = np.zeros((vy, vx), dtype=np.int32)
                valid = 0
                for feat in templ.features:
                    fx, fy = feat.x, feat.y
                    if fy + vy <= sh and fx + vx <= sw and fy >= 0 and fx >= 0:
                        score_map += rmaps[feat.label][fy:fy+vy, fx:fx+vx].astype(np.int32)
                        valid += 1
                if valid > 0:
                    score_map = (score_map * 100.0) / (4 * valid)

        # ---- Orientation colorization ----
        ori_colors = np.array([
            [255,50,50],[255,170,0],[170,255,0],[0,220,0],
            [0,255,170],[0,170,255],[50,50,255],[170,0,255]], dtype=np.uint8)

        def colorize(q):
            vis = np.zeros((*q.shape, 3), dtype=np.uint8)
            for i in range(8):
                mask = (q & (1 << i)) > 0
                vis[mask] = ori_colors[i]
            return vis

        # ---- Feature overlay on template ----
        feat_vis = cv2.cvtColor(self.template_img, cv2.COLOR_GRAY2BGR)
        for f in features:
            color = tuple(int(c) for c in ori_colors[f.label])
            cv2.circle(feat_vis, (f.x, f.y), 3, color, -1)
            cv2.circle(feat_vis, (f.x, f.y), 4, (255,255,255), 1)

        # ---- Create figure ----
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('LINE-2D Pipeline — Every Phase (%s)' % title_prefix,
                     fontsize=16, fontweight='bold')

        # Row 1: Preprocessing (applies to BOTH training & matching)
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('1. Input Image\n(Grayscale)', fontsize=11, fontweight='bold', color='navy')

        axes[0,1].imshow(magnitude, cmap='hot')
        axes[0,1].set_title('2. Gradient Magnitude\n(Sobel)', fontsize=11, fontweight='bold', color='navy')

        axes[0,2].imshow(angle, cmap='hsv', vmin=0, vmax=360)
        axes[0,2].set_title('3. Gradient Angle\n(0\u00b0-360\u00b0)', fontsize=11, fontweight='bold', color='navy')

        axes[0,3].imshow(colorize(quantized))
        axes[0,3].set_title('4. Quantized (8 bins)\n+ Hysteresis Filter', fontsize=11, fontweight='bold', color='navy')

        # Row 2: Matching + Training feature extraction
        axes[1,0].imshow(colorize(spread))
        axes[1,0].set_title('5. Spread (T=%d)\nOR neighbourhood [Matching]' % T, fontsize=11, fontweight='bold', color='red')

        axes[1,1].imshow(response_combined, cmap='hot', vmin=0, vmax=4)
        axes[1,1].set_title('6. Response Maps\nPer-Direction Scores [Matching]', fontsize=11, fontweight='bold', color='red')

        axes[1,2].imshow(cv2.cvtColor(feat_vis, cv2.COLOR_BGR2RGB))
        axes[1,2].set_title('7. Extracted Features\n(%d/%d points) [Training]' % (actual_feats, num_feats), fontsize=11, fontweight='bold', color='green')

        if score_map is not None:
            im = axes[1,3].imshow(score_map, cmap='jet', vmin=0, vmax=100)
            axes[1,3].set_title('8. Score Map\nFeatures × Response [Matching]', fontsize=11, fontweight='bold', color='red')
            plt.colorbar(im, ax=axes[1,3], shrink=0.8)
        else:
            axes[1,3].text(0.5, 0.5, 'Load search image\nto see score map',
                          ha='center', va='center', fontsize=12, transform=axes[1,3].transAxes)
            axes[1,3].set_title('8. Score Map', fontsize=11, fontweight='bold', color='red')

        for ax in axes.flat:
            ax.axis('off')

        # Add phase labels — positioned clearly between rows
        fig.text(0.5, 0.94, 'Steps 1\u20134: Preprocessing (both Training & Matching)',
                ha='center', fontsize=12, color='navy', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
        fig.text(0.25, 0.49, 'Steps 5\u20136, 8: Matching (search image)',
                ha='center', fontsize=10, color='red', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.9))
        fig.text(0.65, 0.49, 'Step 7: Training (from template)',
                ha='center', fontsize=10, color='green', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.subplots_adjust(hspace=0.35)
        plt.show()
        print("Pipeline visualization complete.")

    def update(self, val):
        pass

    def update_rotation(self, val):
        """Rotate search image."""
        if self.original_search_img is None:
            return
        angle = int(self.sliders['ROTATION'].val)
        if angle == 0:
            self.search_img = self.original_search_img.copy()
        else:
            h, w = self.original_search_img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            cos, sin = np.abs(M[0,0]), np.abs(M[0,1])
            nw, nh = int(h*sin + w*cos), int(h*cos + w*sin)
            M[0,2] += (nw - w) / 2; M[1,2] += (nh - h) / 2
            self.search_img = cv2.warpAffine(self.original_search_img, M, (nw, nh))
        self._show_search()
        self.ax_result.set_title("Rotated %d\u00b0" % angle)
        self.fig.canvas.draw_idle()

    def _show_search(self):
        """Display search image."""
        self.ax_result.clear()
        if self.search_img is not None:
            disp = self.search_img if len(self.search_img.shape) == 2 \
                else cv2.cvtColor(self.search_img, cv2.COLOR_BGR2RGB)
            self.ax_result.imshow(disp, cmap='gray' if len(disp.shape)==2 else None)
        self.ax_result.axis('off')

    def update_display(self):
        """Refresh template and search displays."""
        self.ax_template.clear()
        if self.template_img is not None:
            self.ax_template.imshow(self.template_img, cmap='gray')
            self.ax_template.set_title("Template (%dx%d)" % (
                self.template_img.shape[1], self.template_img.shape[0]))
        else:
            self.ax_template.set_title("Template (not loaded)")
        self.ax_template.axis('off')
        self._show_search()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    print("=" * 55)
    print("  LINE-2D Shape-Based Matcher — Interactive Tuner")
    print("=" * 55)
    print()
    print("  Algorithm: SBM LINE-2D (quantized orientations)")
    print("  - 8 orientation bins + spread response maps")
    print("  - Sparse scattered features (LUT scoring)")
    print("  - Vectorized NumPy scoring (~300ms for 1080x1080)")
    print()
    InteractiveLinemodTuner().show()


if __name__ == "__main__":
    main()
