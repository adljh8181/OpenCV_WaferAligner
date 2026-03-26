"""
================================================================================
NCC vs LINE-2D — INTERACTIVE COMPARISON TUNER
================================================================================
Side-by-side comparison of NCC template matching (EmguCV method) vs
LINE-2D shape-based matching. Shows pipeline internals for both methods
and allows loading different search images to demonstrate NCC's
texture-dependency failure.

Key demo:
  - Train template on Image A
  - Search on Image B (different lighting/lot/texture)
  - NCC fails because pixel intensities differ
  - LINE-2D succeeds because edge DIRECTIONS are preserved

Usage:
    python ncc_vs_line2d.py

Author: Wafer Alignment System
================================================================================
"""

import sys
import os

# Add parent directory to path to allow importing from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time

from app.services.linemod_matcher import (
    LinemodMatcher, LinemodConfig,
    _quantize_gradients, _spread, _compute_response_maps,
)


# ======================================================================
# NCC Template Matching (Port of C# WC_REQ)
# ======================================================================

def ncc_match(search_gray, template_gray, threshold=0.7, max_rotation=5):
    """
    Python port of C# EmguCV WC_REQ.
    Uses TM_CCORR_NORMED (same as CcorrNormed in C#).
    """
    best = None
    for deg in range(0, max_rotation + 1):
        if deg == 0:
            tmpl = template_gray
        else:
            h, w = template_gray.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), deg, 1.0)
            tmpl = cv2.warpAffine(template_gray, M, (w, h), borderValue=0)

        result = cv2.matchTemplate(search_gray, tmpl, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        th, tw = tmpl.shape[:2]
        if best is None or max_val > best['score']:
            best = {
                'x': max_loc[0] + tw // 2,
                'y': max_loc[1] + th // 2,
                'score': max_val,
                'angle': float(deg),
                'bbox': (max_loc[0], max_loc[1], tw, th),
                'result_map': result,
            }
        if max_val >= threshold:
            return best
    return best


# ======================================================================
# Interactive Comparison Tuner
# ======================================================================

class ComparisonTuner:
    """
    Interactive GUI comparing NCC (texture-based) vs LINE-2D (edge-based).
    """

    def __init__(self):
        self.template_img = None     # Grayscale template
        self.search_img = None       # BGR search image
        self.search_gray = None      # Grayscale search image

        self.config = LinemodConfig()
        self.config.ANGLE_STEP = 360
        self.config.NUM_FEATURES = 128
        self.config.WEAK_THRESHOLD = 30
        self.config.PYRAMID_LEVELS = 1
        self.config.MATCH_THRESHOLD = 30

        self.matcher = None
        self.last_ncc = None
        self.last_l2d = None

        self._setup_ui()

    def _setup_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle('NCC Template Matching vs LINE-2D — Interactive Comparison',
                         fontsize=14, fontweight='bold')

        # Top row: Template / NCC Result / LINE-2D Result
        self.ax_tmpl = self.fig.add_axes([0.02, 0.52, 0.20, 0.40])
        self.ax_tmpl.set_title('Template', fontsize=10, fontweight='bold')
        self.ax_tmpl.axis('off')

        self.ax_ncc = self.fig.add_axes([0.25, 0.52, 0.35, 0.40])
        self.ax_ncc.set_title('NCC Result', fontsize=10, fontweight='bold', color='steelblue')
        self.ax_ncc.axis('off')

        self.ax_l2d = self.fig.add_axes([0.63, 0.52, 0.35, 0.40])
        self.ax_l2d.set_title('LINE-2D Result', fontsize=10, fontweight='bold', color='coral')
        self.ax_l2d.axis('off')

        # Status text
        self.ax_status = self.fig.add_axes([0.02, 0.35, 0.20, 0.15])
        self.ax_status.axis('off')

        # Buttons
        btn_specs = [
            ([0.02, 0.28, 0.09, 0.04], 'Load Template', 'lightgreen', self._load_template),
            ([0.12, 0.28, 0.09, 0.04], 'Load Search',   'lightyellow', self._load_search),
            ([0.02, 0.23, 0.09, 0.04], 'Crop Template', 'lightcyan', self._crop_template),
            ([0.12, 0.23, 0.09, 0.04], 'Compare!',      'lightcoral', self._run_comparison),
            ([0.02, 0.18, 0.09, 0.04], 'NCC Pipeline',  'lightskyblue', self._show_ncc_pipeline),
            ([0.12, 0.18, 0.09, 0.04], 'L2D Pipeline',  'lightsalmon', self._show_l2d_pipeline),
        ]

        self.buttons = []
        for pos, label, color, cb in btn_specs:
            ax = self.fig.add_axes(pos)
            btn = Button(ax, label, color=color)
            btn.on_clicked(cb)
            self.buttons.append(btn)

        # Brightness slider
        self.ax_bright = self.fig.add_axes([0.25, 0.44, 0.30, 0.025])
        self.sl_bright = Slider(self.ax_bright, 'Brightness', -150, 150,
                               valinit=0, valstep=10, color='gold')

        # Contrast slider
        self.ax_contrast = self.fig.add_axes([0.25, 0.40, 0.30, 0.025])
        self.sl_contrast = Slider(self.ax_contrast, 'Contrast', 0.1, 3.0,
                                 valinit=1.0, valstep=0.1, color='gold')

        # Gamma slider
        self.ax_gamma = self.fig.add_axes([0.63, 0.44, 0.30, 0.025])
        self.sl_gamma = Slider(self.ax_gamma, 'Gamma', 0.2, 4.0,
                              valinit=1.0, valstep=0.1, color='gold')

        # Info text at bottom
        self.ax_info = self.fig.add_axes([0.02, 0.02, 0.96, 0.14])
        self.ax_info.axis('off')
        self._draw_info_table()

        self._refresh_display()

    def _draw_info_table(self):
        self.ax_info.clear()
        self.ax_info.axis('off')
        info = (
            "WORKFLOW:  1) Load Template (from Image A)  →  "
            "2) Load Search (different Image B)  →  "
            "3) Click Compare!  →  "
            "4) Click NCC/L2D Pipeline to see internals\n\n"
            "WHY NCC FAILS:  NCC compares pixel brightness directly. If Image B has different "
            "lighting/texture, the same pattern looks 'different' to NCC → wrong position or low score.\n"
            "WHY LINE-2D WORKS:  LINE-2D compares edge DIRECTIONS only. Same pattern always has the "
            "same edge directions regardless of brightness/contrast → finds correct position."
        )
        self.ax_info.text(0.02, 0.95, info, transform=self.ax_info.transAxes,
                         fontsize=8.5, verticalalignment='top', fontfamily='sans-serif',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    def _pick_file(self, title="Select Image"):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        root.destroy()
        return path

    def _load_template(self, event):
        path = self._pick_file("Select TEMPLATE Image (Image A)")
        if not path: return
        self.template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded template: {path} ({self.template_img.shape[1]}x{self.template_img.shape[0]})")
        self._refresh_display()

    def _load_search(self, event):
        path = self._pick_file("Select SEARCH Image (can be DIFFERENT image)")
        if not path: return
        self.search_img = cv2.imread(path)
        self.search_gray = cv2.cvtColor(self.search_img, cv2.COLOR_BGR2GRAY) \
            if len(self.search_img.shape) == 3 else self.search_img.copy()
        print(f"Loaded search: {path} ({self.search_gray.shape[1]}x{self.search_gray.shape[0]})")
        self._refresh_display()

    def _crop_template(self, event):
        if self.search_img is None:
            print("Load a search image first!"); return
        print("\nDraw rectangle → ENTER to confirm, C to cancel")
        win = "Crop Template Region"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 800)
        roi = cv2.selectROI(win, self.search_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)
        x, y, w, h = roi
        if w > 0 and h > 0:
            self.template_img = self.search_gray[y:y+h, x:x+w].copy()
            print(f"Cropped template: {w}x{h} from ({x},{y})")
            self._refresh_display()

    def _get_modified_search(self):
        """Apply brightness/contrast/gamma sliders to the search image."""
        if self.search_gray is None:
            return None
        img = self.search_gray.astype(np.float32)

        # Contrast
        contrast = self.sl_contrast.val
        mean = np.mean(img)
        img = (img - mean) * contrast + mean

        # Brightness
        brightness = self.sl_bright.val
        img = img + brightness

        # Gamma
        gamma = self.sl_gamma.val
        if gamma != 1.0:
            img = np.clip(img, 0, 255)
            img = ((img / 255.0) ** gamma) * 255.0

        return np.clip(img, 0, 255).astype(np.uint8)

    def _run_comparison(self, event):
        if self.template_img is None or self.search_gray is None:
            print("Load both template and search images!"); return

        modified = self._get_modified_search()

        # ---- NCC ----
        t0 = time.time()
        self.last_ncc = ncc_match(modified, self.template_img, threshold=0.5)
        ncc_ms = (time.time() - t0) * 1000

        # ---- LINE-2D ----
        self.config.NUM_FEATURES = 128
        self.config.WEAK_THRESHOLD = 30
        self.matcher = LinemodMatcher(self.config)
        self.matcher.load_template(self.template_img)
        self.matcher.generate_templates()

        t0 = time.time()
        self.last_l2d = self.matcher.match(modified, threshold=30)
        l2d_ms = (time.time() - t0) * 1000

        # ---- Draw NCC ----
        self.ax_ncc.clear()
        vis_ncc = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)
        if self.last_ncc:
            bx, by, bw, bh = self.last_ncc['bbox']
            score_pct = self.last_ncc['score'] * 100
            cv2.rectangle(vis_ncc, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)
            cv2.circle(vis_ncc, (self.last_ncc['x'], self.last_ncc['y']), 6, (0, 0, 255), -1)
            self.ax_ncc.set_title(
                f"NCC: Score={score_pct:.1f}% at ({self.last_ncc['x']},{self.last_ncc['y']}) [{ncc_ms:.0f}ms]",
                fontsize=10, fontweight='bold', color='steelblue')
        else:
            self.ax_ncc.set_title("NCC: No match", fontsize=10, color='red')
        self.ax_ncc.imshow(cv2.cvtColor(vis_ncc, cv2.COLOR_BGR2RGB))
        self.ax_ncc.axis('off')

        # ---- Draw LINE-2D ----
        self.ax_l2d.clear()
        vis_l2d = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)
        if self.last_l2d:
            bx, by, bw, bh = self.last_l2d['bbox']
            cv2.rectangle(vis_l2d, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)
            cv2.circle(vis_l2d, (self.last_l2d['x'], self.last_l2d['y']), 6, (0, 0, 255), -1)
            self.ax_l2d.set_title(
                f"LINE-2D: Score={self.last_l2d['score']:.1f}% at ({self.last_l2d['x']},{self.last_l2d['y']}) [{l2d_ms:.0f}ms]",
                fontsize=10, fontweight='bold', color='coral')
        else:
            self.ax_l2d.set_title("LINE-2D: No match", fontsize=10, color='red')
        self.ax_l2d.imshow(cv2.cvtColor(vis_l2d, cv2.COLOR_BGR2RGB))
        self.ax_l2d.axis('off')

        # ---- Status ----
        self.ax_status.clear(); self.ax_status.axis('off')
        ncc_s = self.last_ncc['score'] * 100 if self.last_ncc else 0
        l2d_s = self.last_l2d['score'] if self.last_l2d else 0
        status = (
            f"NCC Score:    {ncc_s:.1f}%\n"
            f"LINE-2D Score: {l2d_s:.1f}%\n\n"
            f"Brightness: {self.sl_bright.val:+.0f}\n"
            f"Contrast:   {self.sl_contrast.val:.1f}x\n"
            f"Gamma:      {self.sl_gamma.val:.1f}"
        )
        self.ax_status.text(0.05, 0.95, status, transform=self.ax_status.transAxes,
                           fontsize=9, fontfamily='monospace', verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        self.fig.canvas.draw_idle()
        print(f"NCC: {ncc_s:.1f}% in {ncc_ms:.0f}ms | LINE-2D: {l2d_s:.1f}% in {l2d_ms:.0f}ms")

    def _show_ncc_pipeline(self, event):
        """Show NCC internal pipeline."""
        if self.template_img is None or self.search_gray is None:
            print("Load images and run Compare! first"); return

        modified = self._get_modified_search()
        tmpl = self.template_img

        # NCC pipeline stages
        result_map = cv2.matchTemplate(modified, tmpl, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result_map)
        th, tw = tmpl.shape[:2]

        # Product image (what NCC internally computes)
        # NCC correlation = sum(T * I) / (norm(T)*norm(I))
        # Show what the template pixels look like vs search patch
        bx, by = max_loc
        if by + th <= modified.shape[0] and bx + tw <= modified.shape[1]:
            best_patch = modified[by:by+th, bx:bx+tw]
        else:
            best_patch = np.zeros_like(tmpl)

        diff = cv2.absdiff(tmpl, best_patch)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('NCC Template Matching — Internal Pipeline\n'
                     '(Compares PIXEL INTENSITIES directly)',
                     fontsize=14, fontweight='bold', color='steelblue')

        # Row 1: Inputs
        axes[0,0].imshow(tmpl, cmap='gray')
        axes[0,0].set_title('1. Template\n(Training Pixels)', fontsize=11, fontweight='bold')
        axes[0,0].axis('off')

        axes[0,1].imshow(modified, cmap='gray')
        axes[0,1].set_title('2. Search Image\n(Modified Pixels)', fontsize=11, fontweight='bold')
        axes[0,1].axis('off')

        # Template intensities histogram
        axes[0,2].hist(tmpl.ravel(), bins=50, color='steelblue', alpha=0.7, label='Template')
        axes[0,2].hist(modified.ravel(), bins=50, color='coral', alpha=0.5, label='Search')
        axes[0,2].set_title('3. Pixel Intensity Distribution\n(NCC relies on these matching!)',
                           fontsize=11, fontweight='bold')
        axes[0,2].legend(fontsize=9)
        axes[0,2].set_xlabel('Pixel Value (0-255)')

        # Row 2: Results
        im = axes[1,0].imshow(result_map, cmap='jet')
        axes[1,0].set_title(f'4. NCC Correlation Map\n(max={max_val:.4f})',
                           fontsize=11, fontweight='bold')
        axes[1,0].axis('off')
        plt.colorbar(im, ax=axes[1,0], shrink=0.8)

        # Best matched patch vs template
        axes[1,1].imshow(np.hstack([tmpl, best_patch]), cmap='gray')
        axes[1,1].axvline(x=tw, color='red', linewidth=2)
        axes[1,1].set_title(f'5. Template vs Best Patch\n(Left=Template, Right=Found)',
                           fontsize=11, fontweight='bold')
        axes[1,1].axis('off')

        # Pixel difference
        axes[1,2].imshow(diff, cmap='hot')
        mean_diff = np.mean(diff)
        axes[1,2].set_title(f'6. Pixel Difference (|T-I|)\nMean diff = {mean_diff:.1f}',
                           fontsize=11, fontweight='bold',
                           color='red' if mean_diff > 30 else 'green')
        axes[1,2].axis('off')

        # Explanation text
        fig.text(0.5, 0.02,
                '⚠ NCC compares raw pixel values. When brightness/contrast changes, '
                'pixel values shift → correlation drops → wrong match or low score.',
                ha='center', fontsize=11, color='red', fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))

        plt.tight_layout(rect=[0, 0.06, 1, 0.93])
        plt.show()

    def _show_l2d_pipeline(self, event):
        """Show LINE-2D internal pipeline."""
        if self.template_img is None or self.search_gray is None:
            print("Load images and run Compare! first"); return

        modified = self._get_modified_search()
        tmpl = self.template_img
        weak = self.config.WEAK_THRESHOLD
        T = self.config.T_PYRAMID[0]

        # Quantize template
        q_tmpl, mag_tmpl = _quantize_gradients(tmpl, weak)

        # Quantize search
        q_search, mag_search = _quantize_gradients(modified, weak)

        # Spread
        spread = _spread(q_search, T)

        # Response maps
        rmaps = _compute_response_maps(spread)
        response_max = np.max(np.stack(rmaps, axis=0), axis=0)

        # Color map
        ori_colors = np.array([
            [255,50,50],[255,170,0],[170,255,0],[0,220,0],
            [0,255,170],[0,170,255],[50,50,255],[170,0,255]], dtype=np.uint8)

        def colorize(q):
            vis = np.zeros((*q.shape, 3), dtype=np.uint8)
            for i in range(8):
                vis[(q & (1 << i)) > 0] = ori_colors[i]
            return vis

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('LINE-2D Shape-Based Matching — Internal Pipeline\n'
                     '(Compares EDGE DIRECTIONS only, ignores pixel values)',
                     fontsize=14, fontweight='bold', color='coral')

        # Row 1: Template processing
        axes[0,0].imshow(tmpl, cmap='gray')
        axes[0,0].set_title('1. Template Image\n(Raw pixels — IGNORED by LINE-2D)',
                           fontsize=11, fontweight='bold')
        axes[0,0].axis('off')

        axes[0,1].imshow(colorize(q_tmpl))
        axes[0,1].set_title('2. Template Orientations\n(8 direction bins, color-coded)',
                           fontsize=11, fontweight='bold')
        axes[0,1].axis('off')

        axes[0,2].imshow(modified, cmap='gray')
        axes[0,2].set_title('3. Search Image\n(Modified brightness/contrast)',
                           fontsize=11, fontweight='bold')
        axes[0,2].axis('off')

        # Row 2: Search processing + matching
        axes[1,0].imshow(colorize(q_search))
        axes[1,0].set_title('4. Search Orientations\n(Same edge directions despite brightness!)',
                           fontsize=11, fontweight='bold')
        axes[1,0].axis('off')

        axes[1,1].imshow(colorize(spread))
        axes[1,1].set_title(f'5. Spread (T={T})\n(OR over {T}×{T} neighbourhood)',
                           fontsize=11, fontweight='bold')
        axes[1,1].axis('off')

        axes[1,2].imshow(response_max, cmap='hot', vmin=0, vmax=4)
        axes[1,2].set_title('6. Response Map\n(Score 4=exact, 1=neighbor, 0=miss)',
                           fontsize=11, fontweight='bold')
        axes[1,2].axis('off')

        # Explanation text
        fig.text(0.5, 0.02,
                '✓ LINE-2D extracts edge DIRECTIONS (not pixel values). '
                'Even with inverted/brightened/darkened images, edges point the same way → match succeeds.',
                ha='center', fontsize=11, color='green', fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.9))

        plt.tight_layout(rect=[0, 0.06, 1, 0.93])
        plt.show()

    def _refresh_display(self):
        self.ax_tmpl.clear()
        if self.template_img is not None:
            self.ax_tmpl.imshow(self.template_img, cmap='gray')
            h, w = self.template_img.shape[:2]
            self.ax_tmpl.set_title(f'Template ({w}×{h})', fontsize=10, fontweight='bold')
        else:
            self.ax_tmpl.set_title('Template\n(not loaded)', fontsize=10)
        self.ax_tmpl.axis('off')

        self.ax_ncc.clear()
        self.ax_l2d.clear()
        if self.search_gray is not None:
            disp = cv2.cvtColor(self.search_img, cv2.COLOR_BGR2RGB) \
                if len(self.search_img.shape) == 3 else self.search_gray
            self.ax_ncc.imshow(disp if len(disp.shape) == 3 else disp, cmap='gray')
            self.ax_ncc.set_title('NCC — Load & Compare', fontsize=10, color='steelblue')
            self.ax_l2d.imshow(disp if len(disp.shape) == 3 else disp, cmap='gray')
            self.ax_l2d.set_title('LINE-2D — Load & Compare', fontsize=10, color='coral')
        else:
            self.ax_ncc.set_title('NCC Result\n(load search image)', fontsize=10, color='gray')
            self.ax_l2d.set_title('LINE-2D Result\n(load search image)', fontsize=10, color='gray')
        self.ax_ncc.axis('off')
        self.ax_l2d.axis('off')

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NCC vs LINE-2D — Interactive Comparison Tuner")
    print("=" * 60)
    print()
    print("  DEMO WORKFLOW:")
    print("  1. Load Template from Image A")
    print("  2. Load Search — use a DIFFERENT image (different lot)")
    print("  3. Click Compare!")
    print("  4. Adjust Brightness/Contrast/Gamma sliders")
    print("  5. Click Compare! again to see how each method copes")
    print("  6. Click NCC/L2D Pipeline to see internals")
    print()
    ComparisonTuner().show()
