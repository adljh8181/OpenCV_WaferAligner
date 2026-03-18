"""
================================================================================
PATTERN VIEWMODEL  (ViewModel layer)
================================================================================
Holds all business logic for the "Find Pattern" tab:
  - Applying UI config to LinemodMatcher
  - Running template matching (run_find_pattern)
  - Loading and cropping template images
  - Diagnostic visualizations (show_orientations, show_pipeline)

The ViewModel knows nothing about Tkinter widgets.  The View wires buttons
to these methods and supplies Tkinter variables for reading parameters.
================================================================================
"""

import os
import cv2
import numpy as np

from app.services.linemod_matcher import (
    LinemodMatcher, LinemodConfig,
    _quantize_gradients, _spread, _compute_response_maps,
    _extract_scattered_features
)
from app.models.app_state import AppState


class PatternViewModel:
    """
    Orchestrates the Find-Pattern / Template-Matching workflow.

    Parameters injected by the View:
      state        - shared AppState
      log_callback - callable(msg: str) to write to the status log
    """

    def __init__(self, state: AppState, log_callback=None):
        self.state = state
        self._log = log_callback or print

        # Algorithm instance (one per ViewModel, reused across calls)
        self.linemod_matcher = LinemodMatcher(LinemodConfig())

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def apply_ui_configs(self, tk_vars: dict):
        """
        Copy slider values from the View's Tkinter variables into the
        LinemodMatcher config.

        Args:
            tk_vars: dict with keys pattern_thresh_var, pattern_num_var,
                     pattern_weak_var, pattern_tspread_var,
                     pattern_hyst_var, pattern_mode_var
        """
        try:
            mode = tk_vars['pattern_mode_var'].get()
            cfg = self.linemod_matcher.config
            cfg.MATCH_THRESHOLD  = float(tk_vars['pattern_thresh_var'].get())
            cfg.NUM_FEATURES     = int(tk_vars['pattern_num_var'].get())
            cfg.WEAK_THRESHOLD   = -float(tk_vars['pattern_weak_var'].get())
            t_spread = int(tk_vars['pattern_tspread_var'].get())
            cfg.T_PYRAMID        = [t_spread, t_spread * 2]
            cfg.HYSTERESIS_KERNEL = int(tk_vars['pattern_hyst_var'].get())
            cfg.PYRAMID_LEVELS   = 2

            if mode == 'Simple (Fast)':
                cfg.ANGLE_STEP  = 360
                cfg.SCALE_MIN   = cfg.SCALE_MAX = 1.0
            elif mode == 'With Rotation':
                cfg.ANGLE_STEP  = 5
                cfg.SCALE_MIN   = cfg.SCALE_MAX = 1.0
            else:  # Full Search
                cfg.ANGLE_STEP  = 5
                cfg.SCALE_MIN   = 0.8
                cfg.SCALE_MAX   = 1.2
        except Exception as e:
            self._log(f"Pattern config error: {e}")

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def load_template_from_path(self, path: str, tk_vars: dict) -> bool:
        """
        Load a template image from disk and build the template pyramid.

        Returns True on success, False on failure.
        """
        if not os.path.exists(path):
            return False
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        self.apply_ui_configs(tk_vars)
        self.linemod_matcher.load_template(img)
        self.linemod_matcher.generate_templates()
        self.state.template_loaded = True
        self._log("Template natively loaded into matcher.")
        return True

    def crop_template_from_image(self, search_path: str, tk_vars: dict) -> str | None:
        """
        Open an interactive OpenCV ROI selector on *search_path*, crop the
        selected region, save it to disk as temp_cropped_template.png, load it
        into the matcher, and return the output path (or None if cancelled).
        """
        if not search_path or not os.path.exists(search_path):
            return None

        src = cv2.imread(search_path)
        if src is None:
            return None

        sh, sw = src.shape[:2]

        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w = int(user32.GetSystemMetrics(0) * 0.85)
            screen_h = int(user32.GetSystemMetrics(1) * 0.85)
        except Exception:
            screen_w, screen_h = 1280, 900

        roi_scale = min(screen_w / sw, screen_h / sh, 1.0)
        if roi_scale < 1.0:
            display_img = cv2.resize(src,
                                     (int(sw * roi_scale), int(sh * roi_scale)),
                                     interpolation=cv2.INTER_AREA)
        else:
            display_img = src
            roi_scale = 1.0

        win_name = "Select template region (ENTER=confirm, C=cancel)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, display_img.shape[1], display_img.shape[0])
        roi = cv2.selectROI(win_name, display_img, fromCenter=False,
                            showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return None

        # Map back to original image pixels
        orig_x = int(x / roi_scale)
        orig_y = int(y / roi_scale)
        orig_w = int(w / roi_scale)
        orig_h = int(h / roi_scale)

        # Store trained template centre
        self.state.template_crop_cx = orig_x + orig_w / 2.0
        self.state.template_crop_cy = orig_y + orig_h / 2.0

        cropped = src[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]
        out_path = os.path.join(os.getcwd(), "temp_cropped_template.png")
        cv2.imwrite(out_path, cropped)

        img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.apply_ui_configs(tk_vars)
            self.linemod_matcher.load_template(img)
            self.linemod_matcher.generate_templates()
            self.state.template_loaded = True
            self._log(
                f"Cropped template loaded. Trained centre: "
                f"({self.state.template_crop_cx:.1f}, "
                f"{self.state.template_crop_cy:.1f})"
            )

        return out_path

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def run_find_pattern(self, tk_vars: dict):
        """
        Match the template against the search image.

        Returns:
            (resp, orig_color) where resp is the matcher result dict
            (or None if no match) and orig_color is the BGR search image.
        """
        path = tk_vars['pattern_img_var'].get()
        if not path or not os.path.exists(path):
            return None, None

        if not self.state.template_loaded:
            return None, None

        # Apply latest slider values to config (but skip template rebuild —
        # templates are already generated when the user loaded/cropped them).
        self.apply_ui_configs(tk_vars)

        # Lazy build: if templates were never generated (e.g. recipe was loaded
        # but the user hasn't clicked Detect yet), build them now.
        if not self.linemod_matcher.template_pyramids:
            self._log("[Pattern] Building templates (first run after recipe load)…")
            self.linemod_matcher.generate_templates()

        img       = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        orig_color = cv2.imread(path)
        if img is None:
            return None, None

        try:
            rot = int(tk_vars['pattern_rot_var'].get())
            if rot != 0:
                h2, w2 = img.shape[:2]
                M   = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), rot, 1.0)
                cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
                nw  = int(h2 * sin + w2 * cos)
                nh  = int(h2 * cos + w2 * sin)
                M[0, 2] += (nw - w2) / 2
                M[1, 2] += (nh - h2) / 2
                img        = cv2.warpAffine(img, M, (nw, nh))
                orig_color = cv2.warpAffine(orig_color, M, (nw, nh))
        except Exception:
            pass

        resp = self.linemod_matcher.match(img)
        return resp, orig_color

    def compute_delta(self, resp: dict) -> tuple[float, float]:
        """
        Compute Delta X/Y:  match_center − trained_template_center.
        Mirrors dFndXOffset / dFndYOffset in EmguVision.cs.
        """
        match_cx = float(resp['x'])
        match_cy = float(resp['y'])

        cx = self.state.template_crop_cx
        cy = self.state.template_crop_cy

        if cx is not None:
            return match_cx - cx, match_cy - cy
        return match_cx, match_cy

    # ------------------------------------------------------------------
    # Diagnostic visualizations  (open separate Toplevel windows)
    # ------------------------------------------------------------------

    def show_orientations(self, tk_vars: dict):
        """
        Display quantized orientation maps for template, search, and spread images.
        Opens a non-blocking Matplotlib window.
        """
        import matplotlib.pyplot as plt
        from app.views.main_window import _show_figure_in_window

        template_img = self.linemod_matcher.template_image
        search_path  = tk_vars.get('pattern_img_var', None)
        search_img   = (cv2.imread(search_path.get(), cv2.IMREAD_GRAYSCALE)
                        if search_path else None)

        try:
            weak = float(tk_vars['pattern_weak_var'].get())
            T    = int(tk_vars['pattern_tspread_var'].get())
            hk   = int(tk_vars['pattern_hyst_var'].get())
        except Exception as e:
            self._log(f"Invalid parameters: {e}")
            return

        if template_img is None or search_img is None:
            self._log("Load both template and search images first.")
            return

        q_t, _ = _quantize_gradients(template_img, -weak, kernel_size=hk)
        q_s, _ = _quantize_gradients(search_img,   -weak, kernel_size=hk)
        s_s    = _spread(q_s, T)

        colors = np.array([
            [255, 0, 0], [255, 170, 0], [170, 255, 0], [0, 255, 0],
            [0, 255, 170], [0, 170, 255], [0, 0, 255], [170, 0, 255]
        ], dtype=np.uint8)

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
        axes[2].set_title(f'Spread (T={T})'); axes[2].axis('off')
        plt.tight_layout()
        _show_figure_in_window(fig, "Quantized Orientations (8 bins)")

    def show_pipeline(self, tk_vars: dict):
        """
        Display the full LINE-2D processing pipeline in a Matplotlib window.
        """
        import matplotlib.pyplot as plt
        from app.services.linemod_matcher import (
            _crop_templates, TemplatePyr
        )
        from app.views.main_window import _show_figure_in_window

        template_img = self.linemod_matcher.template_image
        search_path  = tk_vars.get('pattern_img_var', None)
        search_img   = (cv2.imread(search_path.get(), cv2.IMREAD_GRAYSCALE)
                        if search_path and search_path.get() else None)

        if template_img is None:
            self._log("Load a template first!")
            return

        try:
            weak     = float(tk_vars['pattern_weak_var'].get())
            T        = int(tk_vars['pattern_tspread_var'].get())
            num_feats = int(tk_vars['pattern_num_var'].get())
            hk       = int(tk_vars['pattern_hyst_var'].get())
        except Exception as e:
            self._log(f"Invalid parameters: {e}")
            return

        if hk <= 0:
            ref_src = search_img if search_img is not None else template_img
            ref_dim = max(ref_src.shape[:2])
            raw_ks  = max(3, int(3 * ref_dim / 1500))
            hk      = min(raw_ks | 1, 9)
        else:
            hk = max(3, hk | 1)

        gray     = (search_img.copy() if search_img is not None
                    else template_img.copy())
        tmpl_gray = template_img.copy()

        original  = gray.copy()
        dx        = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        dy        = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx * dx + dy * dy)
        angle     = np.degrees(np.arctan2(dy, dx)) % 360.0

        quantized, mag = _quantize_gradients(gray,     -weak, kernel_size=hk)
        q_tmpl,   mag_tmpl = _quantize_gradients(tmpl_gray, -weak, kernel_size=hk)
        spread    = _spread(quantized, T)
        rmaps     = _compute_response_maps(spread)
        response_combined = np.max(np.stack(rmaps, axis=0), axis=0)

        features    = _extract_scattered_features(q_tmpl, mag_tmpl, num_feats)
        actual_feats = len(features)

        # ── Score map ────────────────────────────────────────────────────
        score_map = None
        sh_i, sw_i = gray.shape[:2]
        if search_img is not None and features:
            templ = TemplatePyr()
            templ.pyramid_level = 0
            templ.features = [type(features[0])(f.x, f.y, f.label)
                              for f in features]
            _crop_templates([templ])
            tw, th = templ.width, templ.height
            vy, vx = sh_i - th, sw_i - tw
            if vy > 0 and vx > 0:
                score_map = np.zeros((vy, vx), dtype=np.int32)
                valid = 0
                for feat in templ.features:
                    fx, fy = feat.x, feat.y
                    if (fy + vy <= sh_i and fx + vx <= sw_i
                            and fy >= 0 and fx >= 0):
                        score_map += rmaps[feat.label][fy:fy+vy,
                                                       fx:fx+vx].astype(np.int32)
                        valid += 1
                if valid > 0:
                    score_map = (score_map * 100.0) / (4 * valid)

        ori_colors = np.array([
            [255, 50, 50], [255, 170, 0], [170, 255, 0], [0, 220, 0],
            [0, 255, 170], [0, 170, 255], [50, 50, 255], [170, 0, 255]
        ], dtype=np.uint8)

        def colorize(q):
            vis = np.zeros((*q.shape, 3), dtype=np.uint8)
            for i in range(8):
                vis[(q & (1 << i)) > 0] = ori_colors[i]
            return vis

        feat_dot_r = max(2, min(tmpl_gray.shape[:2]) // 80)
        feat_vis   = cv2.cvtColor(tmpl_gray, cv2.COLOR_GRAY2BGR)
        for f in features:
            color = tuple(int(c) for c in ori_colors[f.label])
            cv2.circle(feat_vis, (f.x, f.y), feat_dot_r, color, -1)
            cv2.circle(feat_vis, (f.x, f.y), feat_dot_r + 1, (255, 255, 255), 1)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        title_prefix = "Search Image" if search_img is not None else "Template"
        fig.suptitle(f'LINE-2D Pipeline — Every Phase ({title_prefix})',
                     fontsize=16, fontweight='bold')

        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('1. Input', color='navy')
        axes[0, 1].imshow(magnitude, cmap='hot')
        axes[0, 1].set_title('2. Gradient Magnitude', color='navy')
        axes[0, 2].imshow(angle, cmap='hsv', vmin=0, vmax=360)
        axes[0, 2].set_title('3. Gradient Angle', color='navy')
        axes[0, 3].imshow(colorize(quantized))
        axes[0, 3].set_title('4. Quantized (+Hysteresis)', color='navy')

        axes[1, 0].imshow(colorize(spread))
        axes[1, 0].set_title(f'5. Spread (T={T})', color='red')
        axes[1, 1].imshow(response_combined, cmap='hot', vmin=0, vmax=4)
        axes[1, 1].set_title('6. Response Maps', color='red')
        axes[1, 2].imshow(cv2.cvtColor(feat_vis, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'7. Extracted ({actual_feats}/{num_feats})',
                             color='green')

        if score_map is not None:
            im = axes[1, 3].imshow(score_map, cmap='jet', vmin=0, vmax=100)
            axes[1, 3].set_title('8. Score Map', color='red')
            plt.colorbar(im, ax=axes[1, 3], shrink=0.8)
        else:
            axes[1, 3].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[1, 3].set_title('8. Score Map', color='red')

        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()
        _show_figure_in_window(fig, "LINE-2D Pipeline")
