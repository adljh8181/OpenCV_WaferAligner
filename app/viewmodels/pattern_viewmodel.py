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
        
        # Smart verification to prevent duplicate template generation
        self._last_template_config_str = None

    def _get_current_config_str(self):
        """Returns a string representation of the current config settings"""
        cfg = self.linemod_matcher.config
        img_hash = hash(self.linemod_matcher.template_image.tobytes()) if self.linemod_matcher.template_image is not None else 0
        mask_hash = hash(self.linemod_matcher.detection_mask.tobytes()) if self.linemod_matcher.detection_mask is not None else 0
        return f"{cfg.NUM_FEATURES}_{cfg.WEAK_THRESHOLD}_{cfg.T_PYRAMID}_{cfg.HYSTERESIS_KERNEL}_{cfg.ANGLE_STEP}_{cfg.SCALE_MIN}_{cfg.SCALE_MAX}_{img_hash}_{mask_hash}"

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
            cfg.T_PYRAMID        = [t_spread, t_spread * 2, t_spread * 4]
            cfg.HYSTERESIS_KERNEL = int(tk_vars['pattern_hyst_var'].get())
            cfg.PYRAMID_LEVELS   = 3

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
        Also loads the detection mask if one exists alongside the template.

        Returns True on success, False on failure.
        """
        if not os.path.exists(path):
            return False
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        self.apply_ui_configs(tk_vars)

        # Check for a saved detection mask alongside the template
        mask = self._try_load_mask(path)
        self.state.template_detection_mask = mask

        self.linemod_matcher.load_template(img, detection_mask=mask)
        self.linemod_matcher.generate_templates()
        self._last_template_config_str = self._get_current_config_str()
        self.state.template_loaded = True
        if mask is not None:
            self._log("Template loaded with detection mask.")
        else:
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

        win_name = "Select template region (Drag rect, ENTER=confirm, ESC=cancel)"
        roi = self._select_roi_safe(display_img, win_name)

        if roi is None:
            return None
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

        # Remove previously generated temp files first
        try:
            for f in os.listdir(os.getcwd()):
                if f.startswith("temp_cropped_template_"):
                    os.remove(os.path.join(os.getcwd(), f))
        except Exception:
            pass

        import time
        timestamp = int(time.time() * 1000)
        cropped = src[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]
        out_path = os.path.join(os.getcwd(), f"temp_cropped_template_{timestamp}.png")
        cv2.imwrite(out_path, cropped)

        img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            self.apply_ui_configs(tk_vars)
            # Reset detection mask when a new template is cropped
            self.state.template_detection_mask = None
            self.linemod_matcher.load_template(img, detection_mask=None)
            self.linemod_matcher.generate_templates()
            self._last_template_config_str = self._get_current_config_str()
            self.state.template_loaded = True
            self._log(
                f"Cropped template loaded. Trained centre: "
                f"({self.state.template_crop_cx:.1f}, "
                f"{self.state.template_crop_cy:.1f})"
            )

        return out_path

    def draw_detection_roi(self, template_path: str, tk_vars: dict) -> np.ndarray | None:
        """
        Open the mask editor on the template image so the user can draw a
        polygon detection region.

        If the user confirms a polygon, the mask is saved to disk alongside
        the template, stored in AppState, and templates are regenerated.

        Returns the mask (or None if cancelled).
        """
        if not template_path or not os.path.exists(template_path):
            self._log("No template image to draw ROI on.")
            return None

        template_img = cv2.imread(template_path)
        if template_img is None:
            self._log("Failed to read template image.")
            return None

        from app.views.mask_editor import draw_detection_mask
        mask = draw_detection_mask(template_img,
                                   window_title="Draw Detection Region (L-click=add, R-click=close)")

        if mask is None:
            self._log("ROI drawing cancelled — using full template.")
            return None

        # Save mask to disk next to the template
        mask_path = self._mask_path_for(template_path)
        cv2.imwrite(mask_path, mask)
        self._log(f"Detection mask saved: {mask_path}")

        # Store in state and rebuild templates
        self.state.template_detection_mask = mask
        self.apply_ui_configs(tk_vars)
        img_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.linemod_matcher.load_template(img_gray, detection_mask=mask)
        self.linemod_matcher.generate_templates()
        self._last_template_config_str = self._get_current_config_str()
        self._log("Templates regenerated with detection mask.")
        return mask

    def clear_detection_mask(self, template_path: str, tk_vars: dict):
        """
        Remove the detection mask and regenerate templates using the full
        template image.
        """
        self.state.template_detection_mask = None
        mask_path = self._mask_path_for(template_path)
        if os.path.exists(mask_path):
            os.remove(mask_path)
            self._log("Detection mask removed.")

        if template_path and os.path.exists(template_path):
            self.apply_ui_configs(tk_vars)
            img_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            self.linemod_matcher.load_template(img_gray, detection_mask=None)
            self.linemod_matcher.generate_templates()
            self._last_template_config_str = self._get_current_config_str()
            self._log("Templates regenerated without mask (full template).")

    # ------------------------------------------------------------------
    # Detection mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_path_for(template_path: str) -> str:
        """Return the mask PNG path corresponding to a template path."""
        base, ext = os.path.splitext(template_path)
        return base + "_mask.png"

    def _try_load_mask(self, template_path: str) -> np.ndarray | None:
        """Load a detection mask from disk if it exists."""
        mask_path = self._mask_path_for(template_path)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self._log(f"Loaded detection mask: {mask_path}")
                return mask
        return None

    @staticmethod
    def _select_roi_safe(image, window_title):
        """
        Custom rectangle ROI selector with Save/Close buttons.
        Returns (x, y, w, h) or None if cancelled.
        """
        from app.views.mask_editor import _draw_button, _point_in_rect

        h_i, w_i = image.shape[:2]
        drawing = False
        x0 = y0 = x1 = y1 = 0
        action = None  # 'save' or 'close'
        cursor_pos = (0, 0)

        # Button dimensions
        btn_w, btn_h = 90, 30
        btn_gap = 10
        btn_save_rect = (w_i - 2 * btn_w - btn_gap - 10, h_i - btn_h - 10,
                         btn_w, btn_h)
        btn_close_rect = (w_i - btn_w - 10, h_i - btn_h - 10,
                          btn_w, btn_h)

        def _mouse_cb(event, mx, my, flags, param):
            nonlocal drawing, x0, y0, x1, y1, action, cursor_pos
            cursor_pos = (mx, my)
            if event == cv2.EVENT_LBUTTONDOWN:
                if _point_in_rect(mx, my, btn_save_rect):
                    action = 'save'
                    return
                if _point_in_rect(mx, my, btn_close_rect):
                    action = 'close'
                    return
                drawing = True
                x0, y0 = mx, my
                x1, y1 = mx, my
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                x1, y1 = mx, my
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1 = mx, my

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_title, _mouse_cb)

        confirmed = False
        cancelled = False

        while True:
            vis = image.copy()
            if x0 != x1 and y0 != y1:
                rx = min(x0, x1)
                ry = min(y0, y1)
                rw = abs(x1 - x0)
                rh = abs(y1 - y0)
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh),
                              (0, 255, 0), 2)
                cx, cy = rx + rw // 2, ry + rh // 2
                cv2.line(vis, (rx, cy), (rx + rw, cy), (0, 255, 0), 1)
                cv2.line(vis, (cx, ry), (cx, ry + rh), (0, 255, 0), 1)

            # Instructions
            font_s = max(0.4, 0.5 * (w_i / 1000))
            cv2.putText(vis, "Drag rectangle to select region",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, font_s,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, "Drag rectangle to select region",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, font_s,
                        (255, 255, 0), 1, cv2.LINE_AA)

            # Buttons
            mx, my = cursor_pos
            _draw_button(vis, "Save", *btn_save_rect,
                         color=(40, 120, 40),
                         hover=_point_in_rect(mx, my, btn_save_rect))
            _draw_button(vis, "Close", *btn_close_rect,
                         color=(40, 40, 140),
                         hover=_point_in_rect(mx, my, btn_close_rect))

            cv2.imshow(window_title, vis)
            key = cv2.waitKey(16) & 0xFF

            # Detect X button close
            try:
                if cv2.getWindowProperty(window_title,
                                         cv2.WND_PROP_VISIBLE) < 1:
                    cancelled = True
                    break
            except cv2.error:
                cancelled = True
                break

            # Button clicks
            if action == 'save':
                confirmed = True
                action = None
                break
            elif action == 'close':
                cancelled = True
                action = None
                break

            if key == 27:
                cancelled = True
                break
            elif key in (13, 10):
                confirmed = True
                break

        cv2.destroyWindow(window_title)
        cv2.waitKey(1)

        if cancelled or not confirmed:
            return None

        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = abs(x1 - x0)
        rh = abs(y1 - y0)
        if rw <= 0 or rh <= 0:
            return None
        return (rx, ry, rw, rh)

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

        # Apply latest slider values to config
        self.apply_ui_configs(tk_vars)

        # Smart Verification: Only generate templates if config has changed
        current_config_str = self._get_current_config_str()
        if self._last_template_config_str != current_config_str:
            self._log("[Pattern] Settings changed. Re-building templates...")
            self.linemod_matcher.generate_templates()
            self._last_template_config_str = current_config_str
        else:
            self._log("[Pattern] Using existing cached templates (settings unchanged).")

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

        features    = _extract_scattered_features(
            q_tmpl, mag_tmpl, num_feats,
            mask=self.linemod_matcher.detection_mask)
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
                
                # Pre-cast response maps to int32 once to avoid massive memory allocations in the loop
                rmaps_int32 = [r.astype(np.int32) for r in rmaps]
                
                for feat in templ.features:
                    fx, fy = feat.x, feat.y
                    if (fy + vy <= sh_i and fx + vx <= sw_i
                            and fy >= 0 and fx >= 0):
                        score_map += rmaps_int32[feat.label][fy:fy+vy, fx:fx+vx]
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

        # Overlay detection mask (green tint) so user can verify features
        # are only within the painted region
        det_mask = self.linemod_matcher.detection_mask
        if det_mask is not None and det_mask.shape == tmpl_gray.shape:
            mask_overlay = feat_vis.copy()
            mask_overlay[det_mask > 0] = (
                mask_overlay[det_mask > 0].astype(float) * 0.5 +
                [0, 80, 0]
            ).clip(0, 255).astype('uint8')
            # Dim region OUTSIDE the mask so it's obvious
            mask_overlay[det_mask == 0] = (
                mask_overlay[det_mask == 0].astype(float) * 0.3
            ).clip(0, 255).astype('uint8')
            feat_vis = mask_overlay

        for f in features:
            color = tuple(int(c) for c in ori_colors[f.label])
            cv2.circle(feat_vis, (f.x, f.y), feat_dot_r, color, -1)
            cv2.circle(feat_vis, (f.x, f.y), feat_dot_r + 1, (255, 255, 255), 1)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        title_prefix = "Search Image" if search_img is not None else "Template"
        fig.suptitle(f'LINE-2D Pipeline — Every Phase ({title_prefix})',
                     fontsize=16, fontweight='bold')

        # Helper to avoid Matplotlib memory crashes (ArrayMemoryError) on hi-res sensor images
        def _downsample(img, max_dim=1024):
            if img is None: return None
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                nh, nw = int(h * scale), int(w * scale)
                # Use nearest neighbor to preserve distinct labels/colors without blurring
                return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
            return img

        axes[0, 0].imshow(_downsample(original), cmap='gray')
        axes[0, 0].set_title('1. Input', color='navy')
        axes[0, 1].imshow(_downsample(magnitude), cmap='hot')
        axes[0, 1].set_title('2. Gradient Magnitude', color='navy')
        axes[0, 2].imshow(_downsample(angle), cmap='hsv', vmin=0, vmax=360)
        axes[0, 2].set_title('3. Gradient Angle', color='navy')
        axes[0, 3].imshow(_downsample(colorize(quantized)))
        axes[0, 3].set_title('4. Quantized (+Hysteresis)', color='navy')

        axes[1, 0].imshow(_downsample(colorize(spread)))
        axes[1, 0].set_title(f'5. Spread (T={T})', color='red')
        axes[1, 1].imshow(_downsample(response_combined), cmap='hot', vmin=0, vmax=4)
        axes[1, 1].set_title('6. Response Maps', color='red')
        axes[1, 2].imshow(_downsample(cv2.cvtColor(feat_vis, cv2.COLOR_BGR2RGB)))
        mask_label = " + MASK" if self.linemod_matcher.detection_mask is not None else ""
        axes[1, 2].set_title(f'7. Extracted ({actual_feats}/{num_feats}){mask_label}',
                             color='green')

        if score_map is not None:
            im = axes[1, 3].imshow(_downsample(score_map), cmap='jet', vmin=0, vmax=100)
            axes[1, 3].set_title('8. Score Map', color='red')
            plt.colorbar(im, ax=axes[1, 3], shrink=0.8)
        else:
            axes[1, 3].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[1, 3].set_title('8. Score Map', color='red')

        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()
        _show_figure_in_window(fig, "LINE-2D Pipeline")
