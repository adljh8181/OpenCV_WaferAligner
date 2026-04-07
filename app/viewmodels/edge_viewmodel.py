"""
================================================================================
EDGE VIEWMODEL  (ViewModel layer)
================================================================================
Holds all business logic for the "Find Wafer Edge" tab:
  - Running the EdgeLineFinder algorithm
  - Visualizing the result (overlay image + gradient plots)
  - Reading / writing the per-direction edge config cache in AppState

The ViewModel knows nothing about Tkinter widgets.  The View (EdgeTab) wires
the button command to run_find_edge() and passes in the Tkinter variables it
needs to read parameters from.
================================================================================
"""

import cv2
import numpy as np

from app.services.edge_finder import EdgeLineFinder, EdgeFinderConfig
from app.services.fov_classifier import preprocess_image, create_gradient_kernel
from app.models.app_state import AppState


class EdgeViewModel:
    """
    Orchestrates the Find-Wafer-Edge workflow.

    Parameters injected by the View:
      state        - shared AppState
      log_callback - callable(msg: str) to write to the status log
    """

    def __init__(self, state: AppState, log_callback=None):
        self.state = state
        self._log = log_callback or print

        # Algorithm instance (one per ViewModel, reused across calls)
        self.edge_finder = EdgeLineFinder(EdgeFinderConfig())

    # ------------------------------------------------------------------
    # Public API called by the View
    # ------------------------------------------------------------------

    def run_find_edge(self, tk_vars: dict) -> dict | None:
        """
        Read parameters from Tkinter StringVars supplied by the View,
        run EdgeLineFinder, and return the raw result dict.

        Args:
            tk_vars: dict with keys:
              'edge_img_var'      – StringVar holding image path
              'edge_kernel_var'   – StringVar for kernel size
              'edge_thresh_var'   – StringVar for edge threshold
              'edge_regions_var'  – StringVar for num regions
              'edge_border_var'   – StringVar for border ignore pct
              'edge_ransac_var'   – StringVar for RANSAC threshold
              'edge_dir_var'      – StringVar for scan direction

        Returns:
            Raw result dict from EdgeLineFinder, or None on error.
        """
        path = tk_vars['edge_img_var'].get()
        if not path:
            return None

        import os
        if not os.path.exists(path):
            return None

        try:
            kernel_size = int(tk_vars['edge_kernel_var'].get())
            if kernel_size % 2 == 0:
                kernel_size += 1

            self.edge_finder.config.KERNEL_SIZE      = kernel_size
            self.edge_finder.config.EDGE_THRESHOLD    = int(tk_vars['edge_thresh_var'].get())
            self.edge_finder.config.NUM_REGIONS        = int(tk_vars['edge_regions_var'].get())
            self.edge_finder.config.BORDER_IGNORE_PCT = float(tk_vars['edge_border_var'].get())
            self.edge_finder.config.RANSAC_THRESHOLD  = float(tk_vars['edge_ransac_var'].get())
            self.edge_finder.config.SCAN_DIRECTION    = tk_vars['edge_dir_var'].get()
            self.edge_finder.config.EDGE_POLARITY     = tk_vars['edge_polarity_var'].get()
            self.edge_finder.kernel = create_gradient_kernel(kernel_size)
        except Exception as e:
            self._log(f"Edge config parse error: {e}")
            return None

        return self.edge_finder.find_edge(path, skip_classification=True)

    def build_overlay_func(self, resp: dict, edge_finder=None) -> callable:
        """
        Returns a cv2 overlay draw-function for the result image label.
        The function takes a BGR image and draws the detected edge line + points.
        """
        e = resp['line_endpoints']
        ef = edge_finder or self.edge_finder   # capture for closure

        def draw_edge_overlay(img):
            h, w = img.shape[:2]
            is_vertical = resp.get('is_vertical_edge', True)

            if is_vertical:
                cv2.line(img, (int(e['x_top']), 0), (int(e['x_bot']), h),
                         (0, 255, 0), max(2, int(h / 200)))
            else:
                cv2.line(img, (0, int(e['y_left'])), (w, int(e['y_right'])),
                         (0, 255, 0), max(2, int(h / 200)))

            for pt in resp['detected_points']:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            for pt in resp['inliers']:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), 2)

            border_x = int(w * ef.config.BORDER_IGNORE_PCT)
            border_y = int(h * ef.config.BORDER_IGNORE_PCT)
            if is_vertical:
                cv2.line(img, (border_x, 0), (border_x, h), (100, 100, 100), 1)
                cv2.line(img, (w - border_x, 0), (w - border_x, h), (100, 100, 100), 1)
            else:
                cv2.line(img, (0, border_y), (w, border_y), (100, 100, 100), 1)
                cv2.line(img, (0, h - border_y), (w, h - border_y), (100, 100, 100), 1)
            return img

        return draw_edge_overlay

    def compute_gradient_display(self, resp: dict, edge_finder=None):
        """
        Compute the gradient magnitude image (for the bottom-left panel)
        and the 1-D gradient profile (for the matplotlib graph).

        Args:
            resp:        Result dict from EdgeLineFinder.find_edge()
            edge_finder: Optional EdgeLineFinder to use for kernel/config.
                         If None, uses the VM's own edge_finder.

        Returns:
            (grad_bgr, abs_gradient_1d, cfg)
            grad_bgr          – BGR image with HOT colormap + overlay
            abs_gradient_1d   – 1-D numpy array of gradient magnitudes
            cfg               – EdgeFinderConfig (for threshold + kernel size)
        """
        ef = edge_finder or self.edge_finder
        img_proc = resp['image']
        h_img, w_img = img_proc.shape
        kernel = ef.kernel
        is_vertical = resp.get('is_vertical_edge', True)
        cfg = ef.config
        e = resp['line_endpoints']

        # ── 2-D gradient magnitude ────────────────────────────────────────
        if is_vertical:
            grad_2d = cv2.filter2D(img_proc.astype(np.float64), -1,
                                   kernel.reshape(1, -1))
        else:
            grad_2d = cv2.filter2D(img_proc.astype(np.float64), -1,
                                   kernel.reshape(-1, 1))

        grad_mag = np.abs(grad_2d)
        if grad_mag.max() > 0:
            grad_display = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
        else:
            grad_display = grad_mag.astype(np.uint8)

        grad_bgr = cv2.applyColorMap(grad_display, cv2.COLORMAP_HOT)
        for pt in resp['detected_points']:
            cv2.circle(grad_bgr, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        for pt in resp['inliers']:
            cv2.circle(grad_bgr, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), 2)
        if is_vertical:
            cv2.line(grad_bgr, (int(e['x_top']), 0), (int(e['x_bot']), h_img),
                     (0, 255, 0), 2)
        else:
            cv2.line(grad_bgr, (0, int(e['y_left'])), (w_img, int(e['y_right'])),
                     (0, 255, 0), 2)

        # ── 1-D gradient profile ──────────────────────────────────────────
        if is_vertical:
            profile_1d = np.median(img_proc, axis=0).astype(np.float64)
        else:
            profile_1d = np.median(img_proc, axis=1).astype(np.float64)

        gradient_1d = np.convolve(profile_1d, kernel, mode='same')
        abs_gradient_1d = np.abs(gradient_1d)

        border_ignore = (int(len(abs_gradient_1d) * cfg.BORDER_IGNORE_PCT)
                         + cfg.KERNEL_SIZE)
        abs_gradient_1d[:border_ignore] = 0
        abs_gradient_1d[-border_ignore:] = 0

        return grad_bgr, abs_gradient_1d, cfg

    # ------------------------------------------------------------------
    # Config cache helpers (called by EdgeTab when direction changes)
    # ------------------------------------------------------------------

    def save_current_dir_to_cache(self, direction: str, tk_vars: dict):
        """Persist current slider values for *direction* into AppState."""
        cfg = self.state.edge_configs[direction]
        cfg["KernelSize"]      = tk_vars['edge_kernel_var'].get()
        cfg["EdgeThreshold"]   = tk_vars['edge_thresh_var'].get()
        cfg["NumRegions"]      = tk_vars['edge_regions_var'].get()
        cfg["BorderIgnorePct"] = tk_vars['edge_border_var'].get()
        cfg["RansacThreshold"] = tk_vars['edge_ransac_var'].get()
        cfg["EdgePolarity"]    = tk_vars['edge_polarity_var'].get()

    def get_cache_for_dir(self, direction: str) -> dict:
        """Return the cached config dict for *direction*."""
        return self.state.edge_configs.get(direction, {})
