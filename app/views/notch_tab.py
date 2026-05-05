"""
================================================================================
NOTCH TAB  (View layer)
================================================================================
Builds the "Find Wafer Notch" sub-tab inside the Recipe notebook.

Features:
  - Load any image from disk
  - Sliders for every pipeline parameter:
      Binarization Threshold, Gaussian kernel size, Morph-close kernel size,
      Canny low/high threshold, Hough min-votes / min-length / max-gap,
      Vertical angle limit, Slant angle limit
  - 3×2 live pipeline grid:
      [Original] [Blurred]  [Binary]
      [Morphed]  [Canny+Hough] [Result Overlay]
  - Pipeline updates automatically 300 ms after any slider move
  - Result panel: tip pixel coords, offset mm, pass/fail status
================================================================================
"""

import math
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.models.app_state import AppState


class NotchTab:
    """
    Owns all widgets in the "Find Wafer Notch" sub-tab.

    Parameters:
      parent_nb         - ttk.Notebook (recipe sub-notebook)
      state             - shared AppState
      display_cv2_image - callable(img, label) from main_window
      log_callback      - callable(msg: str)
    """

    def __init__(self, parent_nb: ttk.Notebook, state: AppState,
                 display_cv2_image, log_callback=None):
        self.state             = state
        self._display_cv2_image = display_cv2_image
        self._log              = log_callback or print

        self.tab = ttk.Frame(parent_nb)
        parent_nb.add(self.tab, text="Find Wafer Notch")
        parent_nb.hide(self.tab)

        # ── Image state ──────────────────────────────────────────────────
        self._src_img   = None   # original grayscale ndarray
        self._img_path  = ""

        # ── Debounce timer ───────────────────────────────────────────────
        self._debounce_id = None

        # ── Tkinter variables ────────────────────────────────────────────
        self.img_path_var   = tk.StringVar()

        # Binarization
        self.threshold_var  = tk.IntVar(value=150)
        # Gaussian
        self.k_blur_var     = tk.IntVar(value=0)     # 0 = auto (0.03 × w)
        # Morph close
        self.k_morph_var    = tk.IntVar(value=0)     # 0 = auto (0.048 × w)
        # Canny
        self.canny_low_var  = tk.IntVar(value=10)
        self.canny_high_var = tk.IntVar(value=30)
        # Hough
        self.hough_votes_var  = tk.IntVar(value=50)
        self.hough_len_var    = tk.IntVar(value=50)
        self.hough_gap_var    = tk.IntVar(value=10)
        # Line classification
        self.vert_angle_var   = tk.DoubleVar(value=5.0)
        self.slant_angle_var  = tk.DoubleVar(value=10.0)
        # Pixel resolution for mm conversion
        self.pix_res_var      = tk.DoubleVar(value=0.0065)

        # Results
        self.tip_px_var     = tk.StringVar(value="—")
        self.offset_mm_var  = tk.StringVar(value="—")
        self.status_var     = tk.StringVar(value="—")

        self._build()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build(self):
        # ── Outer split: left controls | right pipeline grid ────────────
        pane = ttk.PanedWindow(self.tab, orient='horizontal')
        pane.pack(expand=True, fill='both')

        # ── LEFT panel ──────────────────────────────────────────────────
        left = ttk.Frame(pane, width=310)
        left.pack_propagate(False)
        pane.add(left, weight=0)

        # Load Image
        lf_load = ttk.LabelFrame(left, text="Image")
        lf_load.pack(fill='x', padx=8, pady=(8, 4))

        self.btn_load = ttk.Button(lf_load, text="Load Image",
                                   command=self._on_load_image)
        self.btn_load.pack(anchor='w', padx=6, pady=(4, 2))
        ttk.Entry(lf_load, textvariable=self.img_path_var,
                  width=38, state='readonly').pack(
                      anchor='w', padx=6, pady=(0, 4))

        # Parameters
        lf_params = ttk.LabelFrame(left, text="Pipeline Parameters")
        lf_params.pack(fill='x', padx=8, pady=4)
        lf_params.columnconfigure(1, weight=1)

        slider_defs = [
            # (label, var, from_, to, resolution, fmt)
            ("Bin. Threshold:",  self.threshold_var,   0,    255, 1,    "{:d}"),
            ("Gaussian k\n(0=auto):", self.k_blur_var, 0,    99,  2,    "{:d}"),
            ("Morph k\n(0=auto):",    self.k_morph_var,0,    99,  1,    "{:d}"),
            ("Canny Low:",       self.canny_low_var,   1,    255, 1,    "{:d}"),
            ("Canny High:",      self.canny_high_var,  1,    255, 1,    "{:d}"),
            ("Hough Votes:",     self.hough_votes_var, 1,    300, 1,    "{:d}"),
            ("Hough Min Len:",   self.hough_len_var,   1,    300, 1,    "{:d}"),
            ("Hough Max Gap:",   self.hough_gap_var,   0,    100, 1,    "{:d}"),
            ("Vert. Angle °:",   self.vert_angle_var,  0.5,  30,  0.5,  "{:.1f}"),
            ("Slant Angle °:",   self.slant_angle_var, 1.0,  89,  0.5,  "{:.1f}"),
            ("Pixel Res mm/px:", self.pix_res_var,     0.0001, 1.0, 0.0001, "{:.4f}"),
        ]
        for row, (label, var, frm, to, res, fmt) in enumerate(slider_defs):
            self._make_slider(lf_params, row, label, var, frm, to, res, fmt)

        # Execute / Results
        lf_exec = ttk.LabelFrame(left, text="Execute")
        lf_exec.pack(fill='x', padx=8, pady=4)
        self.btn_run = ttk.Button(lf_exec, text="Run Pipeline",
                                  command=self._on_run, state='disabled')
        self.btn_run.pack(padx=6, pady=6)

        lf_result = ttk.LabelFrame(left, text="Result")
        lf_result.pack(fill='x', padx=8, pady=(4, 8))
        result_rows = [
            ("Status:",      self.status_var),
            ("Tip (px):",    self.tip_px_var),
            ("Offset (mm):", self.offset_mm_var),
        ]
        for i, (lbl, var) in enumerate(result_rows):
            ttk.Label(lf_result, text=lbl).grid(
                row=i, column=0, padx=6, pady=3, sticky='w')
            ent = ttk.Entry(lf_result, textvariable=var,
                            width=26, state='readonly')
            ent.grid(row=i, column=1, padx=6, pady=3, sticky='w')
        self._status_entry = lf_result.grid_slaves(row=0, column=1)[0] \
            if lf_result.grid_slaves(row=0, column=1) else None

        # ── RIGHT panel: 3×2 pipeline grid ──────────────────────────────
        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        titles = [
            "① Original",
            "② Gaussian Blur",
            "③ Binary Threshold",
            "④ Morph Close",
            "⑤ Canny + All Hough Lines",
            "⑥ Result Overlay",
        ]
        self._pipeline_labels = []

        for r in range(2):
            right.rowconfigure(r, weight=1, uniform="prow")
        for c in range(3):
            right.columnconfigure(c, weight=1, uniform="pcol")

        for idx, title in enumerate(titles):
            r, c = divmod(idx, 3)
            lf = ttk.LabelFrame(right, text=title)
            lf.grid(row=r, column=c, sticky='nsew',
                    padx=3 if c < 2 else 0,
                    pady=3 if r < 1 else 0)
            lf.rowconfigure(0, weight=1)
            lf.columnconfigure(0, weight=1)
            lbl = ttk.Label(lf, text="(No Image)", anchor='center')
            lbl.grid(row=0, column=0, sticky='nsew')
            self._pipeline_labels.append(lbl)

    # ------------------------------------------------------------------
    # Slider factory
    # ------------------------------------------------------------------

    def _make_slider(self, parent, row, label, var, frm, to, res, fmt):
        ttk.Label(parent, text=label, justify='right').grid(
            row=row, column=0, padx=(6, 2), pady=3, sticky='e')

        val_lbl = ttk.Label(parent, width=7, anchor='e', relief='sunken')
        val_lbl.grid(row=row, column=2, padx=(2, 6), pady=3, sticky='e')

        def _update_label(*_):
            try:
                raw = var.get()
                val_lbl.config(text=fmt.format(raw))
            except Exception:
                pass
            self._schedule_update()

        scale = ttk.Scale(parent, from_=frm, to=to,
                          orient='horizontal', variable=var,
                          command=_update_label)
        scale.grid(row=row, column=1, sticky='ew', padx=2, pady=3)

        # Initialise label immediately
        try:
            val_lbl.config(text=fmt.format(var.get()))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_image(self):
        path = filedialog.askopenfilename(
            title="Select Notch Image",
            filetypes=[("Images", "*.png *.bmp *.jpg *.jpeg *.tiff *.tif"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._log(f"[NOTCH TAB] Cannot read image: {path}")
            return
        self._src_img  = img
        self._img_path = path
        self.img_path_var.set(os.path.basename(path))
        self.btn_run.config(state='normal')
        self._log(f"[NOTCH TAB] Loaded: {path}")
        self._run_pipeline()

    def _on_run(self):
        self._run_pipeline()

    def _schedule_update(self):
        """Debounce slider changes — run pipeline 300 ms after last move."""
        if self._src_img is None:
            return
        if self._debounce_id is not None:
            self.tab.after_cancel(self._debounce_id)
        self._debounce_id = self.tab.after(300, self._run_pipeline)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self):
        if self._src_img is None:
            return

        img = self._src_img.copy()
        h, w = img.shape[:2]

        # ── Read params ───────────────────────────────────────────────
        threshold   = self.threshold_var.get()
        k_blur_raw  = self.k_blur_var.get()
        k_morph_raw = self.k_morph_var.get()
        canny_low   = self.canny_low_var.get()
        canny_high  = self.canny_high_var.get()
        hough_votes = self.hough_votes_var.get()
        hough_len   = self.hough_len_var.get()
        hough_gap   = self.hough_gap_var.get()
        vert_angle  = self.vert_angle_var.get()
        slant_angle = self.slant_angle_var.get()
        pix_res     = self.pix_res_var.get()

        # Auto kernel sizes (mirrors C# defaults)
        k_blur = k_blur_raw if k_blur_raw >= 3 else int(0.03 * w)
        if k_blur % 2 == 0:
            k_blur += 1
        k_blur = max(k_blur, 3)

        k_morph = k_morph_raw if k_morph_raw >= 1 else max(int(0.048 * w), 1)

        # ── Step 1a: Gaussian Blur ────────────────────────────────────
        blurred = cv2.GaussianBlur(img, (k_blur, k_blur), 0)

        # ── Step 1b: Binary threshold ─────────────────────────────────
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # ── Step 1c: Morph Close ──────────────────────────────────────
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k_morph, k_morph))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se, iterations=1)

        # ── Step 1d: Canny ────────────────────────────────────────────
        edges = cv2.Canny(morphed, threshold1=canny_low, threshold2=canny_high)

        # ── Step 2: Hough lines ───────────────────────────────────────
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180.0,
            threshold=hough_votes,
            minLineLength=hough_len,
            maxLineGap=hough_gap,
        )

        # ── Step 3: Classify lines ────────────────────────────────────
        vertical_line = None
        slanted_line  = None
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                abs_a = abs(angle)
                if abs_a < vert_angle:
                    vertical_line = (x1, y1, x2, y2)
                elif abs_a > slant_angle:
                    slanted_line  = (x1, y1, x2, y2)

        # ── Step 4: Intersection ──────────────────────────────────────
        tip = None
        if vertical_line and slanted_line:
            def _abc(x1, y1, x2, y2):
                A = float(y2 - y1); B = float(x1 - x2)
                return A, B, A * x1 + B * y1

            A1, B1, C1 = _abc(*vertical_line)
            A2, B2, C2 = _abc(*slanted_line)
            det = A1 * B2 - A2 * B1
            if abs(det) > 1e-10:
                tip = ((B2 * C1 - B1 * C2) / det,
                       (A1 * C2 - A2 * C1) / det)

        # ── Step 5/6: Offset ──────────────────────────────────────────
        cx, cy = w / 2.0, h / 2.0
        result_info = None
        if tip:
            ix, iy   = tip
            offset_x = (ix - cx) * pix_res
            offset_y = (iy - cy) * pix_res
            result_info = (ix, iy, cx, cy, offset_x, offset_y)

        # ── Update result labels ──────────────────────────────────────
        if result_info:
            ix, iy, _, _, ox, oy = result_info
            self.tip_px_var.set(f"({ix:.1f},  {iy:.1f})")
            self.offset_mm_var.set(f"({ox:.4f},  {oy:.4f})")
            self.status_var.set("OK ✓")
        else:
            reason = "No lines" if lines is None else \
                     ("No vertical" if vertical_line is None else
                      ("No slanted"  if slanted_line  is None else
                       "Parallel lines"))
            self.tip_px_var.set("—")
            self.offset_mm_var.set("—")
            self.status_var.set(f"FAILED — {reason}")

        # ── Render pipeline tiles ─────────────────────────────────────
        self._render_tile(img,     self._pipeline_labels[0])   # Original
        self._render_tile(blurred, self._pipeline_labels[1])   # Blurred
        self._render_tile(binary,  self._pipeline_labels[2])   # Binary
        self._render_tile(morphed, self._pipeline_labels[3])   # Morphed

        # Tile 4: Canny + all Hough lines (cyan)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                cv2.line(edges_bgr, (x1, y1), (x2, y2), (255, 200, 0), 1, cv2.LINE_AA)
        self._render_tile(edges_bgr, self._pipeline_labels[4], already_bgr=True)

        # Tile 5: Result overlay on original
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # All Hough lines (dim cyan)
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                cv2.line(overlay, (x1, y1), (x2, y2), (60, 60, 0), 1)
        # Vertical line (green)
        if vertical_line:
            x1, y1, x2, y2 = vertical_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 230, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, "vertical", (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 0), 1, cv2.LINE_AA)
        # Slanted line (yellow)
        if slanted_line:
            x1, y1, x2, y2 = slanted_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(overlay, "slanted", (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA)
        # Centre crosshair
        cv2.drawMarker(overlay, (int(cx), int(cy)), (180, 180, 180),
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
        if result_info:
            ix, iy, _, _, ox, oy = result_info
            # Arrow centre → tip
            cv2.arrowedLine(overlay, (int(cx), int(cy)), (int(ix), int(iy)),
                            (220, 0, 220), 2, cv2.LINE_AA, tipLength=0.06)
            # Tip dot (red)
            cv2.circle(overlay, (int(ix), int(iy)), 8, (0, 0, 255), -1)
            cv2.circle(overlay, (int(ix), int(iy)), 8, (255, 255, 255), 1)
            # Text
            cv2.putText(overlay, f"({ix:.1f}, {iy:.1f}) px",
                        (8, overlay.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"offset ({ox:.3f}, {oy:.3f}) mm",
                        (8, overlay.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 100), 1, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "FAILED",
                        (overlay.shape[1] // 2 - 60, overlay.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2, cv2.LINE_AA)
        self._render_tile(overlay, self._pipeline_labels[5], already_bgr=True)

        self._log(f"[NOTCH TAB] Pipeline updated — "
                  f"{'OK tip=({:.1f},{:.1f})'.format(*result_info[:2]) if result_info else 'FAILED'}")

    # ------------------------------------------------------------------
    # Tile renderer
    # ------------------------------------------------------------------

    def _render_tile(self, img, label, already_bgr=False):
        """Resize img to fit the label widget and display it."""
        try:
            label.update_idletasks()
            p_w = label.winfo_width()
            p_h = label.winfo_height()
            if p_w < 20 or p_h < 20:
                p_w, p_h = 320, 240

            h, w = img.shape[:2]
            scale = min(p_w / w, p_h / h)
            nw = max(int(w * scale), 1)
            nh = max(int(h * scale), 1)

            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

            if not already_bgr:
                if len(resized.shape) == 2:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            pil  = Image.fromarray(rgb)
            tkimg = ImageTk.PhotoImage(pil)
            label.configure(image=tkimg, text="")
            label.image = tkimg   # keep reference
        except Exception as e:
            self._log(f"[NOTCH TAB] Tile render error: {e}")

    # ------------------------------------------------------------------
    # Recipe integration helpers (called by main_window._apply_recipe)
    # ------------------------------------------------------------------

    def load_from_recipe(self, recipe: dict):
        """Apply notch params from a loaded recipe dict."""
        notch = recipe.get("notch", {})
        if "WaferTopEdgeThreshold" in notch:
            self.threshold_var.set(int(notch["WaferTopEdgeThreshold"]))
        if "PixelResolution" in notch:
            self.pix_res_var.set(float(notch["PixelResolution"]))

    def save_to_recipe(self, recipe: dict):
        """Write current notch params back into recipe dict."""
        notch = recipe.setdefault("notch", {})
        notch["WaferTopEdgeThreshold"] = str(self.threshold_var.get())
        notch["PixelResolution"]       = str(self.pix_res_var.get())
