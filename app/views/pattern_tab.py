"""
================================================================================
PATTERN TAB  (View layer)
================================================================================
Builds the "Find Pattern" sub-tab inside the Recipe notebook.

Responsibilities:
  - Widget construction (sliders, template/search image panels, result labels)
  - Routing button clicks → PatternViewModel
  - Updating result labels from ViewModel output
================================================================================
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2

from app.models.app_state import AppState
from app.viewmodels.pattern_viewmodel import PatternViewModel


class PatternTab:
    """
    Owns all widgets in the "Find Pattern" sub-tab.

    Parameters:
      parent_nb         - ttk.Notebook (recipe sub-notebook)
      state             - shared AppState
      display_image     - callable(path, label)
      display_cv2_image - callable(img, label, overlay_func=None)
      log_callback      - callable(msg: str)
    """

    def __init__(self, parent_nb: ttk.Notebook, state: AppState,
                 display_image, display_cv2_image, log_callback=None):
        self.state             = state
        self._display_image    = display_image
        self._display_cv2_image = display_cv2_image
        self._log              = log_callback or print

        self.vm = PatternViewModel(state, log_callback)

        self.tab = ttk.Frame(parent_nb)
        parent_nb.add(self.tab, text="Find Pattern")

        # Tkinter variables (exposed for main_window recipe sync)
        self.pattern_img_var    = tk.StringVar()
        self.template_img_var   = tk.StringVar()
        self.pattern_mode_var   = tk.StringVar(value="Simple (Fast)")
        self.pattern_thresh_var = tk.StringVar(value="50.0")
        self.pattern_num_var    = tk.StringVar(value="128")
        self.pattern_weak_var   = tk.StringVar(value="70.0")
        self.pattern_tspread_var = tk.StringVar(value="4")
        self.pattern_hyst_var   = tk.StringVar(value="0")
        self.pattern_rot_var    = tk.StringVar(value="0")
        self.pattern_score_var  = tk.StringVar(value="0")
        self.pattern_x_var      = tk.StringVar(value="0")
        self.pattern_y_var      = tk.StringVar(value="0")

        self._build()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build(self):
        # ── Left panel (controls) ──────────────────────────────────────
        left = ttk.Frame(self.tab, width=400)
        left.pack(side='left', fill='y', padx=(10, 5), pady=10)
        left.pack_propagate(False)

        # ── Right panel (images) ───────────────────────────────────────
        right = ttk.Frame(self.tab)
        right.pack(side='left', expand=True, fill='both', padx=10, pady=10)

        # Search image
        self.btn_load_pattern_search = ttk.Button(
            left, text="Load Search Image",
            command=self._on_load_search_image,
            state='disabled')
        self.btn_load_pattern_search.pack(anchor='w', pady=5)
        ttk.Entry(left, textvariable=self.pattern_img_var, width=50,
                  state='readonly').pack(anchor='w', pady=5)

        # Template buttons
        tmpl_frame = ttk.Frame(left)
        tmpl_frame.pack(anchor='w', fill='x', pady=(15, 5))
        self.btn_load_template = ttk.Button(tmpl_frame, text="Load Template Image",
                                            command=self._on_load_template,
                                            state='disabled')
        self.btn_load_template.pack(side='left', padx=(0, 5))
        self.btn_crop_template = ttk.Button(tmpl_frame, text="Crop from Search",
                                            command=self._on_crop_template,
                                            state='disabled')
        self.btn_crop_template.pack(side='left')

        # Detection ROI buttons
        roi_frame = ttk.Frame(left)
        roi_frame.pack(anchor='w', fill='x', pady=(2, 5))
        self.btn_draw_roi = ttk.Button(roi_frame, text="Draw Detection ROI",
                                       command=self._on_draw_roi,
                                       state='disabled')
        self.btn_draw_roi.pack(side='left', padx=(0, 5))
        self.btn_clear_mask = ttk.Button(roi_frame, text="Clear Mask",
                                         command=self._on_clear_mask,
                                         state='disabled')
        self.btn_clear_mask.pack(side='left')

        ttk.Entry(left, textvariable=self.template_img_var, width=50,
                  state='readonly').pack(anchor='w', pady=5)

        # Recipe settings
        lf_settings = ttk.LabelFrame(left, text="Recipe Setting")
        lf_settings.pack(fill='x', pady=10)

        ttk.Label(lf_settings, text="Search Mode:").grid(
            row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Combobox(lf_settings, textvariable=self.pattern_mode_var,
                     values=["Simple (Fast)", "With Rotation", "Full Search"],
                     width=15, state="readonly").grid(
                         row=0, column=1, padx=5, pady=2, sticky='w')

        slider_params = [
            ("Threshold:",    self.pattern_thresh_var,  "50.0", 0.0,  100.0, "{:.1f}"),
            ("Num Features:", self.pattern_num_var,    "128",   16,   512,   "{:.0f}"),
            ("Grad Thr %:",   self.pattern_weak_var,   "70.0",  1.0,  99.0,  "{:.1f}"),
            ("T Spread:",     self.pattern_tspread_var, "4",     1,    20,    "{:.0f}"),
            ("Hyst Kernel:",  self.pattern_hyst_var,   "0",     0,    15,    "{:.0f}"),
            ("Search Rot °:", self.pattern_rot_var,    "0",     0,   360,    "{:.0f}"),
        ]
        lf_settings.columnconfigure(1, weight=1)

        for i, (label, str_var, default, vmin, vmax, fmt) in \
                enumerate(slider_params, start=1):
            self._make_slider(lf_settings, i, label, str_var,
                              float(default), vmin, vmax, fmt)

        vis_frame = ttk.Frame(lf_settings)
        vis_frame.grid(row=len(slider_params) + 1, column=0, columnspan=3, pady=5)
        ttk.Button(vis_frame, text="Show Orientations",
                   command=self._on_show_orientations).pack(side='left', padx=2)
        ttk.Button(vis_frame, text="Show Pipeline",
                   command=self._on_show_pipeline).pack(side='left', padx=2)

        # Execute
        lf_exec = ttk.LabelFrame(left, text="Execute")
        lf_exec.pack(fill='x', pady=10)
        btn_frame = ttk.Frame(lf_exec)
        btn_frame.pack(fill='x', padx=5, pady=5)
        self.btn_detect_pattern = ttk.Button(btn_frame, text="Detect!",
                                             command=self._on_find_pattern,
                                             state='disabled')
        self.btn_detect_pattern.pack(side='left', padx=(0, 5))

        # Result labels
        lf_result = ttk.LabelFrame(left, text="Result")
        lf_result.pack(fill='x', pady=10)
        result_rows = [
            ("Similarity Score:", self.pattern_score_var),
            ("Delta X:",          self.pattern_x_var),
            ("Delta Y:",          self.pattern_y_var),
        ]
        for i, (lbl, var) in enumerate(result_rows):
            ttk.Label(lf_result, text=lbl).grid(
                row=i, column=0, padx=5, pady=5, sticky='w')
            ttk.Entry(lf_result, textvariable=var, width=15,
                      state='readonly').grid(row=i, column=1, padx=5, pady=5)

        # ── Right panel image areas ─────────────────────────────────────
        top_frames = ttk.Frame(right)
        top_frames.pack(fill='both', expand=True, pady=(0, 5))
        top_frames.pack_propagate(False)

        lf_template = ttk.LabelFrame(top_frames, text="Template")
        lf_template.pack(side='left', fill='both', expand=True, padx=(0, 5))
        lf_template.pack_propagate(False)
        self.lbl_template_img = ttk.Label(lf_template, text="(No Template)",
                                          anchor='center')
        self.lbl_template_img.pack(expand=True, fill='both')

        lf_input = ttk.LabelFrame(top_frames, text="Input Search Image")
        lf_input.pack(side='left', fill='both', expand=True, padx=(5, 0))
        lf_input.pack_propagate(False)
        self.lbl_pattern_input = ttk.Label(lf_input, text="(No Image)",
                                           anchor='center')
        self.lbl_pattern_input.pack(expand=True, fill='both')

        lf_output = ttk.LabelFrame(right, text="Output Result")
        lf_output.pack(fill='both', expand=True, pady=(5, 0))
        lf_output.pack_propagate(False)
        self.lbl_pattern_output = ttk.Label(lf_output, text="(No Result)",
                                            anchor='center')
        self.lbl_pattern_output.pack(expand=True, fill='both')

    def _make_slider(self, parent, row, label, str_var, default, vmin, vmax, fmt):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, padx=5, pady=4, sticky='w')
        val_lbl = ttk.Label(parent, textvariable=str_var, width=7,
                            anchor='e', relief='sunken')
        val_lbl.grid(row=row, column=2, padx=(2, 5), pady=4, sticky='e')

        def _on_slide(v, sv=str_var, f=fmt):
            sv.set(f.format(float(v)))

        scale = ttk.Scale(parent, from_=vmin, to=vmax,
                          orient='horizontal', command=_on_slide)
        scale.set(default)
        scale.grid(row=row, column=1, padx=5, pady=4, sticky='ew')
        return scale

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_search_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.pattern_img_var.set(path)
            self._display_image(path, self.lbl_pattern_input)

    def _on_load_template(self):
        path = filedialog.askopenfilename()
        if path:
            self.template_img_var.set(path)
            if self.vm.load_template_from_path(path, self._get_tk_vars()):
                self._log("Template loaded.")
                self.btn_draw_roi.config(state='normal')
                self.btn_clear_mask.config(state='normal')
                # Show mask overlay if one was auto-loaded
                if self.state.template_detection_mask is not None:
                    self._display_template_with_mask(path, self.state.template_detection_mask)
                else:
                    self._display_image(path, self.lbl_template_img)

    def _on_crop_template(self):
        search_path = self.pattern_img_var.get()
        if not search_path or not os.path.exists(search_path):
            messagebox.showwarning("Warning", "Load a search image first.")
            return
        out_path = self.vm.crop_template_from_image(
            search_path, self._get_tk_vars())
        if out_path:
            self.template_img_var.set(out_path)
            self._display_image(out_path, self.lbl_template_img)
            self.btn_draw_roi.config(state='normal')
            self.btn_clear_mask.config(state='normal')

    def _on_draw_roi(self):
        template_path = self.template_img_var.get()
        if not template_path or not os.path.exists(template_path):
            messagebox.showwarning("Warning", "Load or crop a template first!")
            return
        mask = self.vm.draw_detection_roi(template_path, self._get_tk_vars())
        if mask is not None:
            self._display_template_with_mask(template_path, mask)

    def _on_clear_mask(self):
        template_path = self.template_img_var.get()
        if not template_path:
            return
        self.vm.clear_detection_mask(template_path, self._get_tk_vars())
        # Refresh template display without mask
        if os.path.exists(template_path):
            self._display_image(template_path, self.lbl_template_img)

    def _display_template_with_mask(self, template_path, mask):
        """Show the template image with a green overlay where the mask is active."""
        import cv2
        img = cv2.imread(template_path)
        if img is None:
            return
        overlay = img.copy()
        overlay[mask > 0] = (
            overlay[mask > 0].astype(float) * 0.6 +
            [0, 140, 0]
        ).clip(0, 255).astype('uint8')
        # Draw mask contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(1, max(img.shape[:2]) // 400))
        self._display_cv2_image(overlay, self.lbl_template_img)

    def _on_find_pattern(self):
        import threading

        if not self.state.template_loaded:
            messagebox.showwarning("Warning", "Load or crop a template first!")
            return

        # Prevent double-click while already running
        self.btn_detect_pattern.config(state='disabled', text="Detecting…")
        self._log("[Pattern] Matching started…")

        tk_vars_snapshot = self._get_tk_vars()

        def _run():
            resp, orig_color = self.vm.run_find_pattern(tk_vars_snapshot)
            # Schedule UI update back on the main thread
            self.tab.after(0, lambda: self._on_find_pattern_done(resp, orig_color))

        threading.Thread(target=_run, daemon=True).start()

    def _on_find_pattern_done(self, resp, orig_color):
        """Called on the main thread after matching completes."""
        self.btn_detect_pattern.config(state='normal', text="Detect!")

        if resp is not None:
            self.pattern_score_var.set(f"{resp['score']:.1f}")
            delta_x, delta_y = self.vm.compute_delta(resp)
            self.pattern_x_var.set(f"{delta_x:.3f}")
            self.pattern_y_var.set(f"{delta_y:.3f}")
            self._log(f"[Pattern] Match found — score={resp['score']:.1f}  "
                      f"dx={delta_x:.1f}  dy={delta_y:.1f}")

            def draw_rect_overlay(img_draw):
                return self.vm.linemod_matcher.visualize_match(
                    img_draw, resp, show=False)

            self._display_cv2_image(orig_color, self.lbl_pattern_output,
                                    overlay_func=draw_rect_overlay)
        else:
            self.pattern_score_var.set("0 (No Match)")
            self._log("[Pattern] No match found.")
            if orig_color is not None:
                self._display_cv2_image(orig_color, self.lbl_pattern_output)

    def _on_show_orientations(self):
        if not self.state.template_loaded:
            messagebox.showwarning("Warning",
                                   "Load both template and search images first!")
            return
        self.vm.show_orientations(self._get_tk_vars())

    def _on_show_pipeline(self):
        if not self.state.template_loaded:
            messagebox.showwarning("Warning", "Load a template first!")
            return
        self.vm.show_pipeline(self._get_tk_vars())

    # ------------------------------------------------------------------
    # Public helpers (called by main_window for recipe sync + server sync)
    # ------------------------------------------------------------------

    def enable_buttons(self):
        for btn in [self.btn_detect_pattern, self.btn_load_pattern_search,
                    self.btn_load_template, self.btn_crop_template,
                    self.btn_draw_roi, self.btn_clear_mask]:
            btn.config(state='normal')

    def _get_tk_vars(self) -> dict:
        return {
            'pattern_img_var':     self.pattern_img_var,
            'template_img_var':    self.template_img_var,
            'pattern_mode_var':    self.pattern_mode_var,
            'pattern_thresh_var':  self.pattern_thresh_var,
            'pattern_num_var':     self.pattern_num_var,
            'pattern_weak_var':    self.pattern_weak_var,
            'pattern_tspread_var': self.pattern_tspread_var,
            'pattern_hyst_var':    self.pattern_hyst_var,
            'pattern_rot_var':     self.pattern_rot_var,
        }
