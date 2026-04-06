"""
================================================================================
EDGE TAB  (View layer)
================================================================================
Builds the "Find Wafer Edge" sub-tab inside the Recipe notebook.

Responsibilities:
  - Widget construction (sliders, labels, image panels, matplotlib canvas)
  - Routing button clicks → EdgeViewModel
  - Updating result labels from ViewModel output
  - Displaying images using the shared display helpers from main_window
================================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.models.app_state import AppState
from app.viewmodels.edge_viewmodel import EdgeViewModel


class EdgeTab:
    """
    Owns all widgets in the "Find Wafer Edge" sub-tab.

    Parameters:
      parent_nb     - ttk.Notebook (recipe sub-notebook) to add this tab into
      state         - shared AppState
      display_image - callable(path, label) from main_window
      display_cv2_image - callable(img, label, overlay_func=None) from main_window
      log_callback  - callable(msg: str) → writes to the Status log
    """

    def __init__(self, parent_nb: ttk.Notebook, state: AppState,
                 display_image, display_cv2_image, log_callback=None):
        self.state = state
        self._display_image     = display_image
        self._display_cv2_image = display_cv2_image
        self._log               = log_callback or print

        self.vm = EdgeViewModel(state, log_callback)

        self.tab = ttk.Frame(parent_nb)
        parent_nb.add(self.tab, text="Find Wafer Edge")

        # Tkinter variables exposed so main_window can read/set them
        self.edge_img_var      = tk.StringVar()
        self.edge_kernel_var   = tk.StringVar(value="9")
        self.edge_thresh_var   = tk.StringVar(value="80")
        self.edge_regions_var  = tk.StringVar(value="30")
        self.edge_border_var   = tk.StringVar(value="0.050")
        self.edge_ransac_var   = tk.StringVar(value="3.0")
        self.edge_dir_var      = tk.StringVar(value="LEFT")

        # Result display vars
        self.edge_delta_x_var = tk.StringVar(value="")
        self.edge_delta_y_var = tk.StringVar(value="")
        self.edge_slope_var   = tk.StringVar(value="")
        self.edge_c_var       = tk.StringVar(value="")
        self.edge_fov_var     = tk.StringVar(value="(not checked)")  # new

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

        # Load image button
        self.btn_load_edge_search = ttk.Button(
            left, text="Load Image",
            command=self._on_load_image,
            state='disabled')
        self.btn_load_edge_search.pack(anchor='w', pady=5)
        ttk.Entry(left, textvariable=self.edge_img_var, width=50,
                  state='readonly').pack(anchor='w', pady=5)

        # Recipe settings frame
        lf_settings = ttk.LabelFrame(left, text="Recipe Setting")
        lf_settings.pack(fill='x', pady=10)

        ttk.Label(lf_settings, text="Scan Direction:").grid(
            row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Combobox(lf_settings, textvariable=self.edge_dir_var,
                     values=["LEFT", "RIGHT", "TOP", "BOTTOM"],
                     width=13, state="readonly").grid(
                         row=0, column=1, padx=5, pady=2, sticky='w')

        self.edge_universal_var = tk.BooleanVar(value=False)
        cb_universal = ttk.Checkbutton(lf_settings, text="Universal Parameters",
                                       variable=self.edge_universal_var,
                                       command=self._on_universal_toggle)
        cb_universal.grid(row=0, column=2, padx=5, pady=2, sticky='e')

        # Slider params: (label, StringVar attr, default, min, max, res, fmt)
        slider_params = [
            ("Kernel Size:",     self.edge_kernel_var,  "9",     3,    201,   2,     "{:.0f}"),
            ("Edge Threshold:",  self.edge_thresh_var,  "80",    1,    5000,  1,     "{:.0f}"),
            ("Scan Regions:",    self.edge_regions_var, "30",    5,    100,   1,     "{:.0f}"),
            ("Border Ignore %:", self.edge_border_var,  "0.050", 0.0,  0.25,  0.005, "{:.3f}"),
            ("RANSAC Thresh:",   self.edge_ransac_var,  "3.0",   0.5,  20.0,  0.5,   "{:.1f}"),
        ]
        self.edge_scales = {}
        lf_settings.columnconfigure(1, weight=1)

        for i, (label, str_var, default, vmin, vmax, res, fmt) in \
                enumerate(slider_params, start=1):
            self._make_slider(lf_settings, i, label, str_var,
                              float(default), vmin, vmax, fmt)

        # Wire direction combobox to cache save/load
        self.edge_dir_var.trace_add("write", self._on_dir_change)

        # Execute frame
        lf_exec = ttk.LabelFrame(left, text="Execute")
        lf_exec.pack(fill='x', pady=10)
        btn_frame = ttk.Frame(lf_exec)
        btn_frame.pack(fill='x', padx=5, pady=5)
        self.btn_find_edge = ttk.Button(btn_frame, text="Find Edge",
                                        command=self._on_find_edge,
                                        state='disabled')
        self.btn_find_edge.pack(side='left', padx=(0, 5))

        # Result frame
        lf_result = ttk.LabelFrame(left, text="Result")
        lf_result.pack(fill='x', pady=10)
        result_rows = [
            ("FOV Type:",   self.edge_fov_var),
            ("Delta X:",    self.edge_delta_x_var),
            ("Delta Y:",    self.edge_delta_y_var),
            ("Slope:",      self.edge_slope_var),
            ("C (offset):", self.edge_c_var),
        ]
        for i, (lbl, var) in enumerate(result_rows):
            ttk.Label(lf_result, text=lbl).grid(
                row=i, column=0, padx=5, pady=3, sticky='w')
            ttk.Entry(lf_result, textvariable=var, width=30,
                      state='readonly').grid(row=i, column=1, padx=5, pady=3, sticky='w')

        # ── Right panel: 2×2 image grid ───────────────────────────────
        right.rowconfigure(0, weight=1, uniform="row")
        right.rowconfigure(1, weight=1, uniform="row")
        right.columnconfigure(0, weight=1, uniform="col")
        right.columnconfigure(1, weight=1, uniform="col")

        lf_input = ttk.LabelFrame(right, text="Input Image")
        lf_input.grid(row=0, column=0, sticky='nsew', padx=(0, 3), pady=(0, 3))
        lf_input.rowconfigure(0, weight=1); lf_input.columnconfigure(0, weight=1)
        self.lbl_edge_input = ttk.Label(lf_input, text="(No Image)", anchor='center')
        self.lbl_edge_input.grid(row=0, column=0, sticky='nsew')

        lf_output = ttk.LabelFrame(right, text="Image Result")
        lf_output.grid(row=0, column=1, sticky='nsew', padx=(3, 0), pady=(0, 3))
        lf_output.rowconfigure(0, weight=1); lf_output.columnconfigure(0, weight=1)
        self.lbl_edge_output = ttk.Label(lf_output, text="(No Result)", anchor='center')
        self.lbl_edge_output.grid(row=0, column=0, sticky='nsew')

        lf_grad_img = ttk.LabelFrame(right, text="Gradient Magnitude")
        lf_grad_img.grid(row=1, column=0, sticky='nsew', padx=(0, 3), pady=(3, 0))
        lf_grad_img.rowconfigure(0, weight=1); lf_grad_img.columnconfigure(0, weight=1)
        self.lbl_grad_magnitude = ttk.Label(lf_grad_img, text="(No Result)", anchor='center')
        self.lbl_grad_magnitude.grid(row=0, column=0, sticky='nsew')

        lf_grad_graph = ttk.LabelFrame(right, text="Gradient Profile")
        lf_grad_graph.grid(row=1, column=1, sticky='nsew', padx=(3, 0), pady=(3, 0))
        lf_grad_graph.rowconfigure(0, weight=1); lf_grad_graph.columnconfigure(0, weight=1)
        self.edge_fig, self.edge_ax2 = plt.subplots(1, 1, figsize=(5, 3))
        self.edge_fig.tight_layout(pad=2.0)
        self.edge_canvas = FigureCanvasTkAgg(self.edge_fig, master=lf_grad_graph)
        self.edge_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    def _make_slider(self, parent, row, label, str_var, default, vmin, vmax, fmt):
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=5, pady=4, sticky='w')
        val_lbl = ttk.Label(parent, textvariable=str_var, width=7,
                            anchor='e', relief='sunken')
        val_lbl.grid(row=row, column=2, padx=(2, 5), pady=4, sticky='e')

        def _on_slide(v, sv=str_var, f=fmt):
            sv.set(f.format(float(v)))

        scale = ttk.Scale(parent, from_=vmin, to=vmax,
                          orient='horizontal', command=_on_slide)
        scale.set(default)
        scale.grid(row=row, column=1, padx=5, pady=4, sticky='ew')
        # Store scale widget keyed by var name for external use
        self.edge_scales[id(str_var)] = scale
        return scale

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.edge_img_var.set(path)
            self._display_image(path, self.lbl_edge_input)

    def _on_find_edge(self):
        """
        Pre-flight FOV classification to detect die images,
        then run edge detection only if the image looks like an edge FOV.
        """
        import os
        from app.services.fov_classifier import FOVClassifier

        path = self.edge_img_var.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # ── 1. FOV Classification ────────────────────────────────────────
        try:
            classifier = FOVClassifier()
            fov_result = classifier.classify(path)
            fov_type   = fov_result.get('fov_type', 'UNCERTAIN')
            peaks      = fov_result.get('peaks', {})
            pk_h       = peaks.get('h_count', '?')
            pk_v       = peaks.get('v_count', '?')
            pk_label   = peaks.get('label', '')
            confidence = fov_result.get('confidence', 0)

            display_str = (f"{fov_type}  "
                           f"(conf={confidence:.2f}, "
                           f"peaks H={pk_h} V={pk_v}, {pk_label})")
            self.edge_fov_var.set(display_str)
            self._log(f"[FOV] {display_str}")

        except Exception as ex:
            self._log(f"[FOV] Classification error: {ex}")
            fov_type = "UNCERTAIN"
            self.edge_fov_var.set(f"Error: {ex}")

        # ── 2. Warn on DIE_FOV (user can override) ───────────────────────
        # Corner/transition FOVs can have die patterns AND a visible edge.
        # A hard block would prevent valid processing, so we ask the user.
        if fov_type == "DIE_FOV":
            proceed = messagebox.askyesno(
                "Die Image Detected",
                "FOV classifier identified this as a DIE image (many repeating peaks).\n\n"
                "If you can see the wafer edge in the image (e.g. a corner or\n"
                "transition FOV), click YES to run Find Edge anyway.\n\n"
                "Click NO to cancel and load a different image.",
                icon="warning"
            )
            if not proceed:
                self.edge_delta_x_var.set("CANCELLED — DIE_FOV")
                self.edge_delta_y_var.set("")
                self.edge_slope_var.set("")
                self.edge_c_var.set("")
                return
            self._log("[FOV] User overrode DIE_FOV warning — proceeding.")

        # ── 3. Run edge detection ────────────────────────────────────────
        resp = self.vm.run_find_edge(self._get_tk_vars())
        self.visualize_edge_result(resp, path, self.vm.edge_finder)

    def _on_dir_change(self, *args):
        """Save previous direction config to cache, load new direction from cache."""
        old_dir = self.state.last_edge_dir
        if not self.state._is_loading_recipe:
            self.vm.save_current_dir_to_cache(old_dir, self._get_tk_vars())
            if self.edge_universal_var.get():
                for d in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
                    if d != old_dir:
                        self.state.edge_configs[d] = self.state.edge_configs[old_dir].copy()
        new_dir = self.edge_dir_var.get()
        self.update_sliders_from_cache(new_dir)
        self.state.last_edge_dir = new_dir

    def _on_universal_toggle(self):
        """If checked, instantly sync the current configuration to all directions."""
        if self.edge_universal_var.get():
            active_dir = self.edge_dir_var.get()
            self.vm.save_current_dir_to_cache(active_dir, self._get_tk_vars())
            for d in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
                if d != active_dir:
                    self.state.edge_configs[d] = self.state.edge_configs[active_dir].copy()
            self._log(f"[Edge] Universal parameters applied from {active_dir} to all directions.")

    # ------------------------------------------------------------------
    # Public helpers (called by main_window for recipe sync + server sync)
    # ------------------------------------------------------------------

    def update_sliders_from_cache(self, direction: str):
        """Populate sliders from AppState cache for *direction*."""
        cfg = self.state.edge_configs.get(direction, {})
        
        def _set_var(var, key):
            if key in cfg:
                val_str = cfg[key]
                var.set(val_str)
                if id(var) in self.edge_scales:
                    try:
                        self.edge_scales[id(var)].set(float(val_str))
                    except ValueError:
                        pass

        _set_var(self.edge_kernel_var,  "KernelSize")
        _set_var(self.edge_thresh_var,  "EdgeThreshold")
        _set_var(self.edge_regions_var, "NumRegions")
        _set_var(self.edge_border_var,  "BorderIgnorePct")
        _set_var(self.edge_ransac_var,  "RansacThreshold")

    def enable_buttons(self):
        self.btn_find_edge.config(state='normal')
        self.btn_load_edge_search.config(state='normal')

    def _get_tk_vars(self) -> dict:
        return {
            'edge_img_var':     self.edge_img_var,
            'edge_kernel_var':  self.edge_kernel_var,
            'edge_thresh_var':  self.edge_thresh_var,
            'edge_regions_var': self.edge_regions_var,
            'edge_border_var':  self.edge_border_var,
            'edge_ransac_var':  self.edge_ransac_var,
            'edge_dir_var':     self.edge_dir_var,
        }

    def visualize_edge_result(self, resp, path: str, edge_finder_inst):
        """Render edge detection result into all four image panels."""
        from app.services.fov_classifier import preprocess_image

        proc_img = resp.get('image') if resp else None
        if proc_img is not None:
            img_color = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
        else:
            img_gray, _, _ = preprocess_image(
                path, edge_finder_inst.config.TARGET_PROCESS_DIM)
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if resp and resp.get("success"):
            self.edge_delta_x_var.set(f"{resp.get('delta_x', 0):.3f}")
            self.edge_delta_y_var.set(f"{resp.get('delta_y', 0):.3f}")
            self.edge_slope_var.set(f"{resp.get('slope', 0):.6f}")
            self.edge_c_var.set(f"{resp.get('intercept_c', 0):.3f}")

            overlay_func = self.vm.build_overlay_func(resp)
            self._display_cv2_image(img_color, self.lbl_edge_output,
                                    overlay_func=overlay_func)

            # Gradient panels
            grad_bgr, abs_gradient_1d, cfg = self.vm.compute_gradient_display(resp)
            self._display_cv2_image(grad_bgr, self.lbl_grad_magnitude)

            self.edge_ax2.clear()
            self.edge_ax2.plot(abs_gradient_1d, color='green',
                               linewidth=1.5, label='Gradient')
            self.edge_ax2.fill_between(range(len(abs_gradient_1d)),
                                       abs_gradient_1d, alpha=0.3, color='green')
            self.edge_ax2.axhline(cfg.EDGE_THRESHOLD, color='orange',
                                  linestyle='--', linewidth=2,
                                  label=f'Threshold = {cfg.EDGE_THRESHOLD}')
            self.edge_ax2.set_title("Gradient Profile (full-image median)", fontsize=9)
            self.edge_ax2.set_xlabel("Position")
            self.edge_ax2.set_ylabel("Gradient Magnitude")
            self.edge_ax2.grid(True, alpha=0.3)
            self.edge_ax2.legend(loc='upper right', fontsize=8)
            self.edge_fig.tight_layout(pad=2.0)
            self.edge_canvas.draw()

        else:
            self.edge_delta_x_var.set("FAILED")
            self.edge_delta_y_var.set("FAILED")
            self.edge_slope_var.set("")
            self.edge_c_var.set("")
            reason = resp.get('reason', 'Unknown') if resp else 'No response'
            self._log(f"Find Edge Failed: {reason}")
            self._display_cv2_image(img_color, self.lbl_edge_output)
