"""
================================================================================
AUTOFOCUS TAB  (View layer)
================================================================================
Builds the "Auto Focus" sub-tab inside the Recipe notebook.

Exposes two settings that control the peak-detection algorithm:
  - PeakHeightThresholdPercentage  (float, 0–1)
  - PeakFilterHalfSize             (int, points)

Also contains a temporary test section: pick a folder of images, click
"Run Test" and the tab runs the full FMI scan + peak detection locally
(no ZMQ) using the current threshold values, then prints results.
================================================================================
"""

import glob
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog

from app.models.app_state import AppState
from app.services.autofocus import compute_fmi, peak_detection


class AutoFocusTab:
    """
    Owns all widgets in the "Auto Focus" sub-tab.

    Parameters:
      parent_nb    - ttk.Notebook (recipe sub-notebook)
      state        - shared AppState
      log_callback - callable(msg: str)
    """

    def __init__(self, parent_nb: ttk.Notebook, state: AppState,
                 log_callback=None):
        self.state = state
        self._log  = log_callback or print

        self.tab = ttk.Frame(parent_nb)
        parent_nb.add(self.tab, text="Auto Focus")

        # Tkinter variables (exposed for main_window recipe sync)
        self.peak_threshold_var  = tk.StringVar(value="0.05")
        self.peak_half_size_var  = tk.StringVar(value="2")

        # Test-section variable
        self.test_images_dir_var = tk.StringVar(value="")

        self._build()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build(self):
        outer = ttk.Frame(self.tab)
        outer.pack(anchor='nw', fill='both', expand=True, padx=10, pady=10)

        # ── Settings group ────────────────────────────────────────────
        grp = ttk.LabelFrame(outer, text="Auto Focus Setting", padding=(10, 6))
        grp.pack(fill='x')

        # Row 0 — Peak Height Threshold Percentage
        ttk.Label(grp, text="Peak Height Threshold Percentage (0 - 1) :").grid(
            row=0, column=0, sticky='w', padx=(0, 10), pady=4)
        ttk.Entry(grp, textvariable=self.peak_threshold_var, width=12).grid(
            row=0, column=1, sticky='w', pady=4)

        # Row 1 — Filter Half Size
        ttk.Label(grp, text="Filter Half Size (points) :").grid(
            row=1, column=0, sticky='w', padx=(0, 10), pady=4)
        ttk.Entry(grp, textvariable=self.peak_half_size_var, width=12).grid(
            row=1, column=1, sticky='w', pady=4)

        # ── Test group ────────────────────────────────────────────────
        tgrp = ttk.LabelFrame(outer, text="Test with Images (Temporary)", padding=(10, 6))
        tgrp.pack(fill='x', pady=(12, 0))

        # Row 0 — folder path + Browse
        ttk.Label(tgrp, text="Images Folder :").grid(
            row=0, column=0, sticky='w', padx=(0, 6), pady=4)
        ttk.Entry(tgrp, textvariable=self.test_images_dir_var, width=50).grid(
            row=0, column=1, sticky='ew', pady=4)
        ttk.Button(tgrp, text="Browse…", command=self._browse_folder).grid(
            row=0, column=2, padx=(6, 0), pady=4)
        tgrp.columnconfigure(1, weight=1)

        # Row 1 — Run Test button
        self._btn_test = ttk.Button(tgrp, text="Run Test", command=self._run_test)
        self._btn_test.grid(row=1, column=0, columnspan=3, pady=(4, 2))

        # ── Result log ────────────────────────────────────────────────
        rgrp = ttk.LabelFrame(outer, text="Result", padding=(6, 4))
        rgrp.pack(fill='both', expand=True, pady=(10, 0))

        self._result_text = tk.Text(rgrp, height=12, state='disabled',
                                    wrap='none', font=('Courier New', 9))
        vsb = ttk.Scrollbar(rgrp, orient='vertical',   command=self._result_text.yview)
        hsb = ttk.Scrollbar(rgrp, orient='horizontal', command=self._result_text.xview)
        self._result_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._result_text.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        rgrp.rowconfigure(0, weight=1)
        rgrp.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select images folder")
        if d:
            self.test_images_dir_var.set(d)

    def _append_result(self, line: str):
        """Append a line to the result Text widget (thread-safe via after)."""
        def _do():
            self._result_text.configure(state='normal')
            self._result_text.insert('end', line + '\n')
            self._result_text.see('end')
            self._result_text.configure(state='disabled')
        self._result_text.after(0, _do)

    def _clear_result(self):
        self._result_text.configure(state='normal')
        self._result_text.delete('1.0', 'end')
        self._result_text.configure(state='disabled')

    def _run_test(self):
        """Launch the scan in a background thread so the UI stays responsive."""
        folder = self.test_images_dir_var.get().strip()
        if not folder or not os.path.isdir(folder):
            self._clear_result()
            self._append_result("ERROR: Please select a valid images folder first.")
            return

        try:
            threshold  = float(self.peak_threshold_var.get())
            half_size  = int(self.peak_half_size_var.get())
        except ValueError:
            self._clear_result()
            self._append_result("ERROR: Invalid threshold or half-size value.")
            return

        self._btn_test.configure(state='disabled')
        self._clear_result()
        t = threading.Thread(
            target=self._scan_worker,
            args=(folder, threshold, half_size),
            daemon=True,
        )
        t.start()

    def _scan_worker(self, folder: str, threshold: float, half_size: int):
        """
        Background worker: compute FMI for every image, run peak detection,
        print a table and the verdict.
        """
        def out(line=""):
            self._append_result(line)

        out(f"Folder    : {folder}")
        out(f"Threshold : {threshold}  (PeakHeightThresholdPercentage)")
        out(f"Half-size : {half_size}  (PeakFilterHalfSize)")
        out("-" * 62)

        # Collect images
        patterns = ["*.bmp", "*.png", "*.tif", "*.tiff", "*.jpg"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = sorted(files)

        if len(files) < 2:
            out(f"ERROR: need at least 2 images, found {len(files)}.")
            self._btn_test.after(0, lambda: self._btn_test.configure(state='normal'))
            return

        out(f"{'Step':>4}  {'Filename':<28}  {'FMI':>10}")
        out("-" * 62)

        fmis = []
        for i, fp in enumerate(files):
            try:
                fmi = compute_fmi(fp)
            except Exception as e:
                out(f"{i:4d}  {os.path.basename(fp):<28}  ERROR: {e}")
                fmis.append(0.0)
                continue
            fmis.append(fmi)
            out(f"{i:4d}  {os.path.basename(fp):<28}  {fmi:10.3f}")

        out("-" * 62)
        out(f"Max FMI at step {fmis.index(max(fmis))} = {max(fmis):.3f}")
        out()

        # Peak detection with the current UI values
        peaks = peak_detection(fmis,
                                window_half_size=half_size,
                                threshold_perc=threshold)
        if peaks:
            best_idx = peaks[0]
            out(f"PEAK FOUND  → step {best_idx}  "
                f"file={os.path.basename(files[best_idx])}  "
                f"FMI={fmis[best_idx]:.3f}")
            if len(peaks) > 1:
                out(f"  (additional peaks at steps: {peaks[1:]})")
        else:
            out("NO PEAK DETECTED with these parameters.")
            out()
            out("Why no peak?  Two common causes:")
            out(f"  1. Threshold too HIGH ({threshold}) — the peak FMI rise above its")
            out(f"     neighbours is < {threshold*100:.1f}%.  Try lowering it (e.g. 0.01).")
            out(f"  2. Half-size too LARGE ({half_size}) — the window spans too many steps")
            out(f"     so the candidate is no longer a strict local maximum.")
            out(f"     Try reducing it (e.g. 1).")

        out()
        out("Done.")
        self._btn_test.after(0, lambda: self._btn_test.configure(state='normal'))
