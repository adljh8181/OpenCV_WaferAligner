"""
================================================================================
SETTINGS TAB  (View layer)
================================================================================
Provides application-level settings:
  - Recipe Folder: browse to any folder; changing it updates the recipe browser
    so the next Load will scan files from the new location.
================================================================================
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class SettingsTab:
    """
    Owns all widgets in the top-level "Settings" tab.

    Parameters
    ----------
    parent_nb         : ttk.Notebook  — the main top-level notebook
    state             : AppState      — shared application state
    on_folder_changed : callable(new_path: str)
                        Called by main_window when the user confirms a new
                        recipe folder so it can rebuild the browser.
    log_callback      : callable(msg: str)
    """

    def __init__(self, parent_nb: ttk.Notebook, state,
                 on_folder_changed=None, log_callback=None):
        self.state              = state
        self._on_folder_changed = on_folder_changed or (lambda p: None)
        self._log               = log_callback or print

        self.tab = ttk.Frame(parent_nb)
        parent_nb.add(self.tab, text="Settings")

        self._recipe_folder_var = tk.StringVar(
            value=self.state.recipe_mgr.recipes_root)

        self._build()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build(self):
        outer = ttk.Frame(self.tab)
        outer.pack(fill='both', expand=True, padx=20, pady=20)

        grp = ttk.LabelFrame(outer, text="Recipe Folder")
        grp.pack(fill='x', pady=(0, 15))

        row = ttk.Frame(grp)
        row.pack(fill='x', padx=10, pady=10)

        self._folder_entry = ttk.Entry(
            row, textvariable=self._recipe_folder_var, state='readonly', width=60)
        self._folder_entry.pack(side='left', fill='x', expand=True, padx=(0, 6))

        ttk.Button(row, text="Browse…", command=self._browse_folder).pack(side='left')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _browse_folder(self):
        current = self._recipe_folder_var.get()
        if not os.path.isdir(current):
            current = os.getcwd()
        chosen = filedialog.askdirectory(
            title="Select Recipe Folder",
            initialdir=current,
        )
        if chosen:
            new_path = os.path.normpath(chosen)
            self._recipe_folder_var.set(new_path)
            self.state.recipe_mgr.recipes_root = new_path
            self._on_folder_changed(new_path)
            self._log(f"[Settings] Recipe folder set to: {new_path}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_recipe_folder(self) -> str:
        return self._recipe_folder_var.get()

    def set_recipe_folder(self, path: str):
        """Called externally if the folder is changed programmatically."""
        self._recipe_folder_var.set(os.path.normpath(path))
