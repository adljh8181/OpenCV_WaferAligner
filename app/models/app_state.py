"""
================================================================================
APP STATE  (Model layer)
================================================================================
Shared mutable state consumed by ViewModels and Views.
Centralising state here avoids passing dozens of variables between
the main window and individual tab classes.
================================================================================
"""

import os
from app.models.recipe_model import RecipeManager, DEFAULT_DIRECTION_PARAMS


class AppState:
    """
    Single source of truth for all mutable UI state.

    Instantiated once by WaferAlignerUI on startup and passed to every
    ViewModel and View that needs it.
    """

    def __init__(self, recipes_root: str):
        # ── Recipe ──────────────────────────────────────────────────────────
        self.recipe_mgr = RecipeManager(recipes_root)
        self.recipe_loaded: bool = False
        self.current_recipe: dict | None = None   # Raw dict from RecipeManager.load()

        # ── Find Wafer Edge ─────────────────────────────────────────────────
        # Per-direction config cache (persists when the user switches directions)
        self.edge_configs: dict[str, dict] = {
            d: dict(DEFAULT_DIRECTION_PARAMS)
            for d in ["LEFT", "RIGHT", "TOP", "BOTTOM"]
        }
        self.last_edge_dir: str = "LEFT"
        # Flag: suppress slider-change callbacks while loading a recipe
        self._is_loading_recipe: bool = False

        # ── Find Pattern ─────────────────────────────────────────────────────
        # Trained template origin (centre of the cropped region in the search image).
        # Mirrors fiducialCenterOffsetX/Y in EmguVision.cs
        self.template_crop_cx: float | None = None
        self.template_crop_cy: float | None = None
        self.template_loaded: bool = False
