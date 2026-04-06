"""
================================================================================
MAIN WINDOW  (View layer — root orchestrator)
================================================================================
WaferAlignerUI:
  - Creates the Tk root notebook with the two top-level tabs
  - Instantiates AppState, ZmqTab, EdgeTab, PatternTab
  - Owns shared helpers: display_image(), display_cv2_image(), log()
  - Handles cross-tab recipe loading / server-sync events via _handle_server_sync()
================================================================================
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import shutil

from app.models.app_state import AppState
from app.models.recipe_model import RecipeBrowserDialog, DEFAULT_DIRECTION_PARAMS
from app.views.zmq_tab import ZmqTab
from app.views.edge_tab import EdgeTab
from app.views.pattern_tab import PatternTab


class WaferAlignerUI:
    """
    Root UI class — creates the main window and wires all tabs together.
    Instantiate with a Tk root widget.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VPServer - Wafer Alignment UI")
        self.root.geometry("1400x900")

        # ── Windows taskbar icon ─────────────────────────────────────────
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'qes.waferaligner.ui.1.0')
        except Exception:
            pass

        try:
            icon_qes = os.path.join(os.getcwd(), "QES.ico")
            icon_png = os.path.join(os.getcwd(), "logo.png")
            icon_ico = os.path.join(os.getcwd(), "logo.ico")
            if os.path.exists(icon_qes):
                self.root.iconbitmap(icon_qes)
            elif os.path.exists(icon_png):
                self.root.iconphoto(False, tk.PhotoImage(file=icon_png))
            elif os.path.exists(icon_ico):
                self.root.iconbitmap(icon_ico)
        except Exception as e:
            print(f"Failed to load custom logo: {e}")

        # ── Shared state ─────────────────────────────────────────────────
        self.state = AppState(os.path.join(os.getcwd(), "recipes"))

        # ── Top-level notebook ───────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # ── ZMQ tab ──────────────────────────────────────────────────────
        self.zmq_tab = ZmqTab(
            self.notebook,
            log_callback=self._log_status,
            ui_sync_callback=self._handle_server_sync,
        )

        # ── Recipe tab ───────────────────────────────────────────────────
        tab_recipe = ttk.Frame(self.notebook)
        self.notebook.add(tab_recipe, text="Recipe")

        # Recipe top bar
        top_frame = ttk.Frame(tab_recipe)
        top_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(top_frame, text="Recipe Name:").pack(side='left', padx=5)
        self.global_recipe_name_var = tk.StringVar(value="No Recipe Loaded")
        ttk.Entry(top_frame, textvariable=self.global_recipe_name_var,
                  width=30, state='readonly').pack(side='left', padx=5)
        ttk.Button(top_frame, text="Load",
                   command=self._open_recipe_browser).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Save",
                   command=self.save_all_config).pack(side='left', padx=5)

        # Recipe sub-notebook
        recipe_nb = ttk.Notebook(tab_recipe)
        recipe_nb.pack(expand=True, fill='both', padx=10, pady=5)

        self.edge_tab = EdgeTab(
            recipe_nb, self.state,
            display_image=self.display_image,
            display_cv2_image=self.display_cv2_image,
            log_callback=self._log_status,
        )
        self.pattern_tab = PatternTab(
            recipe_nb, self.state,
            display_image=self.display_image,
            display_cv2_image=self.display_cv2_image,
            log_callback=self._log_status,
        )

        # Keep references to tab frames for notebook.select()
        self.tab_edge    = self.edge_tab.tab
        self.tab_pattern = self.pattern_tab.tab

        # Auto-start the ZMQ server after UI is fully initialized
        self.root.after(500, self.zmq_tab.on_start_server)

    # ------------------------------------------------------------------
    # Shared display / log helpers
    # ------------------------------------------------------------------

    def _log_status(self, msg: str):
        """Write to the ZMQ tab Status log (thread-safe via root.after)."""
        self.zmq_tab.log(self.zmq_tab.log_status, msg)

    def display_cv2_image(self, img, label, overlay_func=None):
        """Resize and display a BGR/gray OpenCV image in a Tkinter Label."""
        if img is None:
            return
        try:
            parent = label.master
            parent.update_idletasks()
            p_width  = parent.winfo_width()
            p_height = parent.winfo_height()

            if p_width < 50 or p_height < 50:
                toplevel = label.winfo_toplevel()
                toplevel.update_idletasks()
                root_w   = max(toplevel.winfo_width(), 1000)
                root_h   = max(toplevel.winfo_height(), 700)
                p_width  = max((root_w - 400) // 2 - 40, 200)
                p_height = max((root_h - 100) // 2 - 40, 200)

            target_w = max(p_width  - 10, 100)
            target_h = max(p_height - 10, 100)
            h, w     = img.shape[:2]
            if h <= 0 or w <= 0:
                return

            scale    = min(target_w / w, target_h / h)
            img_disp = img.copy()
            if overlay_func:
                img_disp = overlay_func(img_disp)

            new_w    = max(int(w * scale), 1)
            new_h    = max(int(h * scale), 1)
            img_resized = cv2.resize(img_disp, (new_w, new_h))

            if len(img_resized.shape) == 3:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

            pil_img = Image.fromarray(img_rgb)
            tk_img  = ImageTk.PhotoImage(pil_img)
            label.configure(image=tk_img, text="")
            label.image = tk_img
        except Exception as e:
            print(f"Image display error: {e}")

    def display_image(self, path: str, label, overlay_func=None):
        """Load an image from *path* and display it via display_cv2_image."""
        try:
            img = cv2.imread(path)
            self.display_cv2_image(img, label, overlay_func)
        except Exception as e:
            print(f"Image load error: {e}")

    # ------------------------------------------------------------------
    # Recipe system
    # ------------------------------------------------------------------

    def _open_recipe_browser(self):
        dlg  = RecipeBrowserDialog(self.root, self.state.recipe_mgr.recipes_root)
        path = dlg.show()
        if path:
            self.state.current_recipe = self.state.recipe_mgr.load(path)
            self.state.recipe_loaded  = True
            self._apply_recipe(self.state.current_recipe)

    def _apply_recipe(self, recipe: dict):
        name = recipe.get("name", "Unknown")
        self.global_recipe_name_var.set(name)

        # Enable buttons in both tabs
        self.edge_tab.enable_buttons()
        self.pattern_tab.enable_buttons()

        # ── Edge config ──────────────────────────────────────────────────
        self.state._is_loading_recipe = True
        try:
            fe = recipe.get("find_edge", {})
            for d in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
                if d in fe:
                    self.state.edge_configs[d] = fe[d].copy()
            self.edge_tab.update_sliders_from_cache(
                self.edge_tab.edge_dir_var.get())
            self.state.last_edge_dir = self.edge_tab.edge_dir_var.get()

            univ = recipe.get("use_universal_edge_params", False)
            self.edge_tab.edge_universal_var.set(univ)
        finally:
            self.state._is_loading_recipe = False

        # ── Pattern config ───────────────────────────────────────────────
        fp = recipe.get("find_pattern", {})
        pt = self.pattern_tab
        if "MatchThreshold" in fp: pt.set_slider_value(pt.pattern_thresh_var, fp["MatchThreshold"])
        if "NumFeatures"    in fp: pt.set_slider_value(pt.pattern_num_var, fp["NumFeatures"])
        if "GradThrPct"     in fp: pt.set_slider_value(pt.pattern_weak_var, fp["GradThrPct"])
        if "TSpread"        in fp: pt.set_slider_value(pt.pattern_tspread_var, fp["TSpread"])
        if "HystKernel"     in fp: pt.set_slider_value(pt.pattern_hyst_var, fp["HystKernel"])
        if "SearchMode"     in fp: pt.pattern_mode_var.set(fp["SearchMode"])

        cx = float(fp.get("TemplateCropCX", 0.0))
        cy = float(fp.get("TemplateCropCY", 0.0))
        if cx > 0 and cy > 0:
            self.state.template_crop_cx = cx
            self.state.template_crop_cy = cy

        tpath = fp.get("TemplatePath", "")
        if tpath and os.path.exists(tpath):
            pt.template_img_var.set(tpath)

            # Load detection mask if saved with recipe
            mask_path = fp.get("DetectionMaskPath", "")
            detection_mask = None
            if mask_path and os.path.exists(mask_path):
                detection_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                self.state.template_detection_mask = detection_mask
                self._log_status(f"Loaded detection mask from recipe: {mask_path}")
            else:
                self.state.template_detection_mask = None

            img = cv2.imread(tpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pt.vm.linemod_matcher.load_template(img, detection_mask=detection_mask)
                self.state.template_loaded = True
                self._log_status(f"Loaded recipe template from {tpath}")

            # Show mask overlay on template display if mask exists
            if detection_mask is not None:
                pt._display_template_with_mask(tpath, detection_mask)
            else:
                self.display_image(tpath, pt.lbl_template_img)
        else:
            self.state.template_loaded = False
            self.state.template_detection_mask = None
            pt.lbl_template_img.config(image='')
            pt.template_img_var.set("")

        self._log_status(f"Active Recipe: {name}")

    def save_all_config(self):
        if not self.state.recipe_loaded or not self.state.current_recipe:
            messagebox.showwarning(
                "Save",
                "No recipe loaded. Please load a recipe first via the Load button.")
            return

        try:
            et = self.edge_tab
            pt = self.pattern_tab
            active_dir = et.edge_dir_var.get()

            # Save current slider values for active direction into cache
            self.state.edge_configs[active_dir]["KernelSize"]       = et.edge_kernel_var.get()
            self.state.edge_configs[active_dir]["EdgeThreshold"]    = et.edge_thresh_var.get()
            self.state.edge_configs[active_dir]["NumRegions"]       = et.edge_regions_var.get()
            self.state.edge_configs[active_dir]["BorderIgnorePct"]  = et.edge_border_var.get()
            self.state.edge_configs[active_dir]["RansacThreshold"]  = et.edge_ransac_var.get()

            if et.edge_universal_var.get():
                for d in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
                    if d != active_dir:
                        self.state.edge_configs[d] = self.state.edge_configs[active_dir].copy()
                        
            self.state.current_recipe["use_universal_edge_params"] = et.edge_universal_var.get()

            fe = self.state.current_recipe.setdefault("find_edge", {})
            for d in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
                fe[d] = self.state.edge_configs[d].copy()

            fp = self.state.current_recipe.setdefault("find_pattern", {})
            fp["NumFeatures"]    = pt.pattern_num_var.get()
            fp["MatchThreshold"] = pt.pattern_thresh_var.get()
            fp["GradThrPct"]     = pt.pattern_weak_var.get()
            fp["TSpread"]        = pt.pattern_tspread_var.get()
            fp["HystKernel"]     = pt.pattern_hyst_var.get()
            fp["SearchMode"]     = pt.pattern_mode_var.get()
            fp["TemplateCropCX"] = f"{self.state.template_crop_cx or 0.0:.1f}"
            fp["TemplateCropCY"] = f"{self.state.template_crop_cy or 0.0:.1f}"

            tpath = pt.template_img_var.get()
            if tpath and os.path.exists(tpath):
                recipe_dir  = os.path.dirname(self.state.current_recipe["path"])
                recipe_name = self.state.current_recipe["name"]
                dest_path   = os.path.join(recipe_dir, f"{recipe_name}_template.png")
                if os.path.abspath(tpath) != os.path.abspath(dest_path):
                    shutil.copy2(tpath, dest_path)
                    pt.template_img_var.set(dest_path)
                fp["TemplatePath"] = dest_path

                # Save detection mask alongside template
                mask_src = pt.vm._mask_path_for(tpath)
                mask_dest = os.path.join(recipe_dir, f"{recipe_name}_template_mask.png")
                if self.state.template_detection_mask is not None:
                    cv2.imwrite(mask_dest, self.state.template_detection_mask)
                    fp["DetectionMaskPath"] = mask_dest
                    self._log_status(f"Detection mask saved to recipe: {mask_dest}")
                elif os.path.exists(mask_src):
                    shutil.copy2(mask_src, mask_dest)
                    fp["DetectionMaskPath"] = mask_dest
                else:
                    fp["DetectionMaskPath"] = ""

            self.state.recipe_mgr.save(self.state.current_recipe)
            name = self.state.current_recipe['name']
            self._log_status(f"Saved all config to recipe '{name}'")
            messagebox.showinfo("Saved", f"Recipe '{name}' saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save recipe: {e}")

    # ------------------------------------------------------------------
    # Server sync (events fired by background ZMQ server thread)
    # ------------------------------------------------------------------

    def _handle_server_sync(self, payload: dict):
        """
        Routed here from ZmqTab's ui_sync_callback.
        Switches tabs and updates the UI to reflect background server activity.
        """
        event = payload.get("event")

        if event == "WAFER_EDGE_REQ":
            try:
                self.notebook.select(self.tab_edge)
                self.root.update()
            except Exception:
                pass

            def _do():
                recipe_path = payload.get("recipe_path", "")
                if recipe_path:
                    try:
                        self.state.current_recipe = self.state.recipe_mgr.load(recipe_path)
                        self.state.recipe_loaded  = True
                        self._apply_recipe(self.state.current_recipe)
                    except Exception as ex:
                        self._log_status(f"UI Recipe Sync failed: {ex}")

                direction = payload.get("direction", "LEFT")
                self.edge_tab.edge_dir_var.set(direction)

                img_path = payload.get("image_path", "")
                if img_path:
                    self.edge_tab.edge_img_var.set(img_path)
                    try:
                        self.display_image(img_path, self.edge_tab.lbl_edge_input)
                    except Exception:
                        pass

                    result = payload.get("result", {})
                    if result.get("status") == "ok":
                        self.edge_tab.edge_delta_x_var.set(
                            f"{result.get('delta_x', 0):.3f}")
                        self.edge_tab.edge_delta_y_var.set(
                            f"{result.get('delta_y', 0):.3f}")
                        try:
                            srv = self.zmq_tab.server_instance
                            if srv and hasattr(srv.edge_finder, 'last_result'):
                                full_resp = srv.edge_finder.last_result
                                if full_resp and full_resp.get("success"):
                                    self.edge_tab.visualize_edge_result(
                                        full_resp, img_path, srv.edge_finder)
                        except Exception as e:
                            self._log_status(f"Edge viz error: {e}")

            self.root.after(100, _do)

        elif event == "PM_REQ":
            try:
                self.notebook.select(self.tab_pattern)
                self.root.update()
            except Exception:
                pass

            def _do():
                recipe_path = payload.get("recipe_path", "")
                if recipe_path:
                    try:
                        self.state.current_recipe = self.state.recipe_mgr.load(recipe_path)
                        self.state.recipe_loaded  = True
                        self._apply_recipe(self.state.current_recipe)
                    except Exception as ex:
                        self._log_status(f"UI Recipe Sync failed: {ex}")

                img_path = payload.get("image_path", "")
                if img_path:
                    self.pattern_tab.pattern_img_var.set(img_path)
                    try:
                        self.display_image(img_path, self.pattern_tab.lbl_pattern_input)
                    except Exception:
                        pass

                    result = payload.get("result", {})
                    if result.get("status") == "ok":
                        self.pattern_tab.pattern_score_var.set(
                            str(result.get("score", "0")))
                        self.pattern_tab.pattern_x_var.set(
                            str(result.get("delta_x", "0")))
                        self.pattern_tab.pattern_y_var.set(
                            str(result.get("delta_y", "0")))
                        try:
                            orig_color = cv2.imread(img_path)
                            if orig_color is not None:
                                srv = self.zmq_tab.server_instance
                                def draw_rect(img_draw):
                                    if srv and hasattr(srv, 'matcher'):
                                        return srv.matcher.visualize_match(
                                            img_draw, result, show=False)
                                    return img_draw
                                self.display_cv2_image(
                                    orig_color, self.pattern_tab.lbl_pattern_output,
                                    overlay_func=draw_rect)
                        except Exception as ex:
                            self._log_status(f"Visualizer error: {ex}")
                    else:
                        self.pattern_tab.pattern_score_var.set("0 (No Match)")
                        self.pattern_tab.pattern_x_var.set("0")
                        self.pattern_tab.pattern_y_var.set("0")
                        try:
                            self.display_image(img_path,
                                               self.pattern_tab.lbl_pattern_output)
                        except Exception:
                            pass

            self.root.after(100, _do)

        elif event == "TRAIN_REQ":
            try:
                self.notebook.select(self.tab_pattern)
                self.root.update()
            except Exception:
                pass

            def _do():
                recipe_path = payload.get("recipe_path", "")
                if recipe_path:
                    try:
                        self.state.current_recipe = self.state.recipe_mgr.load(recipe_path)
                        self.state.recipe_loaded  = True
                        self._apply_recipe(self.state.current_recipe)
                    except Exception as ex:
                        self._log_status(f"UI Recipe Sync failed: {ex}")

                img_path = payload.get("image_path", "")
                if img_path:
                    self.pattern_tab.pattern_img_var.set(img_path)
                    try:
                        self.display_image(img_path, self.pattern_tab.lbl_pattern_input)
                    except Exception:
                        pass

                    self._log_status(
                        "Operator intervention requested: Please crop template.")
                    self.pattern_tab._on_crop_template()
                    if self.state.template_loaded:
                        self.save_all_config()
                        self._log_status("Training complete and saved to recipe.")
                        messagebox.showinfo(
                            "Training Complete",
                            "Template pattern successfully trained and saved.")

            self.root.after(100, _do)

    def on_stop_server(self):
        """Called by the window close handler."""
        self.zmq_tab.on_stop_server()


# ---------------------------------------------------------------------------
# Helper: embed a matplotlib figure in a non-blocking Tk Toplevel
# ---------------------------------------------------------------------------
def _show_figure_in_window(fig, title="Figure"):
    """Embed a matplotlib figure in a non-blocking Tk Toplevel window."""
    win = tk.Toplevel()
    win.title(title)
    win.resizable(True, True)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))
