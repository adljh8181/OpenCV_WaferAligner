"""
================================================================================
RECIPE MANAGER
================================================================================
Handles reading/writing XML recipe files and provides the Recipe Browser UI.

Recipe XML schema:
    <Recipe name="..." version="1.0">
        <MetaData>...</MetaData>
        <FindWaferEdge>...</FindWaferEdge>
        <FindPattern>...</FindPattern>
    </Recipe>

Results (Delta X/Y, Score) are NOT stored here — they are sent via ZeroMQ.
================================================================================
"""

import os
import shutil
import datetime
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


# ---------------------------------------------------------------------------
# Default parameter values (used when creating a new recipe)
# ---------------------------------------------------------------------------
DEFAULT_DIRECTION_PARAMS = {
    "KernelSize":      "9",
    "EdgeThreshold":   "500",
    "NumRegions":      "30",
    "BorderIgnorePct": "0.05",
    "RansacThreshold": "3.0",
}

DEFAULT_FIND_EDGE = {
    "LEFT":   dict(DEFAULT_DIRECTION_PARAMS),
    "RIGHT":  dict(DEFAULT_DIRECTION_PARAMS),
    "TOP":    dict(DEFAULT_DIRECTION_PARAMS),
    "BOTTOM": dict(DEFAULT_DIRECTION_PARAMS),
}

DIR_XML_MAPPING = {
    "LEFT":   ("WaferAlignLeftParam",  "WaferAlign Left Parm"),
    "RIGHT":  ("WaferAlignRightParam", "WaferAlign Right Parm"),
    "TOP":    ("WaferAlignTopParam",   "WaferAlign Top Parm"),
    "BOTTOM": ("WaferAlignBotParam",   "WaferAlign Bot Parm")
}

DEFAULT_FIND_PATTERN = {
    "MatchThreshold":  "50.0",
    "NumFeatures":     "128",
    "GradThrPct":      "70.0",
    "TSpread":         "4",
    "HystKernel":      "0",
    "SearchMode":      "Simple (Fast)",
    "TemplatePath":    "",
    "TemplateCropCX":  "0.0",
    "TemplateCropCY":  "0.0",
    "DetectionMaskPath": "",
}


# ---------------------------------------------------------------------------
# RecipeManager — XML read/write helpers
# ---------------------------------------------------------------------------
class RecipeManager:
    """Manages recipe XML files in a root recipes directory."""

    def __init__(self, recipes_root: str):
        self.recipes_root = recipes_root
        os.makedirs(recipes_root, exist_ok=True)

    # ── File helpers ─────────────────────────────────────────────────────

    def list_items(self, folder: str = None) -> list:
        """
        Return sorted list of (name, path, is_dir) tuples in *folder*.
        If folder is None, use recipes_root.
        """
        base = folder or self.recipes_root
        items = []
        try:
            for entry in sorted(os.scandir(base), key=lambda e: (not e.is_dir(), e.name.lower())):
                items.append((entry.name, entry.path, entry.is_dir()))
        except OSError:
            pass
        return items

    def create_recipe(self, folder: str, name: str, edge_params: dict = None,
                      pattern_params: dict = None) -> str:
        """Create a new XML recipe file. Returns the file path."""
        if not name.lower().endswith(".xml"):
            name += ".xml"
        path = os.path.join(folder, name)

        edge   = {**DEFAULT_FIND_EDGE,    **(edge_params    or {})}
        patter = {**DEFAULT_FIND_PATTERN, **(pattern_params or {})}

        root = ET.Element("Recipe", name=os.path.splitext(name)[0], version="1.0")

        meta = ET.SubElement(root, "MetaData")
        ET.SubElement(meta, "CreatedAt").text = datetime.datetime.now().isoformat(timespec="seconds")
        ET.SubElement(meta, "Description").text = ""

        edge_el = ET.SubElement(root, "FindWaferEdge")
        for dir_name, (tag_name, attr_name) in DIR_XML_MAPPING.items():
            d_el = ET.SubElement(edge_el, tag_name, Name=attr_name)
            d_params = edge.get(dir_name, DEFAULT_DIRECTION_PARAMS)
            for k, v in d_params.items():
                ET.SubElement(d_el, k).text = str(v)

        patt_el = ET.SubElement(root, "FindPattern")
        for k, v in patter.items():
            ET.SubElement(patt_el, k).text = str(v)

        self._write_pretty(root, path)
        return path

    def create_folder(self, parent: str, name: str) -> str:
        """Create a subfolder. Returns path."""
        path = os.path.join(parent, name)
        os.makedirs(path, exist_ok=True)
        return path

    def rename(self, old_path: str, new_name: str) -> str:
        """Rename a file or folder. Returns new path."""
        parent = os.path.dirname(old_path)
        is_file = os.path.isfile(old_path)
        if is_file and not new_name.lower().endswith(".xml"):
            new_name += ".xml"
        new_path = os.path.join(parent, new_name)
        os.rename(old_path, new_path)
        return new_path

    def duplicate(self, path: str) -> str:
        """Duplicate a file. Returns new path."""
        base, ext = os.path.splitext(path)
        new_path = f"{base}_copy{ext}"
        counter = 1
        while os.path.exists(new_path):
            new_path = f"{base}_copy{counter}{ext}"
            counter += 1
        shutil.copy2(path, new_path)
        return new_path

    def delete(self, path: str):
        """Delete a file or folder (recursively)."""
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    # ── XML load / save ──────────────────────────────────────────────────

    def load(self, path: str) -> dict:
        """
        Load a recipe XML. Returns dict:
        {
            'name': str,
            'path': str,
            'meta': {'created_at': str, 'description': str},
            'find_edge':    {ScanDirection, KernelSize, ...},
            'find_pattern': {MatchThreshold, NumFeatures, TemplatePath, ...},
        }
        """
        tree = ET.parse(path)
        root = tree.getroot()

        def _get(parent_tag, key, default=""):
            el = root.find(f"{parent_tag}/{key}")
            return el.text.strip() if el is not None and el.text else default

        meta = root.find("MetaData")
        created  = (meta.find("CreatedAt").text  if meta is not None and meta.find("CreatedAt")  is not None else "")
        desc     = (meta.find("Description").text if meta is not None and meta.find("Description") is not None else "")

        edge_params = {}
        for dir_name, (tag_name, attr_name) in DIR_XML_MAPPING.items():
            dir_el = root.find(f"FindWaferEdge/{tag_name}")
            d_params = {}
            if dir_el is not None:
                for k, v_def in DEFAULT_DIRECTION_PARAMS.items():
                    el = dir_el.find(k)
                    d_params[k] = el.text.strip() if el is not None and el.text else v_def
            else:
                old_dir_el = root.find(f"FindWaferEdge/{dir_name}")
                if old_dir_el is not None:
                    for k, v_def in DEFAULT_DIRECTION_PARAMS.items():
                        el = old_dir_el.find(k)
                        d_params[k] = el.text.strip() if el is not None and el.text else v_def
                else:
                    for k, v_def in DEFAULT_DIRECTION_PARAMS.items():
                        d_params[k] = _get("FindWaferEdge", k, v_def)
            edge_params[dir_name] = d_params
        patt_params = {k: _get("FindPattern",   k, v) for k, v in DEFAULT_FIND_PATTERN.items()}

        univ_el = root.find("FindWaferEdge/UseUniversalParams")
        use_univ = (univ_el.text.lower() == 'true') if univ_el is not None and univ_el.text else False

        return {
            "name":         root.get("name", os.path.splitext(os.path.basename(path))[0]),
            "path":         path,
            "meta":         {"created_at": created, "description": desc},
            "find_edge":    edge_params,
            "find_pattern": patt_params,
            "use_universal_edge_params": use_univ,
        }

    def save(self, recipe: dict):
        """Save an in-memory recipe dict back to its XML file."""
        path = recipe["path"]
        name = recipe["name"]

        root = ET.Element("Recipe", name=name, version="1.0")

        meta_d = recipe.get("meta", {})
        meta = ET.SubElement(root, "MetaData")
        ET.SubElement(meta, "CreatedAt").text  = meta_d.get("created_at", "")
        ET.SubElement(meta, "Description").text = meta_d.get("description", "")

        edge_el = ET.SubElement(root, "FindWaferEdge")
        ET.SubElement(edge_el, "UseUniversalParams").text = str(recipe.get("use_universal_edge_params", False))
        fe_dict = recipe.get("find_edge", DEFAULT_FIND_EDGE)
        for dir_name, (tag_name, attr_name) in DIR_XML_MAPPING.items():
            d_el = ET.SubElement(edge_el, tag_name, Name=attr_name)
            d_params = fe_dict.get(dir_name, DEFAULT_DIRECTION_PARAMS)
            for k, v in d_params.items():
                ET.SubElement(d_el, k).text = str(v)

        patt_el = ET.SubElement(root, "FindPattern")
        for k, v in recipe.get("find_pattern", DEFAULT_FIND_PATTERN).items():
            ET.SubElement(patt_el, k).text = str(v)

        self._write_pretty(root, path)

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _write_pretty(root: ET.Element, path: str):
        """Write XML with pretty indentation."""
        rough = ET.tostring(root, encoding="unicode")
        pretty = minidom.parseString(rough).toprettyxml(indent="  ")
        # Remove the extra blank line minidom adds on Python 3
        lines = [l for l in pretty.splitlines() if l.strip()]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# RecipeBrowserDialog
# ---------------------------------------------------------------------------
class RecipeBrowserDialog:
    """
    Modal recipe browser window.
    Usage:
        dlg = RecipeBrowserDialog(parent_root, recipes_root)
        path = dlg.show()   # returns selected XML path, or None if cancelled
    """

    def __init__(self, parent, recipes_root: str):
        self.parent   = parent
        self.manager  = RecipeManager(recipes_root)
        self.selected = None

        # Build window
        self.win = tk.Toplevel(parent)
        self.win.title("Recipe Manager")
        self.win.geometry("600x400")
        self.win.resizable(True, True)
        self.win.grab_set()          # make modal
        self.win.transient(parent)

        self._build_ui()
        self._refresh_tree()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Left: tree + scrollbar ───────────────────────────────────────
        left = ttk.Frame(self.win)
        left.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        ttk.Label(left, text="Recipe List", font=("", 9, "bold")).pack(anchor="w")

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=("type",), show="tree",
                                  selectmode="browse")
        self.tree.column("#0",    width=340, minwidth=200)
        self.tree.column("type",  width=0,   minwidth=0, stretch=False)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>",  self._on_double_click)
        self.tree.bind("<<TreeviewOpen>>", self._on_expand)

        # ── Right: buttons ───────────────────────────────────────────────
        right = ttk.Frame(self.win, width=140)
        right.pack(side="right", fill="y", padx=(5, 10), pady=10)
        right.pack_propagate(False)

        lf = ttk.LabelFrame(right, text="Recipe List")
        lf.pack(fill="x", pady=(0, 10))

        for text, cmd in [
            ("New Item",    self._new_item),
            ("New Folder",  self._new_folder),
            ("Rename",      self._rename),
            ("Duplicate",   self._duplicate),
            ("Delete",      self._delete),
        ]:
            ttk.Button(lf, text=text, command=cmd, width=14).pack(padx=5, pady=3, fill="x")

        # Bottom: Select + Cancel
        ttk.Button(right, text="Select", command=self._select, width=14).pack(
            side="bottom", pady=(5, 3), fill="x")
        ttk.Button(right, text="Cancel", command=self._cancel, width=14).pack(
            side="bottom", pady=(0, 3), fill="x")

    # ── Tree helpers ─────────────────────────────────────────────────────

    def _refresh_tree(self, expand_path: str = None):
        """Rebuild entire tree from disk."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._populate_node("", self.manager.recipes_root)

    def _populate_node(self, parent_iid: str, folder: str):
        for name, path, is_dir in self.manager.list_items(folder):
            if is_dir:
                iid = self.tree.insert(parent_iid, "end", text=f"📁 {name}",
                                       values=("dir", path), open=False)
                # Add a dummy child so the expand arrow shows
                self.tree.insert(iid, "end", text="", values=("dummy", ""))
            elif name.lower().endswith(".xml"):
                self.tree.insert(parent_iid, "end", text=f"📄 {name}",
                                 values=("file", path))

    def _on_expand(self, event):
        """Lazy-load children when a folder is opened."""
        iid = self.tree.focus()
        children = self.tree.get_children(iid)
        # If only dummy child, replace with real content
        if len(children) == 1:
            child_vals = self.tree.item(children[0], "values")
            if child_vals and child_vals[0] == "dummy":
                self.tree.delete(children[0])
                folder_path = self.tree.item(iid, "values")[1]
                self._populate_node(iid, folder_path)

    def _selected_path(self) -> str | None:
        iid = self.tree.focus()
        if not iid:
            return None
        vals = self.tree.item(iid, "values")
        return vals[1] if vals else None

    def _selected_folder(self) -> str:
        """Return the folder of the current selection (or recipes_root)."""
        path = self._selected_path()
        if not path:
            return self.manager.recipes_root
        if os.path.isdir(path):
            return path
        return os.path.dirname(path)

    # ── Button handlers ──────────────────────────────────────────────────

    def _new_item(self):
        name = simpledialog.askstring("New Recipe", "Recipe name:", parent=self.win)
        if name:
            folder = self._selected_folder()
            try:
                self.manager.create_recipe(folder, name)
                self._refresh_tree()
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self.win)

    def _new_folder(self):
        name = simpledialog.askstring("New Folder", "Folder name:", parent=self.win)
        if name:
            parent_folder = self._selected_folder()
            try:
                self.manager.create_folder(parent_folder, name)
                self._refresh_tree()
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self.win)

    def _rename(self):
        path = self._selected_path()
        if not path:
            messagebox.showwarning("Rename", "Select a recipe or folder first.", parent=self.win)
            return
        old_name = os.path.basename(path)
        new_name = simpledialog.askstring("Rename", "New name:", initialvalue=old_name, parent=self.win)
        if new_name and new_name != old_name:
            try:
                self.manager.rename(path, new_name)
                self._refresh_tree()
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self.win)

    def _duplicate(self):
        path = self._selected_path()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Duplicate", "Select a recipe file first.", parent=self.win)
            return
        try:
            self.manager.duplicate(path)
            self._refresh_tree()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win)

    def _delete(self):
        path = self._selected_path()
        if not path:
            messagebox.showwarning("Delete", "Select a recipe or folder first.", parent=self.win)
            return
        name = os.path.basename(path)
        if messagebox.askyesno("Delete", f"Delete '{name}'?", parent=self.win):
            try:
                self.manager.delete(path)
                self._refresh_tree()
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self.win)

    def _select(self):
        path = self._selected_path()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Select", "Select a recipe (.xml) file first.", parent=self.win)
            return
        self.selected = path
        self.win.destroy()

    def _cancel(self):
        self.selected = None
        self.win.destroy()

    def _on_double_click(self, event):
        path = self._selected_path()
        if path and os.path.isfile(path):
            self._select()

    # ── Show (modal entry point) ─────────────────────────────────────────

    def show(self) -> str | None:
        """Block until the dialog is closed. Returns selected path or None."""
        self.parent.wait_window(self.win)
        return self.selected
