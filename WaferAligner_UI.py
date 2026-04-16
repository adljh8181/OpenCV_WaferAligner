"""
WaferAligner_UI.py  --  Application entry point
-------------------------------------------------
All implementation has moved into the app/ package.
This file simply launches the application.
"""
import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image
import pystray
from app.views.main_window import WaferAlignerUI


def _cleanup_temp_files():
    """Delete temp_cropped_template_* files left behind from previous sessions."""
    cwd = os.getcwd()
    for f in os.listdir(cwd):
        if f.startswith("temp_cropped_template_"):
            try:
                os.remove(os.path.join(cwd, f))
            except Exception:
                pass


def main():
    root = tk.Tk()

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")

    app = WaferAlignerUI(root)

    # Clean up any temp files left from previous sessions on startup
    _cleanup_temp_files()

    # Manage system tray state
    icon_instance = None

    def on_restore(icon, item):
        icon.stop()
        root.after(0, root.deiconify)

    def on_exit(icon, item):
        icon.stop()
        _cleanup_temp_files()
        root.after(0, lambda: (app.on_stop_server(), root.destroy()))

    def on_closing():
        nonlocal icon_instance
        root.withdraw()

        # Load icon image
        icon_path = os.path.join(os.getcwd(), "QES.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.getcwd(), "logo.png")
            
        if os.path.exists(icon_path):
            try:
                image = Image.open(icon_path)
            except Exception:
                image = Image.new('RGB', (64, 64), color=(73, 109, 137))
        else: 
            image = Image.new('RGB', (64, 64), color=(73, 109, 137))

        menu = pystray.Menu(
            pystray.MenuItem("Restore", on_restore, default=True),
            pystray.MenuItem("Exit", on_exit)
        )
        
        icon_instance = pystray.Icon("VPServer", image, "VPServer - Wafer Alignment UI", menu)
        threading.Thread(target=icon_instance.run, daemon=True).start()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

