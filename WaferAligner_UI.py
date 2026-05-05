"""
WaferAligner_UI.py  --  Application entry point
-------------------------------------------------
All implementation has moved into the app/ package.
This file simply launches the application.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import pystray
from app.views.main_window import WaferAlignerUI

# ── Single-instance guard (Windows named mutex) ───────────────────────────────
_MUTEX_NAME = "Global\\VPServer_WaferAlignerUI_SingleInstance"

def _check_single_instance():
    """
    Creates a named mutex. Returns the mutex handle on success.
    If another instance already holds the mutex, shows a warning and exits.
    """
    import ctypes
    import ctypes.wintypes

    ERROR_ALREADY_EXISTS = 183

    handle = ctypes.windll.kernel32.CreateMutexW(None, True, _MUTEX_NAME)
    if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        # Need a minimal Tk root just to show the messagebox
        _root = tk.Tk()
        _root.withdraw()
        messagebox.showerror(
            "Application Already Running",
            "VPServer – Wafer Alignment UI is already running.\n\n"
            "If you don't see the window, check the system tray (bottom-right corner "
            "of your taskbar), right-click the icon, and choose Exit."
        )
        _root.destroy()
        sys.exit(1)

    return handle  # keep alive for the duration of the process
# ─────────────────────────────────────────────────────────────────────────────


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
    _mutex = _check_single_instance()  # exits here if another instance is running

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
        def _do_exit():
            def _after_stop():
                try:
                    root.quit()
                    root.destroy()
                except Exception:
                    pass
            app.on_stop_server(completion_callback=_after_stop)
        root.after(0, _do_exit)

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