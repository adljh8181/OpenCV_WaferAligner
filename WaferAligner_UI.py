"""
WaferAligner_UI.py  --  Application entry point
-------------------------------------------------
All implementation has moved into the app/ package.
This file simply launches the application.
"""
import tkinter as tk
from tkinter import ttk
from app.views.main_window import WaferAlignerUI


def main():
    root = tk.Tk()

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")

    app = WaferAlignerUI(root)

    def on_closing():
        app.on_stop_server()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
