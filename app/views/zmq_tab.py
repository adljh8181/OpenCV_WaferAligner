"""
================================================================================
ZMQ TAB  (View layer)
================================================================================
Builds the "ZMQ Communication" tab and handles:
  - Server start / stop buttons
  - Log text areas (RX, TX, Status)
  - ZMQ client socket management (send_zmq)
================================================================================
"""

import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import zmq

from app.services.zmq_server import WaferAlignmentServer


class ZmqTab:
    """
    Owns all widgets in the ZMQ Communication tab.

    Parameters:
      notebook      - ttk.Notebook to add the tab into
      log_callback  - callable(msg: str) shared status logger
      ui_sync_callback - callable(payload: dict) routed to the main window
    """

    def __init__(self, notebook: ttk.Notebook, log_callback=None,
                 ui_sync_callback=None):
        self._log_shared = log_callback or print
        self._ui_sync_callback = ui_sync_callback

        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="ZMQ Communication")

        # ZMQ client state
        self.context       = zmq.Context()
        self.socket        = None
        self.is_connected  = False
        self.zmq_timeout_ms = 10000
        self.server_instance = None
        self.server_thread   = None
        self._ping_active    = False   # guard against duplicate PING threads

        self._build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self):
        # Top controls
        top = ttk.Frame(self.tab)
        top.pack(fill='x', padx=10, pady=10)

        ttk.Label(top, text="IP:").grid(row=0, column=0, padx=5)
        self.ip_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(top, textvariable=self.ip_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(top, text="Port:").grid(row=0, column=2, padx=5)
        self.port_var = tk.StringVar(value="5555")
        ttk.Entry(top, textvariable=self.port_var, width=8).grid(row=0, column=3, padx=5)

        self.btn_start = ttk.Button(top, text="Start Server & Connect",
                                    width=25, command=self.on_start_server)
        self.btn_start.grid(row=0, column=4, padx=10)

        self.btn_stop = ttk.Button(top, text="Stop Server", state='disabled',
                                   width=25, command=self.on_stop_server)
        self.btn_stop.grid(row=0, column=5, padx=10)

        # Log areas
        log_frame = ttk.Frame(self.tab)
        log_frame.pack(expand=True, fill='both', padx=10, pady=5)

        ttk.Label(log_frame, text="Message Received").pack(anchor='w')
        self.log_rx = tk.Text(log_frame, height=10, bg="#f8f9fa")
        self.log_rx.pack(fill='x', pady=2)

        ttk.Label(log_frame, text="Message Sent").pack(anchor='w', pady=(10, 0))
        self.log_tx = tk.Text(log_frame, height=10, bg="#e9ecef")
        self.log_tx.pack(fill='x', pady=2)

        ttk.Label(log_frame, text="Status").pack(anchor='w', pady=(10, 0))
        self.log_status = tk.Text(log_frame, height=5, bg="#d4edda")
        self.log_status.pack(fill='x', pady=2)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log(self, area: tk.Text, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        area.insert(tk.END, f"[{timestamp}] {msg}\n")
        area.see(tk.END)

    def log_status(self, msg: str):
        self.log(self.log_status, msg)

    # ------------------------------------------------------------------
    # Server start / stop
    # ------------------------------------------------------------------

    def on_start_server(self):
        # Prevent starting a second server while one is already running
        if self.server_thread and self.server_thread.is_alive():
            self.log(self.log_status, "Server is already running.")
            return

        self.btn_start.config(state='disabled')

        try:
            ui_port = int(self.port_var.get())
            ui_ip   = self.ip_var.get()
            root    = self.tab.winfo_toplevel()

            def safe_log(msg):
                if root.winfo_exists():
                    root.after(0, lambda: self.log(self.log_status, msg))

            def safe_rx_log(msg):
                if root.winfo_exists():
                    root.after(0, lambda: self.log(self.log_rx, msg))

            def safe_tx_log(msg):
                if root.winfo_exists():
                    root.after(0, lambda: self.log(self.log_tx, msg))

            def safe_ui_sync(payload: dict):
                if root.winfo_exists() and self._ui_sync_callback:
                    root.after(0, lambda: self._ui_sync_callback(payload))

            # Fired from inside server.run() once the poll loop is live
            loop_ready = threading.Event()

            def run_server_bg():
                self.server_instance = WaferAlignmentServer(
                    port=ui_port,
                    num_features=200,
                    threshold=50.0,
                    angle_step=5.0,
                    ip=ui_ip,
                    log_callback=safe_log,
                    msg_rx_callback=safe_rx_log,
                    msg_tx_callback=safe_tx_log,
                    ui_sync_callback=safe_ui_sync,
                    ready_event=loop_ready,
                )
                self.server_instance.run()

            self.server_thread = threading.Thread(target=run_server_bg, daemon=True)
            self.server_thread.start()

            addr = f"tcp://{ui_ip}:{ui_port}"

            # --- PING in background thread (never blocks the main thread) ---
            if self._ping_active:
                return   # already waiting for a pong from a previous attempt
            self._ping_active = True

            def _ping_thread():
                try:
                    # Wait until server.run() has started its poll loop
                    loop_ready.wait(timeout=15)
                    if not loop_ready.is_set():
                        root.after(0, lambda: self.log(
                            self.log_status, "Server loop did not start in time."))
                        root.after(0, lambda: self.btn_start.config(state='normal'))
                        return

                    import json as _json
                    sock = self.context.socket(zmq.REQ)
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.connect(addr)
                    sock.send_string("PING")

                    poller = zmq.Poller()
                    poller.register(sock, zmq.POLLIN)
                    socks = dict(poller.poll(8000))   # 8 s — plenty for localhost

                    if sock in socks and socks[sock] == zmq.POLLIN:
                        data = _json.loads(sock.recv_string())
                        if data.get("status") == "pong":
                            self.socket = sock
                            self.is_connected = True
                            
                            def on_ping_success():
                                self.log(self.log_status, f"Connected to server at {addr} (PING ok)")
                                self.btn_stop.config(state='normal')
                                
                            root.after(0, on_ping_success)
                            return

                    sock.close(linger=0)
                    root.after(0, lambda: self.log(
                        self.log_status, "Server started but PING timed out."))
                    root.after(0, lambda: self.btn_start.config(state='normal'))
                except Exception as ex:
                    root.after(0, lambda: self.log(
                        self.log_status, f"PING error: {ex}"))
                    root.after(0, lambda: self.btn_start.config(state='normal'))
                finally:
                    self._ping_active = False   # allow future attempts

            threading.Thread(target=_ping_thread, daemon=True).start()

        except Exception as e:
            self._ping_active = False
            self.log(self.log_status, f"Error starting server: {e}")
            self.btn_start.config(state='normal')


    def on_stop_server(self):
        """Stop the server immediately without blocking the main thread."""
        self.log(self.log_status, "Stopping server...")

        self.btn_stop.config(state='disabled')
        self.btn_start.config(state='normal')

        # Snapshot state so the background thread can clean up safely
        socket_to_close = self.socket
        thread_to_join  = self.server_thread
        was_connected   = self.is_connected

        # Clear state immediately so the UI reflects 'stopped' right away
        self.is_connected  = False
        self.socket        = None
        self.server_thread = None

        def _stop_bg():
            root = self.tab.winfo_toplevel()
            # Send SHUTDOWN via a temporary socket (short timeout)
            if was_connected and socket_to_close is not None:
                try:
                    socket_to_close.send_string("SHUTDOWN")
                    poller = zmq.Poller()
                    poller.register(socket_to_close, zmq.POLLIN)
                    poller.poll(2000)   # 2 s max — don't hang if server already dead
                except Exception:
                    pass
                try:
                    socket_to_close.close(linger=0)
                except Exception:
                    pass

            # Wait for server thread to finish (it will after processing SHUTDOWN)
            if thread_to_join and thread_to_join.is_alive():
                thread_to_join.join(timeout=3.0)

            if root.winfo_exists():
                root.after(0, lambda: self.log(
                    self.log_status, "Server stopped and disconnected."))

        threading.Thread(target=_stop_bg, daemon=True).start()

    # ------------------------------------------------------------------
    # ZMQ client send helper
    # ------------------------------------------------------------------

    def send_zmq(self, payload: str) -> dict | None:
        if not self.is_connected:
            messagebox.showerror("Error", "Server is not connected!")
            return None
        try:
            msg_str = str(payload)
            self.log(self.log_tx, msg_str)
            self.socket.send_string(msg_str)

            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            socks = dict(poller.poll(self.zmq_timeout_ms))

            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                reply_str = self.socket.recv_string()
                self.log(self.log_rx, reply_str)
                return json.loads(reply_str)
            else:
                self.log(self.log_status, "TIMEOUT receiving from server")
                self.socket.close(linger=0)
                addr = f"tcp://{self.ip_var.get()}:{self.port_var.get()}"
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(addr)
                return None
        except Exception as e:
            self.log(self.log_status, f"ZMQ Error: {e}")
            return None
