"""
================================================================================
ZeroMQ Wafer Alignment Server
================================================================================
Wraps LinemodMatcher in a ZeroMQ REP socket so that C#, C++ or any other
language can request wafer alignment results over TCP sockets.

Protocol (all messages are UTF-8 JSON strings):
-------------------------------------------------
  PING
    Request:  { "cmd": "PING" }
    Response: { "status": "pong" }

  SET_CONFIG  (optional reconfigure the matcher at runtime)
    Request:  { "cmd": "SET_CONFIG",
                "num_features": 200,
                "threshold":    50.0,
                "angle_step":   5.0  }
    Response: { "status": "ok" }

  LOAD_TEMPLATE
    Request:  { "cmd": "LOAD_TEMPLATE", "template_path": "C:\\img\\tmpl.png" }
    Response: { "status": "ok" }
           or { "status": "error", "message": "..." }

  MATCH
    Request:  { "cmd": "MATCH", "search_path": "C:\\img\\search.png" }
    Response: { "status": "ok",
                "x": 512, "y": 310,
                "angle": -2.5, "score": 87.3 }
           or { "status": "no_match" }
           or { "status": "error", "message": "..." }

  TEACH_REQ
    Trigger an interactive teach session. Python opens the image in an
    OpenCV window, lets the operator crop a template ROI and optionally
    draw a detection mask, then saves the results and updates the recipe.

    Request:  TEACH_REQ "<image_path>" "<recipe_path>"
              Parameters (space-separated, paths may be quoted):
                [0] image_path   – source image the operator crops from
                [1] recipe_path  – recipe XML to update (name or full path)
    Response: { "status": "ok",
                "template_path": "C:\\...\\template.png",
                "mask_path":     "C:\\...\\template_mask.png",  // or ""
                "crop_cx":       512.0,
                "crop_cy":       310.0 }
           or { "status": "cancelled" }          // operator pressed ESC / Close
           or { "status": "error", "message": "..." }

  WAFER_EDGE_REQ
    Find the wafer edge and return its offset from the image centre.

    Request:  WAFER_EDGE_REQ "<image_path>" "<recipe_path>" <direction> [<polarity>] [FORCE_RUN]
              Parameters (space-separated, paths must be quoted):
                [0] image_path   – full path to the image to analyse
                [1] recipe_path  – recipe XML to use (name or full path)
                [2] direction    – LEFT | RIGHT | TOP | BOTTOM
                [3] polarity     – ANY | LIGHT_TO_DARK | DARK_TO_LIGHT  (optional)
                                   If omitted, the polarity saved in the recipe is used.
                [4] FORCE_RUN    – literal string "FORCE_RUN"  (optional)
                                   Skips FOV classification and runs edge detection
                                   regardless of image type. Use after the operator
                                   acknowledges a die_fov_warning and chooses to proceed.
    Response: { "status": "ok",
                "delta_x": -123.4, "delta_y": 5.6,
                "fov_type": "EDGE_FOV", "fov_confidence": 0.95,
                "a": .., "b": .., "c": ..,
                "x_top": .., "x_bot": ..   }   // vertical edge (LEFT/RIGHT)
         or   { ..., "y_left": .., "y_right": .. }  // horizontal edge (TOP/BOTTOM)
           or { "status": "die_fov_warning",
                "fov_type": "DIE_FOV", "fov_confidence": 0.91,
                "message": "Image classified as DIE_FOV. Resend with FORCE_RUN to override." }
           or { "status": "no_edge",  "reason": "...", "fov_type": "EDGE_FOV" }
           or { "status": "error",    "message": "..." }

  SHUTDOWN
    Request:  { "cmd": "SHUTDOWN" }
    Response: { "status": "ok" }  (then server exits)

Usage:
    python zmq_server.py [--port 5555] [--num-features 200] [--threshold 50]
================================================================================
"""

import argparse
import json
import re
import os
import sys
import traceback

import cv2
import zmq

# local import
from app.services.linemod_matcher import LinemodMatcher, LinemodConfig
from app.services.edge_finder import EdgeLineFinder, EdgeFinderConfig
from app.models.recipe_model import RecipeManager


# Server

class WaferAlignmentServer:
    """
    ZeroMQ REP server.  One instance of LinemodMatcher is reused across
    requests so that template data built by LOAD_TEMPLATE stays in memory.
    """

    def __init__(self, port: int = 5555,
                 num_features: int = 200,
                 threshold: float = 50.0,
                 angle_step: float = 5.0,
                 ip: str = "*",
                 log_callback=None,
                 msg_rx_callback=None,
                 msg_tx_callback=None,
                 ui_sync_callback=None,
                 ready_event=None):      # threading.Event fired from run() when polling
        self.port = port
        self.ip = ip
        self.log_callback = log_callback
        self.msg_rx_callback = msg_rx_callback
        self.msg_tx_callback = msg_tx_callback
        self.ui_sync_callback = ui_sync_callback
        self._ready_event = ready_event

        # Matcher and EdgeFinder are created lazily on first use so that
        # __init__ stays fast and the PING round-trip is not delayed.
        self._num_features = num_features
        self._threshold    = threshold
        self._angle_step   = angle_step
        self._matcher      = None   # created on first _match / _load_recipe call
        self._edge_finder  = None   # created on first _find_edge call
        self._template_loaded = False

        # Recipe Manager
        self.recipe_mgr = RecipeManager(os.path.join(os.getcwd(), "recipes"))
        self.current_recipe_name = None
        self.template_crop_cx = 0.0
        self.template_crop_cy = 0.0
        self.edge_configs = {}

        # ZMQ setup
        self.context = zmq.Context()
        self.socket  = self.context.socket(zmq.REP)
        bind_addr = f"tcp://{self.ip}:{port}"
        self.socket.bind(bind_addr)
        self._log(f"[SERVER] Wafer Alignment Server listening on {bind_addr}")

    # ------------------------------------------------------------------
    # Logging Helper
    # ------------------------------------------------------------------
    def _log(self, message: str):
        """Prints to stdout and optionally calls the UI log callback."""
        print(message)
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass  # Ignore callback failures (e.g. if UI is closed)

    # ------------------------------------------------------------------
    # Lazy algorithm accessors
    # ------------------------------------------------------------------
    @property
    def matcher(self):
        if self._matcher is None:
            cfg = LinemodConfig()
            cfg.NUM_FEATURES    = self._num_features
            cfg.MATCH_THRESHOLD = self._threshold
            cfg.ANGLE_STEP      = self._angle_step
            self._matcher = LinemodMatcher(cfg)
        return self._matcher

    @property
    def edge_finder(self):
        if self._edge_finder is None:
            self._edge_finder = EdgeLineFinder(EdgeFinderConfig())
        return self._edge_finder

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        """Blocking event loop."""
        # Signal that the poll loop is about to start.  This is the correct
        # moment for the client to attempt a PING — the socket is bound AND
        # we are about to call poll() for the first time.
        self._log("[SERVER] Ready. Waiting for commands")
        if self._ready_event:
            self._ready_event.set()
        try:
            while True:
                # Poll with a 200ms timeout so Python can process KeyboardInterrupt (Ctrl+C) on Windows
                if self.socket.poll(timeout=200):
                    raw = self.socket.recv_string()
                    
                    if self.msg_rx_callback:
                        try: self.msg_rx_callback(raw)
                        except: pass
                        
                    response = self._handle(raw)
                    resp_str = json.dumps(response)
                    
                    if self.msg_tx_callback:
                        try: self.msg_tx_callback(resp_str)
                        except: pass
                        
                    self.socket.send_string(resp_str)
                    
                    if response.get("status") == "ok" and \
                            raw.strip().upper() == "SHUTDOWN":
                        self._log("[SERVER] Shutdown requested. Exiting.")
                        break
        except KeyboardInterrupt:
            self._log("\n[SERVER] Interrupted by user.")
        finally:
            self.socket.close()
            self.context.term()

    # ------------------------------------------------------------------
    # Internal dispatcher
    # ------------------------------------------------------------------
    def _handle(self, raw: str) -> dict:
        # 1. Split into Command and Arguments
        match = re.match(r'^(\S+)\s*(.*)$', raw.strip())
        if not match:
            self._log("[SERVER] Invalid input format")
            return {"status": "error", "message": "Invalid input format"}

        cmd = match.group(1).upper()
        args_part = match.group(2)

        # 2. Extract parameters (Handles both quoted paths and unquoted words)
        # This precisely mimics the C# Regex: [^\s"']+|"([^"]*)"
        raw_matches = re.findall(r'"([^"]*)"|([^\s"\']+)', args_part)
        parameters = [m[0] if m[0] else m[1] for m in raw_matches]

        self._log(f"[SERVER] CMD={cmd} | PARAMS={parameters}")

        # 3. Switch statement routing
        if cmd == "PM_REQ":
            if len(parameters) >= 2:
                # First load the recipe, then run the match
                self._load_recipe({"recipe_path": parameters[1]})
                # Match and get result
                result = self._match({"search_path": parameters[0]})
                
                if self.ui_sync_callback:
                    try:
                        self.ui_sync_callback({
                            "event": "PM_REQ",
                            "image_path": parameters[0],
                            "recipe_path": parameters[1],
                            "result": result
                        })
                    except: pass
                    
                return result
            return {"status": "error", "message": "PM_REQ requires 2 parameters"}

        elif cmd == "TRAIN_REQ":
            if len(parameters) >= 2:
                img_path = parameters[0]
                recipe_path = parameters[1]
                
                if not os.path.isfile(img_path):
                    return {"status": "error", "message": f"Image not found: {img_path}"}
                
                if self.ui_sync_callback:
                    try:
                        self.ui_sync_callback({
                            "event": "TRAIN_REQ",
                            "image_path": img_path,
                            "recipe_path": recipe_path
                        })
                    except: pass
                
                return {"status": "ok", "message": "train_reply OK"}
            return {"status": "error", "message": "TRAIN_REQ requires 2 parameters"}

        elif cmd == "TEACH_REQ":
            if len(parameters) >= 2:
                return self._teach_template({
                    "image_path":  parameters[0],
                    "recipe_path": parameters[1]
                })
            return {"status": "error", "message": "TEACH_REQ requires 2 parameters: <image_path> <recipe_path>"}

        elif cmd == "WAFER_EDGE_REQ":
            if len(parameters) >= 3:
                # parameters[0] = Image Path
                # parameters[1] = Recipe Name
                # parameters[2] = Scan Direction ("LEFT", "RIGHT", "TOP", "BOTTOM")
                # parameters[3] = Edge Polarity  ("ANY", "LIGHT_TO_DARK", "DARK_TO_LIGHT") — optional
                # parameters[4] = "FORCE_RUN" — skip FOV check (optional)
                # Note: FORCE_RUN is detected anywhere in parameters[3:] so polarity can be omitted.

                self._load_recipe({"recipe_path": parameters[1]})

                # Detect FORCE_RUN anywhere in the optional parameters
                force_run = any(p.upper() == "FORCE_RUN" for p in parameters[3:])

                find_edge_args = {
                    "search_path":    parameters[0],
                    "scan_direction": parameters[2],
                    "force_run":      force_run,
                }
                # Optional polarity override — only if parameters[3] is a polarity value
                if len(parameters) >= 4 and parameters[3].upper() != "FORCE_RUN":
                    find_edge_args["polarity_override"] = parameters[3].upper()

                result = self._find_edge(find_edge_args)

                # Only push UI sync for a real edge result (not a die_fov_warning)
                if result.get("status") == "ok" and self.ui_sync_callback:
                    try:
                        self.ui_sync_callback({
                            "event":       "WAFER_EDGE_REQ",
                            "image_path":  parameters[0],
                            "recipe_path": parameters[1],
                            "direction":   parameters[2].upper(),
                            "polarity":    parameters[3].upper() if (len(parameters) >= 4 and parameters[3].upper() != "FORCE_RUN") else None,
                            "edge_config": {
                                "KernelSize":      str(self.edge_finder.config.KERNEL_SIZE),
                                "EdgeThreshold":   str(self.edge_finder.config.EDGE_THRESHOLD),
                                "NumRegions":      str(self.edge_finder.config.NUM_REGIONS),
                                "BorderIgnorePct": str(self.edge_finder.config.BORDER_IGNORE_PCT),
                                "RansacThreshold": str(self.edge_finder.config.RANSAC_THRESHOLD),
                                "EdgePolarity":    self.edge_finder.config.EDGE_POLARITY,
                            },
                            "result":      result
                        })
                    except: pass

                return result
            return {"status": "error", "message": "WAFER_EDGE_REQ requires 3 parameters: <image_path> <recipe_path> <direction> [polarity] [FORCE_RUN]"}

        elif cmd == "LOADR_REQ":
            if len(parameters) >= 1:
                return self._load_recipe({"recipe_path": parameters[0]})
            return {"status": "error", "message": "LOADR_REQ requires 1 parameter"}

        elif cmd == "PING":
            return {"status": "pong"}

        elif cmd == "SHUTDOWN":
            return {"status": "ok"}

        else:
            self._log(f"[SERVER] Unknown command: {cmd}")
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _set_config(self, msg: dict) -> dict:
        """Reconfigure the matcher without restarting the server."""
        try:
            cfg = self.matcher.config
            if "num_features" in msg:
                cfg.NUM_FEATURES = int(msg["num_features"])
            if "threshold" in msg:
                cfg.MATCH_THRESHOLD = float(msg["threshold"])
            if "angle_step" in msg:
                cfg.ANGLE_STEP = float(msg["angle_step"])
            # Reset template so the new params are used on next LOAD_TEMPLATE
            self._template_loaded = False
            self._log(f"[SERVER] Config updated: num_features={cfg.NUM_FEATURES} "
                  f"threshold={cfg.MATCH_THRESHOLD} angle_step={cfg.ANGLE_STEP}")
            return {"status": "ok"}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

    def _load_template(self, msg: dict) -> dict:
        path = msg.get("template_path", "")
        if not path:
            return {"status": "error", "message": "template_path is required"}
        if not os.path.isfile(path):
            return {"status": "error", "message": f"File not found: {path}"}

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"status": "error",
                        "message": f"cv2.imread failed for: {path}"}

            # Optional ROI from message
            roi = msg.get("roi")  # [x, y, w, h] or null
            if roi:
                x, y, w, h = roi
                img = img[y:y+h, x:x+w]

            self.matcher.load_template(img)
            self.matcher.generate_templates()
            self._template_loaded = True
            self._log(f"[SERVER] Template loaded: {path}  shape={img.shape}")
            return {"status": "ok"}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

    def _match(self, msg: dict) -> dict:
        if not self._template_loaded:
            return {"status": "error",
                    "message": "No template loaded. Send LOAD_TEMPLATE first."}

        path = msg.get("search_path", "")
        if not path:
            return {"status": "error", "message": "search_path is required"}
        if not os.path.isfile(path):
            return {"status": "error", "message": f"File not found: {path}"}

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"status": "error",
                        "message": f"cv2.imread failed for: {path}"}

            result = self.matcher.match(img)

            if result is None:
                self._log("[SERVER] No match found.")
                return {"status": "no_match"}

            resp = {
                "status": "ok",
                "x":      float(result['x']),
                "y":      float(result['y']),
                "angle":  float(result['angle']),
                "score":  float(result['score']),
                "delta_x": float(result['x']) - self.template_crop_cx,
                "delta_y": float(result['y']) - self.template_crop_cy,
                # Additional fields for UI visualization:
                "bbox":   [int(v) for v in result['bbox']],
                "template_id": int(result['template_id']),
                "scale":  float(result.get('scale', 1.0))
            }
            if self.current_recipe_name:
                resp["recipe_name"] = self.current_recipe_name
            self._log(f"[SERVER] Match -> x={resp['x']:.1f}  y={resp['y']:.1f}"
                  f"  angle={resp['angle']:.2f} deg  score={resp['score']:.1f}")
            return resp
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

    def _find_edge(self, msg: dict) -> dict:
        path = msg.get("search_path", "")
        if not path:
            return {"status": "error", "message": "search_path is required"}
        if not os.path.isfile(path):
            return {"status": "error", "message": f"File not found: {path}"}

        try:
            direction = msg.get("scan_direction", "LEFT").upper()
            force_run = msg.get("force_run", False)
            self.edge_finder.config.SCAN_DIRECTION = direction

            # Apply specific direction parameters if loaded from recipe
            if hasattr(self, 'edge_configs') and direction in self.edge_configs:
                d_cfg = self.edge_configs[direction]
                if "KernelSize" in d_cfg:      self.edge_finder.config.KERNEL_SIZE = int(d_cfg["KernelSize"])
                if "EdgeThreshold" in d_cfg:   self.edge_finder.config.EDGE_THRESHOLD = int(d_cfg["EdgeThreshold"])
                if "NumRegions" in d_cfg:      self.edge_finder.config.NUM_REGIONS = int(d_cfg["NumRegions"])
                if "BorderIgnorePct" in d_cfg: self.edge_finder.config.BORDER_IGNORE_PCT = float(d_cfg["BorderIgnorePct"])
                if "RansacThreshold" in d_cfg: self.edge_finder.config.RANSAC_THRESHOLD = float(d_cfg["RansacThreshold"])
                if "EdgePolarity" in d_cfg:    self.edge_finder.config.EDGE_POLARITY = d_cfg["EdgePolarity"]

            # Optional per-call polarity override (from C# WAFER_EDGE_REQ parameter[3])
            # This takes priority over the recipe value set above.
            polarity_override = msg.get("polarity_override", "").upper()
            if polarity_override in ("ANY", "LIGHT_TO_DARK", "DARK_TO_LIGHT"):
                self.edge_finder.config.EDGE_POLARITY = polarity_override
                self._log(f"[SERVER] Polarity overridden by command: {polarity_override}")

            # Update kernel in case KernelSize was changed
            if hasattr(self.edge_finder.config, 'KERNEL_SIZE'):
                k_size = self.edge_finder.config.KERNEL_SIZE
                if k_size % 2 == 0: k_size += 1
                from app.services.fov_classifier import create_gradient_kernel
                self.edge_finder.kernel = create_gradient_kernel(k_size)

            # ── FOV Classification (skipped when FORCE_RUN is set) ────────
            fov_type       = "EDGE_FOV"
            fov_confidence = 1.0
            if not force_run:
                from app.services.fov_classifier import FOVClassifier, preprocess_image
                fov_img, _, _ = preprocess_image(
                    path, self.edge_finder.config.TARGET_PROCESS_DIM)
                fov_result    = FOVClassifier(self.edge_finder.config).classify(fov_img)
                fov_type      = fov_result.get('fov_type', 'UNCERTAIN')
                fov_confidence = fov_result.get('confidence', 0.0)

                if fov_type == "DIE_FOV":
                    self._log(
                        f"[SERVER] DIE_FOV detected (confidence={fov_confidence:.2f}) "
                        f"— returning warning. C# should resend with FORCE_RUN to override."
                    )
                    return {
                        "status":         "die_fov_warning",
                        "fov_type":       fov_type,
                        "fov_confidence": round(fov_confidence, 3),
                        "message":        "Image classified as DIE_FOV. "
                                          "Resend with FORCE_RUN to override."
                    }
                self._log(f"[SERVER] FOV check passed: {fov_type} (conf={fov_confidence:.2f})")
            else:
                self._log("[SERVER] FORCE_RUN set — skipping FOV classification.")

            # ── Run edge detection ────────────────────────────────────────
            result = self.edge_finder.find_edge(path, skip_classification=True)
            self.edge_finder.last_result = result

            if not result['success']:
                self._log(f"[SERVER] Edge not found: {result.get('reason')}, "
                          f"delta_x={result.get('delta_x', 0)}, delta_y={result.get('delta_y', 0)}")
                return {
                    "status":         "no_edge",
                    "reason":         result.get('reason'),
                    "delta_x":        0,
                    "delta_y":        0,
                    "fov_type":       fov_type,
                    "fov_confidence": round(fov_confidence, 3),
                }

            p = result['line_params']
            e = result['line_endpoints']

            resp = {
                "status":         "ok",
                "a": float(p['a']), "b": float(p['b']), "c": float(p['c']),
                "vx": float(p['vx']), "vy": float(p['vy']),
                "x0": float(p['x0']), "y0": float(p['y0']),
                "delta_x":        float(result.get('delta_x', 0)),
                "delta_y":        float(result.get('delta_y', 0)),
                "fov_type":       fov_type,
                "fov_confidence": round(fov_confidence, 3),
            }
            if self.current_recipe_name:
                resp["recipe_name"] = self.current_recipe_name

            if result['is_vertical_edge']:
                resp["x_top"] = float(e['x_top'])
                resp["x_bot"] = float(e['x_bot'])
            else:
                resp["y_left"] = float(e['y_left'])
                resp["y_right"] = float(e['y_right'])

            self._log(f"[SERVER] Edge found -> Points: {result['num_points']}, Inliers: {result['num_inliers']}")
            return resp
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

    def _load_recipe(self, msg: dict) -> dict:
        """Loads all parameters and the template image from an XML recipe."""
        path = msg.get("recipe_path", "")
        if not path:
            return {"status": "error", "message": "recipe_path is required", "delta_x": 0, "delta_y": 0}
        
        # If just the filename was passed, try to find it in the recipes directory
        if not os.path.isfile(path):
            probe_path = os.path.join(self.recipe_mgr.recipes_root, path)
            if not probe_path.endswith(".xml"): probe_path += ".xml"
            if os.path.isfile(probe_path):
                path = probe_path
            else:
                return {"status": "error", "message": f"Recipe file not found: {path}"}

        try:
            r = self.recipe_mgr.load(path)
            self.current_recipe_name = r.get("name", "Unknown")

            # 1. Apply Pattern params
            fp = r.get("find_pattern", {})
            m_cfg = self.matcher.config
            if "NumFeatures" in fp:    m_cfg.NUM_FEATURES = int(fp["NumFeatures"])
            if "MatchThreshold" in fp: m_cfg.MATCH_THRESHOLD = float(fp["MatchThreshold"])
            if "GradThrPct" in fp:     m_cfg.WEAK_THRESHOLD = -float(fp["GradThrPct"])
            if "TSpread" in fp:
                t = int(fp["TSpread"])
                m_cfg.T_PYRAMID = [t, t*2, t*4]
                m_cfg.PYRAMID_LEVELS = 3
            if "HystKernel" in fp:     m_cfg.HYSTERESIS_KERNEL = int(fp["HystKernel"])
            
            mode = fp.get("SearchMode", "Simple (Fast)")
            if mode == 'Simple (Fast)':
                m_cfg.ANGLE_STEP = 360; m_cfg.SCALE_MIN = m_cfg.SCALE_MAX = 1.0
            elif mode == 'With Rotation':
                m_cfg.ANGLE_STEP = 5; m_cfg.SCALE_MIN = m_cfg.SCALE_MAX = 1.0
            else:
                m_cfg.ANGLE_STEP = 5; m_cfg.SCALE_MIN = 0.8; m_cfg.SCALE_MAX = 1.2

            self.template_crop_cx = float(fp.get("TemplateCropCX", 0.0))
            self.template_crop_cy = float(fp.get("TemplateCropCY", 0.0))

            # 2. Apply Edge params
            self.edge_configs = r.get("find_edge", {})

            # 3. Load Template Image
            tpath = fp.get("TemplatePath", "")
            if tpath:
                if not os.path.isfile(tpath):
                    tpath_abs = os.path.join(os.getcwd(), tpath)
                    if os.path.isfile(tpath_abs):
                        tpath = tpath_abs

                if os.path.isfile(tpath):
                    img = cv2.imread(tpath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.matcher.load_template(img)
                        self.matcher.generate_templates()
                        self._template_loaded = True
                    else:
                        self._log(f"[SERVER] Warning: Failed cv2.imread on template {tpath}")
                else:
                    self._log(f"[SERVER] Warning: Template file not found: {tpath}")

            self._log(f"[SERVER] Loaded recipe: {self.current_recipe_name}")
            return {"status": "ok", "recipe": self.current_recipe_name}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": f"Failed to load recipe: {exc}"}

    def _teach_template(self, msg: dict) -> dict:
        """
        Interactive template-teaching workflow triggered by TEACH_REQ.

        Steps:
          1. Open the source image in an OpenCV window so the operator can
             drag-select the template crop region.
          2. After confirming the crop, open the mask-drawing window so the
             operator can paint the detection ROI (polygon) on the template.
          3. Save the cropped template PNG (and mask PNG if drawn) into the
             same folder as the recipe.
          4. Update the recipe XML with the new TemplatePath, MaskPath and
             crop-centre values.
          5. Reload the matcher with the new template so MATCH commands work
             immediately without a server restart.

        Returns a JSON-serialisable dict:
          { "status": "ok",
            "template_path": "...",
            "mask_path": "...",   # empty string if no mask drawn
            "crop_cx": float,
            "crop_cy": float }
          or { "status": "cancelled" }
          or { "status": "error", "message": "..." }
        """
        import time
        img_path    = msg.get("image_path",  "")
        recipe_path = msg.get("recipe_path", "")

        # ── Validate inputs ──────────────────────────────────────────────
        if not img_path:
            return {"status": "error", "message": "image_path is required"}
        if not os.path.isfile(img_path):
            return {"status": "error", "message": f"Image not found: {img_path}"}

        # Resolve recipe path (bare name → recipes directory)
        if recipe_path and not os.path.isfile(recipe_path):
            probe = os.path.join(self.recipe_mgr.recipes_root, recipe_path)
            if not probe.lower().endswith(".xml"):
                probe += ".xml"
            if os.path.isfile(probe):
                recipe_path = probe
            else:
                return {"status": "error", "message": f"Recipe not found: {recipe_path}"}

        # ── Step 1: Load source image ────────────────────────────────────
        try:
            src_bgr = cv2.imread(img_path)
            if src_bgr is None:
                return {"status": "error", "message": f"cv2.imread failed for: {img_path}"}
        except Exception as exc:
            return {"status": "error", "message": f"Failed to read image: {exc}"}

        sh, sw = src_bgr.shape[:2]

        # Scale the display to fit 85 % of the screen
        try:
            import ctypes
            user32   = ctypes.windll.user32
            scr_w    = int(user32.GetSystemMetrics(0) * 0.85)
            scr_h    = int(user32.GetSystemMetrics(1) * 0.85)
        except Exception:
            scr_w, scr_h = 1280, 900

        roi_scale = min(scr_w / sw, scr_h / sh, 1.0)
        display_img = (cv2.resize(src_bgr,
                                  (int(sw * roi_scale), int(sh * roi_scale)),
                                  interpolation=cv2.INTER_AREA)
                       if roi_scale < 1.0 else src_bgr.copy())

        # ── Step 2: Let operator drag-select template crop region ────────
        self._log("[TEACH] Opening crop window — drag a rectangle and click Save.")
        from app.viewmodels.pattern_viewmodel import PatternViewModel
        roi = PatternViewModel._select_roi_safe(
            display_img,
            "TEACH: Select template region (drag rect, Save=confirm, Close=cancel)"
        )

        if roi is None:
            self._log("[TEACH] Cancelled by operator (crop step).")
            return {"status": "cancelled"}

        rx, ry, rw, rh = roi
        if rw <= 0 or rh <= 0:
            return {"status": "error", "message": "Invalid crop region (zero size)."}

        # Map display-space ROI back to original image pixels
        orig_x = int(rx / roi_scale)
        orig_y = int(ry / roi_scale)
        orig_w = int(rw / roi_scale)
        orig_h = int(rh / roi_scale)

        crop_cx = orig_x + orig_w / 2.0
        crop_cy = orig_y + orig_h / 2.0
        cropped_bgr = src_bgr[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]

        # ── Step 3: Save cropped template PNG ────────────────────────────
        # Name the template after the recipe so each recipe has its own file
        # and cleanup is scoped to that specific recipe only.
        recipe_dir  = os.path.dirname(recipe_path) if recipe_path else os.getcwd()
        recipe_name = os.path.splitext(os.path.basename(recipe_path))[0] if recipe_path else "teach"
        tmpl_filename = f"{recipe_name}_template.png"
        tmpl_path    = os.path.join(recipe_dir, tmpl_filename)
        mask_filename = f"{recipe_name}_template_mask.png"

        # Clean up only the previous template/mask for THIS recipe
        try:
            for f in [tmpl_filename, mask_filename]:
                fp = os.path.join(recipe_dir, f)
                if os.path.exists(fp):
                    os.remove(fp)
        except Exception:
            pass

        try:
            cv2.imwrite(tmpl_path, cropped_bgr)
            self._log(f"[TEACH] Template saved → {tmpl_path}")
        except Exception as exc:
            return {"status": "error", "message": f"Failed to save template: {exc}"}

        # ── Step 4: Let operator draw detection mask (optional) ──────────
        self._log("[TEACH] Opening mask window — draw polygon then Save, or Close to skip.")
        mask_path = ""
        mask_img  = None
        try:
            from app.views.mask_editor import draw_detection_mask
            template_bgr = cv2.imread(tmpl_path)
            if template_bgr is not None:
                mask_img = draw_detection_mask(
                    template_bgr,
                    window_title="TEACH: Draw Detection Region (L-click=add, R-click=close poly, Save=confirm)"
                )
                if mask_img is not None:
                    mask_path = os.path.join(recipe_dir, mask_filename)
                    cv2.imwrite(mask_path, mask_img)
                    self._log(f"[TEACH] Mask saved → {mask_path}")
                else:
                    self._log("[TEACH] No mask drawn — using full template for detection.")
        except Exception as exc:
            self._log(f"[TEACH] Mask step skipped due to error: {exc}")

        # ── Step 5: Update recipe XML ────────────────────────────────────
        if recipe_path and os.path.isfile(recipe_path):
            try:
                recipe = self.recipe_mgr.load(recipe_path)
                fp = recipe.setdefault("find_pattern", {})
                fp["TemplatePath"]      = tmpl_path
                fp["TemplateCropCX"]    = str(crop_cx)
                fp["TemplateCropCY"]    = str(crop_cy)
                fp["DetectionMaskPath"] = mask_path
                self.recipe_mgr.save(recipe)
                self._log(f"[TEACH] Recipe updated: {recipe_path}")
            except Exception as exc:
                self._log(f"[TEACH] Warning: could not update recipe — {exc}")

        # ── Step 6: Reload matcher in server so MATCH works immediately ──
        try:
            img_gray = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is not None:
                if mask_img is not None:
                    self.matcher.load_template(img_gray, detection_mask=mask_img
                                               if len(mask_img.shape) == 2
                                               else cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY))
                else:
                    self.matcher.load_template(img_gray)
                self.matcher.generate_templates()
                self._template_loaded = True
                self.template_crop_cx  = crop_cx
                self.template_crop_cy  = crop_cy
                self._log("[TEACH] Matcher reloaded with new template.")
        except Exception as exc:
            self._log(f"[TEACH] Warning: matcher reload failed — {exc}")

        # ── Notify UI (if embedded) ──────────────────────────────────────
        if self.ui_sync_callback:
            try:
                self.ui_sync_callback({
                    "event":         "TEACH_REQ",
                    "image_path":    img_path,
                    "recipe_path":   recipe_path,
                    "template_path": tmpl_path,
                    "mask_path":     mask_path,
                    "crop_cx":       crop_cx,
                    "crop_cy":       crop_cy,
                })
            except Exception:
                pass

        return {
            "status":        "ok",
            "template_path": tmpl_path,
            "mask_path":     mask_path,
            "crop_cx":       crop_cx,
            "crop_cy":       crop_cy,
        }

# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="ZeroMQ wafer-alignment server (LINE-2D backend)")
    parser.add_argument("--port",         type=int,   default=5555)
    parser.add_argument("--num-features", type=int,   default=200,
                        help="Number of template features (default 200)")
    parser.add_argument("--threshold",    type=float, default=50.0,
                        help="Match score threshold 0-100 (default 50)")
    parser.add_argument("--angle-step",   type=float, default=5.0,
                        help="Degrees between template rotations (default 5)")
    args = parser.parse_args()

    server = WaferAlignmentServer(
        port=args.port,
        num_features=args.num_features,
        threshold=args.threshold,
        angle_step=args.angle_step,
    )
    server.run()


if __name__ == "__main__":
    main()

