"""
================================================================================
Autofocus Command Test Client
================================================================================
Tests FM_INDEX_REQ, START_AUTOFOCUS_REQ, and FOCUS_REQ without a machine.

How it works
------------
1. Spawns WaferAlignmentServer in a background thread (same process).
2. Generates synthetic grayscale images with a controllable sharpness level
   (Gaussian blur: less blur = sharper = higher FMI).
3. Sends all three command types over ZMQ and prints the server replies.

Run
---
    python test_autofocus.py

Optional flags
--------------
    --port  5556        # use a different port if 5555 is busy
    --steps 10          # number of FOCUS_REQ steps in the scan
    --peak-at 6         # which step index has the sharpest image (0-based)
================================================================================
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_image(sharpness: float = 1.0) -> np.ndarray:
    """
    Return a 256×256 grayscale image.
    sharpness = 1.0  → sharp (no blur, high FMI)
    sharpness = 0.0  → very blurry (heavy Gaussian, low FMI)
    """
    # Random noise pattern — gives the Laplacian something to measure
    img = np.random.randint(80, 180, (256, 256), dtype=np.uint8)
    # Draw a simple edge grid so the image has real structure
    img[::32, :] = 220
    img[:, ::32] = 220

    # sigma: 0.5 (sharp) → 15.0 (blurry), inverted from sharpness
    sigma = 0.5 + (1.0 - sharpness) * 14.5
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return blurred


def save_tmp_image(img: np.ndarray, prefix: str = "af_test") -> str:
    """Save ndarray to a temp BMP and return the path."""
    fd, path = tempfile.mkstemp(suffix=".bmp", prefix=prefix + "_")
    os.close(fd)
    cv2.imwrite(path, img)
    return path


def send_command(sock: zmq.Socket, command: str, timeout_ms: int = 5000) -> str:
    """Send a plain-text command and return the reply string."""
    sock.send_string(command)
    if sock.poll(timeout_ms):
        return sock.recv_string()
    return "TIMEOUT"


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def start_server(port: int) -> threading.Event:
    """Start WaferAlignmentServer in a daemon thread. Returns ready_event."""
    # Import here so the script can be run from any cwd
    sys.path.insert(0, os.path.dirname(__file__))
    from app.services.zmq_server import WaferAlignmentServer

    ready = threading.Event()
    srv = WaferAlignmentServer(port=port, ready_event=ready)

    t = threading.Thread(target=srv.run, daemon=True)
    t.start()

    if not ready.wait(timeout=5):
        print("[TEST] ERROR: server did not become ready in 5 s")
        sys.exit(1)
    print(f"[TEST] Server ready on port {port}\n")
    return ready


def show_image(image_path: str, label: str, fmi: float | None = None,
               wait_ms: int = 1500) -> None:
    """
    Display an image in a cv2 window with an overlay label.
    wait_ms=0 → wait for a keypress; any other value → auto-close after that many ms.
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    # Scale down if image is very large so it fits on screen
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Overlay text
    overlay_lines = [label]
    if fmi is not None:
        overlay_lines.append(f"FMI = {fmi:.3f}")
    for idx, line in enumerate(overlay_lines):
        y = 30 + idx * 30
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0),   3, cv2.LINE_AA)   # shadow
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 1, cv2.LINE_AA)  # text

    cv2.imshow("Autofocus Scan", img)
    cv2.waitKey(wait_ms)


def copy_to_tmp(src_path: str, prefix: str = "real") -> str:
    """
    Copy a real image to a temp file so the server can delete it safely
    without destroying the original.
    """
    ext = os.path.splitext(src_path)[1] or ".bmp"
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix=prefix + "_")
    os.close(fd)
    shutil.copy2(src_path, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Real-image tests
# ---------------------------------------------------------------------------

def test_real_fm_index(sock: zmq.Socket, image_path: str):
    """FM_INDEX_REQ on a single real image."""
    print("=" * 60)
    print("TEST: FM_INDEX_REQ (real image)")
    print("=" * 60)

    if not os.path.isfile(image_path):
        print(f"  ERROR: file not found: {image_path}")
        return

    tmp = copy_to_tmp(image_path, "fmi_real")
    cmd = f'FM_INDEX_REQ "{tmp}"'
    print(f"  Image : {image_path}")
    print(f"  Send  : FM_INDEX_REQ \"<tmp_copy>\"")
    reply = send_command(sock, cmd)
    print(f"  Reply : {reply}")

    if reply.startswith("FM_INDEX_OK"):
        score = float(reply.split()[1])
        print(f"  Score : {score:.3f}")
        print(f"  Note  : FMI is relative — compare across a Z-scan to find the peak.")
        show_image(image_path, os.path.basename(image_path), fmi=score, wait_ms=0)
    cv2.destroyAllWindows()
    print()


def test_real_autofocus_scan(sock: zmq.Socket, images_dir: str,
                             z_start: float = 100.0, z_step: float = 1.0):
    """
    Full autofocus scan using real images from a folder.

    Images are loaded in alphabetical order — name them so they sort
    in scan order, e.g.:
        step_001.bmp, step_002.bmp, ... step_010.bmp
    or  img_z100.bmp, img_z101.bmp, ...

    The server DELETES files after reading — this test copies each image
    to a temp file first so your originals are preserved.
    """
    print("=" * 60)
    print("TEST: Full autofocus scan (real images)")
    print(f"      Folder : {images_dir}")
    print("=" * 60)

    # Collect images in sorted order
    patterns = ["*.bmp", "*.png", "*.tif", "*.tiff", "*.jpg"]
    image_files = []
    for pat in patterns:
        image_files.extend(glob.glob(os.path.join(images_dir, pat)))
    image_files = sorted(image_files)

    if not image_files:
        print(f"  ERROR: no images found in {images_dir}")
        return
    if len(image_files) < 2:
        print(f"  ERROR: need at least 2 images for a scan, found {len(image_files)}")
        return

    print(f"  Found {len(image_files)} images\n")

    step_count = len(image_files)

    print("  (Press any key in the image window to advance to the next step)\n")

    # --- Step 0: START_AUTOFOCUS_REQ ---
    tmp0 = copy_to_tmp(image_files[0], "real_s0")
    cmd  = f'START_AUTOFOCUS_REQ "{tmp0}" {z_start} {step_count}'
    print(f"  [0] {os.path.basename(image_files[0])}  z={z_start:.1f}")
    print(f"      Send  : START_AUTOFOCUS_REQ ... {z_start} {step_count}")
    reply = send_command(sock, cmd)
    print(f"      Reply : {reply}")
    fmi0 = float(reply.split()[1]) if reply.startswith("START_AUTOFOCUS_OK") else None
    show_image(image_files[0],
               f"[0] {os.path.basename(image_files[0])}  z={z_start:.1f}",
               fmi=fmi0, wait_ms=0)
    print()

    # --- Steps 1 .. N-1: FOCUS_REQ ---
    final_reply = ""
    for i, img_path in enumerate(image_files[1:], start=1):
        z = z_start + i * z_step
        tmp = copy_to_tmp(img_path, f"real_step{i}")
        cmd = f'FOCUS_REQ "{tmp}" {z}'
        reply = send_command(sock, cmd)
        print(f"  [{i:2d}] {os.path.basename(img_path):<30s}  z={z:.1f}  Reply: {reply}")
        parts = reply.split()
        fmi_i = float(parts[1]) if reply.startswith("FOCUS_OK") and len(parts) >= 3 else None
        status = " ".join(parts[2:]) if fmi_i is not None else reply
        show_image(img_path,
                   f"[{i}] {os.path.basename(img_path)}  z={z:.1f}  {status}",
                   fmi=fmi_i, wait_ms=0)
        final_reply = reply

    cv2.destroyAllWindows()
    print()
    if "STOP" in final_reply:
        parts = final_reply.split()
        best_z = parts[-1] if len(parts) >= 4 else "?"
        print(f"  PASS : Peak found — best focus Z = {best_z}")
    elif final_reply == "FOCUS_ERR":
        print("  INFO : FOCUS_ERR — no peak detected in this image set.")
        print("         Tips: use more steps, ensure images span the full focus range.")
    else:
        print(f"  INFO : Last reply was '{final_reply}'")
    print()


def test_fm_index_req(sock: zmq.Socket):
    """FM_INDEX_REQ — one-shot FMI on a sharp and a blurry image."""
    print("=" * 60)
    print("TEST: FM_INDEX_REQ")
    print("=" * 60)

    for label, sharpness in [("sharp", 1.0), ("blurry", 0.05)]:
        img  = make_synthetic_image(sharpness)
        path = save_tmp_image(img, f"fmi_{label}")
        cmd  = f'FM_INDEX_REQ "{path}"'
        print(f"  Send : {cmd}")
        reply = send_command(sock, cmd)
        print(f"  Reply: {reply}")
        # File should be deleted by the server
        print(f"  File deleted by server: {not os.path.exists(path)}")
        print()


def test_start_autofocus_req(sock: zmq.Socket):
    """START_AUTOFOCUS_REQ — initialises a scan session."""
    print("=" * 60)
    print("TEST: START_AUTOFOCUS_REQ")
    print("=" * 60)

    img  = make_synthetic_image(sharpness=0.2)   # blurry start
    path = save_tmp_image(img, "af_start")
    z    = 100.0
    steps = 5
    cmd  = f'START_AUTOFOCUS_REQ "{path}" {z} {steps}'
    print(f"  Send : {cmd}")
    reply = send_command(sock, cmd)
    print(f"  Reply: {reply}")
    expected_prefix = "START_AUTOFOCUS_OK"
    ok = reply.startswith(expected_prefix)
    print(f"  PASS : {ok}  (expected prefix '{expected_prefix}')")
    print()


def test_focus_req_scan(sock: zmq.Socket, steps: int = 10, peak_at: int = 6):
    """
    Full autofocus scan:
      1. START_AUTOFOCUS_REQ to reset session.
      2. <steps> FOCUS_REQ calls with a synthetic FMI curve that peaks at
         step index <peak_at>.

    Expected behaviour
    ------------------
    - Replies "FOCUS_OK <fmi> UP" until step count == steps.
    - Replies "FOCUS_OK <fmi> STOP <bestHeight>" when peak is found.
    """
    print("=" * 60)
    print(f"TEST: FOCUS_REQ scan  (steps={steps}, peak_at={peak_at})")
    print("=" * 60)

    # --- Step 0: START ---
    start_img  = make_synthetic_image(sharpness=0.1)
    start_path = save_tmp_image(start_img, "af_s0")
    z_start    = 100.0
    cmd = f'START_AUTOFOCUS_REQ "{start_path}" {z_start} {steps}'
    print(f"  [0] Send : {cmd}")
    reply = send_command(sock, cmd)
    print(f"  [0] Reply: {reply}\n")

    # --- Steps 1 .. steps-1: FOCUS_REQ ---
    # Build a sharpness curve: rises to peak_at then falls (Gaussian-ish)
    sharpness_curve = [
        float(np.exp(-0.5 * ((i - peak_at) / 2.0) ** 2))
        for i in range(1, steps)
    ]

    final_reply = ""
    for i, sharpness in enumerate(sharpness_curve, start=1):
        img  = make_synthetic_image(sharpness=sharpness)
        path = save_tmp_image(img, f"af_step{i}")
        z    = z_start + i * 1.0          # 1 µm per step
        cmd  = f'FOCUS_REQ "{path}" {z}'
        reply = send_command(sock, cmd)
        print(f"  [{i:2d}] z={z:.1f}  sharpness={sharpness:.2f}  Reply: {reply}")
        final_reply = reply

    print()
    if "STOP" in final_reply:
        print(f"  PASS : scan completed with STOP reply")
    elif final_reply == "FOCUS_ERR":
        print(f"  INFO : FOCUS_ERR — no peak detected (try adjusting --peak-at)")
    else:
        print(f"  INFO : last reply was '{final_reply}'")
    print()


def test_missing_image(sock: zmq.Socket):
    """All three commands should return ERR ImageNotFound for a missing file."""
    print("=" * 60)
    print("TEST: Missing image handling")
    print("=" * 60)

    fake = r"C:\does\not\exist\image.bmp"
    for cmd in [
        f'FM_INDEX_REQ "{fake}"',
        f'START_AUTOFOCUS_REQ "{fake}" 100.0 5',
        f'FOCUS_REQ "{fake}" 101.0',
    ]:
        print(f"  Send : {cmd}")
        reply = send_command(sock, cmd)
        print(f"  Reply: {reply}")
        assert "ERR" in reply, f"Expected ERR, got: {reply}"
    print("  PASS : all returned ERR\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autofocus command test client")
    parser.add_argument("--port",       type=int,   default=5556,
                        help="ZMQ port (default 5556)")
    parser.add_argument("--steps",      type=int,   default=10,
                        help="Synthetic scan steps (default 10)")
    parser.add_argument("--peak-at",    type=int,   default=6,
                        help="Synthetic peak step index (default 6)")
    # Real-image options
    parser.add_argument("--image",      type=str,   default=None,
                        help="Single real image for FM_INDEX_REQ test")
    parser.add_argument("--images-dir", type=str,   default=None,
                        help="Folder of real images for full autofocus scan "
                             "(sorted alphabetically = scan order)")
    parser.add_argument("--z-start",    type=float, default=100.0,
                        help="Starting Z height for real scan (default 100.0)")
    parser.add_argument("--z-step",     type=float, default=1.0,
                        help="Z increment per step for real scan (default 1.0)")
    args = parser.parse_args()

    # Start server
    start_server(args.port)

    # Connect client
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    time.sleep(0.1)

    try:
        if args.image:
            # --- Real single-image FMI test ---
            test_real_fm_index(sock, args.image)

        elif args.images_dir:
            # --- Real full autofocus scan ---
            test_real_autofocus_scan(sock, args.images_dir,
                                     z_start=args.z_start,
                                     z_step=args.z_step)

        else:
            # --- Default: synthetic tests ---
            test_fm_index_req(sock)
            test_start_autofocus_req(sock)
            test_focus_req_scan(sock, steps=args.steps, peak_at=args.peak_at)
            test_missing_image(sock)

    finally:
        sock.close()
        ctx.term()
        print("[TEST] Done.")


if __name__ == "__main__":
    main()
