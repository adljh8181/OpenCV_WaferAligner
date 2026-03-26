"""
================================================================================
LIGHTING VARIATION TEST FOR LINE-2D
================================================================================
Generates images with different lighting conditions and tests whether
LINE-2D can still detect the pattern.

Variations:
  1. Brightness (darker / brighter)
  2. Contrast (lower / higher)
  3. Gamma correction (non-linear)
  4. Gaussian noise
  5. Gradient lighting (uneven illumination)

Usage:
    python light_variation_test.py

Author: Wafer Alignment System
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from linemod_matcher import LinemodMatcher, LinemodConfig


# ======================================================================
# Lighting Variation Functions
# ======================================================================

def adjust_brightness(img, value):
    """Add/subtract constant brightness. value: -100 to +100"""
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)


def adjust_contrast(img, factor):
    """Scale contrast around mean. factor: 0.3 to 3.0"""
    mean = np.mean(img)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def adjust_gamma(img, gamma):
    """Apply gamma correction. gamma < 1 = brighter, gamma > 1 = darker"""
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def add_noise(img, sigma):
    """Add Gaussian noise with given standard deviation."""
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gradient_lighting(img, direction='horizontal', strength=80):
    """Simulate uneven illumination with a linear gradient."""
    h, w = img.shape[:2]
    if direction == 'horizontal':
        grad = np.linspace(-strength, strength, w).reshape(1, -1)
        grad = np.repeat(grad, h, axis=0)
    else:
        grad = np.linspace(-strength, strength, h).reshape(-1, 1)
        grad = np.repeat(grad, w, axis=1)
    return np.clip(img.astype(np.float32) + grad, 0, 255).astype(np.uint8)


def invert_image(img):
    """Invert (negative) of the image."""
    return 255 - img


# ======================================================================
# Test Runner
# ======================================================================

def run_test(search_img_path=None, template_roi=None):
    """
    Run LINE-2D matching on multiple lighting variations.
    
    Args:
        search_img_path: Path to search image (or None for file dialog)
        template_roi: (x, y, w, h) for template crop (or None for interactive)
    """
    # ---- Load image ----
    if search_img_path and os.path.exists(search_img_path):
        img = cv2.imread(search_img_path)
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Search Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        root.destroy()
        if not path:
            print("No image selected!"); return
        img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # ---- Crop template ----
    if template_roi:
        x, y, w, h = template_roi
        template = gray[y:y+h, x:x+w].copy()
    else:
        print("\n  Draw rectangle around the pattern → ENTER to confirm")
        win = "Select Template"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 800)
        roi = cv2.selectROI(win, img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)
        x, y, w, h = roi
        if w == 0 or h == 0:
            print("No template selected!"); return
        template = gray[y:y+h, x:x+w].copy()

    print(f"\n  Template: {w}x{h} from ({x},{y})")
    print(f"  Search:  {gray.shape[1]}x{gray.shape[0]}")

    # ---- Define lighting variations ----
    variations = [
        ("Original",             gray),
        ("Bright +60",           adjust_brightness(gray, 60)),
        ("Bright +120",          adjust_brightness(gray, 120)),
        ("Dark -60",             adjust_brightness(gray, -60)),
        ("Dark -120",            adjust_brightness(gray, -120)),
        ("Contrast 0.3x",        adjust_contrast(gray, 0.3)),
        ("Contrast 0.5x",        adjust_contrast(gray, 0.5)),
        ("Contrast 2.0x",        adjust_contrast(gray, 2.0)),
        ("Gamma 0.4 (bright)",   adjust_gamma(gray, 0.4)),
        ("Gamma 2.5 (dark)",     adjust_gamma(gray, 2.5)),
        ("Noise σ=25",           add_noise(gray, 25)),
        ("Noise σ=50",           add_noise(gray, 50)),
        ("Gradient H ±80",       gradient_lighting(gray, 'horizontal', 80)),
        ("Gradient V ±80",       gradient_lighting(gray, 'vertical', 80)),
        ("Inverted",             invert_image(gray)),
    ]

    # ---- Configure LINE-2D ----
    config = LinemodConfig()
    config.ANGLE_STEP = 360        # No rotation search
    config.NUM_FEATURES = 128
    config.WEAK_THRESHOLD = 30
    config.PYRAMID_LEVELS = 1
    config.MATCH_THRESHOLD = 30

    matcher = LinemodMatcher(config)
    matcher.load_template(template)
    matcher.generate_templates()

    # ---- Run matching on each variation ----
    print("\n" + "=" * 75)
    print(f"  {'Variation':<25} {'Score':>8} {'Position':>15} {'Time':>10} {'Result':>8}")
    print("=" * 75)

    results = []
    for name, varied_img in variations:
        t0 = time.time()
        match = matcher.match(varied_img, threshold=config.MATCH_THRESHOLD)
        elapsed = (time.time() - t0) * 1000

        if match:
            score = match['score']
            pos = f"({match['x']}, {match['y']})"
            status = "✓ PASS" if score >= 50 else "~ WEAK"
        else:
            score = 0.0
            pos = "---"
            status = "✗ FAIL"

        results.append((name, varied_img, score, pos, elapsed, status, match))
        print(f"  {name:<25} {score:>7.1f}% {pos:>15} {elapsed:>8.0f}ms {status:>8}")

    print("=" * 75)

    # ---- Summary ----
    passed = sum(1 for r in results if r[5] == "✓ PASS")
    weak = sum(1 for r in results if r[5] == "~ WEAK")
    failed = sum(1 for r in results if r[5] == "✗ FAIL")
    print(f"\n  Summary: {passed} passed, {weak} weak, {failed} failed out of {len(results)}")

    # ---- Visualization ----
    n = len(variations)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle('LINE-2D Lighting Robustness Test', fontsize=16, fontweight='bold')

    for idx, (name, varied_img, score, pos, elapsed, status, match) in enumerate(results):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]

        # Draw result on the image
        vis = cv2.cvtColor(varied_img, cv2.COLOR_GRAY2BGR) if len(varied_img.shape) == 2 else varied_img.copy()
        if match:
            bx, by, bw, bh = match['bbox']
            color = (0, 255, 0) if score >= 50 else (0, 255, 255)
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, 2)
            cv2.circle(vis, (match['x'], match['y']), 5, (0, 0, 255), -1)

        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        # Title color based on result
        title_color = 'green' if status == "✓ PASS" else ('orange' if status == "~ WEAK" else 'red')
        ax.set_title(f"{name}\nScore: {score:.1f}% {status}",
                    fontsize=9, fontweight='bold', color=title_color)
        ax.axis('off')

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return results


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  LINE-2D Lighting Robustness Test")
    print("=" * 55)
    print()
    print("  This test generates 15 lighting variations and")
    print("  checks if LINE-2D can still detect the pattern.")
    print()

    # You can hardcode a path and ROI for quick testing:
    # run_test("Images/Alignment/Alignment Image2.bmp", (333, 346, 100, 100))

    # Or use interactive mode:
    run_test()
