import cv2
import numpy as np
import time
import os
import argparse
import math
from app.services.linemod_matcher import LinemodMatcher, LinemodConfig

def test_pyramid_levels(image_path, template_path, max_level=4):
    print(f"=== Deep Pyramid Sub-Search Position Delta Strategy ===")
    print(f"Target Image: {image_path}")
    print(f"Template Image: {template_path}")
    print(f"Testing Pyramid Levels: L0 (Baseline) up to L{max_level}")
    print("-" * 75)
    
    orig_img = cv2.imread(image_path)
    orig_tmpl = cv2.imread(template_path)

    if orig_img is None or orig_tmpl is None:
        print("Error: Could not read image or template.")
        return

    config = LinemodConfig()
    config.MATCH_THRESHOLD = 30.0
    config.WEAK_THRESHOLD = -70.0
    config.ANGLE_STEP = 360 # Test single template for speed
    config.HYSTERESIS_KERNEL = 7
    config.PYRAMID_LEVELS = max_level + 1 # Need enough levels generated
    
    # Initialize matcher and build deep templates
    matcher = LinemodMatcher(config)
    t_start = time.perf_counter()
    matcher.load_template(orig_tmpl)
    matcher.generate_templates()
    t_end = time.perf_counter()
    print(f"Generated deep templates up to L{max_level} in {(t_end - t_start)*1000:.1f} ms\n")

    baseline_x, baseline_y = None, None

    print(f"{'Pyramid Lvl':>12} | {'Eff. Area':>13} | {'Found X, Y':>16} | {'Delta px':>10} | {'Time (ms)':>10}")
    print("-" * 75)

    for lvl in range(max_level + 1):
        if lvl == 0:
            config.FORCE_COARSE_LEVEL = None  # pristine baseline
            start_t = time.perf_counter()
            results = matcher._match_single_level(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY), config.MATCH_THRESHOLD)
            end_t = time.perf_counter()
        else:
            config.FORCE_COARSE_LEVEL = lvl
            start_t = time.perf_counter()
            results = matcher._match_pyramid(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY), config.MATCH_THRESHOLD)
            end_t = time.perf_counter()

        elapsed_ms = (end_t - start_t) * 1000.0
        
        # Calculate effective coarse search area
        scale_fac = 1.0 / (2**lvl)
        eff_w = int(orig_img.shape[1] * scale_fac)
        eff_h = int(orig_img.shape[0] * scale_fac)
        eff_str = f"{eff_w}x{eff_h}"

        if not results:
            print(f"Level {lvl:<6} | {eff_str:>13} | {'FAILED':>16} | {'N/A':>10} | {elapsed_ms:>10.1f}")
            continue

        # Get top result
        top_res = results if isinstance(results, dict) else (results[0] if isinstance(results, list) else results)
        res_x = float(top_res.get('x', getattr(top_res, 'x', 0)))
        res_y = float(top_res.get('y', getattr(top_res, 'y', 0)))

        if lvl == 0:
            baseline_x, baseline_y = res_x, res_y
            err_str = "0.00"
            dist = 0.0
        else:
            dist = np.sqrt((res_x - baseline_x)**2 + (res_y - baseline_y)**2)
            err_str = f"{dist:.2f}"

        print(f"Level {lvl:<6} | {eff_str:>13} | {f'{res_x:.1f}, {res_y:.1f}':>16} | {err_str:>10} | {elapsed_ms:>10.1f}")

    print("-" * 75)
    print("Done.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--max_level", type=int, default=4)
    args = parser.parse_args()
    test_pyramid_levels(args.image, args.template, args.max_level)
