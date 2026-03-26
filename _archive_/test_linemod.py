"""
Test script for LINE-MOD matcher improvements.
Tests: sub-pixel refinement, pyramid search, angular refinement,
       absolute gradient thresholds, and timing comparison.
"""
import cv2
import numpy as np
import time
from linemod_matcher import LinemodMatcher, LinemodConfig


def test_basic_match():
    """Test basic matching: crop a region from an image and find it back."""
    print("=" * 60)
    print("  TEST 1: Basic Match (No Rotation)")
    print("=" * 60)
    
    search_img_path = r"Images\TOP_EDGE_PNG\Copy of TopEdge.png"
    search_img = cv2.imread(search_img_path)
    
    if search_img is None:
        print(f"[SKIP] Test image not found: {search_img_path}")
        return False
    
    # Crop template from known location
    h, w = search_img.shape[:2]
    cx, cy = w // 2, h // 2
    template_img = search_img[cy-100:cy+100, cx-100:cx+100].copy()
    
    config = LinemodConfig()
    config.ANGLE_STEP = 360  # Only 0° for speed
    config.SCALE_MIN = 1.0
    config.SCALE_MAX = 1.0
    config.USE_PYRAMID = False  # Brute force for baseline
    config.ANGLE_REFINE = False
    config.SUBPIXEL_REFINE = False
    
    matcher = LinemodMatcher(config)
    matcher.load_template(template_img)
    matcher.generate_templates(verbose=False)
    
    matches = matcher.match(search_img, return_all=True)
    
    print(f"  Found {len(matches)} matches")
    if len(matches) > 0:
        best = matches[0]
        print(f"  Best: ({best['x']}, {best['y']}) score={best['score']:.1f}%")
        print(f"  Expected center: ({cx}, {cy})")
        dist = np.sqrt((best['x'] - cx)**2 + (best['y'] - cy)**2)
        print(f"  Distance from expected: {dist:.1f} px")
        passed = dist < 5
        print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
        return passed
    else:
        print("  RESULT: FAIL (no matches)")
        return False


def test_subpixel():
    """Test sub-pixel refinement produces float coordinates."""
    print("\n" + "=" * 60)
    print("  TEST 2: Sub-Pixel Refinement")
    print("=" * 60)
    
    search_img_path = r"Images\TOP_EDGE_PNG\Copy of TopEdge.png"
    search_img = cv2.imread(search_img_path)
    
    if search_img is None:
        print(f"[SKIP] Test image not found")
        return False
    
    h, w = search_img.shape[:2]
    cx, cy = w // 2, h // 2
    template_img = search_img[cy-100:cy+100, cx-100:cx+100].copy()
    
    config = LinemodConfig()
    config.ANGLE_STEP = 360
    config.SCALE_MIN = 1.0
    config.SCALE_MAX = 1.0
    config.USE_PYRAMID = False
    config.ANGLE_REFINE = False
    config.SUBPIXEL_REFINE = True  # Enable sub-pixel
    
    matcher = LinemodMatcher(config)
    matcher.load_template(template_img)
    matcher.generate_templates(verbose=False)
    
    match = matcher.match(search_img)
    
    if match is None:
        print("  RESULT: FAIL (no match)")
        return False
    
    is_float_x = isinstance(match['x'], float)
    is_float_y = isinstance(match['y'], float)
    is_subpixel = match.get('subpixel', False)
    
    print(f"  Position: ({match['x']}, {match['y']})")
    print(f"  Float coords: x={is_float_x}, y={is_float_y}")
    print(f"  Subpixel flag: {is_subpixel}")
    
    passed = is_subpixel and is_float_x and is_float_y
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_pyramid_vs_bruteforce():
    """Compare pyramid search vs brute-force: correctness and speed."""
    print("\n" + "=" * 60)
    print("  TEST 3: Pyramid vs Brute-Force (Speed + Correctness)")
    print("=" * 60)
    
    search_img_path = r"Images\TOP_EDGE_PNG\Copy of TopEdge.png"
    search_img = cv2.imread(search_img_path)
    
    if search_img is None:
        print(f"[SKIP] Test image not found")
        return False
    
    h, w = search_img.shape[:2]
    cx, cy = w // 2, h // 2
    template_img = search_img[cy-80:cy+80, cx-80:cx+80].copy()
    
    # --- Brute force ---
    config_bf = LinemodConfig()
    config_bf.ANGLE_STEP = 45  # 8 angles
    config_bf.SCALE_MIN = 1.0
    config_bf.SCALE_MAX = 1.0
    config_bf.USE_PYRAMID = False
    config_bf.ANGLE_REFINE = False
    config_bf.SUBPIXEL_REFINE = False
    
    matcher_bf = LinemodMatcher(config_bf)
    matcher_bf.load_template(template_img)
    matcher_bf.generate_templates(verbose=False)
    
    t0 = time.perf_counter()
    match_bf = matcher_bf.match(search_img)
    time_bf = time.perf_counter() - t0
    
    # --- Pyramid ---
    config_py = LinemodConfig()
    config_py.ANGLE_STEP = 45
    config_py.SCALE_MIN = 1.0
    config_py.SCALE_MAX = 1.0
    config_py.USE_PYRAMID = True
    config_py.PYRAMID_LEVELS = 2
    config_py.PYRAMID_SCALE = 0.5
    config_py.ANGLE_REFINE = False
    config_py.SUBPIXEL_REFINE = False
    
    matcher_py = LinemodMatcher(config_py)
    matcher_py.load_template(template_img)
    matcher_py.generate_templates(verbose=False)
    
    t0 = time.perf_counter()
    match_py = matcher_py.match(search_img)
    time_py = time.perf_counter() - t0
    
    print(f"  Brute-force: {time_bf*1000:.1f} ms", end="")
    if match_bf:
        print(f"  pos=({match_bf['x']}, {match_bf['y']}) score={match_bf['score']:.1f}%")
    else:
        print("  (no match)")
    
    print(f"  Pyramid:     {time_py*1000:.1f} ms", end="")
    if match_py:
        print(f"  pos=({match_py['x']}, {match_py['y']}) score={match_py['score']:.1f}%")
    else:
        print("  (no match)")
    
    if match_bf and match_py:
        dist = np.sqrt((match_bf['x'] - match_py['x'])**2 + 
                       (match_bf['y'] - match_py['y'])**2)
        print(f"  Position difference: {dist:.1f} px")
        print(f"  Speedup: {time_bf/time_py:.1f}x")
        passed = dist < 20  # Allow some tolerance since pyramid is approximate
    elif not match_bf and not match_py:
        print("  Both found no match (consistent)")
        passed = True
    else:
        print("  Inconsistent results!")
        passed = False
    
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gradient_threshold_modes():
    """Test absolute vs relative gradient threshold modes."""
    print("\n" + "=" * 60)
    print("  TEST 4: Gradient Threshold Modes")
    print("=" * 60)
    
    search_img_path = r"Images\TOP_EDGE_PNG\Copy of TopEdge.png"
    search_img = cv2.imread(search_img_path)
    
    if search_img is None:
        print(f"[SKIP] Test image not found")
        return False
    
    h, w = search_img.shape[:2]
    cx, cy = w // 2, h // 2
    template_img = search_img[cy-100:cy+100, cx-100:cx+100].copy()
    
    results = {}
    for mode in ["absolute", "relative"]:
        config = LinemodConfig()
        config.ANGLE_STEP = 360
        config.SCALE_MIN = 1.0
        config.SCALE_MAX = 1.0
        config.USE_PYRAMID = False
        config.ANGLE_REFINE = False
        config.SUBPIXEL_REFINE = False
        config.GRADIENT_THRESHOLD_MODE = mode
        
        matcher = LinemodMatcher(config)
        matcher.load_template(template_img)
        matcher.generate_templates(verbose=False)
        
        match = matcher.match(search_img)
        if match:
            results[mode] = match
            print(f"  {mode:10s}: pos=({match['x']}, {match['y']}) score={match['score']:.1f}%")
        else:
            print(f"  {mode:10s}: no match")
    
    passed = len(results) > 0  # At least one mode should work
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_full_pipeline():
    """Test the full pipeline: pyramid + angle refine + sub-pixel."""
    print("\n" + "=" * 60)
    print("  TEST 5: Full Pipeline (Pyramid + Angle Refine + Sub-Pixel)")
    print("=" * 60)
    
    search_img_path = r"Images\TOP_EDGE_PNG\Copy of TopEdge.png"
    search_img = cv2.imread(search_img_path)
    
    if search_img is None:
        print(f"[SKIP] Test image not found")
        return False
    
    h, w = search_img.shape[:2]
    cx, cy = w // 2, h // 2
    template_img = search_img[cy-80:cy+80, cx-80:cx+80].copy()
    
    config = LinemodConfig()
    config.ANGLE_STEP = 30      # 12 coarse angles
    config.SCALE_MIN = 1.0
    config.SCALE_MAX = 1.0
    config.USE_PYRAMID = True
    config.PYRAMID_LEVELS = 2
    config.ANGLE_REFINE = True
    config.ANGLE_REFINE_RANGE = 10
    config.ANGLE_REFINE_STEP = 1.0
    config.SUBPIXEL_REFINE = True
    
    matcher = LinemodMatcher(config)
    matcher.load_template(template_img)
    
    t0 = time.perf_counter()
    matcher.generate_templates(verbose=False)
    match = matcher.match(search_img)
    elapsed = time.perf_counter() - t0
    
    if match is None:
        print("  RESULT: FAIL (no match)")
        return False
    
    print(f"  Position: ({match['x']:.2f}, {match['y']:.2f})")
    print(f"  Angle: {match['angle']:.1f}°")
    print(f"  Scale: {match['scale']:.2f}")
    print(f"  Score: {match['score']:.1f}%")
    print(f"  Sub-pixel: {match.get('subpixel', False)}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    
    dist = np.sqrt((match['x'] - cx)**2 + (match['y'] - cy)**2)
    print(f"  Distance from expected ({cx}, {cy}): {dist:.2f} px")
    
    passed = dist < 10 and match.get('subpixel', False)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LINE-MOD Matcher Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Basic Match", test_basic_match()))
    results.append(("Sub-Pixel Refinement", test_subpixel()))
    results.append(("Pyramid vs Brute-Force", test_pyramid_vs_bruteforce()))
    results.append(("Gradient Threshold Modes", test_gradient_threshold_modes()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}  {status}")
    
    total_pass = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  {total_pass}/{total} tests passed")
    print("=" * 60)
