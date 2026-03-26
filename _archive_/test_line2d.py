"""Quick test for the LINE-2D matcher."""
import cv2
import numpy as np
import time
from linemod_matcher import (
    LinemodMatcher, LinemodConfig,
    _quantize_gradients, _spread, _compute_response_maps,
    _extract_scattered_features, _RESPONSE_LUTS,
)

# Create synthetic test image
img = np.zeros((400, 400), dtype=np.uint8)
cv2.rectangle(img, (150, 150), (250, 250), 200, -1)
cv2.circle(img, (200, 200), 30, 255, -1)
cv2.line(img, (160, 200), (240, 200), 255, 3)
cv2.line(img, (200, 160), (200, 240), 255, 3)
tmpl = img[145:260, 145:260].copy()

# Test 1: Import
print("1. Import OK")

# Test 2: Quantization
q, mag = _quantize_gradients(tmpl, 20.0)
print("2. Quantized: non-zero=%d" % np.count_nonzero(q))

# Test 3: Features
feats = _extract_scattered_features(q, mag, 32)
print("3. Features: %d extracted" % len(feats))

# Test 4: Spread
s = _spread(q, 4)
print("4. Spread: non-zero=%d" % np.count_nonzero(s))

# Test 5: Response maps
rmaps = _compute_response_maps(s)
print("5. Response maps: %d maps" % len(rmaps))
print("   LUT check: exact=%d, neighbor=%d" % (_RESPONSE_LUTS[0, 1], _RESPONSE_LUTS[0, 2]))

# Test 6: Generate
config = LinemodConfig()
config.ANGLE_STEP = 360
config.NUM_FEATURES = 32
config.WEAK_THRESHOLD = 20
config.PYRAMID_LEVELS = 2

matcher = LinemodMatcher(config)
matcher.load_template(tmpl)
matcher.generate_templates()
print("6. Templates: %d" % len(matcher.template_pyramids))

# Test 7: Manual response check at known position
q2, _ = _quantize_gradients(img, 20)
s2 = _spread(q2, 4)
rm2 = _compute_response_maps(s2)
tp = matcher.template_pyramids[0]
t0_templ = tp["templates"][0]
print("   Template tl=(%d,%d) size=(%d,%d) feats=%d" % (
    t0_templ.tl_x, t0_templ.tl_y, t0_templ.width, t0_templ.height,
    len(t0_templ.features)))

for ox in range(142, 155):
    sc = 0
    for f in t0_templ.features:
        fx = t0_templ.tl_x + f.x + ox
        fy = t0_templ.tl_y + f.y + ox
        if 0 <= fx < 400 and 0 <= fy < 400:
            sc += int(rm2[f.label][fy, fx])
    sim = sc * 100.0 / (4 * len(t0_templ.features))
    if sim > 10:
        print("   Manual at offset (%d,%d): raw=%d sim=%.1f%%" % (ox, ox, sc, sim))

# Test 8: Full match
t0 = time.time()
match = matcher.match(img, threshold=30)
elapsed = (time.time() - t0) * 1000

if match:
    print("7. MATCH found:")
    print("   Position: (%d, %d)" % (match["x"], match["y"]))
    print("   Score: %.1f%%" % match["score"])
    print("   Time: %.0f ms" % elapsed)
    print("PASS")
else:
    print("7. No match found (time=%.0f ms)" % elapsed)
    print("FAIL")
