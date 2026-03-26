"""Quick test: auto-downsample on 5120x5120."""
import numpy as np, cv2, time
from linemod_matcher import LinemodMatcher, LinemodConfig

# Create synthetic 5120x5120 with cross
img = np.ones((5120, 5120), dtype=np.uint8) * 128
cv2.rectangle(img, (2400, 2500), (2700, 2600), 50, -1)
cv2.rectangle(img, (2500, 2400), (2600, 2700), 50, -1)

template = img[2380:2720, 2380:2720].copy()
print(f"Template: {template.shape}, Search: {img.shape}")

config = LinemodConfig()
config.ANGLE_STEP = 360
config.PYRAMID_LEVELS = 1
config.NUM_FEATURES = 128
config.WEAK_THRESHOLD = 30
config.MATCH_THRESHOLD = 30

matcher = LinemodMatcher(config)
matcher.load_template(template)
matcher.generate_templates()

t0 = time.time()
result = matcher.match(img, threshold=30)
t_total = (time.time() - t0) * 1000

if result:
    print(f"Match: ({result['x']}, {result['y']}) Score={result['score']:.1f}%")
    print(f"Time: {t_total:.0f}ms")
    # Expected position: ~2550, ~2550
    dx = abs(result['x'] - 2550)
    dy = abs(result['y'] - 2550)
    print(f"Position error: dx={dx}, dy={dy}")
else:
    print(f"No match in {t_total:.0f}ms")
