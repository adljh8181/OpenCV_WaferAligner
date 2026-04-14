from app.services.linemod_matcher import LinemodMatcher, LinemodConfig
import cv2, time, numpy as np

img = np.zeros((3000, 3000), dtype=np.uint8)
cv2.rectangle(img, (200, 200), (800, 800), 255, -1)

config = LinemodConfig()
config.USE_GPU = True
config.ANGLE_STEP = 360

matcher = LinemodMatcher(config)
matcher.load_template(img[150:850, 150:850])
matcher.generate_templates()

t0 = time.time()
res = matcher.match(img, threshold=40.0, return_all=True)
timing_match = time.time() - t0

print(f'Match pipeline full eval: {timing_match:.3f}s')
