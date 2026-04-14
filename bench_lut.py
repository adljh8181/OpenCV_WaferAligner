import cv2, time, numpy as np
lut = np.random.randint(0, 5, 256, dtype=np.uint8)
spread_img = np.random.randint(0, 255, (2500, 2500), dtype=np.uint8)

t0=time.time()
r=[]
for label in range(8):
    r.append(lut[spread_img])
print(f'Numpy: {time.time()-t0:.3f}s')

t0=time.time()
r2=[]
for label in range(8):
    r2.append(cv2.LUT(spread_img, lut))
print(f'cv2.LUT: {time.time()-t0:.3f}s')
