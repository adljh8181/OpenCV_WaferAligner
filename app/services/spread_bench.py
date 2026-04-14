import cv2, time, numpy as np
h, w = 5120, 5120
kernel = np.ones((15, 15), dtype=np.uint8)
q = np.random.randint(0, 255, (h, w), dtype=np.uint8)

t0=time.time()
res_c = np.zeros_like(q)
for b in range(8):
    p = ((q >> b)&1)
    res_c |= (cv2.dilate(p, kernel)<<b)
print(f'CPU: {time.time()-t0:.3f}s')

# GPU path 4-channel
morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC4, kernel)
t0=time.time()
lut_0 = np.array([[(x >> i) & 1 for i in range(4)] for x in range(256)], dtype=np.uint8)
lut_1 = np.array([[(x >> (i+4)) & 1 for i in range(4)] for x in range(256)], dtype=np.uint8)
q4_0 = lut_0[q]
q4_1 = lut_1[q]
g0 = cv2.cuda_GpuMat()
g1 = cv2.cuda_GpuMat()
g0.upload(q4_0)
g1.upload(q4_1)
d0 = morph.apply(g0).download()
d1 = morph.apply(g1).download()
res_g = np.zeros_like(q)
for i in range(4): res_g |= (d0[...,i] << i)
for i in range(4): res_g |= (d1[...,i] << (i+4))
print(f'GPU 4ch: {time.time()-t0:.3f}s')
