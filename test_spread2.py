import cv2, time, numpy as np
h, w = 5000, 5000
kernel = np.ones((31, 31), dtype=np.uint8)
q = np.random.randint(0, 255, (h, w), dtype=np.uint8)
t0 = time.time()
res_c = np.zeros_like(q)
for b in range(8):
    p = ((q >> b)&1)
    res_c |= (cv2.dilate(p, kernel)<<b)
print(f'CPU: {time.time()-t0:.3f}s')

# GPU path
morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC4, kernel)
t0 = time.time()
q4_0 = np.empty((h, w, 4), dtype=np.uint8)
q4_1 = np.empty((h, w, 4), dtype=np.uint8)
for i in range(4): q4_0[...,i] = (q >> i)&1
for i in range(4): q4_1[...,i] = (q >> (i+4))&1
g0 = cv2.cuda_GpuMat()
g1 = cv2.cuda_GpuMat()
g0.upload(q4_0)
g1.upload(q4_1)
d0 = morph.apply(g0).download()
d1 = morph.apply(g1).download()
res_g = np.zeros_like(q)
for i in range(4): res_g |= (d0[...,i] << i)
for i in range(4): res_g |= (d1[...,i] << (i+4))
print(f'GPU: {time.time()-t0:.3f}s')
print(np.array_equal(res_c, res_g))
