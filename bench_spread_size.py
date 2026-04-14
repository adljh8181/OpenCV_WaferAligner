import cv2, time, numpy as np
h, w = 5120, 5120
kernel = np.ones((7, 7), dtype=np.uint8)
q = np.random.randint(0, 255, (h, w), dtype=np.uint8)
t0=time.time()
res_c = np.zeros_like(q)
for b in range(8):
    p = ((q >> b)&1)
    res_c |= (cv2.dilate(p, kernel)<<b)
print(f'CPU: {time.time()-t0:.3f}s')

morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8U, kernel)
t0=time.time()
res_g = np.zeros_like(q)
for bit in range(8):
    p = ((q >> bit) & 1).astype(np.uint8)
    gpu_plane = cv2.cuda_GpuMat()
    gpu_plane.upload(p)
    gpu_out = morph.apply(gpu_plane)
    dilated = gpu_out.download()
    res_g |= (dilated.astype(np.uint8) << bit)
print(f'GPU loop: {time.time()-t0:.3f}s')
