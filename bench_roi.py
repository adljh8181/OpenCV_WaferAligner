import cv2, time, numpy as np

def cpu_quantize(roi):
    t0 = time.perf_counter()
    gray = cv2.GaussianBlur(roi, (3, 3), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    ang = cv2.phase(dx, dy, angleInDegrees=True) % 360.0
    return (time.perf_counter()-t0)*1000

def gpu_quantize(roi):
    t0 = time.perf_counter()
    g_g = cv2.cuda_GpuMat()
    g_g.upload(roi)
    bf = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (3, 3), 0)
    g_g = bf.apply(g_g)
    sx = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 0, 1, ksize=3)
    dx = sx.apply(g_g)
    dy = sy.apply(g_g)
    mag = cv2.cuda.magnitude(dx, dy, cv2.cuda_GpuMat()).download()
    ang = cv2.cuda.phase(dx, dy, cv2.cuda_GpuMat(), angleInDegrees=True).download()
    return (time.perf_counter()-t0)*1000

rois = [np.random.randint(0, 255, (200, 200), dtype=np.uint8).astype(np.float32) for _ in range(40)]

cpu_t = sum(cpu_quantize(r) for r in rois)
gpu_t = sum(gpu_quantize(r) for r in rois)

print(f'CPU 40 ROIs: {cpu_t:.1f}ms')
print(f'GPU 40 ROIs: {gpu_t:.1f}ms')
