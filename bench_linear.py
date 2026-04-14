import time, numpy as np
h, w, T = 2500, 2500, 8
W = w // T
H = h // T
response_map = np.random.randint(0, 255, (h, w), dtype=np.uint8)

t0=time.time()
mem1 = np.zeros((T*T, H*W), dtype=np.uint8)
for gy in range(T):
    for gx in range(T):
        mem1[gy*T + gx] = response_map[gy:gy+H*T:T, gx:gx+W*T:T].ravel()
print(f'Loop: {time.time()-t0:.4f}s')

t0=time.time()
reshaped = response_map[:H*T, :W*T].reshape(H, T, W, T)
mem2 = reshaped.transpose(1, 3, 0, 2).reshape(T*T, H*W)
print(f'Reshape: {time.time()-t0:.4f}s')
print(np.array_equal(mem1, mem2))
