import cv2
import numpy as np

src_path = r"C:\Users\adrain.lim\OneDrive - QES (Asia-Pacific) Sdn Bhd\Desktop\EmguFindEdge\recipes\TKX3_template.png"
dst_path = r"C:\Users\adrain.lim\OneDrive - QES (Asia-Pacific) Sdn Bhd\Desktop\EmguFindEdge\recipes\TKX3_template_bright.png"

img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

# Step 1: CLAHE — boosts local contrast, preserves edge transitions
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)

# Step 2: Gentle gamma correction to lift overall brightness further
gamma = 1.8
lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                for i in range(256)], dtype=np.uint8)
brightened = cv2.LUT(enhanced, lut)

# Step 3: Unsharp mask — recover any edge softness introduced by brightening
blur = cv2.GaussianBlur(brightened, (0, 0), sigmaX=2)
sharpened = cv2.addWeighted(brightened, 1.5, blur, -0.5, 0)

cv2.imwrite(dst_path, sharpened)
print(f"Saved → {dst_path}")