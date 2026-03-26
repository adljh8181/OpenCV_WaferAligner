# Edge Detection Method Comparison

## OLD Method vs NEW Method for Wafer Edge Detection

**Author:** Adrain Lim  
**Date:** February 2026  
**Tool:** `edge_compare.py`

---

## 1. Purpose

This document compares two edge detection approaches used for wafer alignment:
- **OLD Method**: Threshold + Canny + `cv2.fitLine` (Least Squares)
- **NEW Method**: Custom Gradient Kernel + Region-based Peak Detection + RANSAC Line Fitting

---

## 2. Method Comparison Summary

| Feature | OLD Method (Canny + fitLine) | NEW Method (Gradient + RANSAC) |
|---------|------------------------------|-------------------------------|
| **Edge Detection** | Canny edge detector | Custom gradient kernel convolution |
| **Pre-processing** | Gaussian blur → Binary threshold → Morphological closing | Median profile per scan region |
| **Line Fitting** | `cv2.fitLine` (Least Squares L2) | RANSAC + `cv2.fitLine` on inliers |
| **Outlier Handling** | ❌ None — all edge pixels contribute equally | ✅ RANSAC rejects outlier points |
| **Accuracy** | Integer pixel only | Sub-pixel (parabola refinement) |
| **Edge Points** | All Canny edge pixels (thousands) | One point per scan region (typically 40) |
| **Robustness** | Sensitive to noise and false edges | Robust to isolated noise |

---

## 3. Processing Pipeline Comparison

### 3.1 OLD Method Pipeline

```
Input Image (Grayscale)
    │
    ▼
Step 1: Gaussian Blur (kernel = 3% of image width)
    │
    ▼
Step 2: Binary Threshold (user-adjustable)
    │    - Pixels above threshold → 255 (white)
    │    - Pixels below threshold → 0 (black)
    │
    ▼
Step 3: Morphological Closing (kernel = 4.8% of width)
    │    - Fills small gaps in the binary image
    │
    ▼
Step 4: Canny Edge Detection (thresholds: 30, 100)
    │    - Extracts all edge pixels
    │
    ▼
Step 5: cv2.fitLine (Least Squares, L2 distance)
    │    - Fits a single line through ALL edge pixels
    │    - No outlier rejection
    │
    ▼
Output: Edge line (integer pixel endpoints)
```

### 3.2 NEW Method Pipeline

```
Input Image (Grayscale)
    │
    ▼
Step 1: Divide image into N scan regions (default: 40)
    │
    ▼
Step 2: For each region:
    │    a. Compute median intensity profile (1D)
    │    b. Convolve with custom gradient kernel
    │    c. Find gradient peaks above threshold
    │    d. Cluster peaks and select correct cluster
    │       based on scan direction (LEFT/RIGHT/TOP/BOTTOM)
    │    e. Sub-pixel refinement using parabola fitting
    │
    ▼
Step 3: Collect all detected edge points (one per region)
    │
    ▼
Step 4: RANSAC Line Fitting (2000 iterations)
    │    - Randomly samples 2 points per iteration
    │    - Finds consensus line with most inliers
    │    - Rejects outlier points automatically
    │
    ▼
Step 5: Final cv2.fitLine on RANSAC inliers only
    │
    ▼
Output: Edge line (sub-pixel endpoints) + inlier/outlier info
```

---

## 4. Key Differences Explained

### 4.1 Edge Detection Approach

| Aspect | OLD | NEW |
|--------|-----|-----|
| **How edges are found** | Canny finds ALL edge pixels globally (thousands of points) | Custom gradient kernel finds ONE strongest edge per scan region |
| **Noise sensitivity** | Canny can detect false edges from scratches, debris, or texture | Median profile per region averages out local noise |
| **Direction awareness** | Not direction-aware — detects edges everywhere | Selects edge cluster based on expected direction (LEFT → first cluster, RIGHT → last cluster) |

### 4.2 Line Fitting

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Algorithm** | Least Squares (L2) via `cv2.fitLine` | RANSAC → then Least Squares on inliers |
| **Outlier impact** | A single false edge pixel shifts the fitted line | RANSAC ignores outlier points automatically |
| **Typical point count** | Hundreds to thousands of edge pixels | ~40 carefully selected points |
| **Failure mode** | Line pulled toward noise clusters | Only fails if too few inlier points (<3) |

**Example:**  
If 5 out of 40 detected points are on a scratch instead of the wafer edge:
- **OLD**: All 5 noisy points affect the line → line angle is wrong
- **NEW**: RANSAC rejects the 5 noisy points → line fits only the 35 true edge points

### 4.3 Sub-pixel Accuracy

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Detection precision** | Integer pixel (Canny outputs binary edge map) | Sub-pixel via parabola fitting on gradient peak |
| **How it works** | Edge at pixel (x, y) has no fractional part | Fits parabola to 3 points around gradient peak → vertex gives sub-pixel position |
| **Typical improvement** | — | ±0.5 pixel refinement from integer position |

---

## 5. When to Use Each Method

### Use OLD Method When:
- Quick sanity check is needed
- Image has very clean, high-contrast edges
- Sub-pixel accuracy is not required
- You want to see the Canny edge map for debugging

### Use NEW Method When:
- **Production/alignment use** — sub-pixel accuracy matters
- Image has noise, scratches, or debris near the edge
- You need robust results that ignore outliers
- You need to know which points are inliers vs outliers
- Accurate angle measurement is required

---

## 6. Adjustable Parameters

| Parameter | Range | OLD Method Effect | NEW Method Effect |
|-----------|-------|-------------------|-------------------|
| **Threshold** | 5 – 200 | Binary threshold cutoff — higher removes more dark areas | Minimum gradient magnitude — higher requires stronger edges |
| **Kernel Size** | 3 – 15 | *Not used* | Gradient kernel width — larger gives smoother detection |
| **Scan Regions** | 10 – 80 | *Not used* | Number of edge points detected — more regions = more points |
| **RANSAC Thresh** | 1.0 – 20.0 | *Not used* | Inlier distance — lower = stricter line fitting |

---

## 7. Scan Direction Logic

Both methods support 4 scan directions:

| Direction | Wafer Position | Edge Position | Scan Axis | Line Type |
|-----------|---------------|---------------|-----------|-----------|
| **LEFT** | Wafer on right | Edge on left | Along X (horizontal) | Vertical line |
| **RIGHT** | Wafer on left | Edge on right | Along X (horizontal) | Vertical line |
| **TOP** | Wafer on bottom | Edge on top | Along Y (vertical) | Horizontal line |
| **BOTTOM** | Wafer on top | Edge on bottom | Along Y (vertical) | Horizontal line |

**NEW method cluster selection:**
- LEFT/TOP → Selects the **first** gradient cluster (nearest to image origin)
- RIGHT/BOTTOM → Selects the **last** gradient cluster (farthest from image origin)

---

## 8. Output Comparison

### Visual Markers on Result Image:

| Marker | OLD Method | NEW Method |
|--------|-----------|------------|
| 🟢 Green line | Fitted edge line | Fitted edge line |
| 🔴 Red dots | Canny edge pixels (sampled, max 500) | All detected edge points (one per region) |
| 🟡 Cyan circles | *Not shown* | RANSAC inlier points |

### Status Panel:

| Field | OLD Method | NEW Method |
|-------|-----------|------------|
| Edge Pixels / Points | Total Canny edge pixels | Total detected points (= num regions) |
| Inliers | *N/A* | Points fitting the RANSAC line model |
| Angle | From `cv2.fitLine` (integer precision) | From `cv2.fitLine` on inliers (sub-pixel precision) |
| Sub-pixel | NO | YES |

### Pipeline Visualization:

| OLD Method | NEW Method |
|-----------|------------|
| Binary threshold image (white/black) | Gradient magnitude heatmap |

### Profile Plot:

| OLD Method | NEW Method |
|-----------|------------|
| Edge pixel density distribution (how many Canny edge pixels at each position) | Gradient magnitude profile (median across image) with threshold line |

---

## 9. Experimental Results — Copy of LeftKW.png (Broken Wafer Edge)

This experiment tests both methods on a wafer image where the **wafer edge is broken/chipped** — some regions along the edge are damaged, missing, or irregular. This is a critical real-world test because broken edges are common in production environments.

### 9.1 OLD Method Result

| Parameter | Value |
|-----------|-------|
| **Threshold** | 17 |
| **Edge Pixels** | 1,617 |
| **Angle** | 172.487° |
| **Sub-pixel** | NO |

**Observation:**
- With a low threshold of 17, the binary threshold step turns almost the entire image white. Canny then finds edges **everywhere** — on the real wafer edge, on scratches, on debris, and on the broken/chipped sections.
- The **Edge Pixel Distribution** plot shows spikes at multiple X positions (around X=600–700 and a large spike near X=950–1000), confirming that Canny detected edges in many different locations, not just the wafer edge.
- `cv2.fitLine` (Least Squares) fits a single line through **all 1,617 edge pixels**. Since many of those pixels come from the broken sections and noise, the fitted line is **pulled away** from the true edge position.
- **No outlier rejection** — every detected pixel has equal weight, including pixels from the broken/chipped areas. The broken sections corrupt the line fitting result.

**Verdict:** ❌ The OLD method **cannot distinguish** between real edge pixels and noise from the broken edge. The fitted line is inaccurate.

---

### 9.2 NEW Method Result

| Parameter | Value |
|-----------|-------|
| **Threshold** | 172 |
| **Points Detected** | 34 out of 40 regions |
| **Inliers** | 22 |
| **Outliers Rejected** | 12 |
| **Angle** | 174.172° |
| **Sub-pixel** | YES |

**Observation:**
- The gradient threshold of 172 filters out weak edges, keeping only the **strongest gradient transitions** corresponding to the actual wafer edge.
- Out of 40 scan regions, 34 found a valid edge point. Of those 34 points, some fall on the **broken/chipped sections** of the edge where the edge position is irregular or shifted.
- **RANSAC identifies the 22 points that lie on the true edge line** and automatically **rejects the 12 points** from the broken/damaged sections as outliers.
- The final line is fitted using **only the 22 inlier points**, producing an accurate edge line that represents the **true wafer edge position**, ignoring the broken sections entirely.

**Verdict:** ✅ **Even though the wafer edge is broken, the NEW method still successfully finds the correct edge line.** RANSAC treats the broken/chipped sections as outliers and fits the line through only the intact edge points.

---

### 9.3 Why the NEW Method Works on Broken Edges

The key insight is that a broken wafer edge creates **local disruptions** — only a few scan regions are affected. The majority of regions still detect the correct edge position. RANSAC exploits this by:

1. **Broken edge sections** → Edge points in these regions are shifted or missing → RANSAC rejects them as outliers
2. **Intact edge sections** → Edge points align on the true edge line → RANSAC keeps them as inliers
3. **Final fit** → Only the intact-edge inlier points contribute to the line → Accurate result

```
Wafer Edge (with break):

    ●──●──●──●──╳──╳──╳──●──●──●──●──●──●
    inlier      outlier (broken)     inlier
                rejected by RANSAC

    Result: Line fitted through ● points only → correct edge
```

This is a **major advantage** over the OLD method, which has no concept of inliers/outliers and treats every edge pixel equally — including pixels from the broken sections.

| Scenario | OLD Method | NEW Method |
|----------|-----------|------------|
| Clean edge | ✅ Works | ✅ Works |
| Edge with scratches | ❌ Line pulled by scratches | ✅ Scratches rejected as outliers |
| **Broken/chipped edge** | ❌ **Line corrupted by broken sections** | ✅ **Broken sections rejected, edge still found** |
| Edge with debris | ❌ Debris detected as false edges | ✅ Debris points rejected as outliers |

---

## 10. Conclusion

The **NEW method** (Gradient + RANSAC) is recommended for production wafer alignment because:

1. ✅ **Handles broken/chipped wafer edges** — RANSAC rejects damaged sections and still finds the true edge line from intact regions
2. ✅ **Sub-pixel accuracy** — critical for precise wafer positioning
3. ✅ **Outlier rejection** — RANSAC automatically handles noise, defects, and edge damage
4. ✅ **Controlled point sampling** — one point per region avoids bias from dense edge clusters
5. ✅ **Direction-aware** — correctly selects the appropriate edge cluster for each scan direction

The **OLD method** (Canny + fitLine) remains useful as a **baseline comparison** and for visual debugging of the binary threshold step, but it **cannot handle broken or damaged wafer edges** reliably.

---

## 10. How to Run the Comparison Tool

```bash
python edge_compare.py
```

1. Select **METHOD** (NEW or OLD) using the radio buttons
2. Select **DIRECTION** (LEFT, RIGHT, TOP, BOTTOM)
3. Adjust **sliders** to tune parameters in real-time
4. Click **Load Image** to test with different wafer images
5. Click **Reset** to restore default parameter values
