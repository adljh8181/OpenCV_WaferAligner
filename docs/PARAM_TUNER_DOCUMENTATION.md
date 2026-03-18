# Interactive Parameter Tuner Documentation

## Overview

The `param_tuner.py` is an **interactive calibration tool** that wraps the `edge_finder.py` pipeline to allow real-time parameter adjustment with live visual feedback. It uses matplotlib sliders to dynamically modify edge detection parameters and immediately visualize the impact on detection results.

---

## Relationship to Edge Finder

| Component | `edge_finder.py` | `param_tuner.py` |
|-----------|------------------|------------------|
| **Purpose** | Core edge detection algorithm | Interactive GUI for parameter tuning |
| **Configuration** | Fixed values in `EdgeFinderConfig` | Dynamic values from sliders |
| **Output** | Edge line detection results | Live preview with adjustable parameters |
| **Use Case** | Production/programmatic use | Development/calibration phase |

> **Key Point**: Both files use the **exact same pipeline** (`EdgeLineFinder.find_edge()`). The only difference is that `param_tuner.py` allows you to adjust parameters in real-time via sliders.

---

## Edge Detection Pipeline - Detailed Code Walkthrough

The complete pipeline flows through these stages:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE EDGE DETECTION PIPELINE                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   STAGE 1   │───▶│   STAGE 2   │───▶│   STAGE 3   │───▶│   STAGE 4   │   │
│  │  Preprocess │    │  Classify   │    │  Gradient   │    │   Detect    │   │
│  │    Image    │    │    FOV      │    │ Computation │    │   Points    │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Load image  │    │ Is EDGE_FOV │    │Create kernel│    │  Find peak  │   │
│  │ Scale to    │    │     ?       │    │ Convolve    │    │  per region │   │
│  │  1000px     │    │ ✓ Continue  │    │   profile   │    │  Sub-pixel  │   │
│  └─────────────┘    │ ✗ Abort     │    └─────────────┘    │  refinement │   │
│                     └─────────────┘                       └─────────────┘   │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐                                         │
│  │   STAGE 5   │───▶│   STAGE 6   │                                         │
│  │   RANSAC    │    │  Visualize  │                                         │
│  │ Line Fitting│    │   Results   │                                         │
│  └─────────────┘    └─────────────┘                                         │
│         │                  │                                                 │
│         ▼                  ▼                                                 │
│  ┌─────────────┐    ┌─────────────┐                                         │
│  │Filter points│    │ Draw line,  │                                         │
│  │Find inliers │    │   points,   │                                         │
│  │  Fit line   │    │  gradient   │                                         │
│  └─────────────┘    └─────────────┘                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## STAGE 1: Image Preprocessing

**File**: `fov_classifier.py` → `preprocess_image()` (Lines 89-116)

### What It Does
Loads and scales the image to a consistent size for processing.

### Code
```python
def preprocess_image(img_or_path, target_dim=1000):
    """
    Load and preprocess image for classification.
    Returns: Tuple of (processed_img, original_img, scale)
    """
    # Load image as grayscale
    if isinstance(img_or_path, str):
        original_img = cv2.imread(img_or_path, 0)  # 0 = grayscale
    else:
        original_img = img_or_path.copy()

    # Calculate scale factor
    h_orig, w_orig = original_img.shape
    scale = target_dim / max(h_orig, w_orig)
    
    # Resize to target dimension
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    processed_img = cv2.resize(original_img, (new_w, new_h), 
                               interpolation=cv2.INTER_AREA)

    return processed_img, original_img, scale
```

### Data Flow
```
Input Image (e.g., 4000x3000 pixels)
         │
         ▼
    Scale Factor = 1000 / 4000 = 0.25
         │
         ▼
Output Image (1000x750 pixels, grayscale)
```

### Why Scale?
- **Consistency**: All images processed at same resolution
- **Speed**: Smaller images = faster processing
- **Memory**: Less RAM usage

---

## STAGE 2: FOV Classification

**File**: `fov_classifier.py` → `FOVClassifier.classify()` (Lines 137-177)

### What It Does
Determines if the image contains a wafer edge (EDGE_FOV) or not.

### Code
```python
def classify(self, img_or_path):
    # Preprocess image
    img, original_img, scale = preprocess_image(img_or_path, 
                                                 self.config.TARGET_PROCESS_DIM)

    # Detect edge using intensity profile analysis
    edge_result = self._detect_edge(img)

    # Final classification
    if edge_result['has_edge']:
        fov_type = "EDGE_FOV"     # ✓ Proceed with edge finding
    else:
        fov_type = "DIE_FOV" or "WAFER_FOV"  # ✗ Abort

    return {
        'fov_type': fov_type,
        'edge': edge_result,
        ...
    }
```

### Edge Detection Logic (`_detect_edge`)
```python
def _detect_edge(self, img):
    # 1. Extract horizontal intensity profile from multiple bands
    bands = [
        img[int(h*0.2):int(h*0.35), :],   # Top-middle
        img[int(h*0.35):int(h*0.5), :],   # Upper-middle
        img[int(h*0.5):int(h*0.65), :],   # Lower-middle
        img[int(h*0.65):int(h*0.8), :]    # Bottom-middle
    ]
    avg_profile = np.mean([np.mean(band, axis=0) for band in bands], axis=0)
    
    # 2. Smooth and calculate gradient
    smoothed = gaussian_filter1d(avg_profile, sigma=30)
    gradient = np.gradient(smoothed)
    
    # 3. Find rising/falling regions (edge transition)
    rising_region, falling_region = self._find_monotonic_regions(gradient)
    
    # 4. Classify edge type based on intensity change and region length
    if rising_change >= 30 and rising_ratio >= 0.15:
        edge_type = "RIGHT_EDGE"  # Wafer on LEFT side
        wafer_side = "LEFT"
```

### Decision Criteria
| Criterion | Condition | Edge Type |
|-----------|-----------|-----------|
| Curved Rising | Rising ≥15% width, ΔIntensity ≥30 | RIGHT_EDGE |
| Curved Falling | Falling ≥15% width, ΔIntensity ≥30 | LEFT_EDGE |
| Sharp Rising | Rising ≥5% width, ΔIntensity ≥50 | RIGHT_EDGE |
| Sharp Falling | Falling ≥5% width, ΔIntensity ≥50 | LEFT_EDGE |
| L-R Difference | Left-Right ≥40 intensity units | Based on direction |

---

## STAGE 3: Gradient Kernel Creation

**File**: `fov_classifier.py` → `create_gradient_kernel()` (Lines 55-86)

### What It Does
Creates a convolution kernel to detect intensity transitions (edges).

### Code
```python
def create_gradient_kernel(size):
    """
    Creates a gradient kernel with specified window size.
    
    Examples:
        size=3:  [-1, 0, 1]
        size=5:  [-1, -1, 0, 1, 1]
        size=7:  [-1, -1, -1, 0, 1, 1, 1]
        size=9:  [-1, -1, -1, -1, 0, 1, 1, 1, 1]
    """
    # Ensure size is odd and at least 3
    size = max(3, size)
    if size % 2 == 0:
        size += 1
    
    # Calculate half width
    half_width = size // 2
    
    # Create kernel: [-1, -1, ..., 0, ..., 1, 1]
    kernel = np.concatenate([
        -np.ones(half_width),   # Left side: negative
        np.zeros(1),            # Center: zero
        np.ones(half_width)     # Right side: positive
    ])
    
    return kernel
```

### Visual Explanation
```
Kernel Size = 7:  [-1, -1, -1, 0, 1, 1, 1]

                    ◄── Left side ──►   ◄── Right side ──►
                    (darker values)     (brighter values)
                           ▼                   ▼
Image Profile:    [..., 50, 50, 50, 100, 200, 200, 200, ...]
                              │
                              ▼
Convolution:      (-1×50) + (-1×50) + (-1×50) + (0×100) + (1×200) + (1×200) + (1×200)
                = -150 + 0 + 600 = 450    ← HIGH GRADIENT = EDGE FOUND!
```

### Effect of Kernel Size
| Kernel Size | Formula | Effect |
|-------------|---------|--------|
| 3 | `[-1, 0, 1]` | Very sensitive, detects small edges, more noise |
| 7 | `[-1,-1,-1,0,1,1,1]` | Balanced, good noise resistance (DEFAULT) |
| 15 | `[-1×7, 0, 1×7]` | Very smooth, only detects broad edges, resistant to noise |

---

## STAGE 4: Edge Point Detection Per Region

**File**: `edge_finder.py` → `EdgeLineFinder.find_edge()` (Lines 160-259)

### What It Does
Divides the image into horizontal regions and finds the edge point in each region.

### Code
```python
def find_edge(self, img_or_path, edge_info=None):
    # Preprocess
    img, original_img, scale = preprocess_image(img_or_path, 
                                                 self.config.TARGET_PROCESS_DIM)
    h, w = img.shape
    
    # Calculate border to ignore (avoid false edges at image boundaries)
    border_ignore = int(w * cfg.BORDER_IGNORE_PCT) + cfg.KERNEL_SIZE
    
    # Calculate region height
    region_h = h // cfg.NUM_REGIONS  # e.g., 1000 / 40 = 25 pixels per region
    
    detected_points = []
    
    # Process each horizontal region
    for i in range(cfg.NUM_REGIONS):
        y_start = i * region_h
        y_end = (i + 1) * region_h
        
        # 1. Extract horizontal slice
        region = img[y_start:y_end, :]
        
        # 2. Get median profile (reduces noise)
        profile = np.median(region, axis=0)
        
        # 3. Calculate gradient using kernel convolution
        gradient = np.convolve(profile, self.kernel, mode='same')
        abs_gradient = np.abs(gradient)
        
        # 4. Ignore borders
        abs_gradient[:border_ignore] = 0
        abs_gradient[-border_ignore:] = 0
        
        # 5. Detect edge point in this region
        result = self._detect_edge_point(abs_gradient, profile, y_start, region_h)
        
        if result['found']:
            detected_points.append((result['x'], result['y']))
```

### `_detect_edge_point()` (Lines 261-305)
```python
def _detect_edge_point(self, abs_gradient, profile, y_start, region_h, edge_info=None):
    cfg = self.config
    
    # 1. Find pixels where gradient exceeds threshold
    potential_indices = np.where(abs_gradient > cfg.EDGE_THRESHOLD)[0]
    
    if len(potential_indices) == 0:
        return {'found': False}  # No edge in this region
    
    # 2. Find clusters (groups of adjacent edge pixels)
    jumps = np.where(np.diff(potential_indices) > cfg.MAX_CLUSTER_GAP)[0]
    
    # 3. Take first cluster (usually the real edge)
    if len(jumps) > 0:
        edge_cluster = potential_indices[:jumps[0]+1]
    else:
        edge_cluster = potential_indices
    
    # 4. Find peak within cluster
    peak_int_x = edge_cluster[np.argmax(abs_gradient[edge_cluster])]
    
    # 5. Sub-pixel refinement (parabola fitting)
    x_subpixel = refine_peak_subpixel(abs_gradient, peak_int_x)
    y = y_start + (region_h // 2)
    
    return {
        'found': True,
        'x': x_subpixel,
        'y': y,
        'x_int': peak_int_x,
        'gradient_value': abs_gradient[peak_int_x]
    }
```

### Visual: Region Processing
```
Image Height = 1000px, NUM_REGIONS = 40
Each region = 25px tall

┌────────────────────────────────────────────────────────────┐
│ Region 0  (y: 0-25)     →  Find edge point  →  (x=653, y=12)   │
├────────────────────────────────────────────────────────────┤
│ Region 1  (y: 25-50)    →  Find edge point  →  (x=654, y=37)   │
├────────────────────────────────────────────────────────────┤
│ Region 2  (y: 50-75)    →  Find edge point  →  (x=652, y=62)   │
├────────────────────────────────────────────────────────────┤
│    ...        ...                ...              ...           │
├────────────────────────────────────────────────────────────┤
│ Region 39 (y: 975-1000) →  Find edge point  →  (x=660, y=987)  │
└────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    40 detected points (x, y)
```

### Sub-Pixel Refinement (Lines 51-84)
```python
def refine_peak_subpixel(y_data, peak_idx):
    """
    Fits a parabola to 3 points around the peak to find
    the TRUE peak position between pixels.
    """
    # Get 3 points around peak
    left = y_data[peak_idx - 1]
    center = y_data[peak_idx]
    right = y_data[peak_idx + 1]
    
    # Parabola vertex formula: x = (left - right) / (2 * (left - 2*center + right))
    denom = left - 2 * center + right
    offset = (left - right) / (2 * denom)
    
    return peak_idx + offset  # e.g., 653.23 instead of 653
```

### Visual: Sub-Pixel Refinement
```
      Gradient Values
         │
    250 ─┼─────────────────●───────  ← True peak at 653.23
         │               ╱   ╲
    200 ─┼─────────────●─────●─────  ← Measured at pixels 652, 653, 654
         │           ╱         ╲
    150 ─┼─────────●─────────────●─
         │       ╱                 ╲
    100 ─┼─────●───────────────────●
         │   ╱
     50 ─┼──●
         │
      0 ─┼─────────────────────────────
         └───┬───┬───┬───┬───┬───┬───
            650 651 652 653 654 655 656
                          │
                          ▼
              Parabola fitting gives x = 653.23
```

---

## STAGE 5: RANSAC Line Fitting

**File**: `edge_finder.py` → `fit_line_ransac()` (Lines 87-137)

### What It Does
Fits a line through detected points while rejecting outliers.

### Code
```python
def fit_line_ransac(points, iterations=2000, threshold=5.0):
    """
    RANSAC = RANdom SAmple Consensus
    
    Finds the best line by:
    1. Randomly selecting 2 points
    2. Creating a line through them
    3. Counting how many other points are close to this line (inliers)
    4. Keeping the line with the most inliers
    """
    best_line = None
    best_inliers = []
    max_count = 0
    points_array = np.array(points)

    for _ in range(iterations):
        # 1. Random sample 2 points
        idx1, idx2 = random.sample(range(len(points)), 2)
        p1, p2 = points_array[idx1], points_array[idx2]

        # 2. Line equation: Ax + By + C = 0
        A = p1[1] - p2[1]   # y1 - y2
        B = p2[0] - p1[0]   # x2 - x1
        C = -A * p1[0] - B * p1[1]

        # 3. Calculate distance from each point to line
        norm = np.sqrt(A*A + B*B)
        distances = np.abs(A * points_array[:, 0] + B * points_array[:, 1] + C) / norm

        # 4. Count inliers (points within threshold distance)
        inlier_mask = distances < threshold
        current_inliers = points_array[inlier_mask]

        # 5. Keep best result
        if len(current_inliers) > max_count:
            max_count = len(current_inliers)
            best_inliers = current_inliers
            best_line = (A, B, C)

    return best_line, best_inliers
```

### RANSAC Visual
```
Iteration 1:                    Iteration 500:                 Iteration 2000 (BEST):
                                                               
   ●                               ●                              ●
    ╲                               ╲                              │
     ●                               ●                             ● ← Inlier
      ╲                               ╲                            │
       ●                               ●                           ● ← Inlier
        ╲    ○ ← Outlier                │   ○                      │   ○ ← Ignored
         ●                              ●                          ● ← Inlier
          ╲                              ╲                         │
           ●                              ●                        ● ← Inlier

  Inliers: 4                    Inliers: 5                    Inliers: 36 ✓ BEST!
```

### `_fit_edge_line()` (Lines 307-350)
```python
def _fit_edge_line(self, points, img_height):
    # Run RANSAC
    _, inliers = fit_line_ransac(
        points,
        cfg.RANSAC_ITERATIONS,   # 2000 iterations
        cfg.RANSAC_THRESHOLD     # 5.0 pixel tolerance
    )

    if len(inliers) < 3:
        return {'success': False, 'reason': 'RANSAC rejected points'}

    # Final line fitting with cv2.fitLine (uses all inliers)
    inliers_reshaped = np.array(inliers).reshape((-1, 1, 2)).astype(np.float32)
    line_result = cv2.fitLine(inliers_reshaped, cv2.DIST_L2, 0, 0.01, 0.01)
    
    vx, vy = line_result[0], line_result[1]  # Direction vector
    x0, y0 = line_result[2], line_result[3]  # Point on line
    
    # Calculate line endpoints at top and bottom of image
    x_top = int(x0 + (0 - y0) * (vx/vy))        # X at y=0
    x_bot = int(x0 + (img_height - y0) * (vx/vy))  # X at y=height

    return {
        'success': True,
        'endpoints': {'x_top': x_top, 'x_bot': x_bot},
        'inliers': inliers
    }
```

---

## STAGE 6: Visualization (param_tuner.py)

**File**: `param_tuner.py` → `update()` (Lines 109-218)

### What It Does
Draws the results on screen with live updates as sliders change.

### Code (Simplified)
```python
def update(self, val):
    # 1. Create custom config with current slider values
    class TunerConfig(EdgeFinderConfig):
        KERNEL_SIZE = int(self.sliders['KERNEL_SIZE'].val)
        EDGE_THRESHOLD = int(self.sliders['EDGE_THRESHOLD'].val)
        NUM_REGIONS = int(self.sliders['NUM_REGIONS'].val)
        BORDER_IGNORE_PCT = self.sliders['BORDER_IGNORE_PCT'].val
        RANSAC_THRESHOLD = self.sliders['RANSAC_THRESHOLD'].val

    # 2. Run edge finder with new config
    finder = EdgeLineFinder(TunerConfig())
    result = finder.find_edge(self.img)

    # 3. Draw results
    result_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
    
    if result['success']:
        # Draw edge line (GREEN)
        cv2.line(result_img, 
                 (endpoints['x_top'], 0), 
                 (endpoints['x_bot'], self.h),
                 (0, 255, 0), 3)
        
        # Draw all detected points (RED dots)
        for point in result['detected_points']:
            cv2.circle(result_img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
        
        # Draw inliers (CYAN circles)
        for point in result['inliers']:
            cv2.circle(result_img, (int(point[0]), int(point[1])), 6, (255, 255, 0), 2)

    # 4. Update display
    self.ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    self.fig.canvas.draw_idle()
```

### Output Legend
| Visual Element | Color | Meaning |
|----------------|-------|---------|
| Line | 🟢 Green | Detected edge line |
| Small dots | 🔴 Red | All detected points (40) |
| Large circles | 🟡 Cyan | Inlier points used for fitting |
| Vertical lines | ⬜ Gray | Border ignore zones |

---

## Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Wafer Image (4000x3000 px)                                          │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────┐                                        │
│  │ PREPROCESS                      │                                        │
│  │ • Scale to 1000px              │                                        │
│  │ • Convert to grayscale         │                                        │
│  └─────────────────────────────────┘                                        │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────┐                                        │
│  │ CLASSIFY FOV                    │                                        │
│  │ • Analyze intensity profile    │                                        │
│  │ • Detect edge transition       │                                        │
│  │ • Return: EDGE_FOV / DIE_FOV   │                                        │
│  └─────────────────────────────────┘                                        │
│               │                                                             │
│               ▼ (if EDGE_FOV)                                               │
│  ┌─────────────────────────────────┐                                        │
│  │ FOR EACH REGION (40 regions)   │◄─── KERNEL_SIZE affects gradient       │
│  │ • Extract horizontal slice     │                                        │
│  │ • Compute median profile       │                                        │
│  │ • Convolve with gradient kernel│                                        │
│  │ • Apply EDGE_THRESHOLD         │◄─── EDGE_THRESHOLD filters noise       │
│  │ • Find peak (sub-pixel)        │                                        │
│  │ • Add to detected_points[]     │◄─── NUM_REGIONS controls point count   │
│  └─────────────────────────────────┘                                        │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────┐                                        │
│  │ RANSAC LINE FITTING             │                                        │
│  │ • 2000 random iterations       │                                        │
│  │ • Find line with most inliers  │◄─── RANSAC_THRESHOLD controls          │
│  │ • cv2.fitLine() for final fit  │     inlier/outlier classification     │
│  └─────────────────────────────────┘                                        │
│               │                                                             │
│               ▼                                                             │
│  Output: Edge Line (x_top, x_bot) + Angle + Inlier Count                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Code Sections

### 1. Import Dependencies & Core Modules

```python
from fov_classifier import FOVClassifier, ClassificationConfig, create_gradient_kernel, preprocess_image
from edge_finder import EdgeLineFinder, EdgeFinderConfig
```

The tuner imports the core classes from both `fov_classifier.py` and `edge_finder.py`.

---

### 2. Dynamic Configuration (Lines 120-127)

When a slider value changes, a new configuration class is created dynamically:

```python
class TunerConfig(EdgeFinderConfig):
    KERNEL_SIZE = kernel_size                              # From slider
    EDGE_THRESHOLD = int(self.sliders['EDGE_THRESHOLD'].val)
    NUM_REGIONS = int(self.sliders['NUM_REGIONS'].val)
    BORDER_IGNORE_PCT = self.sliders['BORDER_IGNORE_PCT'].val
    RANSAC_THRESHOLD = self.sliders['RANSAC_THRESHOLD'].val

config = TunerConfig()
```

---

### 3. Execute Pipeline (Lines 129-131)

The same pipeline as `edge_finder.py` is executed:

```python
# Run edge finder
finder = EdgeLineFinder(config)
result = finder.find_edge(self.img)
```

---

### 4. Visualize Results (Lines 139-217)

Results are drawn on the image:
- **Green line**: Detected edge line
- **Red dots**: All detected edge points
- **Cyan circles**: Inlier points (used for line fitting)
- **Gray lines**: Border ignore zones

---

## Adjustable Parameters

### Default Values (from `EdgeFinderConfig`)

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `KERNEL_SIZE` | **7** | 3-15 | Size of gradient kernel (must be odd) |
| `EDGE_THRESHOLD` | **25** | 5-100 | Minimum gradient value to detect edge |
| `NUM_REGIONS` | **40** | 10-80 | Number of vertical scan regions |
| `BORDER_IGNORE_PCT` | **0.02** | 0.01-0.15 | Percentage of image width to ignore at borders |
| `RANSAC_THRESHOLD` | **5.0** | 1.0-20.0 | Distance threshold for RANSAC inliers |

---

## Parameter Impact Analysis

### Experiment: Effect of Kernel Size and Edge Threshold

#### Configuration 1: Default Values
| Parameter | Value |
|-----------|-------|
| Kernel Size | 7 |
| Edge Threshold | 25 |
| Scan Regions | 40 |
| Border % | 0.02 |
| RANSAC Thresh | 5 |

**Result:**
- Points Detected: 40
- Inliers (used): **22**
- Line: **653 → 660**
- Angle: **0.46°**

#### Configuration 2: Adjusted Values
| Parameter | Value |
|-----------|-------|
| Kernel Size | **15** |
| Edge Threshold | **100** |
| Scan Regions | 40 |
| Border % | 0.02 |
| RANSAC Thresh | 5 |

**Result:**
- Points Detected: 40
- Inliers (used): **36**
- Line: **678 → 678**
- Angle: **180.00°** (perfectly vertical)

---

### Parameter Impact Summary

| Parameter | Low Value Effect | High Value Effect |
|-----------|------------------|-------------------|
| **Kernel Size** | More sensitive to noise, local edges | Smoother gradient, detects broader edges |
| **Edge Threshold** | Detects weak edges (more noise) | Only strong edges pass (less noise) |
| **Scan Regions** | Fewer detection points | More detection points for better fitting |
| **Border %** | Uses more edge pixels | Ignores more border area |
| **RANSAC Threshold** | Stricter inlier selection | More tolerant of outliers |

---

## Why Results Differ Between Configurations

### 1. Kernel Size Effect (7 → 15)

```
Kernel Size = 7:  Smaller window → Detects local gradient changes
                  More sensitive to noise and small variations

Kernel Size = 15: Larger window → Averages over wider area
                  Smoother gradient, less noise, more consistent detection
```

### 2. Edge Threshold Effect (25 → 100)

```
Threshold = 25:   Low bar → Many gradient peaks pass
                  May include noise and secondary edges

Threshold = 100:  High bar → Only strongest gradients pass
                  Filters out noise, keeps only true edge
```

### 3. Combined Effect

| Config 1 (7, 25) | Config 2 (15, 100) |
|------------------|---------------------|
| 22/40 inliers | 36/40 inliers |
| Line: 653 → 660 | Line: 678 → 678 |
| Angle: 0.46° (slight tilt) | Angle: 180° (perfectly vertical) |

**Conclusion**: Larger kernel + Higher threshold = **Cleaner edge detection** with more consistent inliers and a more accurate vertical line.

---

## How to Use the Tuner

### Basic Usage

```bash
# Use default image
python param_tuner.py

# Use custom image
python param_tuner.py path/to/image.png
```

### Workflow

1. **Run the tuner** → Opens matplotlib window with sliders
2. **Adjust sliders** → Observe live changes in edge detection
3. **Note optimal values** → Record parameter values that give best results
4. **Update `EdgeFinderConfig`** → Apply optimal values to `edge_finder.py` for production

---

## Output Visualization

The tuner displays four key elements:

| Area | Content |
|------|---------|
| **Main Image** | Processed image with edge line, detected points, and inliers |
| **Gradient Profile** | Gradient values for middle region with threshold line |
| **Status Box** | Detection status, point count, line endpoints, angle |
| **Sliders** | Interactive parameter controls |

### Legend

| Visual | Meaning |
|--------|---------|
| 🟢 Green Line | Detected edge line |
| 🔴 Red Dots | All detected edge points |
| 🟡 Cyan Circles | Inlier points (used for fitting) |
| ⬜ Gray Lines | Border ignore zones |
| 🟠 Orange Line | Edge threshold level |

---

## File Structure

```
EmguFindEdge/
├── fov_classifier.py     # FOV classification (shared functions)
├── edge_finder.py        # Core edge detection (EdgeLineFinder class)
├── param_tuner.py        # Interactive parameter tuning GUI
├── main.py               # Main entry point
└── docs/
    └── PARAM_TUNER_DOCUMENTATION.md  # This documentation
```

---

## Conclusion

The `param_tuner.py` is a development-time tool designed to help calibrate the edge detection parameters. Once optimal values are found through experimentation:

1. Update `EdgeFinderConfig` in `edge_finder.py` with the new values
2. Run `edge_finder.py` or `main.py` for production use

This separation allows for:
- **Rapid experimentation** without modifying production code
- **Visual validation** of parameter effects
- **Consistent results** once parameters are finalized
