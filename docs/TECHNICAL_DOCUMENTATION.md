# Technical Documentation: Wafer Edge Detection System

## Module Overview

This document provides technical API documentation for the wafer edge detection system, covering the interactive parameter tuner, edge finding algorithms, and configuration classes.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         param_tuner.py                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │  InteractiveParameterTuner                                      │   │ │
│  │  │  • Matplotlib GUI with sliders                                  │   │ │
│  │  │  • Real-time parameter adjustment                               │   │ │
│  │  │  • Live visualization of results                                │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────┬─────────────────────────────────────┘ │
│                                      │ imports                               │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          edge_finder.py                                 │ │
│  │  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │ │
│  │  │  EdgeFinderConfig     │  │  EdgeLineFinder                       │  │ │
│  │  │  • NUM_REGIONS        │  │  • find_edge()                        │  │ │
│  │  │  • EDGE_THRESHOLD     │  │  • _detect_edge_point()               │  │ │
│  │  │  • RANSAC_THRESHOLD   │  │  • _fit_edge_line()                   │  │ │
│  │  │  • BORDER_IGNORE_PCT  │  │  • visualize()                        │  │ │
│  │  └───────────────────────┘  └───────────────────────────────────────┘  │ │
│  │  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │ │
│  │  │ refine_peak_subpixel  │  │  fit_line_ransac                      │  │ │
│  │  │ (helper function)     │  │  (helper function)                    │  │ │
│  │  └───────────────────────┘  └───────────────────────────────────────┘  │ │
│  └───────────────────────────────────┬─────────────────────────────────────┘ │
│                                      │ imports                               │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         fov_classifier.py                               │ │
│  │  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │ │
│  │  │  ClassificationConfig │  │  FOVClassifier                        │  │ │
│  │  │  • KERNEL_SIZE        │  │  • classify()                         │  │ │
│  │  │  • TARGET_PROCESS_DIM │  │  • _detect_edge()                     │  │ │
│  │  └───────────────────────┘  └───────────────────────────────────────┘  │ │
│  │  ┌───────────────────────┐  ┌───────────────────────────────────────┐  │ │
│  │  │ create_gradient_kernel│  │  preprocess_image                     │  │ │
│  │  └───────────────────────┘  └───────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## File: `param_tuner.py`

### Module Description
Interactive GUI application for real-time tuning of wafer edge detection parameters using matplotlib sliders.

### Dependencies
```python
import sys, os, cv2, numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from fov_classifier import FOVClassifier, ClassificationConfig, create_gradient_kernel, preprocess_image
from edge_finder import EdgeLineFinder, EdgeFinderConfig
```

---

### Class: `InteractiveParameterTuner`

**Purpose**: Provides a matplotlib-based GUI for adjusting edge detection parameters with live preview.

#### Constructor

```python
def __init__(self, image_path=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` or `None` | `None` | Path to image file. If `None`, uses default image. |

**Initialization Steps**:
1. Load `EdgeFinderConfig` as default configuration
2. Determine image path (default or provided)
3. Preprocess image using `preprocess_image()`
4. Setup matplotlib UI with sliders
5. Trigger initial `update(None)` to render first frame

**Attributes Created**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `EdgeFinderConfig` | Default configuration object |
| `image_path` | `str` | Path to loaded image |
| `img` | `ndarray` | Preprocessed grayscale image |
| `original_img` | `ndarray` | Original image before preprocessing |
| `scale` | `float` | Scale factor applied during preprocessing |
| `h, w` | `int` | Height and width of preprocessed image |
| `fig` | `Figure` | Matplotlib figure object |
| `ax_img` | `Axes` | Axes for image display |
| `ax_gradient` | `Axes` | Axes for gradient profile plot |
| `ax_status` | `Axes` | Axes for status summary |
| `sliders` | `dict` | Dictionary of slider widgets |
| `reset_button` | `Button` | Reset button widget |

---

#### Method: `setup_ui()`

```python
def setup_ui(self)
```

**Purpose**: Creates the matplotlib figure with sliders and display areas.

**UI Layout**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Wafer Edge Detection - Parameter Tuner                   │
├────────────────────────────────────────┬────────────────────────────────────┤
│                                        │  ┌──────────────────────────────┐  │
│                                        │  │ STATUS: ✓ SUCCESS            │  │
│                                        │  │ Points Detected: 40          │  │
│        Edge Detection Result           │  │ Inliers: 36                  │  │
│        (Main Image Display)            │  └──────────────────────────────┘  │
│                                        │                                    │
│                                        │  EDGE FINDER PARAMETERS            │
│                                        │  ├─ Kernel Size      [====  ] 7    │
│                                        │  ├─ Edge Threshold   [===   ] 25   │
│                                        │  ├─ Scan Regions     [=====] 40    │
│                                        │  ├─ Border %         [=     ] 0.02 │
│                                        │  └─ RANSAC Thresh    [===   ] 5.0  │
│                                        │                                    │
│                                        │        [ Reset ]                   │
├────────────────────────────────────────┴────────────────────────────────────┤
│                         Gradient Profile                                    │
│  200 ─┬───────────────▲─────────────────────────────────────────────────    │
│       │              ╱ ╲                                                    │
│  100 ─┤─────────────╱───╲────── Threshold = 25 ─────────────────────────    │
│       │            ╱     ╲                                                  │
│    0 ─┴───────────────────────────────────────────────────────────────────  │
│       0          200         400         600         800        1000        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Slider Configuration**:
| Slider Name | Min | Max | Default | Step | Label |
|-------------|-----|-----|---------|------|-------|
| `KERNEL_SIZE` | 3 | 15 | 7 | 2 | Kernel Size |
| `EDGE_THRESHOLD` | 5 | 100 | 25 | 1 | Edge Threshold |
| `NUM_REGIONS` | 10 | 80 | 40 | 5 | Scan Regions |
| `BORDER_IGNORE_PCT` | 0.01 | 0.15 | 0.02 | 0.01 | Border % |
| `RANSAC_THRESHOLD` | 1.0 | 20.0 | 5.0 | 0.5 | RANSAC Thresh |

---

#### Method: `update(val)`

```python
def update(self, val)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `val` | `any` | Slider value (not used directly, triggers refresh) |

**Purpose**: Called when any slider changes. Re-runs edge detection with current parameters.

**Algorithm**:
```python
# 1. Read current slider values
kernel_size = int(self.sliders['KERNEL_SIZE'].val)
if kernel_size % 2 == 0:
    kernel_size += 1  # Ensure odd

# 2. Create dynamic config class
class TunerConfig(EdgeFinderConfig):
    KERNEL_SIZE = kernel_size
    EDGE_THRESHOLD = int(self.sliders['EDGE_THRESHOLD'].val)
    NUM_REGIONS = int(self.sliders['NUM_REGIONS'].val)
    BORDER_IGNORE_PCT = self.sliders['BORDER_IGNORE_PCT'].val
    RANSAC_THRESHOLD = self.sliders['RANSAC_THRESHOLD'].val

# 3. Run edge finder
finder = EdgeLineFinder(TunerConfig())
result = finder.find_edge(self.img)

# 4. Visualize results
# ... (drawing code)
```

**Visualization Elements**:
| Element | Color (BGR) | Description |
|---------|-------------|-------------|
| Edge Line | `(0, 255, 0)` Green | Detected edge line from top to bottom |
| Detected Points | `(0, 0, 255)` Red | All 40 detected edge points |
| Inliers | `(255, 255, 0)` Cyan | Points used for final line fitting |
| Border Zones | `(100, 100, 100)` Gray | Areas ignored during detection |

---

#### Method: `reset(event)`

```python
def reset(self, event)
```

**Purpose**: Resets all sliders to default values from `EdgeFinderConfig`.

---

#### Method: `show()`

```python
def show(self)
```

**Purpose**: Displays the interactive tuner window. Blocks until window is closed.

---

## File: `edge_finder.py`

### Module Description
Core edge detection module providing the `EdgeLineFinder` class and configuration.

### Dependencies
```python
import cv2, numpy as np, matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import random, os
from fov_classifier import create_gradient_kernel, preprocess_image, FOVClassifier, ClassificationConfig
```

---

### Class: `EdgeFinderConfig`

**Purpose**: Configuration class for edge finding parameters. Extends `ClassificationConfig`.

**Inheritance**:
```
ClassificationConfig (fov_classifier.py)
        │
        └──▶ EdgeFinderConfig (edge_finder.py)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUM_REGIONS` | `int` | `40` | Number of horizontal scan regions to divide image |
| `EDGE_THRESHOLD` | `int` | `25` | Minimum gradient magnitude to consider as edge |
| `MAX_CLUSTER_GAP` | `int` | `5` | Maximum pixel gap between points in same cluster |
| `BORDER_IGNORE_PCT` | `float` | `0.02` | Percentage of image width to ignore at borders (2%) |
| `RANSAC_ITERATIONS` | `int` | `2000` | Number of RANSAC iterations for line fitting |
| `RANSAC_THRESHOLD` | `float` | `5.0` | Maximum distance (pixels) for point to be inlier |

**Inherited from `ClassificationConfig`**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IMAGE_FOLDER` | `str` | `'Images/LEFT_EDGE_PNG'` | Default image folder path |
| `DEFAULT_IMAGE` | `str` | `'Copy of sample_wafer.jpg'` | Default image filename |
| `TARGET_PROCESS_DIM` | `int` | `1000` | Target dimension for image scaling |
| `KERNEL_SIZE` | `int` | `7` | Gradient kernel window size |

---

### Class: `EdgeLineFinder`

**Purpose**: Finds precise edge lines in wafer images using gradient-based detection and RANSAC line fitting.

#### Constructor

```python
def __init__(self, config=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `EdgeFinderConfig` or `None` | `None` | Configuration object. Uses default if `None`. |

**Attributes Created**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `EdgeFinderConfig` | Active configuration |
| `kernel` | `ndarray` | Gradient convolution kernel |
| `classifier` | `FOVClassifier` | FOV classifier instance |

---

#### Method: `find_edge(img_or_path, edge_info=None)`

```python
def find_edge(self, img_or_path, edge_info=None) -> dict
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_or_path` | `str` or `ndarray` | - | Image file path or grayscale numpy array |
| `edge_info` | `dict` or `None` | `None` | Optional edge hints from FOVClassifier |

**Returns**: `dict` with edge detection results

**Return Dictionary Structure**:
```python
{
    'success': bool,              # True if edge found successfully
    'line_params': {              # Line equation parameters (if success)
        'a': float,               # Line equation: ax + by + c = 0
        'b': float,
        'c': float,
        'vx': float,              # Direction vector x
        'vy': float               # Direction vector y
    },
    'line_endpoints': {           # Line endpoints (if success)
        'x_top': int,             # X coordinate at y=0
        'x_bot': int              # X coordinate at y=height
    },
    'detected_points': list,      # List of (x, y) tuples for all detected points
    'inliers': list,              # List of inlier points after RANSAC
    'num_points': int,            # Total number of detected points
    'num_inliers': int,           # Number of inliers used for fitting
    'region_data': list,          # Per-region detection data
    'edge_info': dict,            # Edge classification info from FOVClassifier
    'fov_type': str,              # 'EDGE_FOV', 'DIE_FOV', or 'WAFER_FOV'
    'image': ndarray,             # Preprocessed image
    'original_image': ndarray,    # Original image
    'scale': float,               # Scale factor
    'reason': str                 # Failure reason (if not success)
}
```

**Algorithm**:
1. **Preprocess**: Load and scale image to `TARGET_PROCESS_DIM`
2. **Classify FOV**: Run `FOVClassifier.classify()` to check if EDGE_FOV
3. **Region Detection**: For each of `NUM_REGIONS` horizontal bands:
   - Extract region slice
   - Compute median intensity profile
   - Convolve with gradient kernel
   - Apply `EDGE_THRESHOLD` filter
   - Find peak gradient location
   - Apply sub-pixel refinement
4. **RANSAC Fitting**: Fit line through detected points
5. **Final Fit**: Use `cv2.fitLine()` on inliers

---

#### Method: `_detect_edge_point(abs_gradient, profile, y_start, region_h, edge_info=None)`

```python
def _detect_edge_point(self, abs_gradient, profile, y_start, region_h, edge_info=None) -> dict
```

**Purpose**: Detects edge point in a single horizontal region.

| Parameter | Type | Description |
|-----------|------|-------------|
| `abs_gradient` | `ndarray` | Absolute gradient values for region |
| `profile` | `ndarray` | Intensity profile of region |
| `y_start` | `int` | Starting Y coordinate of region |
| `region_h` | `int` | Height of region in pixels |
| `edge_info` | `dict` | Optional edge hints |

**Returns**:
```python
{
    'found': bool,           # True if edge point detected
    'x': float,              # Sub-pixel X coordinate
    'y': int,                # Y coordinate (center of region)
    'x_int': int,            # Integer X coordinate
    'gradient_value': float  # Gradient magnitude at peak
}
```

---

#### Method: `_fit_edge_line(points, img_height)`

```python
def _fit_edge_line(self, points, img_height) -> dict
```

**Purpose**: Fits a line to detected points using RANSAC algorithm.

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `list` | List of (x, y) tuples |
| `img_height` | `int` | Image height for endpoint calculation |

**Returns**:
```python
{
    'success': bool,
    'line_params': dict,      # {a, b, c, vx, vy}
    'endpoints': dict,        # {x_top, x_bot}
    'inliers': list,
    'reason': str             # If failed
}
```

---

#### Method: `visualize(result, save_path=None)`

```python
def visualize(self, result, save_path=None)
```

**Purpose**: Visualizes edge detection result with 4-panel display.

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `dict` | Result dictionary from `find_edge()` |
| `save_path` | `str` or `None` | Optional path to save figure |

---

#### Method: `print_results(result)`

```python
def print_results(self, result)
```

**Purpose**: Prints edge detection results to console.

---

### Helper Function: `refine_peak_subpixel(y_data, peak_idx)`

```python
def refine_peak_subpixel(y_data, peak_idx) -> float
```

**Purpose**: Refines integer peak position to sub-pixel accuracy using parabola fitting.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_data` | `ndarray` | 1D array of values |
| `peak_idx` | `int` | Integer index of detected peak |

**Returns**: `float` - Sub-pixel peak position

**Algorithm**:
```
Given 3 points: (peak_idx-1, left), (peak_idx, center), (peak_idx+1, right)

Parabola vertex formula:
    offset = (left - right) / (2 * (left - 2*center + right))
    
Sub-pixel position = peak_idx + offset
```

---

### Helper Function: `fit_line_ransac(points, iterations=1000, threshold=5.0)`

```python
def fit_line_ransac(points, iterations=1000, threshold=5.0) -> tuple
```

**Purpose**: Fits a line to points using RANSAC algorithm.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | `list` | - | List of (x, y) tuples |
| `iterations` | `int` | `1000` | Number of RANSAC iterations |
| `threshold` | `float` | `5.0` | Inlier distance threshold |

**Returns**: `tuple` of `(line_params, inlier_points)`
- `line_params`: `(A, B, C)` for line equation `Ax + By + C = 0`
- `inlier_points`: `ndarray` of inlier coordinates

**Algorithm**:
```
for iteration in range(iterations):
    1. Randomly select 2 points
    2. Compute line equation through points
    3. Calculate distance of all points to line
    4. Count inliers (distance < threshold)
    5. Keep best line (most inliers)
```

---

## Usage Examples

### Example 1: Basic Parameter Tuner Usage

```python
from param_tuner import InteractiveParameterTuner

# Use default image
tuner = InteractiveParameterTuner()
tuner.show()

# Use custom image
tuner = InteractiveParameterTuner("path/to/wafer_image.png")
tuner.show()
```

### Example 2: Programmatic Edge Finding

```python
from edge_finder import EdgeLineFinder, EdgeFinderConfig

# Create custom config
class MyConfig(EdgeFinderConfig):
    KERNEL_SIZE = 15
    EDGE_THRESHOLD = 100
    NUM_REGIONS = 60

# Find edge
finder = EdgeLineFinder(MyConfig())
result = finder.find_edge("path/to/image.png")

if result['success']:
    print(f"Edge X: {result['line_endpoints']['x_top']} → {result['line_endpoints']['x_bot']}")
    print(f"Inliers: {result['num_inliers']}/{result['num_points']}")
else:
    print(f"Failed: {result['reason']}")
```

### Example 3: Batch Processing Multiple Images

```python
from edge_finder import EdgeLineFinder, EdgeFinderConfig
import os

config = EdgeFinderConfig()
finder = EdgeLineFinder(config)

image_folder = "Images/LEFT_EDGE_PNG"
results = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg')):
        path = os.path.join(image_folder, filename)
        result = finder.find_edge(path)
        results.append({
            'file': filename,
            'success': result['success'],
            'x_top': result['line_endpoints']['x_top'] if result['success'] else None,
            'inliers': result['num_inliers'] if result['success'] else 0
        })

# Summary
for r in results:
    status = "✓" if r['success'] else "✗"
    print(f"{status} {r['file']}: x={r['x_top']}, inliers={r['inliers']}")
```

---

## Error Handling

### Common Failure Reasons

| Reason | Description | Solution |
|--------|-------------|----------|
| `Not an edge FOV` | Image classified as DIE_FOV or WAFER_FOV | Ensure image contains visible wafer edge |
| `Not enough points` | Less than 3 edge points detected | Lower `EDGE_THRESHOLD` or increase `NUM_REGIONS` |
| `RANSAC rejected points` | Line fitting failed | Adjust `RANSAC_THRESHOLD` or check edge quality |
| `Vertical line error` | Line fitting resulted in undefined slope | Check image orientation |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-03 | Initial release with basic edge detection |
| 1.2 | 2026-02-06 | Simplified param_tuner with focused parameters |

---

## Author
Auto-generated for Wafer Alignment System
