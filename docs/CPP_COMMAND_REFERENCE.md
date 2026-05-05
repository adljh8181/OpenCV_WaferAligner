# ZMQ Server — C++ Command Reference

All messages are **plain UTF-8 text strings** (not JSON).  
Paths containing spaces **must** be wrapped in `"double quotes"`.  
The server listens on **TCP port 5555** (REQ-REP pattern).

---

## Command Quick Reference

| # | Command | Description |
|---|---------|-------------|
| 1 | `PING` | Health check |
| 2 | `LOADR_REQ` | Load recipe |
| 3 | `PM_REQ` | Pattern match |
| 4 | `TRAIN_REQ` | Train (triggers UI) |
| 5 | `TEACH_REQ` | Teach template (interactive) |
| 6 | `WAFER_EDGE_REQ` | Wafer edge detection |
| 7 | `START_AUTOFOCUS_REQ` | Start autofocus session |
| 8 | `FOCUS_REQ` | Autofocus step |
| 9 | `FM_INDEX_REQ` | Focus measure index (one-shot) |
| 10 | `WAFER_ANGLE_REQ` | Find wafer angle |
| 11 | `WAFER_CENTER_REQ` | Calculate wafer centre |
| 12 | `WAFER_NOTCH_REQ` | Find wafer notch edge |
| 13 | `WAFER_TEMPLATE_REQ` | Find wafer template |
| 14 | `SHUTDOWN` | Shut down server |

---

## 1. PING

Health check. Use this to verify the server is alive before sending real commands.

**Send:**
```
PING
```

**Response:**
```json
{"status": "pong"}
```

---

## 2. LOADR_REQ — Load Recipe

Loads a recipe XML and applies all its parameters (pattern, edge, autofocus).  
Must be called before `PM_REQ`, `WAFER_EDGE_REQ`, or `WAFER_TEMPLATE_REQ` if the recipe has not been loaded yet.

**Send:**
```
LOADR_REQ "TKX2"
```
or with a full path:
```
LOADR_REQ "C:\recipes\TKX2.xml"
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `recipe` | Recipe name (short) or full `.xml` path |

**Response (success):**
```json
{"status": "ok", "recipe": "TKX2"}
```

**Response (error):**
```json
{"status": "error", "message": "Recipe file not found: TKX2"}
```

---

## 3. PM_REQ — Pattern Match

Runs template matching on a live image. Automatically loads the recipe if not already loaded.

**Send:**
```
PM_REQ "C:\Images\live.png" "TKX2"
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to the live camera image |
| 1 | `recipe` | Recipe name or full `.xml` path |

**Response (match found):**
```json
{
  "status": "ok",
  "x": 512.0,
  "y": 310.0,
  "angle": -2.5,
  "score": 87.3,
  "delta_x": 12.0,
  "delta_y": -5.0
}
```

**Response (no match):**
```json
{"status": "no_match"}
```

**Response (error):**
```json
{"status": "error", "message": "..."}
```

---

## 4. TRAIN_REQ — Train

Signals the Python UI to trigger the training workflow for the given image and recipe.  
Does **not** perform actual training on the server; it triggers the UI callback.

**Send:**
```
TRAIN_REQ "C:\Images\live.png" "TKX2"
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to the image |
| 1 | `recipe` | Recipe name or full `.xml` path |

**Response (success):**
```json
{"status": "ok", "message": "train_reply OK"}
```

**Response (image not found):**
```json
{"status": "error", "message": "Image not found: C:\Images\live.png"}
```

---

## 5. TEACH_REQ — Teach Template (Interactive)

Opens interactive OpenCV windows on the **Python machine** so the operator can:
1. Drag-select the template crop region
2. Draw a detection mask polygon (optional)

The server saves the template PNG, updates the recipe XML, and reloads the matcher immediately.

> **Note:** This command blocks until the operator finishes. Use a long timeout (e.g. 5 minutes).

**Send:**
```
TEACH_REQ "C:\Images\live.png" "TKX2"
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to the source image |
| 1 | `recipe` | Recipe name or full `.xml` path |

**Response (success):**
```json
{
  "status": "ok",
  "template_path": "C:\\recipes\\TKX2_template.png",
  "mask_path": "C:\\recipes\\TKX2_template_mask.png",
  "crop_cx": 512.0,
  "crop_cy": 310.0
}
```
> `mask_path` is an empty string `""` if the operator skipped mask drawing.

**Response (cancelled by operator):**
```json
{"status": "cancelled"}
```

**Response (error):**
```json
{"status": "error", "message": "..."}
```

---

## 6. WAFER_EDGE_REQ — Wafer Edge Detection

Detects the wafer edge in the given direction and returns its pixel offset from the image centre.

**Send (basic):**
```
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" LEFT
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" RIGHT
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" TOP
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" BOTTOM
```

**Send (with optional polarity):**
```
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" LEFT LIGHT_TO_DARK
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" LEFT DARK_TO_LIGHT
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" LEFT ANY
```

**Send (skip FOV classification):**
```
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" LEFT FORCE_RUN
WAFER_EDGE_REQ "C:\Images\live.png" "TKX2" LEFT ANY FORCE_RUN
```

**Parameters:**

| # | Value | Required | Description |
|---|-------|----------|-------------|
| 0 | `image_path` | Yes | Full path to image |
| 1 | `recipe` | Yes | Recipe name or full `.xml` path |
| 2 | `direction` | Yes | `LEFT` \| `RIGHT` \| `TOP` \| `BOTTOM` |
| 3 | `polarity` | No | `ANY` \| `LIGHT_TO_DARK` \| `DARK_TO_LIGHT` (uses recipe value if omitted) |
| 4 | `FORCE_RUN` | No | Skips FOV classification. Required after a `die_fov_warning`. |

**Response (edge found — vertical: LEFT/RIGHT):**
```json
{
  "status": "ok",
  "delta_x": -123.4,
  "delta_y": 0.0,
  "fov_type": "EDGE_FOV",
  "fov_confidence": 0.95,
  "a": 1.0, "b": 0.0, "c": -512.0,
  "x_top": 388.5,
  "x_bot": 389.1
}
```

**Response (edge found — horizontal: TOP/BOTTOM):**
```json
{
  "status": "ok",
  "delta_x": 0.0,
  "delta_y": 56.7,
  "fov_type": "EDGE_FOV",
  "fov_confidence": 0.93,
  "y_left": 200.3,
  "y_right": 201.0
}
```

**Response (die FOV warning — resend with FORCE_RUN):**
```json
{
  "status": "die_fov_warning",
  "fov_type": "DIE_FOV",
  "fov_confidence": 0.91,
  "message": "Image classified as DIE_FOV. Resend with FORCE_RUN to override."
}
```

**Response (edge not found):**
```json
{"status": "no_edge", "reason": "...", "delta_x": 0, "delta_y": 0, "fov_type": "EDGE_FOV"}
```

---

## 7. START_AUTOFOCUS_REQ — Start Autofocus Session

Resets the autofocus session and measures the FMI of the first image.  
Must be called once before the `FOCUS_REQ` loop.

**Send:**
```
START_AUTOFOCUS_REQ "C:\Images\img0.png" 0.0 50
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to image at starting Z height |
| 1 | `zHeightStart` | Starting Z height (float) |
| 2 | `stepNumber` | Total number of Z steps to collect before peak detection |

> The image file is **deleted** by the server after reading.

**Response (success):**
```
START_AUTOFOCUS_OK 1234.567 UP
```

**Response (error):**
```
ERR ImageNotFound
```

---

## 8. FOCUS_REQ — Autofocus Step

Appends one Z-step measurement to the session. Call repeatedly (once per Z step) after `START_AUTOFOCUS_REQ`.

**Send:**
```
FOCUS_REQ "C:\Images\img1.png" 0.1
FOCUS_REQ "C:\Images\img2.png" 0.2
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to image at current Z height |
| 1 | `zHeightCurrent` | Current Z height (float) |

> The image file is **deleted** by the server after reading.

**Response (still collecting — move Z up):**
```
FOCUS_OK 1234.567 UP
```

**Response (peak found — move to best height):**
```
FOCUS_OK 1234.567 STOP 0.300
```
> `0.300` is the best Z height to move to.

**Response (no peak after full scan):**
```
FOCUS_ERR
```

**Response (image missing):**
```
ERR ImageNotFound
```

---

## 9. FM_INDEX_REQ — Focus Measure Index (One-Shot)

Returns the sharpness score (Laplacian variance) of a single image.  
No session state — the machine controller manages Z-stage logic.

**Send:**
```
FM_INDEX_REQ "C:\Images\live.png"
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to the image |

> The image file is **deleted** by the server after reading.

**Response (success):**
```
FM_INDEX_OK 1234.567
```

**Response (error):**
```
ERR ImageNotFound
```

---

## 10. WAFER_ANGLE_REQ — Find Wafer Angle

> **Not yet implemented.** Returns `Command Not Supported.`

**Send:**
```
WAFER_ANGLE_REQ "C:\Images\live.png"
```

**Response:**
```
Command Not Supported.
```

---

## 11. WAFER_CENTER_REQ — Calculate Wafer Centre

Computes the circumcircle centre of three wafer edge points (perpendicular bisector intersection).

**Send:**
```
WAFER_CENTER_REQ 100,500 500,900 900,500
```

**Parameters:**

| # | Format | Description |
|---|--------|-------------|
| 0 | `x1,y1` | First edge point (no spaces inside pair) |
| 1 | `x2,y2` | Second edge point |
| 2 | `x3,y3` | Third edge point |

**Response (success):**
```
WAFER_CENTER_OK 512.000 500.000
```

**Response (points are collinear):**
```
WAFER_CENTER_ERR Points are collinear
```

---

## 12. WAFER_NOTCH_REQ — Find Wafer Notch Edge

> **Not yet implemented.** Returns `WAFER_NOTCH_ERR Not yet implemented`.

**Send:**
```
WAFER_NOTCH_REQ "C:\Images\live.png" 0 512.0 512.0 0.0065
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to image |
| 1 | `side` | `0`=Left, `1`=Right, `2`=Top, `3`=Bottom |
| 2 | `posX` | Expected notch X position (float) |
| 3 | `posY` | Expected notch Y position (float) |
| 4 | `pixelResolution` | mm/pixel resolution (float, must be > 0) |

**Response (not implemented):**
```
WAFER_NOTCH_ERR Not yet implemented
```

**Response (validation errors):**
```
ERR ImageNotFound
ERR Invalid Pixel Resolution
```

---

## 13. WAFER_TEMPLATE_REQ — Find Wafer Template

Runs template matching and returns the final real-world position.  
Uses the template loaded from the current recipe.

**Send:**
```
WAFER_TEMPLATE_REQ "C:\Images\live.png" 512.0 512.0 0.0065
```

**Parameters:**

| # | Value | Description |
|---|-------|-------------|
| 0 | `image_path` | Full path to live image |
| 1 | `posX` | Expected X position in real-world units (float) |
| 2 | `posY` | Expected Y position in real-world units (float) |
| 3 | `pixelResolution` | mm/pixel resolution (float, must be > 0) |

> Requires `LOADR_REQ` to be called first (recipe must have a `TemplatePath`).

**Response (success):**
```
WAFER_TEMPLATE_OK 0.952 513.250 511.820
```
> Format: `WAFER_TEMPLATE_OK <score> <finalX> <finalY>`

**Response (errors):**
```
ERR ImageNotFound
ERR Invalid Pixel Resolution
WAFER_TEMPLATE_ERR No recipe loaded
WAFER_TEMPLATE_ERR No template loaded
WAFER_TEMPLATE_ERR ...
```

---

## 14. SHUTDOWN

Asks the server to exit cleanly after sending the response.

**Send:**
```
SHUTDOWN
```

**Response:**
```json
{"status": "ok"}
```

---

## C++ Example (using ZeroMQ cppzmq)

```cpp
#include <zmq.hpp>
#include <string>
#include <iostream>

int main() {
    zmq::context_t ctx(1);
    zmq::socket_t  sock(ctx, zmq::socket_type::req);
    sock.connect("tcp://127.0.0.1:5555");

    auto send_recv = [&](const std::string& cmd) -> std::string {
        zmq::message_t req(cmd.begin(), cmd.end());
        sock.send(req, zmq::send_flags::none);
        zmq::message_t rep;
        sock.recv(rep, zmq::recv_flags::none);
        return rep.to_string();
    };

    // 1. Health check
    std::cout << send_recv("PING") << "\n";

    // 2. Load recipe
    std::cout << send_recv("LOADR_REQ \"TKX2\"") << "\n";

    // 3. Pattern match
    std::cout << send_recv("PM_REQ \"C:\\Images\\live.png\" \"TKX2\"") << "\n";

    // 4. Wafer edge
    std::cout << send_recv("WAFER_EDGE_REQ \"C:\\Images\\live.png\" \"TKX2\" LEFT") << "\n";

    // 5. Shutdown
    std::cout << send_recv("SHUTDOWN") << "\n";

    return 0;
}
```

---

## Error Handling Summary

| Response | Meaning |
|----------|---------|
| `{"status": "error", "message": "..."}` | Command failed — check `message` |
| `{"status": "no_match"}` | Pattern match found nothing |
| `{"status": "no_edge", ...}` | Edge not found in image |
| `{"status": "die_fov_warning", ...}` | Image is a die FOV — resend with `FORCE_RUN` |
| `{"status": "cancelled"}` | Operator cancelled interactive window |
| `ERR ImageNotFound` | Image file does not exist at given path |
| `ERR Invalid Pixel Resolution` | `pixRes` parameter is zero or negative |
| `FOCUS_ERR` | Autofocus complete but no peak detected |
