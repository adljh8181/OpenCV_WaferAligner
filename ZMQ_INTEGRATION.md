# ZeroMQ Integration Guide – Python Server ↔ C# Client

## Overview

```
C# Machine Code (x86 or x64)
  └── WaferAlignmentClient.cs
        │  (TCP / ZeroMQ REQ-REP)
        ▼
  zmq_server.py  ──► LinemodMatcher  ──► alignment result
```

The Python server runs continuously. The C# side sends JSON commands over a
TCP socket; the server reads the image from disk by file path and returns the
alignment result as JSON. No raw image bytes are sent over the wire.

---

## 1. Install prerequisites

### Python side
```powershell
pip install pyzmq opencv-python numpy
```

### C# side (NuGet)
```
Install-Package NetMQ          # pure-managed ZeroMQ – works x86 and x64
Install-Package System.Text.Json   # only needed for .NET Framework 4.x
```

NetMQ is **pure managed C#** — no native `zmq.dll` — so the same build runs
on both 32-bit and 64-bit host processes without any recompile.

---

## 2. Start the Python server

```powershell
# Default options (port 5555, 200 features, threshold 50, angle step 5°)
python zmq_server.py

# Custom options
python zmq_server.py --port 5555 --num-features 300 --threshold 60 --angle-step 3
```

Keep this process running while your machine software operates.

---

## 3. Message protocol (all messages are UTF-8 JSON)

### PING
Checks that the server is alive.
```json
// Request
{ "cmd": "PING" }
// Response
{ "status": "pong" }
```

### SET_CONFIG *(optional)*
Reconfigures the matcher at runtime. Call before `LOAD_TEMPLATE` if you need
different settings than the server defaults.
```json
// Request
{ "cmd": "SET_CONFIG", "num_features": 200, "threshold": 50.0, "angle_step": 5.0 }
// Response
{ "status": "ok" }
```

### LOAD_TEMPLATE
Loads the reference (golden) wafer image. The `template_path` must be
accessible by the **Python process** (local path or shared network drive).
```json
// Request – full image
{ "cmd": "LOAD_TEMPLATE", "template_path": "C:\\Images\\template.png" }

// Request – with optional ROI [x, y, width, height]
{ "cmd": "LOAD_TEMPLATE", "template_path": "C:\\Images\\template.png",
  "roi": [100, 80, 400, 400] }

// Response
{ "status": "ok" }
// or
{ "status": "error", "message": "File not found: ..." }
```

### MATCH
Runs alignment on a live camera image.
```json
// Request
{ "cmd": "MATCH", "search_path": "C:\\Images\\live_001.png" }

// Response – match found
{ "status": "ok", "x": 512.0, "y": 310.0, "angle": -2.5, "score": 87.3 }

// Response – no match
{ "status": "no_match" }

// Response – error
{ "status": "error", "message": "..." }
```

| Field   | Type   | Description                              |
|---------|--------|------------------------------------------|
| `x`     | float  | Horizontal offset in pixels              |
| `y`     | float  | Vertical offset in pixels                |
| `angle` | float  | Rotation in degrees (+ = CCW)            |
| `score` | float  | Match confidence 0 – 100                 |

### TEACH_REQ  *(interactive – opens OpenCV windows on the Python machine)*
Triggers the full interactive template-teaching workflow.  Python opens two
OpenCV windows sequentially on the machine running `zmq_server.py`:

1. **Crop window** – the operator drags a rectangle over the reference pattern
   on the source image and clicks **Save**.
2. **Mask window** – the operator left-clicks to add polygon vertices around
   the exact region to detect, right-clicks to close the polygon, then clicks
   **Save**.  Closing the window without saving skips mask creation (full
   template is used).

After both windows are closed the server saves the PNG files, updates the
recipe XML, and **immediately reloads the matcher** so subsequent `MATCH`
commands use the new template without a restart.

```
// Request  (space-separated, paths MUST be quoted if they contain spaces)
TEACH_REQ "<image_path>" "<recipe_path>"

// Response – operator confirmed crop (mask is optional)
{
  "status":        "ok",
  "template_path": "C:\\recipes\\MyRecipe\\template_1712345678000.png",
  "mask_path":     "C:\\recipes\\MyRecipe\\template_1712345678000_mask.png",
  "crop_cx":       512.0,
  "crop_cy":       310.0
}

// Response – operator pressed Close / ESC on the crop window
{ "status": "cancelled" }

// Response – file/recipe error
{ "status": "error", "message": "..." }
```

| Field           | Type   | Description                                         |
|-----------------|--------|-----------------------------------------------------|
| `template_path` | string | Absolute path to the saved template PNG             |
| `mask_path`     | string | Absolute path to the mask PNG; **empty string** if no mask was drawn |
| `crop_cx`       | float  | X centre of the crop in the original image (pixels) |
| `crop_cy`       | float  | Y centre of the crop in the original image (pixels) |

> **Important**: `TEACH_REQ` blocks the ZMQ server until the operator
> finishes (or cancels) both interactive windows.  Your C# timeout for this
> command should be generous (e.g. 5 minutes).

---

### SHUTDOWN
Asks the server to exit cleanly.
```json
{ "cmd": "SHUTDOWN" }
```

---

## 4. C# usage (minimal example)

```csharp
using WaferAlignment;

using var client = new WaferAlignmentClient("localhost", 5555);

// 1. Health-check
client.Ping();

// 2. Load template once (e.g. recipe change)
client.LoadTemplate(@"C:\Images\wafer_template.png");

// 3. For every camera capture:
AlignmentResult r = client.Match(@"C:\Images\live_capture.png");
if (r.Success)
{
    // Feed to motion controller
    Console.WriteLine($"X={r.X:F2}  Y={r.Y:F2}  Angle={r.Angle:F3}°");
}
```

---

## 5. Shared-folder setup (server and C# on same PC)

If both processes run on the same Windows machine, use ordinary absolute paths:

```
C:\Images\template.png      ← saved by your recipe manager
C:\Images\live.png          ← saved by your camera driver before each Match()
```

The Python server reads the file **after** C# has finished writing it, so no
locking issues arise as long as C# does a full `File.WriteAllBytes()` before
calling `Match()`.

---

## 6. Files added

| File | Purpose |
|------|---------|
| `CSharpClient\WaferAlignmentClient.cs` | Drop-in C# client class |
| `CSharpClient\Program.cs` | Demo / integration example |
| `CSharpClient\WaferAlignmentClient.csproj` | Project file (targets .NET 4.8; change as needed) |
| `zmq_server.py` | Python server (updated with `SET_CONFIG` command) |
