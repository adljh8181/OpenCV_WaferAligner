# C++ ZMQ Wafer Alignment Client — Setup Guide

## Prerequisites

- **Visual Studio 2022** (Community or higher)
- **vcpkg** package manager — [https://github.com/microsoft/vcpkg](https://github.com/microsoft/vcpkg)

Install vcpkg (run once in PowerShell):
```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg integrate install
```

---

## Step 1 — Create the Visual Studio Project

1. Open **Visual Studio 2022**
2. Click **Create a new project**
3. Filter language to **C++**
4. Select **Console App** → click **Next**
5. Name the project: `WaferClient`
6. Click **Create**

---

## Step 2 — Install NuGet / vcpkg Dependencies

In the project root folder, create `vcpkg.json`:

```json
{
  "dependencies": [
    "cppzmq",
    "nlohmann-json"
  ]
}
```

Then in Visual Studio:
**Tools → Command Line → Developer Command Prompt**
```
vcpkg install cppzmq nlohmann-json --triplet x64-windows
```

---

## Step 3 — Files to Create

Create the following files inside your Visual Studio project folder:

```
WaferClient/
├── vcpkg.json          ← dependency manifest (Step 2)
├── WaferClient.h       ← client class declaration
├── WaferClient.cpp     ← client class implementation
└── main.cpp            ← console UI (replaces the auto-generated one)
```

### File Descriptions

| File | Purpose |
|------|---------|
| `vcpkg.json` | Declares `cppzmq` and `nlohmann-json` as dependencies |
| `WaferClient.h` | Header — declares `WaferAlignmentClient` class with all command methods |
| `WaferClient.cpp` | Implementation — builds ZMQ command strings, sends/receives, returns parsed JSON |
| `main.cpp` | Console menu UI — loops through all 13 commands with user input prompts |

---

## Step 4 — Commands Covered

The client covers every command in the Python `ProcessCommand`:

| Menu # | Command | Method |
|--------|---------|--------|
| 1 | `PING` | `Ping()` |
| 2 | `LOADR_REQ` | `LoadRecipe()` |
| 3 | `PM_REQ` | `PatternMatch()` |
| 4 | `TRAIN_REQ` | `TrainReq()` |
| 5 | `TEACH_REQ` | `TeachReq()` |
| 6 | `WAFER_EDGE_REQ` | `WaferEdgeReq()` |
| 7 | `START_AUTOFOCUS_REQ` | `StartAutofocus()` |
| 8 | `FOCUS_REQ` | `FocusReq()` |
| 9 | `FM_INDEX_REQ` | `FmIndexReq()` |
| 10 | `WAFER_ANGLE_REQ` | `WaferAngleReq()` |
| 11 | `WAFER_CENTER_REQ` | `WaferCenterReq()` |
| 12 | `WAFER_NOTCH_REQ` | `WaferNotchReq()` |
| 13 | `WAFER_TEMPLATE_REQ` | `WaferTemplateReq()` |
| 14 | `SHUTDOWN` | `Shutdown()` |

---

## Step 5 — Build & Run

1. Start the Python server first:
   ```powershell
   python zmq_server.py --port 5555
   ```

2. In Visual Studio, press **F5** (or **Ctrl+F5**) to build and run

3. The console menu will appear — enter a number to call the desired service

---

## Notes

- Default server address is `tcp://localhost:5555` — change in `main.cpp` if needed
- Receive timeout is set to **10 seconds** in `WaferClient.cpp`
- JSON responses are pretty-printed to console (2-space indent)
- Plain-text responses (autofocus commands) are printed as-is
