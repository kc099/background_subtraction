# Background Subtraction Application - Documentation

## Overview
A PySide6-based GUI application for detecting objects via background subtraction. Supports three image acquisition modes: uploaded images, RealSense camera, and HTTP-streamed top camera. Includes depth sensing (RealSense), wheel center detection, and interactive parameter tuning.

---

## Classes & Components

### Configuration System
**File:** `config.yaml`  
YAML configuration file with the following sections:
- `http_streamer.url` - HTTP camera stream URL (for Top Camera mode)
- `http_streamer.timeout` - Connection timeout in seconds
- `background_image.path` - File path to default background image
- `background_image.auto_load` - If `true`, loads background on app startup
- `processing` - Default parameter values (min_deviation, max_deviation, min_contour_area)
- `ui` - Window dimensions and update FPS rate

### 1. **BackgroundSubtractionApp** (Main Application)
**File:** `background_subtraction_app.py`  
**Parent:** `QMainWindow` (PySide6)

#### Key Attributes:
- `background_image` - Shared background reference image (numpy array)
- `target_image` - Current target for comparison (from file or camera)
- `captured_depth_frame` - Depth frame captured alongside target (RealSense only)
- `current_mode` - Operating mode: `"uploaded"`, `"realsense"`, or `"top_camera"`
- `streamer` - Active camera streamer instance
- `bg_subtractor` - Instance of `BackgroundSubtraction` processor
- `config` - Loaded configuration dictionary from `config.yaml`
- `min_deviation`, `max_deviation`, `min_contour_area` - Tunable algorithm parameters (loaded from config or defaults)

#### Key Methods:

| Method | Purpose |
|--------|---------|
| `_build_ui()` | Constructs entire PySide6 interface with layouts, buttons, sliders, and display labels |
| `on_mode_changed()` | Handles radio button selection; stops active streamer on mode switch |
| `load_background()` | File dialog to load shared background image (used across all modes) |
| `load_target()` | File dialog to load target image from disk |
| `start_camera()` | Instantiates appropriate streamer with config parameters; starts threaded acquisition |
| `stop_camera()` | Halts streamer and stops UI update timer |
| `update_camera_ui()` | QTimer callback (~20 FPS) to display live camera frames and extract color/depth data |
| `capture_target()` | Freezes current camera frame as target image; stores depth if available |
| `process_background_subtraction()` | Main processing pipeline: loads fallback background from config if needed, calls subtractor, filters contours, detects wheel center, displays results |
| `find_wheel_center_and_depth()` | Locates wheel center using bottom 75% of largest contour; queries depth at center point |
| `get_depth_at_point()` | Retrieves median depth in expanding window around pixel; handles invalid/zero values |
| `update_params()` | Syncs slider/spinbox values to internal parameters and subtractor instance |

---

## Operating Modes

### Mode 1: Uploaded Images
- **Description:** Compare two static images loaded from disk
- **Background:** Loaded via "Load Background Image" button
- **Target:** Loaded via "Load from File" button
- **Depth:** Not available
- **Use Case:** Quick testing, batch processing

### Mode 2: RealSense Camera
- **Description:** Live RGB-D streaming from Intel RealSense camera
- **Background:** Shared image loaded from disk
- **Target:** Captured from live stream via "Capture from Camera" button
- **Depth:** Full depth frame captured alongside RGB
- **Features:** Wheel center depth measurement in mm/meters
- **Use Case:** Real-time inspection, depth-aware object detection

### Mode 3: Top Camera (HTTP)
- **Description:** Live RGB streaming from HTTP endpoint (e.g., network camera)
- **Background:** Shared image loaded from disk
- **Target:** Captured from live stream
- **Depth:** Not supported (RGB only)
- **Use Case:** Integration with IP cameras, networked monitoring

---

## Processing Pipeline

```
1. User selects mode (Uploaded/RealSense/Top Camera)
   ↓
2. Load background image (shared across modes)
   ↓
3. Load or capture target image
   ↓
4. Adjust parameters (sliders/spinbox)
   ↓
5. Run Background Subtraction:
   - BackgroundSubtraction.process() → computes difference heatmap & binary mask
   - Filter contours by minimum area
   - Calculate wheel center from bottom 75% of largest contour
   - Query depth at wheel center (if depth available)
   - Generate overlay with contours, center marker, & depth label
   ↓
6. Display results: heatmap, mask, overlay, wheel info
```

---

## UI Components

### Input Controls
- **Mode Selection:** Radio buttons (Uploaded/RealSense/Top Camera)
- **Background Button:** "Load Background Image" (shared)
- **Target Buttons:** 
  - "Load from File" (for uploaded mode)
  - "Start Camera" / "Stop Camera" (for streaming)
  - "Capture from Camera" (capture live frame as target)

### Parameter Sliders
- **Min Deviation Slider:** Range [0–100]
- **Max Deviation Slider:** Range [0–255]
- **Min Contour Area SpinBox:** Range [0–100000], step 100

### Display Panels
- **Background Image:** Shows loaded reference
- **Target Image:** Shows file or live stream
- **Difference Heatmap:** Color-coded pixel-wise deviation
- **Background Mask:** Binary detection result
- **Overlay Result:** Target + green mask + wheel center marker
- **Wheel Center & Depth:** Text info (X, Y, depth_mm, depth_m)
- **Status Bar:** Real-time feedback (mode, frame count, mask statistics)

---

## External Dependencies

| Module | Purpose |
|--------|---------|
| `cv2` (OpenCV) | Image processing, contour detection, visualization |
| `numpy` | Array manipulation, depth windowing |
| `PySide6` | GUI framework |
| `yaml` | Configuration file parsing |
| `threading` | Concurrent camera streaming |
| `realsense_streamer` | `RealSenseStreamer`, `HttpStreamer` classes |
| `background_subtraction` | `BackgroundSubtraction` processor class |

---

## Helper Functions

### `load_config(config_path: str = "config.yaml")`
Loads and parses the YAML configuration file.
- Returns empty dict if file not found or parse error occurs
- Logs errors/warnings to help debug configuration issues

### `ndarray_to_qpixmap(img, is_bgr=True)`
Converts OpenCV BGR/grayscale numpy arrays to PySide6 QPixmap for display.
- Handles color space conversion (BGR→RGB)
- Supports both 8-bit grayscale and RGB formats
- Returns empty QPixmap if input is None/empty

---

## Entry Point

```python
def main():
    app = QApplication(sys.argv)
    window = BackgroundSubtractionApp()
    window.show()
    sys.exit(app.exec())
```

Launches Qt event loop and displays main window.

---

## Configuration File (config.yaml)

### Example Structure
```yaml
# HTTP Streamer Settings
http_streamer:
  url: "http://192.168.1.100:8080/video_feed"  # HTTP camera stream endpoint
  timeout: 5  # Connection timeout in seconds

# Background Image Path (used as fallback if no image loaded via UI)
background_image:
  path: "./captures/default_background.png"  # Relative or absolute path


# Processing Parameters (defaults)
processing:
  min_deviation: 10
  max_deviation: 255
  min_contour_area: 1000

```

### Configuration Usage
- **HTTP Streamer URL:** Loaded automatically when "Top Camera (HTTP)" mode is activated
- **Background Image Path:** 
  - Used as fallback during processing if no background image is loaded via UI
  - Can use relative paths (resolved from application directory) or absolute paths
  - Applied silently during processing with no UI changes
- **Processing Parameters:** Applied on startup; can be overridden via UI sliders

---

## Key Features & Notes

✅ **Configuration-Driven Setup:** HTTP URL and background path fallback managed via YAML  
✅ **Fallback Background:** Config path used during processing if no image loaded via UI (silent, no UI changes)  
✅ **Shared Background:** Single background image used across all modes  
✅ **Live Streaming:** Real-time camera with ~20 FPS UI updates (configurable)  
✅ **Depth Integration:** RealSense depth measurement at wheel center  
✅ **Interactive Parameters:** Real-time slider adjustment  
✅ **Wheel Detection:** Centers calculated from bottom 75% of contour (ignores top fins)  
✅ **Robust Depth Sampling:** Adaptive window expansion if initial depth window contains too many invalid pixels  
✅ **Error Handling:** Graceful degradation for missing depth, no camera, or invalid masks  

---

## Key Features & Notes

✅ **Shared Background:** Single background image used across all modes  
✅ **Live Streaming:** Real-time camera with ~20 FPS UI updates  
✅ **Depth Integration:** RealSense depth measurement at wheel center  
✅ **Interactive Parameters:** Real-time slider adjustment  
✅ **Wheel Detection:** Centers calculated from bottom 75% of contour (ignores top fins)  
✅ **Robust Depth Sampling:** Adaptive window expansion if initial depth window contains too many invalid pixels  
✅ **Error Handling:** Graceful degradation for missing depth, no camera, or invalid masks  

---

## Common Workflows

### Workflow 1: Quick File Comparison
1. Select "Uploaded Images" mode
2. Load background image
3. Load target image
4. Adjust sliders as needed
5. Click "Run Background Subtraction"

### Workflow 2: Real-Time RealSense Inspection
1. Select "RealSense Camera" mode
2. Load background image
3. Click "Start Camera"
4. Wait for live preview in Target panel
5. Click "Capture from Camera" to freeze target
6. Adjust parameters
7. Click "Run Background Subtraction" → wheel center + depth displayed

### Workflow 3: Network Camera Integration
1. Select "Top Camera (HTTP)" mode
2. Load background image
3. Click "Start Camera"
4. Wait for live HTTP stream preview
5. Proceed as Workflow 2 (no depth available)

---

**Last Updated:** December 30, 2025
