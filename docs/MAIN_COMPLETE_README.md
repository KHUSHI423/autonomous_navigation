# Comprehensive Multi-Model Detection System

## 📋 Overview

`main_complete.py` is the unified entry point that integrates **ALL detection models** in this project with the **enhanced HLS lane detection** system.

## 🚀 Key Features

### 1. Enhanced Lane Detection (HLS + Canny + Hough + Polynomial)
- **HLS Color Space Masking**: Detects lanes using saturation and lightness channels
- **Adaptive Thresholding**: Automatically adjusts to lighting conditions
- **Yellow & White Line Detection**: Detects both lane marking types
- **Temporal Smoothing**: Stable detection across frames
- **6 Presets**: default, highway, city, faded, night, indian_road

### 2. Object Detection (YOLOv8)
- Real-time 2D detection with 3D position estimation
- Models: yolov8n.pt (fast), yolov8m.pt (balanced), yolov8l.pt (accurate)
- Depth fusion using MiDaS

### 3. Traffic Sign Detection
- OpenCV mode: Color segmentation + shape analysis
- YOLO mode: Deep learning-based detection
- Detects: stop, yield, speed limits, warnings, etc.

### 4. Depth Estimation (MiDaS)
- Monocular depth from single RGB image
- Per-pixel depth in meters (0.1-50m range)

### 5. Driver Drowsiness Detection
- Face and eye detection using Haar Cascades
- Eye aspect ratio analysis
- Real-time fatigue level monitoring

### 6. Road Hazard Detection
- Identifies pedestrians, bicycles, motorcycles near vehicle
- Distance-based severity assessment
- Critical/warning level alerts

### 7. BEV Mapping (Bird's Eye View)
- Top-down perspective transformation
- Spatial awareness visualization

### 8. Decision Making
- Fuses all detections for navigation
- Obstacle avoidance logic
- Lane keeping assistance

## 🎮 Usage Examples

### Process Video with All Models
```bash
python main_complete.py video input.mp4
```

### Real-time Webcam Detection
```bash
python main_complete.py webcam --model yolov8n.pt
```

### Process Single Image
```bash
python main_complete.py image photo.jpg --show
```

### Fast Mode (YOLO + Lanes Only)
```bash
python main_complete.py video input.mp4 --fast
```

### Enable All Models Explicitly
```bash
python main_complete.py video input.mp4 --all-models
```

### Custom Configuration
```bash
python main_complete.py video input.mp4 \
    --model yolov8m.pt \
    --lane-preset highway \
    --confidence 0.6 \
    --depth-skip 3
```

### Night Road Conditions
```bash
python main_complete.py video night_drive.mp4 --lane-preset night
```

### Indian Roads
```bash
python main_complete.py video indian_road.mp4 --lane-preset indian_road
```

## ⌨️ Keyboard Shortcuts (Video/Webcam Mode)

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |
| `1` | Toggle Lane Detection ON/OFF |
| `2` | Toggle Traffic Sign Detection ON/OFF |
| `3` | Toggle Depth Estimation ON/OFF |
| `4` | Toggle Drowsiness Detection ON/OFF |
| `p` | Pause/Resume video |
| `s` | Save current frame as screenshot |

## 🔧 Command Line Options

### Model Selection
- `--model, -m`: YOLO model (yolov8n.pt, yolov8m.pt, yolov8l.pt)
- `--confidence, -c`: Detection threshold (default: 0.5)

### Lane Detection
- `--no-lane`: Disable lane detection
- `--lane-preset`: Preset selection (default/highway/city/faded/night/indian_road)

### Sign Detection
- `--no-signs`: Disable traffic sign detection
- `--sign-mode`: Detection mode (opencv/yolo)

### Depth Estimation
- `--no-depth`: Disable depth estimation
- `--depth-skip`: Process depth every N frames (default: 2)
- `--max-depth`: Maximum depth in meters (default: 50.0)

### Special Modes
- `--all-models`: Enable all detection models
- `--fast`: Fast mode (YOLO + Lanes only)

### Output
- `--output, -o`: Output directory (default: output)
- `--show`: Show detection results in window

## 📊 Detection Pipeline Flow

```
Input Frame (BGR)
    ↓
┌─────────────────────────────────────────┐
│  1. Depth Estimation (MiDaS)            │
│     - Monocular depth map               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Object Detection (YOLOv8)           │
│     - 2D bounding boxes                 │
│     - 3D position estimation            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. Lane Detection (ENHANCED)           │
│     - HLS color space transformation    │
│     - S-channel & L-channel masking     │
│     - Canny edge detection              │
│     - Combined binary thresholding      │
│     - ROI masking (trapezoidal)         │
│     - Probabilistic Hough Transform     │
│     - Polynomial fitting                │
│     - Curvature & offset calculation    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. Traffic Sign Detection              │
│     - Color segmentation (HSV)          │
│     - Shape analysis                    │
│     - Classification                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  5. Drowsiness Detection                │
│     - Face detection (Haar Cascade)     │
│     - Eye detection & tracking          │
│     - Fatigue level calculation         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  6. Hazard Detection                    │
│     - Proximity analysis                │
│     - Severity assessment               │
│     - Warning generation                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  7. Decision Making                     │
│     - Multi-sensor fusion               │
│     - Action determination              │
│     - Safety checks                     │
└─────────────────────────────────────────┘
    ↓
Comprehensive Result + Visualizations
```

## 🎯 Enhanced Lane Detection Details

### Processing Pipeline
1. **BGR → HLS Conversion**: Transforms image to HLS color space
2. **Channel Splitting**: Extracts H, L, S channels
3. **CLAHE Enhancement**: Improves L-channel contrast
4. **Adaptive Thresholding**: Adjusts based on lighting
5. **S-Channel Masking**: Detects high saturation (white lines)
6. **L-Channel Masking**: Detects lightness variations
7. **Yellow Detection**: H-channel (20-35) + S-channel (>100)
8. **Combined Binary Mask**: Merges all detections
9. **Canny Edge Detection**: On enhanced L-channel
10. **Edge + Color Fusion**: Combines Canny with HLS mask
11. **ROI Masking**: Trapezoidal region of interest
12. **Hough Transform**: Probabilistic line detection
13. **Line Separation**: Left vs right based on slope/position
14. **Polynomial Fitting**: 2nd-order curve smoothing
15. **Metrics Calculation**: Width, curvature, offset

### Advantages Over Basic Methods
- ✅ Better performance in varying lighting
- ✅ Robust to shadows and glare
- ✅ Detects both yellow and white lines
- ✅ Adaptive to weather conditions
- ✅ Temporal smoothing reduces jitter
- ✅ Accurate curvature estimation

## 📈 Performance Tips

### For Real-time Processing
```bash
# Use fast YOLO model
python main_complete.py video input.mp4 --model yolov8n.pt

# Skip depth frames
python main_complete.py video input.mp4 --depth-skip 5

# Fast mode (only essential models)
python main_complete.py video input.mp4 --fast
```

### For Maximum Accuracy
```bash
# Use larger YOLO model
python main_complete.py video input.mp4 --model yolov8l.pt --confidence 0.6

# Enable all models
python main_complete.py video input.mp4 --all-models
```

## 🐛 Troubleshooting

### Missing Dependencies
```bash
# Install YOLO
pip install ultralytics

# Install MiDaS dependencies
pip install timm

# Install OpenCV
pip install opencv-python opencv-contrib-python
```

### Lane Detection Not Working
- Check video has visible lane markings
- Try different presets: `--lane-preset highway`
- Enable debug mode to see HLS processing
- Adjust `--confidence` threshold

### Slow Performance
- Use `--model yolov8n.pt` for fastest YOLO
- Increase `--depth-skip` value
- Use `--fast` mode
- Reduce input video resolution

## 📝 Example Output

```
======================================================================
INITIALIZING COMPREHENSIVE DETECTION PIPELINE
======================================================================

Engine Configuration:
  YOLO Model: yolov8n.pt
  Confidence: 0.5
  Lane Detection: ✓ (preset: default)
  Sign Detection: ✓ (mode: opencv)
  Depth Estimation: ✓
  Max Depth: 50.0m

======================================================================
Video Properties:
  Resolution: 1920x1080
  FPS: 30.00
  Total Frames: 900
======================================================================

PROCESSING VIDEO - Press 'q' to quit
======================================================================
Keyboard Shortcuts:
  '1' - Toggle Lane Detection (current: ON)
  '2' - Toggle Sign Detection (current: ON)
  '3' - Toggle Depth Estimation (current: ON)
  '4' - Toggle Drowsiness Detection (current: OFF)
  'p' - Pause/Resume
  's' - Save Screenshot
  'q' or 'ESC' - Quit
======================================================================

Frame 30/900 | Avg: 45ms | Objects: 5 | Signs: 2 | Lanes: ✓
Frame 60/900 | Avg: 43ms | Objects: 3 | Signs: 1 | Lanes: ✓
...
```

## 🎓 Understanding the Enhanced Lane Detection

The lane detection now matches the approach from the LinkedIn post:

**Real-Time Road Safety with AI YOLOvX**:
- HLS masking (saturation channel for white/yellow lines)
- Canny edge detection (gradient-based edges)
- Probabilistic Hough Transform (line segment detection)
- Polynomial fitting (smooth curve generation)

This provides robust lane detection for:
- Lane departure warnings
- Lane keeping assistance
- Road curvature analysis
- Vehicle positioning

## 📞 Support

For issues or questions:
1. Check this README
2. Review console output for error messages
3. Ensure all dependencies are installed
4. Try different lane detection presets

---

**Built with ❤️ for Advanced Driver Assistance Systems (ADAS)**
