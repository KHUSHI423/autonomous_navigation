# 🚗 EdgeDrive3D v2.0 - Enhanced Unified Perception System

## What's New in v2.0?

The enhanced version now includes **Lane Detection** and **Traffic Sign Detection** modules integrated into the unified pipeline!

### Complete Feature Set:
- ✅ **Depth Estimation** (MiDaS)
- ✅ **3D Object Detection** (YOLOv8)
- ✅ **Lane Detection** (OpenCV) - **NEW!**
- ✅ **Traffic Sign Detection** - **NEW!**
- ✅ **Bird's Eye View Mapping**
- ✅ **Intelligent Decision Making**
- ✅ **Point Cloud Generation**

---

## 📦 Installation

```bash
cd combined_proj_folder

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Process an Image
```bash
# All modules enabled (default)
python main.py image road.jpg -o output/ --show

# Disable specific modules
python main.py image road.jpg --no-lane --no-signs

# Use specific lane preset
python main.py image highway.jpg --lane-preset highway
```

### Webcam Processing
```bash
# Start webcam with all features
python main.py webcam

# Disable lane detection
python main.py webcam --no-lane

# Use Indian road preset
python main.py webcam --lane-preset indian_road
```

### Video Processing
```bash
python main.py video road_video.mp4 --save --show
```

---

## 🎮 Keyboard Controls (Webcam Mode)

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save snapshot |
| `1` | Toggle lane detection ON/OFF |
| `2` | Toggle traffic sign detection ON/OFF |

---

## 📋 CLI Options

### Image Mode
```bash
python main.py image <input> [options]

Options:
  -o, --output DIR        Output directory (default: output/)
  -m, --model MODEL       YOLO model (default: yolov8m.pt)
  -c, --confidence FLOAT  Detection confidence (default: 0.4)
  -d, --max-depth FLOAT   Maximum depth (default: 50.0)
  -s, --show              Show results
  --no-lane              Disable lane detection
  --no-signs             Disable traffic sign detection
  --lane-preset PRESET   Lane detection preset
                         Options: default, highway, city, faded, night, indian_road
  --sign-mode MODE       Sign detection mode (opencv or yolo)
```

### Webcam Mode
```bash
python main.py webcam [options]

Options:
  -i, --camera-id ID     Camera device ID (default: 0)
  -m, --model MODEL      YOLO model (default: yolov8n.pt)
  -c, --confidence FLOAT Detection confidence (default: 0.4)
  --no-lane              Disable lane detection
  --no-signs             Disable traffic sign detection
  --lane-preset PRESET   Lane detection preset
```

---

## 🔧 Lane Detection Presets

| Preset | Best For | Canny Thresholds |
|--------|----------|------------------|
| `default` | General use | 50-150 |
| `highway` | Clear lane markings | 60-180 |
| `city` | Urban roads | 40-120 |
| `faded` | Worn markings | 30-100 |
| `night` | Low light | 30-80 |
| `indian_road` | Indian road conditions | 35-110 |

---

## 📊 Output

The enhanced system provides:

1. **Annotated Frame** - All detections overlaid
   - 3D bounding boxes (green)
   - Lane markings (green lines)
   - Traffic signs (colored boxes)
   - Decision overlay

2. **Bird's Eye View** - Top-down visualization

3. **Depth Map** - Colored depth visualization

4. **JSON Results** - Structured data including:
   ```json
   {
     "objects": [...],
     "lanes": {
       "lane_width_meters": 3.5,
       "curvature": 0.002,
       "vehicle_offset_meters": 0.15,
       "confidence": 0.85
     },
     "traffic_signs": [
       {
         "sign_type": "stop",
         "sign_class": "regulatory",
         "confidence": 0.92,
         "distance_estimate_m": 5.2
       }
     ],
     "decision": {
       "action": "stop",
       "reason": "stop_sign_detected",
       "warnings": ["STOP SIGN at 5.2m"]
     }
   }
   ```

---

## 🧪 Run Tests

```bash
# Test all modules
python tests/test_enhanced_pipeline.py

# Test individual modules
python -m core.lane_detector
python -m core.sign_detector
```

---

## 📁 Project Structure

```
combined_proj_folder/
│
├── main.py                      # Main CLI entry point
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── core/                        # Core perception modules
│   ├── __init__.py
│   ├── perception_engine.py     # Unified pipeline (v2.0)
│   ├── lane_detector.py         # Lane detection (NEW!)
│   └── sign_detector.py         # Traffic sign detection (NEW!)
│
├── tests/                       # Test scripts
│   └── test_enhanced_pipeline.py
│
├── output/                      # Output directory
│   ├── images/
│   ├── videos/
│   └── results/
│
└── dashboard/                   # Streamlit dashboards
    ├── app.py
    ├── gps_dashboard.py
    └── gps_dashboard_3d.py
```

---

## 🎯 Decision Making Logic

The enhanced decision maker considers:

1. **Traffic Signs** (Highest Priority)
   - Stop sign → STOP immediately
   - Speed limit → Adjust speed limit

2. **Obstacles** (3D Objects)
   - Distance < 2m → STOP
   - Distance < 5m → SLOW + AVOID
   - Distance > 5m → FORWARD

3. **Lane Position**
   - Vehicle offset > 0.5m → Lane correction
   - High curvature → Reduce speed

---

## 🐛 Troubleshooting

### Lane detection not working?
- Try different preset: `--lane-preset highway`
- Ensure good lighting
- Check if lane markings are visible

### No traffic signs detected?
- Lower confidence: `-c 0.3`
- Ensure signs are clearly visible
- Try YOLO mode if model available: `--sign-mode yolo`

### Low FPS?
- Use smaller YOLO model: `-m yolov8n.pt`
- Disable modules: `--no-lane --no-signs`
- Reduce camera resolution

---

## 📈 Performance

### Laptop (Intel i7, GTX 1650)

| Configuration | FPS |
|---------------|-----|
| All modules | 15-20 FPS |
| Without signs | 20-25 FPS |
| Without lanes | 18-22 FPS |
| Base (depth + objects) | 25-30 FPS |

---

## 🤝 Contributing

Areas for improvement:
- Deep learning-based lane detection
- Multi-class traffic sign recognition
- Night mode enhancement
- Rain/fog robustness

---

## 📄 License

MIT License - Free for educational and commercial use.

---

**Built for Indian Roads | Enhanced with Lane & Sign Detection**
