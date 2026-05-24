# 🚦 Real-Time Video Visualization System

## Quick Start Guide

Transform your static 3D models into a **live traffic visualization dashboard** that processes camera video in real-time!

---

## 📋 What You Get

| Component | Description |
|-----------|-------------|
| 🎥 **Video Processing** | Webcam, video file, or RTSP stream input |
| 🔍 **Object Detection** | YOLOv8-based vehicle and pedestrian detection |
| 🎯 **Object Tracking** | Consistent IDs across frames with ByteTrack |
| 🗺️ **3D Mapping** | 2D detections → 3D world coordinates |
| 🌐 **WebSocket Streaming** | Real-time data to web viewer |
| 🎮 **3D Web Viewer** | Interactive Three.js visualization |
| 📊 **Analytics Dashboard** | Live metrics and charts |

---

## 🚀 Installation

### Step 1: Install Dependencies

```bash
# Install real-time visualization dependencies
pip install -r requirements_realtime.txt
```

**Required packages:**
- `ultralytics` - YOLOv8 detection
- `opencv-python` - Video processing
- `filterpy` - Kalman filter tracking
- `fastapi`, `websockets`, `uvicorn` - WebSocket server
- `plotly` - Analytics charts (optional)

### Step 2: Verify Installation

```bash
# Test if all components are available
python -c "from ultralytics import YOLO; import cv2; import fastapi; print('✅ All dependencies installed!')"
```

---

## 🎬 Quick Start

### Option 1: Full System (Recommended for First Time)

```bash
# Run with default webcam
python main.py
```

This will:
1. Open your webcam
2. Detect and track objects
3. Start WebSocket server
4. Show detection results in a window

**Press 'q' to exit**

### Option 2: WebSocket Server + Web Viewer

```bash
# Terminal 1: Start WebSocket server
python main.py --server-only

# Terminal 2: Open web viewer
# Open viewer_realtime.html in your browser
# Or navigate to: http://localhost:8765
```

### Option 3: Analytics Dashboard

```bash
# Run Streamlit dashboard
python main.py --dashboard

# Or directly:
streamlit run live_dashboard.py
```

---

## 📹 Input Sources

### Webcam

```bash
# Default webcam (device 0)
python main.py --source 0

# Second webcam (device 1)
python main.py --source 1
```

### Video File

```bash
# Process video file
python main.py --source path/to/video.mp4
```

### RTSP Stream (IP Camera)

```bash
# RTSP stream
python main.py --source "rtsp://username:password@ip:port/stream"
```

### HTTP Stream

```bash
# HTTP stream
python main.py --source "http://ip:port/video"
```

---

## 🎮 Using the 3D Web Viewer

1. **Start the system:**
   ```bash
   python main.py
   ```

2. **Open the viewer:**
   - Open `viewer_realtime.html` in Chrome/Edge/Firefox
   - Or navigate to `http://localhost:8765`

3. **Viewer Features:**
   - **🎥 View Modes**: Switch between Perspective and Top-Down views
   - **📊 Live Metrics**: See object counts, FPS, update rate
   - **🎛️ Controls**: Toggle grid, trails, labels, auto-rotate
   - **📋 Connection Log**: Monitor WebSocket connection status

---

## ⚙️ Configuration

Edit `config_realtime.yaml` to customize:

### Camera Settings
```yaml
camera:
  source: 0              # Webcam index or file path
  width: 1280            # Resolution
  height: 720
  fps: 30
```

### Detection Settings
```yaml
detection:
  model: yolov8n.pt      # Model size: n/s/m/l/x
  confidence: 0.5        # Detection threshold
  iou: 0.7               # NMS threshold
  device: cpu            # or 'cuda' for GPU
```

### Tracking Settings
```yaml
tracking:
  tracker: bytetrack     # or 'sort'
  track_threshold: 0.3
  max_age: 30            # Frames before track removal
```

---

## 📁 New Files Created

```
visualization_3d/
├── main.py                    # Main entry point
├── video_processor.py         # Video capture module
├── object_detector.py         # YOLOv8 detection
├── tracker.py                 # Object tracking
├── scene_mapper.py            # 2D to 3D mapping
├── model_selector.py          # GLB model selection
├── websocket_server.py        # Real-time streaming
├── viewer_realtime.html       # 3D web viewer
├── live_dashboard.py          # Analytics dashboard
├── utils_3d.py                # 3D utilities
├── config_realtime.yaml       # Configuration
├── requirements_realtime.txt  # Dependencies
├── REALTIME_VIDEO_PLAN.md     # Detailed architecture
└── README_REALTIME.md         # This file
```

**Existing files unchanged:**
- `dashboard.py` - Original static model viewer
- `generator.py` - 3D model generator
- `viewer.html` - Static model viewer
- `models/` - Your 3D GLB assets

---

## 🔧 Troubleshooting

### "ultralytics not installed"
```bash
pip install ultralytics
```

### "No module named 'fastapi'"
```bash
pip install fastapi websockets uvicorn
```

### Camera not opening
- Check camera permissions
- Try different device index: `--source 1`
- On Windows, ensure no other app is using the camera

### Low FPS
- Use smaller model: Change `model: yolov8n.pt` in config
- Reduce resolution: Set `width: 640, height: 480`
- Use CPU instead of GPU (or vice versa)

### WebSocket connection failed
- Ensure server is running: `python main.py --server-only`
- Check firewall settings
- Try different port in `config_realtime.yaml`

### Models not appearing in 3D viewer
- Verify model paths in `model_selector.py`
- Check that GLB files exist in `models/` directory
- Open browser console (F12) for errors

---

## 🎯 Usage Examples

### Example 1: Test with Video File

```bash
# Process a traffic video
python main.py --source traffic_video.mp4 --no-websocket
```

### Example 2: Full Setup with Dashboard

```bash
# Terminal 1: Run detection pipeline
python main.py --source 0

# Terminal 2: Run dashboard
streamlit run live_dashboard.py
```

### Example 3: RTSP Camera Monitoring

```bash
# Connect to IP camera
python main.py --source "rtsp://admin:password@192.168.1.100:554/stream"
```

### Example 4: GPU-Accelerated Detection

```yaml
# In config_realtime.yaml
detection:
  device: cuda
  model: yolov8m.pt
```

---

## 📊 Understanding the Output

### Console Output
```
12:30:45 | INFO     | RealTimeVisualizer initialized
12:30:46 | INFO     | Using webcam: 0
12:30:46 | INFO     | Initializing object detector...
12:30:47 | INFO     | YOLO model loaded successfully
12:30:48 | INFO     | Scene mapper calibrated
12:30:48 | INFO     | Starting video processing loop...
```

### Detection Window
- **Green boxes**: Detected objects
- **ID labels**: Tracking IDs
- **Red dot**: Ground contact point
- **Green grid**: Ground plane projection

### Web Viewer
- **Status dot**: Green = Connected, Red = Disconnected
- **Object counts**: By class (cars, buses, pedestrians)
- **FPS counter**: Render frame rate
- **3D scene**: Your GLB models positioned in real-time

---

## 🚀 Performance Tips

### For Best FPS:
1. Use `yolov8n.pt` (nano model)
2. Reduce input resolution to 640x480
3. Lower confidence threshold to 0.4
4. Use GPU if available

### For Best Accuracy:
1. Use `yolov8m.pt` or `yolov8l.pt`
2. Increase resolution to 1280x720
3. Set confidence to 0.6
4. Calibrate homography manually

---

## 📝 Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Video      │────▶│   YOLOv8     │────▶│   ByteTrack  │
│   Capture    │     │   Detection  │     │   Tracking   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Three.js   │◀────│   WebSocket  │◀────│   2D→3D      │
│   Viewer     │     │   Server     │     │   Mapping    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 🔮 Future Enhancements

- [ ] Multi-camera support
- [ ] Speed estimation (km/h)
- [ ] License plate recognition
- [ ] Incident detection
- [ ] Historical data playback
- [ ] Mobile app viewer

---

## 📞 Support

### Check Logs
```bash
# View latest logs
tail -f logs/realtime_visualizer.log
```

### Debug Mode
```bash
# Enable debug logging
# Edit config_realtime.yaml:
logging:
  level: DEBUG
```

### Test Individual Components
```bash
# Test video processor
python video_processor.py

# Test detector
python object_detector.py

# Test tracker
python tracker.py
```

---

## 🎉 Success Checklist

- [ ] Dependencies installed
- [ ] Camera accessible
- [ ] Models exist in `models/` folder
- [ ] WebSocket server starts
- [ ] Web viewer connects
- [ ] Detections appear
- [ ] 3D models render

---

**Created**: 20 March 2026  
**Version**: 1.0  
**Status**: Ready to Use! 🚀

For detailed architecture, see [`REALTIME_VIDEO_PLAN.md`](REALTIME_VIDEO_PLAN.md)
