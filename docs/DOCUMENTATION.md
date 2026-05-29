# 🚗 EdgeDrive3D - Complete System Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Hardware Integration](#hardware-integration)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Performance](#performance)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**EdgeDrive3D** is a production-ready autonomous vehicle perception system that combines cutting-edge computer vision algorithms with real-time hardware control. Designed for edge devices, it processes camera streams from a Raspberry Pi 3B+ and makes intelligent decisions for robot navigation.

### Key Capabilities

- **Monocular Depth Estimation** - MiDaS/Depth Anything models
- **3D Object Detection** - YOLOv8/v11 with 3D positioning
- **Bird's Eye View Mapping** - Real-time BEV generation
- **Lane Detection** - Robust detection for various road conditions
- **Point Cloud Generation** - Open3D visualization
- **Hardware Control** - ESP32 motor integration
- **Web Dashboard** - Streamlit-based control interface

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICAL LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐         UDP Stream        ┌──────────────────┐ │
│  │ Raspberry   │  ─────────────────────>   │   Laptop/PC      │ │
│  │ Pi 3B+      │      640x480 @ 30fps      │   (Main Compute) │ │
│  │             │                           │                  │ │
│  │  ┌───────┐  │                           │  ┌────────────┐  │ │
│  │  │Camera │  │                           │  │Perception  │  │ │
│  │  └───┬───┘  │                           │  │Engine      │  │ │
│  │      │      │                           │  │            │  │ │
│  │  ┌───┴───┐  │                           │  │- Depth     │  │ │
│  │  │UDP    │  │                           │  │- 3D Detect │  │ │
│  │  │Sender │  │                           │  │- BEV Map   │  │ │
│  │  └───────┘  │                           │  │- Decision  │  │ │
│  └──────┬──────┘                           │  └─────┬──────┘  │ │
│         │                                   └───────┼─────────┘ │
│  ┌──────┴──────┐                                    │           │
│  │   ESP32     │  <────────────────────────────────  │           │
│  │   Motor Ctrl │       HTTP Commands                │           │
│  └─────────────┘                           ┌─────────┴─────────┐│
│                                            │  Streamlit        ││
│                                            │  Dashboard        ││
│                                            └───────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Perception Pipeline

| Module | Technology | Output |
|--------|------------|--------|
| Depth Estimation | MiDaS Hybrid | Dense depth map (640x480) |
| Object Detection | YOLOv8m | 3D bounding boxes |
| BEV Mapping | Custom projection | Top-down view |
| Lane Detection | Edge + Hough | Lane curves |
| Point Cloud | Depth projection | 3D points (XYZRGB) |

### Hardware Support

- **Raspberry Pi 3B+** - Camera streaming via UDP
- **ESP32** - Motor control via HTTP
- **Webcam/USB Camera** - Direct processing
- **Video Files** - Offline processing

### Dashboard Features

- Real-time 3D visualization (Plotly)
- Interactive BEV map
- System metrics tracking
- Hardware control panel
- Recording and export

---

## Installation

### Quick Setup (Windows)

```bash
# Run automated setup
setup.bat

# Or manual installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Manual Setup (Linux/Mac)

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- Ultralytics YOLO 8.0+
- Streamlit 1.28+
- Plotly 5.15+

---

## Usage Guide

### Command Line Interface

```bash
# Show help
python main.py --help

# Process image
python main.py image road.jpg -o output/ --show

# Process video
python main.py video road.mp4 --save

# Webcam
python main.py webcam -i 0

# Pi stream
python main.py pi-stream --port 5000

# With auto control
python main.py pi-stream --auto --esp32-ip 192.168.4.1

# Dashboard
python main.py dashboard --port 8501
```

### Python API

```python
from core.perception_engine import PerceptionEngine

# Initialize
engine = PerceptionEngine({
    'yolo_model': 'yolov8m.pt',
    'confidence': 0.4,
    'max_depth': 50.0
})

# Process frame
import cv2
frame = cv2.imread('image.jpg')
result = engine.process_frame(frame)

# Access results
print(f"Objects: {len(result.objects_3d)}")
print(f"Decision: {result.decision['action']}")

# Save results
engine.save_results(result, 'output/')
```

### Dashboard Usage

1. Start dashboard: `streamlit run dashboard/app.py`
2. Open browser: http://localhost:8501
3. Configure settings in sidebar
4. Click "Start" to begin processing
5. View real-time visualizations

---

## Hardware Integration

### ESP32 Setup

**1. Flash Firmware:**
```bash
# Open Arduino IDE
# Load hardware/hardware.ino
# Select board: DOIT ESP32 DEVKIT V1
# Upload
```

**2. Wiring:**
```
Motor A:
  - AIN1 → GPIO 26
  - AIN2 → GPIO 27
  - PWMA → GPIO 25

Motor B:
  - BIN1 → GPIO 14
  - BIN2 → GPIO 13
  - PWMB → GPIO 33

STBY → GPIO 32 (HIGH to enable)
```

**3. Power:**
- 5V 2A power supply
- Motor driver: L298N or similar

**4. Test:**
- Connect to WiFi: `EdgeDrive3D_Robot` / `edgedrive123`
- Open browser: http://192.168.4.1
- Test motor control via web interface

### Raspberry Pi Setup

**1. Install dependencies:**
```bash
pip install opencv-python numpy
```

**2. Run sender:**
```bash
python hardware/pi_sender.py 192.168.1.100 -p 5000
```

**3. Optimize for Pi 3B+:**
- Resolution: 640x480
- Quality: 70-80
- FPS target: 30

---

## API Reference

### PerceptionEngine

```python
class PerceptionEngine:
    def __init__(self, config: Dict)
    def process_frame(self, frame: np.ndarray) -> PerceptionResult
    def save_results(self, result: PerceptionResult, output_dir: str)
```

### PerceptionResult

```python
@dataclass
class PerceptionResult:
    timestamp: float
    fps: float
    depth_map: np.ndarray
    objects_3d: List[Object3D]
    depth_colored: np.ndarray
    detections_overlay: np.ndarray
    bev_image: np.ndarray
    point_cloud: Tuple[np.ndarray, np.ndarray]
    decision: Dict[str, Any]
```

### Object3D

```python
@dataclass
class Object3D:
    class_id: int
    class_name: str
    confidence: float
    bbox_2d: Tuple[int, int, int, int]
    position_3d: np.ndarray
    distance: float
    dimensions: Dict[str, float]
    bbox_3d: np.ndarray
```

---

## Configuration

### Settings File (config/settings.yaml)

```yaml
perception:
  yolo_model: yolov8m.pt
  confidence_threshold: 0.4
  max_depth_meters: 50.0
  fov_degrees: 70.0

hardware:
  udp_port: 5000
  esp32_ip: 192.168.4.1
  auto_control: false
  safety_stop_distance: 2.0

visualization:
  dashboard_port: 8501
  point_size: 2.0
  save_snapshots: true
```

### Preset Configurations

```python
from config.settings import PresetConfigs

# Fast (60+ FPS)
config = PresetConfigs.fast_config()

# Balanced (30 FPS)
config = PresetConfigs.balanced_config()

# Accurate (15 FPS)
config = PresetConfigs.accurate_config()
```

---

## Performance

### Benchmarks (RTX 3060)

| Configuration | FPS | Latency | Accuracy |
|--------------|-----|---------|----------|
| YOLOv8n + MiDaS Small | 45 | 22ms | Good |
| YOLOv8m + MiDaS Hybrid | 28 | 36ms | Better |
| YOLOv8l + MiDaS Large | 15 | 67ms | Best |

### Optimization Tips

**For Higher FPS:**
1. Use smaller models (yolov8n.pt)
2. Reduce resolution (640x480)
3. Enable CUDA
4. Increase skip_frames

**For Better Accuracy:**
1. Use larger models (yolov8l.pt)
2. Increase resolution (1280x720)
3. Lower confidence threshold
4. Calibrate camera FOV

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Use smaller model
python main.py webcam -m yolov8n.pt

# Or reduce batch size
export CUDA_VISIBLE_DEVICES=0
```

**"UDP connection timeout"**
- Check firewall settings
- Verify IP addresses
- Ensure same network subnet

**"ESP32 not responding"**
- Check power (5V 2A minimum)
- Reset board
- Re-flash firmware

**"Dashboard won't load"**
```bash
# Check port availability
netstat -ano | findstr :8501

# Try different port
streamlit run dashboard/app.py --server.port 8502
```

### Logs

Check logs in: `output/logs/`

Enable debug mode:
```python
config.debug_mode = True
```

---

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- **MiDaS** - Intel ISL
- **YOLO** - Ultralytics
- **Open3D** - 3D data processing
- **Streamlit** - Dashboard framework
- **Plotly** - Visualization

---

**Built with ❤️ for autonomous navigation**

For support: Create a GitHub issue with logs and system info.
