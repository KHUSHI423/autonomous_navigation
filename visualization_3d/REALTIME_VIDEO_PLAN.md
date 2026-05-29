# 🎥 Real-Time Video Visualization System

## Mission
Transform static 3D model viewer into a **live traffic visualization dashboard** that processes camera video in real-time, detects objects, and renders them in a 3D scene using the existing GLB models.

---

## 🎯 Problem Statement

**Current System:**
- Static 3D model browser (dashboard.py, viewer.html)
- Pre-generated GLB assets (47 models in models/)
- No real-time data processing
- No video integration

**Goal:**
- Process live camera feed or video file
- Detect vehicles, pedestrians, and objects in real-time
- Map detections to 3D positions
- Visualize in an interactive 3D dashboard
- Show live metrics (counts, speeds, trajectories)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  📹 Webcam  │  📼 Video File  │  🌐 RTSP Stream  │  📱 IP Camera   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Frame      │  │   Object     │  │   Object     │              │
│  │   Capture    │─▶│   Detection  │─▶│   Tracking   │              │
│  │   (OpenCV)   │  │   (YOLOv8)   │  │   (ByteTrack)│              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MAPPING LAYER                                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   2D to 3D   │  │   Model      │  │   Position   │              │
│  │   Transform  │─▶│   Selection  │─▶│   Mapping    │              │
│  │              │  │   (GLB)      │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Three.js 3D Scene (viewer_realtime.html)        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │  │
│  │  │  Bird's    │  │  Perspective│  │  Split     │             │  │
│  │  │  Eye View  │  │  View       │  │  View      │             │  │
│  │  └────────────┘  └────────────┘  └────────────┘             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ANALYTICS LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Live       │  │   Speed      │  │   Trajectory │              │
│  │   Counts     │  │   Estimation │  │   Heatmaps   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 New Files to Create

### Core Processing Pipeline
| File | Purpose | Key Features |
|------|---------|--------------|
| `video_processor.py` | Video capture & preprocessing | Webcam/file/RTSP input, frame buffering |
| `object_detector.py` | Object detection | YOLOv8, custom Indian vehicle classes |
| `tracker.py` | Object tracking | ByteTrack, unique ID assignment |
| `scene_mapper.py` | 2D→3D position mapping | Homography, ground plane projection |

### Visualization & Dashboard
| File | Purpose | Key Features |
|------|---------|--------------|
| `model_selector.py` | GLB model matching | Maps detection class to best GLB model |
| `websocket_server.py` | Real-time data streaming | FastAPI + WebSocket for live updates |
| `viewer_realtime.html` | 3D web visualizer | Three.js with live model updates |
| `live_dashboard.py` | Analytics dashboard | Streamlit with metrics, charts, controls |

### Configuration & Utilities
| File | Purpose | Key Features |
|------|---------|--------------|
| `config_realtime.yaml` | System configuration | Camera settings, thresholds, mappings |
| `requirements_realtime.txt` | New dependencies | YOLO, OpenCV, tracking libs |
| `utils_3d.py` | 3D transformation utilities | Coordinate transforms, scaling |
| `recorder.py` | Session recording | Video export, data logging |

---

## 🔧 Technology Stack

### Detection & Tracking
```yaml
ultralytics: 8.0+      # YOLOv8 for object detection
opencv-python: 4.8+    # Video capture and preprocessing
filterpy: 1.4+         # Kalman filter for tracking
lapx: 0.5+             # Linear assignment for tracking
```

### Real-time Communication
```yaml
fastapi: 0.104+        # WebSocket server
websockets: 12.0+      # WebSocket protocol
uvicorn: 0.24+         # ASGI server
```

### Visualization
```yaml
three.js: 0.160+       # 3D rendering (existing)
streamlit: 1.28+       # Dashboard (existing)
plotly: 5.18+          # Analytics charts
```

---

## 🎨 Visualization Modes

### Mode 1: Dual View (Default)
```
┌─────────────────────────────┬─────────────────────────────┐
│     📹 Original Video       │     🗺️ 3D Reconstruction   │
│                             │                             │
│   [Live camera feed with    │   [GLB models positioned    │
│    detection boxes]         │    in 3D scene]             │
│                             │                             │
│   Vehicle: 15               │   🚗 Cars: 8                │
│   Pedestrian: 8             │   🚌 Buses: 2               │
│   Two-wheeler: 5            │   🛵 Scooters: 3            │
│                             │   🚶 Pedestrians: 2         │
└─────────────────────────────┴─────────────────────────────┘
```

### Mode 2: Bird's Eye View
```
                    ┌─────────────────────┐
                    │                     │
                    │    🗺️ Top-Down 3D   │
                    │    Reconstruction   │
                    │                     │
                    │   Shows all objects │
                    │   from above        │
                    │                     │
                    └─────────────────────┘
```

### Mode 3: Analytics Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  📊 Live Traffic Analytics                                   │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   🚗 Total   │   ⚡ Avg     │   🔥 Peak    │   📈 Density  │
│   Vehicles   │   Speed      │   Hour       │   Index       │
│     1,247    │   32 km/h    │   5:30 PM    │     0.78      │
├──────────────┴──────────────┴──────────────┴───────────────┤
│  📉 Vehicle Count Over Time (Live Chart)                    │
│  ╭────────────────────────────────────────────╮             │
│  │    ╭─╮     ╭─╮                             │             │
│  │   ╭╯ ╰╮   ╭╯ ╰╮    ╭─╮                     │             │
│  │  ╭╯   ╰╮ ╭╯   ╰╮  ╭╯ ╰╮                    │             │
│  ╰──╯     ╰─╯     ╰──╯     ╰──────────────────╯             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Step 1: Video Capture
```python
# video_processor.py
cap = cv2.VideoCapture(0)  # or file/RTSP URL
while True:
    ret, frame = cap.read()
    yield frame
```

### Step 2: Object Detection
```python
# object_detector.py
model = YOLO('yolov8n.pt')  # or custom trained model
results = model(frame)
# Returns: [class, confidence, bbox] for each detection
```

### Step 3: Object Tracking
```python
# tracker.py
tracker = ByteTrack()
tracks = tracker.update(detections)
# Returns: [track_id, class, bbox] for each tracked object
```

### Step 4: 2D to 3D Mapping
```python
# scene_mapper.py
def bbox_to_ground_plane(bbox, homography_matrix):
    # Extract bottom center of bbox (contact point with ground)
    contact_point = (bbox[0] + bbox[2]) / 2, bbox[3]
    # Transform to world coordinates
    world_x, world_z = homography_transform(contact_point, H)
    return world_x, 0, world_z  # Y=0 for ground plane
```

### Step 5: Model Selection
```python
# model_selector.py
MODEL_MAP = {
    'car': 'models/vehicles/veh_car_sedan_blue.glb',
    'bus': 'models/vehicles/veh_bus_city_red.glb',
    'motorcycle': 'models/vehicles/veh_bike_motorcycle.glb',
    'person': 'models/humans/hum_pedestrian_blue.glb',
    # ... more mappings
}
```

### Step 6: Real-time Rendering
```javascript
// viewer_realtime.html
ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateScene(data.objects);  // Update Three.js scene
};
```

---

## 📊 Dashboard Features

### Live Metrics Panel
- **Vehicle Count by Type**: Cars, buses, trucks, two-wheelers
- **Pedestrian Count**: People detected and tracked
- **Average Speed**: Estimated from trajectory analysis
- **Traffic Density**: Objects per unit area
- **Peak Detection**: Busiest time window

### Interactive Controls
- **View Mode Switch**: Dual/Bird's Eye/Analytics
- **Model Visibility**: Toggle object categories
- **Trail Display**: Show/hide trajectory trails
- **Speed Filter**: Filter by minimum speed threshold
- **Recording**: Start/stop session recording

### Analytics Charts
- **Time Series**: Vehicle count over time
- **Speed Distribution**: Histogram of vehicle speeds
- **Heat Map**: High-density areas visualization
- **Class Distribution**: Pie chart of object types

---

## 🎯 Model Mapping Strategy

### Detection Class → GLB Model
| YOLO Class | Selected GLB Model | Variation Logic |
|------------|-------------------|-----------------|
| car | veh_car_sedan_blue.glb | Random color variation |
| car (large) | veh_car_suv_dark.glb | Based on bbox aspect ratio |
| bus | veh_bus_city_red.glb | Red for city bus |
| truck | veh_truck_delivery.glb | Based on size |
| motorcycle | veh_bike_motorcycle.glb | Default bike |
| bicycle | veh_bicycle_blue.glb | Direct mapping |
| person | hum_pedestrian_blue.glb | Random color |
| person (moving fast) | hum_cyclist.glb | Based on speed |

### Auto-Selection Algorithm
```python
def select_best_model(detection):
    cls = detection['class']
    bbox = detection['bbox']
    speed = detection.get('speed', 0)
    
    aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    
    if cls == 'car':
        if aspect_ratio > 1.3:  # Wider = SUV
            return 'veh_car_suv_dark.glb'
        else:
            return random.choice(SEDAN_MODELS)
    
    if cls == 'person' and speed > 5:  # Fast moving = cyclist
        return 'hum_cyclist.glb'
    
    return DEFAULT_MAP.get(cls, 'mrk_pin_location.glb')
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements_realtime.txt
```

### Step 2: Configure Camera
```yaml
# config_realtime.yaml
camera:
  source: 0  # or "video.mp4" or "rtsp://..."
  width: 1280
  height: 720
  fps: 30
```

### Step 3: Run Detection Pipeline
```bash
python object_detector.py --source 0  # webcam
python object_detector.py --source traffic.mp4  # video file
```

### Step 4: Start Visualization
```bash
# Option A: Web viewer with WebSocket
python websocket_server.py
# Open viewer_realtime.html in browser

# Option B: Streamlit dashboard
streamlit run live_dashboard.py
```

---

## 📈 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection FPS | 30+ | On GPU, 1080p input |
| Tracking Latency | <50ms | End-to-end |
| WebSocket Update Rate | 15-30 Hz | Smooth visualization |
| 3D Scene FPS | 60+ | Three.js rendering |
| Max Tracked Objects | 100+ | Simultaneous tracks |

---

## 🔮 Future Enhancements

### Phase 2 Features
1. **Multi-Camera Support**: Fuse feeds from multiple cameras
2. **License Plate Recognition**: ANPR integration
3. **Incident Detection**: Accident, wrong-way, congestion alerts
4. **3D Gaussian Splatting**: Photorealistic scene reconstruction
5. **Mobile App**: React Native viewer app

### Phase 3 Features
1. **Predictive Analytics**: Traffic flow prediction
2. **Signal Optimization**: Adaptive traffic light control
3. **Digital Twin**: Full city-scale 3D replica
4. **AR Overlay**: Mobile AR for on-site visualization

---

## 📝 Implementation Checklist

- [ ] Create `requirements_realtime.txt`
- [ ] Create `config_realtime.yaml`
- [ ] Create `video_processor.py`
- [ ] Create `object_detector.py`
- [ ] Create `tracker.py`
- [ ] Create `scene_mapper.py`
- [ ] Create `model_selector.py`
- [ ] Create `websocket_server.py`
- [ ] Create `viewer_realtime.html`
- [ ] Create `live_dashboard.py`
- [ ] Create `utils_3d.py`
- [ ] Create `recorder.py`
- [ ] Test with webcam input
- [ ] Test with video file
- [ ] Test with RTSP stream
- [ ] Performance optimization
- [ ] Documentation

---

## 🎬 Demo Scenarios

### Scenario 1: Webcam Testing
- Use laptop webcam to capture desk/room
- Detect and track objects in real-time
- Visualize as mini 3D scene

### Scenario 2: Traffic Video
- Load pre-recorded traffic video
- Show vehicle detection and tracking
- Display count and speed analytics

### Scenario 3: RTSP Stream
- Connect to IP camera stream
- Process remote location feed
- Remote 3D visualization

---

## 📞 Support & Troubleshooting

### Common Issues
1. **Low FPS**: Reduce input resolution, use YOLOv8n (nano model)
2. **Tracking ID Switches**: Adjust tracker confidence threshold
3. **3D Models Not Appearing**: Check file paths in model_selector.py
4. **WebSocket Connection Failed**: Ensure server is running on correct port

### Debug Mode
```bash
python object_detector.py --source 0 --debug
# Shows detection boxes, tracking IDs, and timing info
```

---

**Created**: 20 March 2026  
**Version**: 1.0  
**Status**: Ready for Implementation
