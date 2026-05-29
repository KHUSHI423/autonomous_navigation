# EdgeDrive3D - System Architecture

## Component Overview

```
combined_proj_folder/
│
├── 📄 main.py                          # Main entry point (CLI)
├── 📄 README.md                        # Project overview
├── 📄 QUICKSTART.md                    # Quick start guide
├── 📄 DOCUMENTATION.md                 # Full documentation
├── 📄 requirements.txt                 # Python dependencies
├── 📄 setup.bat                        # Windows setup script
│
├── 📁 core/                            # Core perception modules
│   ├── perception_engine.py            # Main unified pipeline
│   ├── __init__.py
│
├── 📁 hardware/                        # Hardware integration
│   ├── hardware_integration.py         # Pi sender + Laptop receiver
│   ├── hardware.ino                    # ESP32 Arduino code
│   ├── pi_sender.py                    # Raspberry Pi UDP sender
│   ├── __init__.py
│
├── 📁 dashboard/                       # Streamlit dashboard
│   ├── app.py                          # Main dashboard application
│   ├── __init__.py
│
├── 📁 config/                          # Configuration
│   ├── settings.py                     # System configuration
│   ├── __init__.py
│
├── 📁 utils/                           # Utilities (to be added)
│   ├── data_structures.py
│   ├── visualization.py
│   └── logger.py
│
├── 📁 output/                          # Output directory
│   ├── recordings/                     # Video recordings
│   ├── snapshots/                      # Image snapshots
│   ├── pointclouds/                    # PLY point clouds
│   └── maps/                           # Generated maps
│
└── 📁 tests/                           # Test scripts
    ├── test_perception.py
    ├── test_hardware.py
    └── test_dashboard.py
```

---

## Data Flow

### 1. Image Capture
```
Raspberry Pi Camera
        ↓
    [UDP Stream - 640x480 @ 30fps]
        ↓
Laptop Network Interface
        ↓
    [JPEG Decode]
        ↓
    OpenCV BGR Image
```

### 2. Perception Pipeline
```
Input Frame (640x480)
        ↓
┌───────────────────────────────────┐
│  Depth Estimation (MiDaS)         │
│  Output: Depth Map (320x240)      │
│  Time: ~50ms                      │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  3D Object Detection (YOLOv8)     │
│  Output: List[Object3D]           │
│  Time: ~30ms                      │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  BEV Mapping                       │
│  Output: BEV Image (800x600)      │
│  Time: ~5ms                       │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  Decision Making                   │
│  Output: {action, speed, reason}  │
│  Time: <1ms                       │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  Point Cloud Generation            │
│  Output: Points (N×3), Colors     │
│  Time: ~10ms                      │
└───────────────────────────────────┘
```

### 3. Output & Control
```
Perception Result
        ↓
    ┌───┴───┬───────────┬────────────┐
    ↓       ↓           ↓            ↓
Dashboard  Save     ESP32       Log
Visualization  Output  Control    Metrics
```

---

## Module Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
├─────────────────────────────────────────────────────────┤
│  main.py  │  dashboard/app.py  │  hardware_integration │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Perception Layer                       │
├─────────────────────────────────────────────────────────┤
│           core/perception_engine.py                      │
│  ┌──────────────┬──────────────┬─────────────────────┐  │
│  │DepthEstimator│ObjectDetector│  BEVMapper          │  │
│  │              │              │  DecisionMaker      │  │
│  └──────────────┴──────────────┴─────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Model Layer                           │
├─────────────────────────────────────────────────────────┤
│   YOLOv8 (Ultralytics)  │  MiDaS (Torch Hub)           │
│   OpenCV                │  NumPy                        │
│   Plotly                │  Open3D (optional)            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Hardware Layer                         │
├─────────────────────────────────────────────────────────┤
│   Raspberry Pi (UDP)  │  ESP32 (HTTP)  │  Webcams      │
└─────────────────────────────────────────────────────────┘
```

---

## Communication Protocols

### UDP Streaming (Pi → Laptop)

```
Frame Format:
┌─────────────┬─────────────────────────────────┐
│ Size (4B)   │ JPEG Data (variable)            │
│ Little-endian│ Compressed RGB image           │
└─────────────┴─────────────────────────────────┘

Packet Structure:
  - Header: 4 bytes (frame size)
  - Payload: JPEG compressed image
  - Protocol: UDP/IP
  - Port: 5000 (default)
```

### HTTP Control (Laptop → ESP32)

```
Commands:
  GET /forward     - Move forward
  GET /backward    - Move backward
  GET /left        - Turn left
  GET /right       - Turn right
  GET /stop        - Stop motors
  GET /speed?v=180 - Set speed

Auto Command:
  GET /auto_command?action=forward&speed=180
```

---

## Performance Characteristics

### Latency Breakdown

| Stage | Time | Cumulative |
|-------|------|------------|
| Camera capture | 33ms | 33ms |
| UDP transmission | 5ms | 38ms |
| JPEG decode | 10ms | 48ms |
| Depth estimation | 50ms | 98ms |
| Object detection | 30ms | 128ms |
| BEV generation | 5ms | 133ms |
| Decision making | 1ms | 134ms |
| Visualization | 10ms | 144ms |
| **Total** | | **~144ms (7 FPS)** |

### Optimization Strategies

1. **Pipeline Parallelism**
   - Run depth and detection concurrently
   - Overlap processing with visualization

2. **Frame Skipping**
   - Process every Nth frame
   - Use previous results for skipped frames

3. **Model Selection**
   - YOLOv8n for speed
   - YOLOv8m for balance
   - YOLOv8l for accuracy

4. **Resolution Scaling**
   - Process at lower resolution
   - Scale results to original size

---

## Security Considerations

### Network Security

- UDP streaming: Local network only
- ESP32 AP: WPA2 password protected
- Dashboard: localhost by default

### Recommendations

1. Use firewall rules to restrict UDP port
2. Change default ESP32 password
3. Enable dashboard authentication for remote access
4. Use HTTPS for production deployment

---

## Scalability

### Single System → Multi-System

```
Current: Single Pi → Single Laptop
Future: Multiple Pis → Central Server → Multiple Dashboards

Architecture:
  [Pi 1] ─┐
  [Pi 2] ─┼──> [Message Queue] ──> [Processing Server] ──> [Dashboard Cluster]
  [Pi N] ─┘
```

### Cloud Deployment

- Containerize with Docker
- Deploy on Kubernetes
- Use Redis for message queue
- Scale horizontally with load balancer

---

## Future Enhancements

### Planned Features

1. **Multi-Camera Fusion**
   - Stereo depth estimation
   - 360° surround view

2. **Advanced Tracking**
   - Kalman filter tracking
   - Multi-object tracking (DeepSORT)

3. **Sensor Fusion**
   - LiDAR integration
   - IMU data fusion
   - GPS localization

4. **Path Planning**
   - A* pathfinding
   - Dynamic obstacle avoidance
   - Trajectory optimization

5. **Machine Learning**
   - Custom model training
   - Online learning
   - Federated learning

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-15 | Initial release |
| | | - Unified perception engine |
| | | - Hardware integration |
| | | - Streamlit dashboard |
| | | - ESP32 motor control |

---

**EdgeDrive3D - Built for the Future of Autonomous Navigation**
