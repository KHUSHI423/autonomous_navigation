# 🚗 EdgeDrive3D — Robot Car Perception & Control System

> **Location:** `combined_final/combined_proj_folder/`
> **Entry Point:** `python main.py dashboard` (Streamlit) or `python dashboard/realtime_app.py` (Flask)
> **YOLO Models:** `yolov8m.pt` (primary), `yolov8n.pt` (lightweight)

---

## 1. Overview

EdgeDrive3D is a **distributed perception and control system** for an autonomous robot car. It uses a laptop as the "brain" to run YOLOv8 object detection, depth estimation, and decision-making, while a Raspberry Pi 5 streams camera video over UDP, and two ESP32 microcontrollers manage real-time motor control (via 2× TB6612FNG drivers) and ultrasonic obstacle sensing.

---

## 2. Full File Tree

```
combined_proj_folder/
├── main.py                          # Entry point — launches dashboard or stream
│
├── dashboard/
│   ├── app.py                       # Primary Streamlit dashboard (5 views)
│   ├── realtime_app.py              # Flask + Socket.IO real-time dashboard (:5000)
│   ├── realtime_fixed.py            # Fixed variant of realtime dashboard
│   ├── simple_dashboard.py          # Minimal Flask dashboard (:5001)
│   ├── templates/index.html         # HTML template used by realtime dashboards
│   └── __init__.py
│
├── core/
│   ├── perception_engine.py         # YOLOv8 detection + depth + sensor fusion
│   └── __init__.py
│
├── config/
│   ├── settings.py                  # Dataclass-based config (Perception, Hardware, Viz)
│   └── __init__.py
│
├── hardware/
│   ├── hardware_integration.py      # PiCameraSender + LaptopReceiver + ESP32Controller
│   ├── pi_sender.py                 # Standalone Pi camera UDP sender script
│   ├── hardware.ino                 # ESP32 firmware (AP mode, HTTP motor control)
│   └── __init__.py
│
├── output/                          # Runtime outputs (generated)
│   ├── snapshots/                   # Captured frames
│   ├── recordings/                  # Video recordings
│   ├── pointclouds/                 # 3D point cloud data (.ply)
│   ├── maps/                        # Generated maps
│   ├── uploads/                     # Uploaded videos/images
│   └── *.jpg / *.json               # Processed detection results
│
├── requirements.txt                 # Full Python dependency list
├── setup.bat                        # One-click Windows setup
├── yolov8m.pt / yolov8n.pt         # YOLOv8 model files
├── DOCUMENTATION.md                 # Extended technical documentation
├── PROJECT_DOCUMENTATION.md         # Alternative project docs
├── README.md                        # Short README
├── QUICKSTART.md                    # Quick start guide
├── ARCHITECTURE.md                  # Architecture notes
├── improvisation_plan.md            # Future improvements log
├── bento.html / bento - Copy.html  # Standalone HTML dashboard
├── background_hero.jpg              # Hero image
├── dashboard_ui.jpg                 # UI reference image
├── hardware_robot.jpg/.png          # Hardware photos
├── point_cloud.jpg/.png             # Point cloud examples
├── test_road_scene.jpg              # Sample test image
├── demo_video1-4.mp4                # Demo videos
├── __init__.py
│
└── .qwen/                           # IDE settings (internal)
```

> **Note:** Running `python config/settings.py` generates `config/settings.yaml` at runtime — this file is not committed to the repository.


---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Wi-Fi Network (SANJEEVI)                     │
└─────────────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────┐
│   ESP32 #1       │   │   ESP32 #2      │   │  Raspberry Pi 5     │
│  Motor Controller│   │ Ultrasonic Sensor│   │  Camera Streamer    │
│                  │   │                  │   │                      │
│  2× TB6612FNG    │   │  HC-SR04         │   │  Pi Camera / Webcam │
│  4 DC Motors     │   │  TRIG=GPIO21     │   │  640×480, 30 FPS    │
│  Differential    │   │  ECHO=GPIO18     │   │  JPEG over UDP:5000 │
│  steer (no servo)│   │  5V, GND         │   │                      │
└────────┬─────────┘   └────────┬─────────┘   └──────────┬───────────┘
         │                      │                        │
         │  UDP :9000           │                        │
         │  ← throttle:steer   │                        │
         │                      │  UDP :9002             │
         │  ← "ULTRA:XX.XXcm"  │                        │
         │                      │                        │
         │  UDP :9001           │                        │
         │  → STATUS broadcast  │                        │
         │                      │                        │
         ▼                      ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LAPTOP — Controller ("Brain")                    │
│                                                                      │
│  • Receives UDP video stream on :5000 from Pi 5                     │
│  • YOLOv8 object detection (vehicles, persons, traffic signs)       │
│  • Distance estimation from bounding box geometry                    │
│  • Sensor fusion (ultrasonic < 30cm → priority override)            │
│  • Sends throttle:steering commands via UDP :9000 to ESP32 #1       │
│  • Receives ESP32 status via UDP :9001                               │
│  • Displays modern OpenCV dashboard or Streamlit/Flask web UI       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown

### 4.1 `main.py` — Entry Point

Parses CLI arguments and launches one of:
- `python main.py dashboard` — Runs `dashboard/app.py` via Streamlit
- `python main.py <other>` — Other operational modes

### 4.2 `core/perception_engine.py` — Perception Pipeline

The core ML processing module:

| Feature | Details |
|---------|---------|
| **Object Detection** | YOLOv8m/YOLOv8n on images, video, webcam, or UDP stream |
| **Depth Estimation** | Monocular depth from bounding box geometry |
| **3D Positioning** | Converts 2D detections to 3D world coordinates |
| **Decision Engine** | Rule-based: ultrasonic priority (<10cm critical, <30cm caution) |
| **BEV Generation** | Bird's Eye View with SVG icons |
| **Point Cloud** | 3D visualization points for Plotly |
| **Sensor Fusion** | Merges YOLO + ultrasonic data with priority logic |

**Detection classes:** car, truck, bus, motorcycle, bicycle, person, and more COCO classes.

### 4.3 `dashboard/` — Visualization (3 variants)

#### `app.py` — Streamlit Dashboard (primary)
```
python main.py dashboard
→ Opens http://localhost:8501
```
- **5 input modes:**
  - 🎮 **Demo** — Simulated objects + BEV + 3D point cloud
  - 📹 **Webcam** — Local camera OR robot UDP stream (:5000)
  - 🖼️ **Upload Image** — Single-frame YOLO processing
  - 🎬 **Upload Video** — Frame-by-frame YOLO processing with seek/play
  - 📡 **Pi Stream** — Real-time UDP stream from Pi 5 camera
- **5 simultaneous output views:** Detection overlay → Depth map (colormap) → BEV map (SVG) → 3D point cloud (Plotly) → Distance data + System telemetry charts

#### `realtime_app.py` — Flask + Socket.IO Dashboard
```
python dashboard/realtime_app.py
→ Opens http://localhost:5000
```
- **Real-time WebSocket updates** (no page refresh)
- **30 FPS smooth video** with base64 JPEG encoding
- **Same 5 modes** as Streamlit version
- **Live metrics:** FPS, Object count, Decision action, Uptime
- **Plotly analytics chart** with rolling FPS/object history
- **Warning system** with color-coded alert boxes

#### `simple_dashboard.py` — Minimal Flask version
```
python dashboard/simple_dashboard.py
→ Opens http://localhost:5001
```
- Lightweight, guaranteed-to-work version for testing

### 4.4 `hardware/` — Hardware Integration

| File | Purpose |
|------|---------|
| `hardware_integration.py` | `PiCameraSender` (UDP streamer) + `LaptopReceiver` (UDP receiver + YOLO) + `ESP32Controller` (HTTP commands) + `IntegratedSystem` (full pipeline) |
| `pi_sender.py` | Standalone Pi camera → UDP sender script (640×480, JPEG q80, 30 FPS) |
| `hardware.ino` | ESP32 firmware — WiFi AP mode, web UI on port 80, HTTP motor control |

**ESP32 firmware (`hardware.ino`):**
- WiFi Access Point: `EdgeDrive3D_Robot` / `edgedrive123`
- Pinout: Motor A (AIN1=26, AIN2=27, PWMA=25), Motor B (BIN1=14, BIN2=13, PWMB=33), STBY=32
- PWM: 1 kHz, 8-bit resolution
- Web UI: D-pad + speed slider + auto mode toggle
- Auto-timeout: 2s (stops motors if no command received)
- HTTP endpoints: `/forward`, `/backward`, `/left`, `/right`, `/stop`, `/speed`, `/toggle_auto`, `/auto_command`

### 4.5 `config/settings.py` — System Configuration

Dataclass-based config with 4 sub-configs:

| Config Section | Key Settings |
|----------------|-------------|
| `PerceptionConfig` | YOLO model, confidence/IoU thresholds, depth model, FOV, resolution, GPU toggle |
| `HardwareConfig` | UDP port (:5000), ESP32 IP, motor speed range, safety distances |
| `VisualizationConfig` | Dashboard port, point size, BEV range/size, snapshot/video saving |
| `OutputConfig` | Base dir, recordings, snapshots, pointclouds, maps paths |

**Presets:** `fast_config()` (YOLOv8n + small depth), `balanced_config()` (default), `accurate_config()` (YOLOv8l + large depth), `hardware_config()` (auto-control enabled), `demo_config()` (demo mode).

---

## 5. Hardware Bill of Materials

| Component | Description | Role |
|-----------|-------------|------|
| **Raspberry Pi 5** | Camera streamer | Runs `pi_camera_v2.py`, sends UDP video |
| **ESP32 #1** | Motor controller | Runs `esp32_motor_controller.ino`, receives UDP commands :9000, broadcasts status :9001, receives ultrasonic :9002 |
| **ESP32 #2** | Ultrasonic sensor | Runs `esp32_ultrasonic.ino`, reads HC-SR04, sends distance via UDP :9002 |
| **2× TB6612FNG** | Dual H-bridge drivers | 4 channels total (left/right motor pairs) |
| **4× DC Motors** | Robot drive | Differential steering (no servo) |
| **HC-SR04** | Ultrasonic range finder | TRIG=GPIO21, ECHO=GPIO18, 5V power |
| **Pi Camera / USB Webcam** | Vision sensor | 640×480, 30 FPS |
| **GPS USB Module** | Navigation (optional) | Connected to Pi 5 |
| **7.4V LiPo Battery** | Power source | Powers ESP32 ×2 + TB6612FNG ×2 |

**Network:** All devices on same Wi-Fi — SSID: `SANJEEVI` (phone hotspot / router). Both ESP32s in Station mode (no AP mode for this architecture).

---

## 6. UDP Communication Map

| Port | From → To | Format | Content |
|------|-----------|--------|---------|
| **:5000** | Pi 5 → Laptop | `[4-byte size][JPEG data]` | Video stream |
| **:9000** | Laptop → ESP32 #1 | `throttle:steering` string | Movement commands |
| **:9001** | ESP32 #1 → Laptop | `STATUS:AUTO:150:0.02:25.5` | Status broadcast |
| **:9002** | ESP32 #2 → ESP32 #1 | `ULTRA:25.50` | Ultrasonic distance |
| **:8080** | Laptop → ESP32 #1 (web) | HTTP | Web UI (direct control) |

---

## 7. Decision Logic (Ultrasonic Priority)

```
Distance Range              Action
──────────────────────────────────────────────────
< 10 cm                     → STOP immediately (critical)
10 – 30 cm                  → ULTRASONIC PRIORITY:
                              • Reverse + turn away from obstacle
                              • YOLO detections ignored
30 – 100 cm                 → YOLO-based navigation:
                              • Object left → steer right
                              • Object right → steer left
                              • Object center → stop or slow turn
> 100 cm                    → Full speed forward (differential steering)
No obstacles                → Forward at max steer-neutral speed
```

---

## 8. ESP32 #1 — Motor Pinout (2× TB6612FNG)

| ESP32 Pin | TB6612FNG #1 (Left) | Function |
|-----------|---------------------|----------|
| GPIO 26   | AIN1                | Motor A direction 1 |
| GPIO 27   | AIN2                | Motor A direction 2 |
| GPIO 25   | PWMA                | Motor A PWM speed (1kHz, 8-bit) |
| GPIO 14   | BIN1                | Motor B direction 1 |
| GPIO 13   | BIN2                | Motor B direction 2 |
| GPIO 33   | PWMB                | Motor B PWM speed (1kHz, 8-bit) |
| GPIO 32   | STBY                | Standby enable (HIGH=active) |

| ESP32 Pin | TB6612FNG #2 (Right) | Function |
|-----------|----------------------|----------|
| GPIO 4    | AIN1                 | Motor A direction 1 |
| GPIO 5    | AIN2                 | Motor A direction 2 |
| GPIO 18   | PWMA                 | Motor A PWM speed |
| GPIO 19   | BIN1                 | Motor B direction 1 |
| GPIO 21   | BIN2                 | Motor B direction 2 |
| GPIO 22   | PWMB                 | Motor B PWM speed |
| GPIO 23   | STBY                 | Standby enable (HIGH=active) |

**Steering:** Differential drive — slows one side's motors relative to the other. No servo steering.

---

## 9. ESP32 #2 — Ultrasonic Pinout

| HC-SR04 Pin | ESP32 #2 Pin |
|-------------|--------------|
| TRIG        | GPIO 21      |
| ECHO        | GPIO 18      |
| VCC         | 5V           |
| GND         | GND          |

---

## 10. How to Run

### Laptop (Streamlit Dashboard)
```bash
python main.py dashboard
# → http://localhost:8501
```

### Laptop (Flask Real-Time Dashboard)
```bash
python dashboard/realtime_app.py
# → http://localhost:5000
```

### Laptop (Simple Test Dashboard)
```bash
python dashboard/simple_dashboard.py
# → http://localhost:5001
```

### Laptop (UDP Receiver + OpenCV Display)
```bash
python hardware/hardware_integration.py laptop --port 5000 --model yolov8m.pt
```

### Pi 5 (Camera Streamer)
```bash
python pi_sender.py <LAPTOP_IP> -p 5000
```

### First-Time Setup
```bash
setup.bat                        # Windows — installs all dependencies
```

---

## 11. Dependencies (`requirements.txt`)

| Package | Usage |
|---------|-------|
| `streamlit` | Primary dashboard framework |
| `torch`, `torchvision` | PyTorch for YOLOv8 |
| `ultralytics` | YOLOv8 model loading and inference |
| `opencv-python` | Video capture, image processing, display |
| `plotly` | 3D point cloud and analytics charts |
| `flask` | Real-time dashboard server |
| `flask-socketio` | WebSocket real-time updates |
| `pillow` | Image file handling |
| `numpy` | Numerical arrays |
| `pandas` | Data manipulation |
| `PyYAML` | Config file serialization |
| `open3d` | 3D point cloud processing (optional) |

---

## 12. Key Features

- **Real-time YOLOv8 object detection** on live video stream, webcam, images, or video files
- **Monocular depth estimation** from bounding box geometry
- **3D world positioning** with Plotly point cloud visualization
- **Bird's Eye View** with SVG icons showing detected objects
- **Sensor fusion** — ultrasonic distance takes priority over vision-based estimates
- **5 dashboard views** simultaneously: detection, depth, BEV, 3D, telemetry
- **Dual dashboard options**: Streamlit (comprehensive) or Flask+Socket.IO (real-time)
- **Demo mode** generates synthetic objects for testing without hardware
- **ESP32 command timeout** safety — stops motors after 500ms without command
- **WiFi auto-reconnect** — both ESP32s recover connection every 5s if lost
- **Differential steering** — no servo required
- **Brownout detection disabled** on ESP32 to prevent dropout during motor spin-up
