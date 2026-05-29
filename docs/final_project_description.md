# Intelligent Autonomous Robot Car System — Final Project Description

> **Project Root:** `combined_final_new_gps/combined_proj_folder/`  
> **Last Updated:** 2026-03-16  
> **Platform:** Raspberry Pi 5 + Dual ESP32 + Laptop Brain Architecture

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Hardware Components](#3-hardware-components)
4. [Software Components](#4-software-components)
5. [Operating Modes](#5-operating-modes)
6. [Decision Engine](#6-decision-engine)
7. [GPS & Navigation System](#7-gps--navigation-system)
8. [Digital Twin](#8-digital-twin)
9. [Dashboard & Visualization](#9-dashboard--visualization)
10. [Telegram Bot & Remote Control](#10-telegram-bot--remote-control)
11. [Perception & Computer Vision](#11-perception--computer-vision)
12. [Communication Protocols](#12-communication-protocols)
13. [Configuration & Settings](#13-configuration--settings)
14. [Setup & Deployment](#14-setup--deployment)
15. [Complete File Structure](#15-complete-file-structure)
16. [Development Roadmap](#16-development-roadmap)

---

## 1. Project Overview

An **intelligent autonomous robot car system** that combines computer vision, GPS navigation, real-time obstacle avoidance, and multi-mode operation into a single cohesive platform. The system uses a **distributed processing architecture**: a **Laptop** runs the brain (YOLOv8 detection, decision engine, sensor fusion), a **Raspberry Pi 5** streams camera video, and two **ESP32** units handle real-time motor control and ultrasonic sensing. All devices communicate over the same Wi-Fi network (SSID: `SANJEEVI`).

### Key Capabilities

- **Autonomous Navigation** — GPS waypoint following with real-time obstacle avoidance
- **Multi-Mode Operation** — 11+ operating modes (follow-me, guidance, summon, delivery, escort, ambulance priority, elder assist, hospital assist, medical delivery, auto-park, crash response)
- **Real-Time Perception** — YOLOv8 object detection on Laptop (video from Pi 5), ultrasonic ranging, sensor fusion
- **Digital Twin** — Live 3D virtual representation synchronized with physical hardware
- **Remote Control** — Telegram bot for command & control, web dashboard for monitoring
- **Decision Engine** — Rule-based + priority-weighted decision making for autonomous behavior
- **GPS Waypoint System** — Pre-planned routes, POI navigation, perimeter zone detection
- **3D Mapping** — Real-time 3D scene reconstruction from depth estimation

---

## 2. System Architecture

### 2.1 System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    SAME WiFi (SANJEEVI)                        │
│                 (Phone Hotspot / Router)                       │
└────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│   ESP32 #1      │  │   ESP32 #2      │  │   Raspberry Pi 5    │
│ Motor Controller│  │ Ultrasonic Sensor│  │   Camera Sender     │
│                 │  │                 │  │                     │
│ Controls 4 motors│  │ HC-SR04 sensor  │  │ Pi Camera / Webcam  │
│ via 2x TB6612FNG│  │ TRIG=GPIO21     │  │                     │
│ drivers         │  │ ECHO=GPIO18     │  │ Streams 640x480     │
│                 │  │                 │  │ JPEG over UDP :5000 │
│ WiFi Station    │  │                 │  │                     │
│ Web UI :8080    │  │                 │  │                     │
│ UDP :9000 / :9001│  │ UDP :9002       │  │                     │
│ UDP Ultra :9002  │  │                 │  │                     │
└────────┬────────┘  └────────┬────────┘  └─────────────────────┘
         │                     │                     │
         │   UDP :9000         │                     │
         │ ← throttle:steer   │                     │
         │                     │  HTTP / UDP :9002   │
         │ ← "ULTRA:XX.XXcm"  │                     │
         │                     │                     │
         │   UDP :9001         │                     │
         │ → STATUS broadcast  │                     │
         │                     │                     │
         ▼                     ▼                     ▼
┌────────────────────────────────────────────────────────────────┐
│                    LAPTOP (Controller/Brain)                    │
│  - YOLOv8 object detection on video stream (UDP :5000)         │
│  - Sensor fusion (ultrasonic < 30cm → priority)               │
│  - Sends throttle:steering commands via UDP :9000              │
│  - Receives ESP32 status via UDP :9001                         │
│  - Receives ultrasonic data via UDP :9002                      │
│  - Modern OpenCV dashboard with info overlay                   │
│  - Optional Flask web server for phone control                 │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
Physical World
     │
     ▼
┌──────────────────────┐     ┌──────────────────────┐
│  Pi 5 Camera         │────►│  Laptop (Brain)      │
│  (UDP video stream)  │     │                      │
└──────────────────────┘     │  YOLOv8 Detection    │
                             │  Sensor Fusion       │
┌──────────────────────┐     │  Decision Engine     │
│  ESP32 #2 Ultrasonic │────►│  Motor Commands      │
│  (UDP :9002)         │     └──────────┬───────────┘
└──────────────────────┘                │ commands
                                         ▼
┌──────────────────────┐     ┌──────────────────────┐
│  ESP32 #1 Motor Ctrl │◄────│  LAN (UDP :9000)     │
│  (2x TB6612FNG / 4 motors)│    └──────────────────────┘
└──────────────────────┘              │
                                       │ state (UDP :9001)
                                       ▼
┌──────────────────────────────────────────────────┐
│  Digital Twin / Dashboard (laptop Flask server)  │
└──────────────────────────────────────────────────┘
```

---

## 3. Hardware Components

### 3.1 Bill of Materials

| Component | Description | Notes |
|-----------|-------------|-------|
| Raspberry Pi 5 | Camera processor | Runs `pi_camera_v2.py` |
| ESP32 #1 | Motor Controller | Runs `esp32_motor_controller.ino` |
| ESP32 #2 | Ultrasonic Sensor | Runs `esp32_ultrasonic.ino` |
| HC-SR04 | Ultrasonic module | TRIG=GPIO21, ECHO=GPIO18, 5V power |
| 2× TB6612FNG | Motor drivers | 4 channels total (left/right pair per driver) |
| 4× DC Motors | Robot drive | Differential steering (skid-steer) |
| Pi Camera / USB Webcam | Vision | 640×480, 30 FPS |

### 3.2 Raspberry Pi 5 — Camera Streamer

| Component | Specification |
|-----------|---------------|
| **Model** | Raspberry Pi 5 |
| **OS** | Raspberry Pi OS (Debian-based) |
| **Role** | Dedicated camera streamer — captures 640×480 video, compresses as JPEG (quality 75%), sends via UDP to laptop on port `:5000` |
| **Camera** | Pi Camera v2 (CSI, 8MP) or USB Webcam |
| **Frame Format** | `[4-byte size][JPEG data]` over UDP |

**Script:** `pi_camera_v2.py`
- Captures 640×480 video at ~30 FPS
- Compresses frames as JPEG (quality 75%)
- Sends over UDP to laptop IP on port `:5000`
- Frame format: `[4-byte size][JPEG data]`
- Prints FPS and size stats every second

### 3.3 ESP32 #1 — Motor Controller (`esp32_motor_controller.ino`)

| Feature | Detail |
|---------|--------|
| **WiFi Mode** | Station mode (connects to SSID: `SANJEEVI`) — no AP mode |
| **Web Interface** | Port `:8080` for manual testing |
| **Command Port** | UDP `:9000` — receives `throttle:steering` format |
| **Status Port** | UDP `:9001` — broadcasts `STATUS:AUTO:150:0.02:25.5` |
| **Ultrasonic Port** | UDP `:9002` — receives `ULTRA:25.50` from ESP32 #2 |
| **Brownout Detection** | Disabled (prevents WiFi dropout when motors spin up) |
| **WiFi Monitor** | Auto-reconnects every 5s if connection lost |
| **Command Timeout** | Stops motors if no command received in 500ms |

**Motor Wiring (4 motors, 2x TB6612FNG drivers):**

| Pin (Driver 1 - LEFT) | Function | Pin (Driver 2 - RIGHT) | Function |
|----------------------|----------|----------------------|----------|
| GPIO 26 | Left Forward IN1 | GPIO 4 | Right Forward IN1 |
| GPIO 27 | Left Forward IN2 | GPIO 5 | Right Forward IN2 |
| GPIO 25 | Left Forward PWM | GPIO 18 | Right Forward PWM |
| GPIO 14 | Left Reverse IN1 | GPIO 19 | Right Reverse IN1 |
| GPIO 13 | Left Reverse IN2 | GPIO 21 | Right Reverse IN2 |
| GPIO 33 | Left Reverse PWM | GPIO 22 | Right Reverse PWM |
| GPIO 32 | STBY (enable) | GPIO 23 | STBY (enable) |

**Steering:** Differential drive — slows one side's motors to turn (no servo steering).

### 3.4 ESP32 #2 — Ultrasonic Sensor (`esp32_ultrasonic.ino`)

| Feature | Detail |
|---------|--------|
| **WiFi Mode** | Station mode (same SSID: `SANJEEVI`) |
| **Sensor** | HC-SR04 (2–400 cm range) |
| **Read Interval** | Every 200ms |
| **Data Sending** | Sends to ESP32 #1 via HTTP GET (`/distance?value=XX`) |
| **Fallback** | UDP on port `:9002` (`ULTRA:25.50` format) |

**Wiring:**

| Pin | Connection |
|-----|-----------|
| GPIO 21 | TRIG (output) |
| GPIO 18 | ECHO (input) |
| 5V | VCC |
| GND | GND |

### 3.5 Motor System

- **Type:** 2× TB6612FNG Dual H-Bridge DC Motor Drivers
- **Total Channels:** 4 (left pair + right pair, each pair on its own driver)
- **Control:** PWM speed control, directional control via IN pins
- **Power:** 7.4V LiPo battery (separate from Pi power)
- **Wheel Configuration:** 4-wheel differential drive (skid-steer)
- **Motor Testing Script:** `check_motors/check_motors.ino` — independent motor test sketch

### 3.6 Sensor Suite

| Sensor | Interface | Purpose |
|--------|-----------|---------|
| Pi Camera v2 / USB Webcam | CSI / USB | Video stream for YOLOv8 detection |
| HC-SR04 Ultrasonic | GPIO (ESP32 #2) | Obstacle distance measurement |
| USB GPS Module | UART (USB) | Position tracking, waypoint navigation |
| (Optional) LiDAR | UART/I2C | Advanced depth mapping (planned) |

---

## 4. Software Components

### 4.1 Laptop — The Brain (`robot_car_v2_modern.py`)

The laptop runs the primary control software:
1. **Receives video** via UDP port `:5000` from Pi 5
2. **YOLOv8 object detection** — detects vehicles (car, motorcycle, bus, truck) and persons
3. **Distance estimation** — from bounding box height: `1.5 × frame_h × 0.8 / bbox_h`
4. **Sensor fusion** — ultrasonic takes priority under 30cm
5. **Sends commands** via UDP port `:9000` to ESP32 #1 (`throttle:steering` format)
6. **Receives status** via UDP port `:9001` from ESP32 #1
7. **Receives ultrasonic data** via UDP port `:9002` from ESP32 #2
8. **Modern OpenCV dashboard** with info overlay

**Decision Tree (Ultrasonic Priority):**
| Distance Range | Action |
|---------------|--------|
| < 15 cm | Immediate STOP (emergency) |
| 15–30 cm | Slow reverse + turn |
| 30–50 cm | Slow forward with caution |
| > 50 cm | Normal operation (YOLO-based) |

### 4.2 Core Project Modules (`core/`)

#### `perception_engine.py`
- Real-time object detection using YOLOv8 (model: `yolov8m.pt` or `yolov8n.pt`)
- Ultrasonic distance fusion
- Obstacle classification and tracking
- Threaded capture with frame queue management

#### `perception_engine_gps.py`
- Extended perception engine with GPS integration
- Maps object detections to GPS coordinates
- Adds spatial context to detected objects

### 4.3 Decision Engine (`decision_engine/`)

Located in `decision_engine/core/`:

#### `decision_maker.py`
- Central decision-making pipeline
- Processes perception data + GPS data + current mode
- Outputs motor commands and mode transitions
- Priority-based action selection

#### `rule_engine.py`
- Deterministic rule-based behavior system
- Safety overrides (e.g., STOP if obstacle < threshold distance)
- Mode-specific rule sets
- Configurable priority levels per rule type

### 4.4 Mode Controller (`hardware_setup/modes/`)

A multi-mode operating system with 11+ distinct behaviors. Each mode is implemented as an HTML/JS file with visualization:

| Mode | File | Description |
|------|------|-------------|
| **Follow Me** | `follow_me_mode.html` | Tracks and follows a person |
| **Guidance** | `guidance_mode.html` | Guides user along path |
| **Summon** | `summon_mode.html` | Robot comes to user's location |
| **Delivery** | `delivery_mode.html` | Autonomous delivery to waypoint |
| **Escort** | `escort_mode.html` | Escorts person/group |
| **Ambulance Priority** | `ambulance_mode.html` | Emergency lane clearance |
| **Auto-Park** | `autopark_mode.html` | Automatic parking |
| **Crash Response** | `crash_response_mode.html` | Post-collision protocol |
| **Elder Assist** | `elder_assist_mode.html` | Support for elderly users |
| **Hospital Assist** | `hospital_assist_mode.html` | Hospital navigation aid |
| **Medical Delivery** | `medical_delivery_mode.html` | Urgent medical transport |

**Mode Selection:** Web UI (`index.html`) + Telegram Bot commands  
**Launch Script:** `START_MODES.bat`

### 4.5 Telemetry & Monitoring (`dashboard/`)

- **`gps_dashboard.py`** — Flask-based GPS monitoring dashboard with real-time map rendering
- **`gps_dashboard_3d.py`** — Three.js 3D visualization of robot position and environment

### 4.6 Main Entry Points

#### `main.py` (project root)
- Primary orchestration script
- Initializes all subsystems: camera, GPS, serial/UDP, decision engine
- Main control loop with sensor fusion and command dispatch

#### `setup.bat`
- Windows batch setup script for dependencies and configuration

### 4.7 System Role Definition (`Role.md`)

Defines the operational persona, behavioral constraints, and default configuration parameters for the autonomous system.

---

## 5. Operating Modes

Each mode represents a distinct autonomous behavior pattern. Modes are selected either via the **web UI** (`modes/index.html`), **Telegram Bot**, or automatically by the decision engine based on context.

### 5.1 Mode State Machine

```
                    ┌──────────────┐
                    │    Idle      │
                    └──────┬───────┘
                           │ mode select
         ┌─────────────────┼──────────────────┐
         │                 │                  │
    ┌────▼────┐     ┌─────▼──────┐     ┌─────▼──────┐
    │ Follow  │     │  Guidance  │     │  Delivery  │
    │   Me    │     │            │     │            │
    └────┬────┘     └─────┬──────┘     └─────┬──────┘
         │                 │                  │
    ┌────▼────┐     ┌─────▼──────┐     ┌─────▼──────┐
    │ Summon  │     │  Escort    │     │  Auto-Park │
    └────┬────┘     └─────┬──────┘     └─────┬──────┘
         │                 │                  │
    ┌────▼──────────────────▼──────────────────▼──────┐
    │              Emergency Override                 │
    │  (Crash Response / Obstacle Avoidance / Stop)   │
    └─────────────────────┬───────────────────────────┘
                          │
                    ┌─────▼──────┐
                    │   Return   │
                    │   to Mode  │
                    └────────────┘
```

### 5.2 Mode Selection Logic

Modes are selected based on:
1. **User Command** (Telegram or Web UI) — highest priority manual override
2. **Context Detection** (e.g., crash detected → crash response, elderly person detected → elder assist)
3. **Scheduled Tasks** (planned delivery routes)
4. **GPS Proximity** (entering hospital zone → hospital assist)

---

## 6. Decision Engine

### 6.1 Architecture

```
┌──────────────────────────────────────────┐
│           Decision Engine (Laptop)         │
│                                          │
│  ┌────────────┐    ┌───────────────┐     │
│  │  Sensor    │    │   Mode        │     │
│  │  Input     │───►│   Context     │     │
│  │  Queue     │    │   Manager     │     │
│  └────────────┘    └───────┬───────┘     │
│                            │             │
│  ┌─────────────────────────▼──────────┐  │
│  │         Rule Engine                 │  │
│  │  ┌──────────┐  ┌──────────┐       │  │
│  │  │ Safety   │  │ Behavior │       │  │
│  │  │ Rules    │  │ Rules    │       │  │
│  │  └──────────┘  └──────────┘       │  │
│  │  ┌──────────┐  ┌──────────┐       │  │
│  │  │ Mode     │  │ Priority │       │  │
│  │  │ Rules    │  │ Arbiter  │       │  │
│  │  └──────────┘  └──────────┘       │  │
│  └────────────────────────────────────┘  │
│                    │                      │
│  ┌─────────────────▼──────────────────┐  │
│  │      Command Generator              │  │
│  │  (throttle:steering for ESP32)     │  │
│  └─────────────────┬──────────────────┘  │
└────────────────────┼─────────────────────┘
                     │ UDP :9000
              ┌──────▼──────┐
              │  ESP32 #1   │
              │  Motor Ctrl │
              └─────────────┘
```

### 6.2 Priority Levels

| Level | Type | Example |
|-------|------|---------|
| 1 (Highest) | Collision Avoidance | Ultrasonic < 15cm → STOP |
| 2 | Emergency Override | Crash detected → HALT + alert |
| 3 | Mode-Specific Task | Delivery: follow GPS waypoints |
| 4 | Navigation | Standard path following |
| 5 (Lowest) | Exploration | Random wander / patrol |

### 6.3 Safety Mechanisms

- **Emergency Distance:** Hard stop if ultrasonic < 15cm
- **Caution Zone:** Slow reverse + turn at 15–30cm
- **Watch Slow:** Slow forward with caution at 30–50cm
- **Timeout:** No command in 500ms → ESP32 stops motors independently
- **WiFi Monitor:** ESP32 auto-reconnects every 5s if connection lost
- **Brownout Protection:** Disabled on ESP32 to prevent motor-spinup WiFi dropout

---

## 7. GPS & Navigation System

### 7.1 GPS Reader (`hardware/gps_reader.py`)

- Reads NMEA sentences from USB GPS receiver (9600 baud)
- Extracts $GPGGA and $GPRMC sentences
- Outputs: latitude, longitude, altitude, speed, heading, fix quality, satellite count
- Configurable COM port and baud rate via `config/gps_settings.py`

### 7.2 GPS Settings (`config/gps_settings.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `GPS_PORT` | "COM3" (Win) / "/dev/ttyACM0" (Pi) | Serial port |
| `GPS_BAUD` | 9600 | Baud rate |
| `GPS_TIMEOUT` | 1.0s | Read timeout |
| `HOME_LAT`, `HOME_LON` | configurable | Home base coordinates |
| `WAYPOINT_TOLERANCE` | 3.0m | Distance to consider waypoint reached |
| `MAX_WAYPOINTS` | 50 | Max route waypoints |
| `ZONE_RADIUS` | 100m | POI detection radius |

### 7.3 Navigation Features

- **Waypoint Following:** Autonomous travel through ordered GPS waypoints
- **POI (Point of Interest) Management:** Named locations with configurable radius
- **Zone Detection:** Geofenced areas trigger mode changes
- **Route Planning:** Pre-planned routes loaded from GPX files
- **Return-to-Home:** Automatic return to base coordinates
- **Speed Adjustment:** Speed modulated by distance to next waypoint

### 7.4 GPS Dashboard (`dashboard/gps_dashboard.py`)

- Flask web server displaying real-time robot position on map
- Waypoint list with status indicators
- Path history with polyline overlay
- Current telemetry: speed, heading, satellite count, fix quality

---

## 8. Digital Twin

### 8.1 Overview

The Digital Twin system creates a **real-time synchronized virtual replica** of the physical robot car. It enables remote monitoring, simulation, and debugging without needing physical access to the hardware.

### 8.2 Components (`hardware_setup/digital_twin/`)

#### `digital_twin_server.py`
- Flask + WebSocket server
- Receives telemetry from the physical robot
- Broadcasts state to connected web clients
- API endpoints: `/api/state`, `/api/command`, `/api/telemetry`
- WebSocket for real-time updates (50ms update interval)

#### `hardware_sync_bridge.py`
- Bridge between physical hardware and digital twin
- Reads from: ESP32 motor controller state, ultrasonic sensor, GPS, Pi camera
- Writes to: Digital Twin state model
- Handles serialization/deserialization of hardware protocol
- Provides bidirectional sync (commands can be sent to physical robot from twin)

### 8.3 Twin State Model

```json
{
  "robot": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "speed": 0.0,
    "heading": 0.0,
    "throttle": 0,
    "steering": 0.0
  },
  "sensors": {
    "ultrasonic": {"distance": 45.2, "timestamp": 1234567890},
    "gps": {"lat": 12.34, "lon": 56.78, "altitude": 100.0, "speed": 0.5}
  },
  "status": {
    "mode": "autonomous",
    "battery": 85.0,
    "connection": "active",
    "last_command": "150:0.02",
    "uptime": 3600
  },
  "environment": {
    "detections": [
      {"class": "person", "confidence": 0.95, "bbox": [100, 200, 300, 400]}
    ],
    "obstacles": [
      {"distance": 45.2, "angle": 0}
    ]
  }
}
```

### 8.4 Benefits

| Feature | Benefit |
|---------|---------|
| Real-time 3D visualization | Monitor robot without line-of-sight |
| State logging & playback | Debug past behaviors |
| Hardware-in-the-loop simulation | Test algorithms safely |
| Remote command injection | Control from anywhere |
| Performance metrics | Track speed, accuracy, battery life |

---

## 9. Dashboard & Visualization

### 9.1 Web Dashboard (Flask)

The main dashboard application provides:
- **Live Video Feed** — Stream from on-board camera with YOLO detection overlays
- **GPS Map** — Real-time position on interactive map (OpenStreetMap)
- **Telemetry Panel** — Speed, heading, battery, sensor readings
- **Mode Controls** — Switch between operating modes
- **Command Console** — Manual override controls
- **System Status** — Connection status, uptime, errors

### 9.2 Bento Grid Dashboard (`bento.html`)

A modern bento-grid style HTML dashboard featuring:
- Responsive card-based layout
- Real-time data visualization
- Camera feed, GPS map, telemetry, and controls in one view
- Dark theme with gradient accents
- Animated transitions and hover effects

### 9.3 OpenCV Dashboard

The primary operator dashboard is built with **OpenCV** (`robot_car_v2_modern.py`):
- Live camera feed with YOLO bounding boxes and labels
- Ultrasonic distance overlay
- Throttle/steering indicator
- FPS counter
- Detection stats (objects found, confidence levels)

---

## 10. Telegram Bot & Remote Control

### 10.1 Overview (`hardware_setup/telegram_bot/`)

The Telegram Bot provides **remote command and control** capabilities via any Telegram client (phone, desktop, web). It runs as a standalone Python process on the laptop or Raspberry Pi.

### 10.2 Core Files

| File | Purpose |
|------|---------|
| `telegram_bot_main.py` | Main bot entry point with command handlers |
| `yolo_detector.py` | YOLO inference wrapper for image processing |
| `advanced_features.py` | Extended commands: photo, video, mode control |
| `main_with_dist.py` | Alternative entry point with distance fusion |

### 10.3 Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message + mode selection |
| `/status` | Current robot state, battery, GPS position |
| `/photo` | Capture and send real-time photo |
| `/video` | Record and send short video clip |
| `/mode <name>` | Switch operating mode |
| `/goto <lat>,<lon>` | Send robot to GPS coordinates |
| `/stop` | Emergency stop |
| `/home` | Return to home base |
| `/speed <0-255>` | Set movement speed |
| `/detect` | Run object detection and report results |
| `/path` | Show current planned path |
| `/battery` | Battery level report |
| `/ping` | Connection test (latency check) |

---

## 11. Perception & Computer Vision

### 11.1 Object Detection (YOLOv8)

Runs on the **laptop** (not Pi) for maximum processing power:

- **Model:** Ultralytics YOLOv8 (`yolov8m.pt` primary, `yolov8n.pt` for lightweight)
- **Classes Detected:** COCO dataset (80 classes including person, car, bicycle, traffic signs)
- **Video Source:** UDP stream from Pi 5 camera (640×480, ~30 FPS)
- **Frame Rate:** Depends on laptop hardware (typically 15–30 FPS with GPU)
- **Confidence Threshold:** 0.5 (configurable)

### 11.2 Distance Estimation

Distance is estimated from bounding box height:
```
distance = 1.5 × frame_height × 0.8 / bbox_height
```

### 11.3 Sensor Fusion

```
YOLOv8 Detections ──┐
(Distance from bbox) ├──► Sensor Fusion ──► Decision Engine
                     │        (Laptop)
Ultrasonic ──────────┘
(UDP :9002, priority < 30cm)
```

- YOLO detections correlated with ultrasonic distance measurements
- **Ultrasonic takes priority under 30cm** (more reliable for close range)
- Distance estimation from bounding box used for objects > 30cm
- GPS coordinates tag static obstacles for future navigation

### 11.4 Lane Detection

- Lane detection implemented in earlier iterations (`lane_detection_claude/`, `land_detection_claude_2/`)
- Basic visual odometry from camera frame differences
- Traffic sign detection via YOLO fine-tuning (Indian Traffic Sign Dataset)

---

## 12. Communication Protocols

### 12.1 Network Architecture

All devices connect to the **same Wi-Fi network** (`SANJEEVI` — phone hotspot or router):

```
┌────────────────────────────────────────────────────────────┐
│                    WiFi SSID: SANJEEVI                      │
└────────────────────────────────────────────────────────────┘
```

### 12.2 Pi 5 → Laptop (Video Stream)

```
Pi 5 ──(UDP port :5000)──► Laptop

Frame Format: [4-byte frame size (uint32)] [JPEG data]
Video: 640×480, ~30 FPS, JPEG quality 75
```

### 12.3 Laptop → ESP32 #1 (Motor Commands)

```
Laptop ──(UDP port :9000)──► ESP32 #1 (Motor Controller)

Command Format: "throttle:steering"
  throttle:  0-255 (forward/backward intensity)
  steering:  -1.0 to 1.0 (negative = left, positive = right)

Examples:
  150:0.00    → Forward at speed 150, straight
  100:-0.50  → Forward at speed 100, turning left
  0:0.00     → Stop
  -120:0.30  → Reverse at speed 120, turning right
```

### 12.4 ESP32 #1 → Laptop (Status Broadcast)

```
ESP32 #1 ──(UDP port :9001)──► Laptop

Status Format: "STATUS:<mode>:<throttle>:<steering>:<battery>"
Example: "STATUS:AUTO:150:0.02:25.5"
```

### 12.5 ESP32 #2 → ESP32 #1 / Laptop (Ultrasonic Data)

```
ESP32 #2 ──(HTTP GET /distance?value=XX)──► ESP32 #1
     or ──(UDP port :9002)──► Laptop (fallback for newer variants)

Format: "ULTRA:25.50"  (distance in cm, 2 decimal places)
Example: "ULTRA:45.23"
```

### 12.6 Network Services Summary

| Service | Port | Protocol | Direction |
|---------|------|----------|-----------|
| Video Stream | 5000 | UDP | Pi 5 → Laptop |
| Motor Commands | 9000 | UDP | Laptop → ESP32 #1 |
| Status Broadcast | 9001 | UDP | ESP32 #1 → Laptop |
| Ultrasonic Data | 9002 | UDP | ESP32 #2 → ESP32 #1/Laptop |
| ESP32 Web UI | 8080 | HTTP | Browser → ESP32 #1 (manual test) |
| Flask Dashboard | 5000 | HTTP | Laptop → Browser |
| Telegram Bot | 443 | HTTPS (outbound) | Bot → API |

---

## 13. Configuration & Settings

### 13.1 Main Settings (`config/settings.py`)

```python
# General
DEBUG = False
LOG_LEVEL = "INFO"
UPDATE_INTERVAL = 0.1  # seconds

# Camera
CAMERA_ID = 0  # /dev/video0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Motor
MOTOR_MIN_SPEED = 80
MOTOR_MAX_SPEED = 255
DEFAULT_SPEED = 150

# Ultrasonic
USONIC_THRESHOLD = 50  # cm — obstacle avoidance trigger
USONIC_EMERGENCY = 15  # cm — hard stop

# UDP (Updated: Dual ESP32 v2)
MOTOR_UDP_PORT = 9000       # throttle:steering commands
STATUS_UDP_PORT = 9001      # ESP32 status broadcast
USONIC_UDP_PORT = 9002      # Ultrasonic data
VIDEO_UDP_PORT = 5000       # Pi 5 video stream
MOTOR_UDP_IP = "192.168.x.x"  # ESP32 #1 IP address
CAM_UDP_IP = "192.168.x.x"    # Pi 5 IP address

# YOLO
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
YOLO_IOU = 0.45
```

### 13.2 GPS Settings (`config/gps_settings.py`)

```python
GPS_PORT = "COM3"  # Windows; "/dev/ttyACM0" on Pi
GPS_BAUD = 9600
GPS_TIMEOUT = 1.0

HOME_LAT = 12.9716   # Default: Bangalore
HOME_LON = 77.5946

WAYPOINT_TOLERANCE = 3.0  # meters
MAX_WAYPOINTS = 50
ZONE_RADIUS = 100  # meters
```

---

## 14. Setup & Deployment

### 14.1 Prerequisites

- **Hardware:**
  - Raspberry Pi 5
  - ESP32 Dev Board (x2)
  - Pi Camera v2 or USB Webcam
  - HC-SR04 Ultrasonic Sensor
  - TB6612FNG Motor Driver (x2)
  - DC Motors (x4)
  - 7.4V LiPo battery
  - GPS USB Module
  - Laptop (Windows/Linux) with Python

- **Software:**
  - Python 3.8+ (on laptop)
  - pip
  - PlatformIO / Arduino IDE (for ESP32)
  - OpenCV, Ultralytics YOLOv8 (on laptop)
  - Raspberry Pi OS (on Pi 5)
  - WiFi hotspot or router (SSID: `SANJEEVI`)

### 14.2 Quick Setup

```batch
:: On laptop — install dependencies & run
pip install -r requirements.txt
python robot_car_v2_modern.py
```

### 14.3 ESP32 Flashing

1. Open each `.ino` file in Arduino IDE (or PlatformIO)
2. Configure Wi-Fi credentials (SSID: `SANJEEVI`, password, target laptop IP)
3. Flash to respective ESP32 boards
4. Verify UDP communication: ESP32 #1 on ports `:9000`/`:9001`, ESP32 #2 on `:9002`

**Note:** ESP32 #1 must have brownout detection disabled in code to prevent WiFi dropout during motor spinup.

### 14.4 Pi 5 Setup

1. Install Raspberry Pi OS (with desktop or lite)
2. Install Python dependencies
3. Enable Camera interface (`raspi-config`)
4. Run camera streamer: `python pi_camera_v2.py`
5. Set static IP or use DHCP reservation

### 14.5 Startup Sequence

```
1. Power on ESP32 boards (wait ~10s for WiFi connection to SANJEEVI)
2. Power on Pi 5 → auto-starts pi_camera_v2.py (or run manually)
3. On laptop: python robot_car_v2_modern.py
4. Wait for "Connected" confirmation in terminal
5. Use arrow keys or gamepad for manual control
6. Switch to autonomous mode via Telegram /mode or key press
```

### 14.6 Raspberry Pi Headless Setup

Refer to `jetson_nano_headless_inst.md` for headless configuration guidance:
- Enable SSH, configure Wi-Fi, set static IP
- Enable Camera interface (`raspi-config`)
- Auto-start camera script via systemd or cron

---

## 15. Complete File Structure

```
combined_final_new_gps/combined_proj_folder/
│
├── main.py                          # Primary orchestration entry point
├── setup.bat                        # Windows setup script
├── requirements.txt                 # Python dependencies
├── Role.md                          # System role definition
├── README.md                        # Project readme
│
├── config/
│   ├── settings.py                  # Main configuration
│   └── gps_settings.py             # GPS-specific configuration
│
├── core/
│   ├── perception_engine.py         # YOLO + ultrasonic fusion
│   └── perception_engine_gps.py     # Perception with GPS integration
│
├── decision_engine/
│   └── core/
│       ├── decision_maker.py        # Central decision pipeline
│       └── rule_engine.py           # Rule-based behavior engine
│
├── hardware/
│   ├── hardware_integration.py      # Hardware abstraction layer
│   ├── gps_reader.py                # NMEA GPS parser
│   └── pi_sender.py                 # UDP command sender to ESP32
│
├── dashboard/
│   ├── gps_dashboard.py             # Flask GPS monitoring
│   └── gps_dashboard_3d.py          # Three.js 3D visualization
│
├── hardware_setup/
│   ├── check_motors/
│   │   └── check_motors.ino         # Motor test sketch
│   │
│   ├── digital_twin/
│   │   ├── digital_twin_server.py   # Flask+WS digital twin server
│   │   ├── hardware_sync_bridge.py  # Physical↔Digital sync bridge
│   │   ├── README_DIGITAL_TWIN.md
│   │   ├── STEP_BY_STEP_SETUP.md
│   │   ├── QUICK_SETUP.md
│   │   ├── FINAL_STARTUP_GUIDE.md
│   │   └── QUICK_START_FIXED.md
│   │
│   ├── robot_car_and_pi_v2_2esp/
│   │   ├── esp32_motor_controller/
│   │   │   └── esp32_motor_controller.ino  # 4-motor, 2-TB6612FNG PWM control
│   │   │
│   │   ├── esp32_ultrasonic_udp_v1/
│   │   │   ├── esp32_ultrasonic_udp_v1.ino # HC-SR04 UDP sender
│   │   │   └── SETUP_GUIDE.md
│   │   │
│   │   ├── pi_camera_v2.py               # Pi 5 Camera UDP streamer
│   │   ├── robot_car_v2_modern.py        # Laptop brain (YOLO+fusion)
│   │   ├── robot_car_v2_with_digital_twin.py
│   │   ├── robot_car_v2_modern_WITH_DIGITAL_TWIN - Copy.py
│   │   ├── START_HERE_V2.bat
│   │   ├── START_HERE_V4.bat
│   │   ├── robot_car_v2_demo_path.py
│   │   ├── context.txt                   # Dev notes & debug logs
│   │   ├── Resumes/
│   │   ├── Back/
│   │   ├── Tries/
│   │   └── Camera/
│   │
│   ├── telegram_bot/
│   │   ├── telegram_bot_main.py
│   │   ├── yolo_detector.py
│   │   ├── advanced_features.py
│   │   ├── main_with_dist.py
│   │   ├── requirements.txt
│   │   ├── SETUP.md
│   │   ├── README.md
│   │   ├── FEATURES.md
│   │   ├── telegram_bot_cmnds.md
│   │   ├── HACKATHON_SUMMARY.md
│   │   └── INDEX.md
│   │
│   └── modes/
│       ├── index.html
│       ├── README.md
│       ├── START_MODES.bat
│       ├── HACKATHON_GUIDE.md
│       ├── follow_me_mode.html
│       ├── guidance_mode.html
│       ├── summon_mode.html
│       ├── delivery_mode.html
│       ├── escort_mode.html
│       ├── ambulance_mode.html
│       ├── autopark_mode.html
│       ├── crash_response_mode.html
│       ├── elder_assist_mode.html
│       ├── hospital_assist_mode.html
│       └── medical_delivery_mode.html
│
├── bento.html                        # Bento-grid dashboard
├── DECISION_ENGINE_SUMMARY.md
├── new_plan_gps.md
├── test_map_3d.html
├── test_map_3d - Copy.html
├── topoexport_3D_modeling.dxf
├── yolov8m.pt                        # YOLOv8 medium model
├── yolov8n.pt                        # YOLOv8 nano model
└── __init__.py
```

---

## 16. Development Roadmap

### Completed ✅
- ✅ Dual-ESP32 architecture (motor control + ultrasonic sensor)
- ✅ YOLOv8 object detection on laptop (video from Pi 5)
- ✅ GPS waypoint navigation system
- ✅ Decision engine with rule-based priority arbitration
- ✅ Multi-mode operating system (11 modes)
- ✅ Telegram bot for remote control
- ✅ Digital twin with 3D visualization
- ✅ Web dashboard with GPS map and telemetry
- ✅ Motor testing and calibration scripts
- ✅ Obstacle avoidance via ultrasonic sensor fusion
- ✅ Pi 5 camera UDP streamer

### In Progress 🔄
- 🔄 3D SLAM mapping integration (see `perp_autonomous_slam_detection/`)
- 🔄 Advanced lane detection refinement
- 🔄 Intelligent road system

### Planned 📋
- 📋 LiDAR integration for enhanced depth mapping
- 📋 Multi-robot coordination via mesh networking
- 📋 Machine learning for adaptive mode transitions
- 📋 Voice command interface
- 📋 Real-time path re-planning with dynamic obstacle avoidance
- 📋 Edge AI optimization (TensorFlow Lite / ONNX runtime)
- 📋 Battery management and autonomous charging
- 📋 Cloud integration for fleet management

---

## Appendix A: Network Topology

```
                    ┌──────────────┐
                    │   Internet   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  SSID:       │
                    │  SANJEEVI    │
                    │  (Hotspot)   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐     ┌─────▼──────┐    ┌─────▼──────┐
   │  Phone   │     │  Laptop    │    │  Pi 5      │
   │(Telegram)│     │(Brain)     │    │(Camera)    │
   │          │     │ :5000 rcv  │    │ :5000 snd  │
   │          │     │ :9000 snd  │    │            │
   │          │     │ :9001 rcv  │    │            │
   │          │     │ :9002 rcv  │    │            │
   └──────────┘     └──────┬─────┘    └────────────┘
                           │
          ┌────────────────┼─────┐
          │                │     │
     ┌────▼─────┐    ┌─────▼─────┐
     │ ESP32 #1 │    │ ESP32 #2  │
     │(Motor)   │    │(USonic)   │
     │ :9000 rcv│    │ :9002 snd │
     │ :9001 snd│    │           │
     │ :9002 rcv│    │           │
     │ :8080 Web│    │           │
     └──────────┘    └───────────┘
```

## Appendix B: Pinout Reference

### ESP32 #1 — Motor Controller (2× TB6612FNG, 4 Motors)

**Driver 1 (LEFT pair of motors):**

| ESP32 Pin | TB6612FNG #1 | Function |
|-----------|--------------|----------|
| GPIO 26 | AIN1 | Left Forward Motor — Direction 1 |
| GPIO 27 | AIN2 | Left Forward Motor — Direction 2 |
| GPIO 25 | PWMA | Left Forward Motor — Speed (PWM) |
| GPIO 14 | BIN1 | Left Reverse Motor — Direction 1 |
| GPIO 13 | BIN2 | Left Reverse Motor — Direction 2 |
| GPIO 33 | PWMB | Left Reverse Motor — Speed (PWM) |
| GPIO 32 | STBY | Standby — HIGH = enabled |

**Driver 2 (RIGHT pair of motors):**

| ESP32 Pin | TB6612FNG #2 | Function |
|-----------|--------------|----------|
| GPIO 4 | AIN1 | Right Forward Motor — Direction 1 |
| GPIO 5 | AIN2 | Right Forward Motor — Direction 2 |
| GPIO 18 | PWMA | Right Forward Motor — Speed (PWM) |
| GPIO 19 | BIN1 | Right Reverse Motor — Direction 1 |
| GPIO 21 | BIN2 | Right Reverse Motor — Direction 2 |
| GPIO 22 | PWMB | Right Reverse Motor — Speed (PWM) |
| GPIO 23 | STBY | Standby — HIGH = enabled |

### ESP32 #2 — Ultrasonic Sensor

| ESP32 Pin | HC-SR04 | Function |
|-----------|---------|----------|
| GPIO 21 | Trig | Trigger pulse (10µs) |
| GPIO 18 | Echo | Echo pulse (duration = distance) |
| 5V | VCC | Power |
| GND | GND | Ground |

### Raspberry Pi 5 — Camera & GPS

| Pi Port | Device | Interface |
|---------|--------|-----------|
| CSI-2 | Pi Camera v2 | Camera Serial Interface |
| USB 2.0 | GPS Module | USB-UART (NMEA @ 9600 baud) |
| USB 3.0 | (Future: LiDAR) | USB |
| USB 3.0 | USB Webcam | USB Video Class |

---

*This document is maintained as the authoritative reference for the Intelligent Autonomous Robot Car System. Update this document whenever significant architectural changes are made.*
