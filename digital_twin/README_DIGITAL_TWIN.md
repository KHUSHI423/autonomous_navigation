# 🔮 Digital Twin + Simulation System
## Autonomous Robot Car - Virtual Replica & Route Testing Platform

> **A hackathon-winning Digital Twin solution** that provides a photorealistic 3D virtual replica of your robot car with live telemetry, route simulation, and incident replay capabilities.

---

## 🎯 Overview

The Digital Twin system creates a **real-time virtual replica** of your physical robot car, enabling:

- **🧍 Live 3D Visualization** - See your robot's exact position, orientation, and sensor data in a stunning cyberpunk-themed 3D environment
- **🧠 Simulate Before Action** - Test routes and maneuvers virtually before executing on physical hardware
- **📊 Incident Replay** - Record, analyze, and replay any incident with full state reconstruction
- **🎮 Unity-Style Controls** - Multiple camera modes (Follow, FPV, Top-Down, Cinematic, Orbit)

---

## 🚀 Quick Start

### Step 1: Start the Digital Twin Server

```bash
# Windows
START_DIGITAL_TWIN.bat

# Or manually
python digital_twin_server.py
```

### Step 2: Open the Web Interface

Navigate to: **http://localhost:8080**

### Step 3: Connect Your Robot (Optional)

If you have the physical robot running:
1. Ensure `robot_car_v2_modern.py` is running
2. The Digital Twin will automatically receive telemetry via UDP
3. Watch the 3D model mirror your robot's movements in real-time!

---

## 📁 File Structure

```
digital_twin/
├── index.html              # Main 3D web interface (Three.js)
├── digital_twin_server.py  # WebSocket server for real-time data
├── route_simulator.py      # Route planning & simulation engine
├── START_DIGITAL_TWIN.bat  # Windows launcher
└── README_DIGITAL_TWIN.md  # This file
```

---

## 🎮 Features

### 1. Live 3D Visualization

| Feature | Description |
|---------|-------------|
| **Virtual Robot Model** | Exact 3D replica with chassis, wheels, sensors, camera mount |
| **Real-time Telemetry** | Position, heading, speed, throttle, steering, ultrasonic |
| **Sensor Fusion Display** | Camera, Ultrasonic, GPS, IMU status indicators |
| **Neon Cyberpunk Theme** | Stunning visual design matching project aesthetic |

### 2. Camera Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **🎯 Follow** | Third-person camera trailing robot | General navigation |
| **🌍 Orbit** | Free-rotating orbital camera | Inspection, demos |
| **📍 Top-Down** | Bird's eye view from above | Route planning |
| **👁️ FPV** | First-person from robot camera | Immersive driving |
| **🎬 Cinematic** | Sweeping dramatic angles | Presentations |

### 3. Route Simulation

```python
# Example: Simulate a route before physical execution
python route_simulator.py

> simulate figure_eight
🚀 Starting simulation of 'figure_eight'...
📍 Waypoint 1/9: (0, 0)
  Pos: (0.0, 0.0) | Dist: 42.4cm | Ultra: 95cm
📍 Waypoint 2/9: (30, 30)
  ⚠️ Obstacle at 18.5cm!
  ...
✅ Simulation complete!
```

**Sample Routes Included:**
- `square_loop` - Basic square pattern
- `figure_eight` - Figure-8 navigation
- `obstacle_course` - Complex avoidance course

### 4. Incident Replay

**Automatically Recorded Incidents:**
- ⚠️ Ultrasonic Close (< 30cm)
- 🛑 Emergency Stops
- 🔄 Obstacle Avoidance Maneuvers
- 📡 Sensor Failures
- 🔋 Low Battery

**Timeline Controls:**
- ⏮️ Jump to start
- ▶️ Play/Pause
- ⏹️ Stop
- ⏭️ Jump to end
- 📍 Scrub to any point

---

## 🔧 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   Physical   │────▶│   WebSocket  │────▶│   Web UI     │ │
│  │    Robot     │ UDP │   Server     │ WS  │  (Three.js)  │ │
│  │              │     │  (Port 8765) │     │ (Port 8080)  │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │  Telemetry   │     │   Incident   │     │   Camera     │ │
│  │   Stream     │     │   Recorder   │     │   Controls   │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Physical Robot** → UDP packets (throttle, steering, ultrasonic)
2. **WebSocket Server** → Receives UDP, broadcasts to clients
3. **Web Interface** → Renders 3D scene, updates robot model
4. **Incident Recorder** → Logs state snapshots for replay

---

## 🛠️ Configuration

### Network Settings

Edit `digital_twin_server.py`:

```python
WEBSOCKET_PORT = 8765      # WebSocket server port
HTTP_PORT = 8080           # Web interface port
UDP_ROBOT_PORT = 9001      # Receive robot telemetry
UDP_CAMERA_PORT = 5000     # Receive camera frames
```

### Robot Integration

The Digital Twin automatically receives data from `robot_car_v2_modern.py` via UDP.

**Required format:**
```
STATUS:AUTO:150:0.02:25.5
       │    │   │    └─ Ultrasonic (cm)
       │    │   └────── Steering (-1.0 to 1.0)
       │    └────────── Throttle (0-255)
       └─────────────── Mode (AUTO/MANUAL)
```

### Simulation Mode

If no physical robot is connected, the system runs in **Simulation Mode**:
- Generates realistic movement patterns
- Simulates sensor readings
- Creates sample incidents for demo

---

## 📊 Dashboard Components

### Header Status Bar
- 📍 **Position** - X, Z coordinates
- 🧭 **Heading** - Orientation in degrees
- ⚡ **Speed** - Current velocity (cm/s)
- 📡 **Ultrasonic** - Front obstacle distance
- 🎮 **Mode** - AUTO / MANUAL
- 📶 **Connection** - WebSocket status

### Left Panel - Telemetry
- **Vehicle State** - Throttle, Steering, Battery, Temperature
- **Sensor Fusion** - Camera, Ultrasonic, GPS, IMU status
- **Detection Objects** - YOLO detections with distances

### Right Panel - Simulation
- **Route Simulation** - Start/monitor route tests
- **Incident Replay** - Browse and replay incidents
- **Mini Map** - Real-time position overview

---

## 🎯 Usage Examples

### Example 1: Live Monitoring

```bash
# Start server
python digital_twin_server.py

# Open browser
http://localhost:8080

# Click "Telemetry" to see live sensor data
# Click "Simulation" to view route options
```

### Example 2: Route Simulation

```bash
# Start route simulator
python route_simulator.py

# List available routes
> plan

# Simulate a route
> simulate figure_eight

# View recorded incidents
> incidents
```

### Example 3: Incident Analysis

1. **During Operation**: Incidents auto-recorded when ultrasonic < 30cm
2. **After Operation**: 
   - Open web interface
   - Click "Replay" button
   - Select incident from list
   - Use timeline scrubber to analyze

### Example 4: Export Incident Report

```python
# In route_simulator.py
> incidents export

# Creates: incidents.json
{
  "exported_at": "2026-03-23T12:34:56",
  "total_incidents": 5,
  "incidents": [
    {
      "id": 1,
      "type": "ULTRASONIC_CLOSE",
      "severity": "high",
      "position": {"x": 25.3, "z": 12.1},
      "state": {...}
    }
  ]
}
```

---

## 🎨 Visual Design

The Digital Twin uses a **cyberpunk/neon aesthetic** matching the project theme:

- **Colors**: Neon cyan (#00f3ff), Purple (#bc13fe), Pink (#ff2a6d), Green (#39ff14)
- **Fonts**: Orbitron (headings), Rajdhani (body), Share Tech Mono (data)
- **Effects**: Glass morphism, neon glow, scanlines, particle systems
- **Animations**: Smooth camera transitions, wheel rotation, particle flow

---

## 🔌 API Reference

### WebSocket Messages

**Client → Server:**
```json
{
  "type": "command",
  "command": "start_simulation"
}
```

**Server → Client:**
```json
{
  "type": "state_update",
  "state": {
    "position": {"x": 10.5, "y": 0, "z": 20.3},
    "rotation": 1.57,
    "throttle": 150,
    "steering": 0.02,
    "speed": 75,
    "ultrasonic": 45.2,
    "mode": "AUTO",
    "detections": [],
    "timestamp": 1711180800.123
  }
}
```

### UDP Packet Format

**Receive (from robot):**
```
STATUS:<mode>:<throttle>:<steering>:<ultrasonic>
Example: STATUS:AUTO:150:0.02:25.5
```

---

## 🐛 Troubleshooting

### Issue: Web interface not loading

**Solution:**
1. Check server is running: `python digital_twin_server.py`
2. Verify port 8080 is not in use
3. Try: `http://127.0.0.1:8080`

### Issue: No robot data appearing

**Solution:**
1. Ensure `robot_car_v2_modern.py` is running
2. Check UDP port 9001 is open
3. Verify robot sends STATUS messages

### Issue: 3D model not visible

**Solution:**
1. Check browser console for errors
2. Ensure WebGL is enabled
3. Try different browser (Chrome recommended)

### Issue: Camera controls not working

**Solution:**
1. Click on 3D viewport to focus
2. Try different camera mode
3. Refresh page

---

## 🏆 Hackathon Highlights

### Why This Wins:

1. **🎯 Unique Value Proposition**
   - First Digital Twin for student robot car projects
   - Bridges physical and virtual worlds seamlessly

2. **🔥 Visual Impact**
   - Stunning cyberpunk aesthetics
   - Smooth 60 FPS 3D rendering
   - Professional demo-ready interface

3. **🧠 Technical Depth**
   - Real-time WebSocket communication
   - Sensor fusion visualization
   - Incident recording and replay
   - Route simulation engine

4. **🎮 User Experience**
   - Intuitive controls
   - Multiple camera perspectives
   - Interactive timeline
   - Comprehensive telemetry

5. **📈 Scalability**
   - Can add more sensors (LiDAR, IMU)
   - Support for multiple robots
   - Cloud deployment ready
   - AR/VR integration possible

---

## 🚀 Future Enhancements

- [ ] **Multi-Robot Support** - Control fleet of robots
- [ ] **AR Overlay** - Project digital twin onto real world
- [ ] **VR Mode** - Full immersive virtual reality
- [ ] **ML Prediction** - Predict collisions before they happen
- [ ] **Cloud Sync** - Remote monitoring via web
- [ ] **Voice Commands** - "Show me the last incident"
- [ ] **Haptic Feedback** - Feel obstacles through controller

---

## 📄 License

Part of the Autonomous Robot Car project.

---

## 🙏 Credits

**Developed for:** Autonomous Infrastructure Intelligence System

**Inspired by:** PROJECT NOVA design language

**Technologies:** Three.js, WebSocket, Python, WebGL

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review console logs in browser
3. Verify server is running
4. Ensure network ports are open

---

**🎯 Ready to demo? Run `START_DIGITAL_TWIN.bat` and impress the judges!**
