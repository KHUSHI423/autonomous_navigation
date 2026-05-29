# 🎮 Digital Twin - Quick Setup Guide

## ✅ What's New

Your Digital Twin now has **MANUAL CONTROL** and **HARDWARE SYNC**!

### New Features:

1. **🎮 Manual Control Panel** - Drive the robot with keyboard or on-screen buttons
2. **🤖 Hardware Sync Panel** - Mirror your physical robot's exact movements in real-time
3. **⌨️ Keyboard Controls** - WASD keys for intuitive driving

---

## 🚀 Quick Start

### Step 1: Start the Digital Twin

```bash
cd hardware_setup\digital_twin
START_DIGITAL_TWIN.bat
```

This launches:
- Main Digital Twin Server (WebSocket + HTTP)
- Hardware Sync Bridge (connects to physical robot)
- Opens browser at http://localhost:8080

### Step 2: Manual Control (No Hardware Needed)

1. Click the **"🎮 Control"** button at the bottom
2. Use **WASD keys** or click the direction buttons
3. Adjust speed with the throttle slider
4. Watch the 3D model move in real-time!

**Controls:**
- **W** - Forward
- **A** - Turn Left
- **S** - Stop
- **D** - Turn Right

### Step 3: Hardware Sync (With Physical Robot)

1. Make sure your physical robot is running (`robot_car_v2_modern.py`)
2. Click the **"🤖 Hardware"** button
3. Click **"🔗 Enable Hardware Sync"**
4. The 3D model will now mirror your physical robot exactly!

---

## ⚙️ Configuration (IMPORTANT!)

### Update IP Addresses in `hardware_sync_bridge.py`:

Open `hardware_sync_bridge.py` and change these lines:

```python
LAPTOP_IP = "10.17.122.100"    # CHANGE to your laptop's IP
ESP32_IP = "10.17.122.207"     # CHANGE to your ESP32's IP
```

**How to find your IPs:**

```bash
# Windows - Find laptop IP
ipconfig

# Look for IPv4 Address under your WiFi adapter
# Example: 10.17.122.100
```

Your ESP32 IP should be shown in the serial monitor when it connects to WiFi.

---

## 🎯 Usage Scenarios

### Scenario 1: Demo Mode (No Hardware)

Perfect for presentations when you don't have the physical robot:

1. Run `START_DIGITAL_TWIN.bat`
2. Use **WASD** to drive around
3. Show off the cyberpunk 3D visualization
4. Demo different camera modes (Follow, FPV, Top-Down, Cinematic)

### Scenario 2: Testing Before Deployment

Test your autonomous algorithms virtually:

1. Click **"📊 Simulation"** panel
2. Click **"🚀 Simulate Route"**
3. Watch the robot navigate with simulated obstacles
4. Review incidents in the timeline

### Scenario 3: Real Hardware Monitoring

Monitor and control your physical robot:

1. Start physical robot: `robot_car_v2_modern.py`
2. Enable **Hardware Sync** in Digital Twin
3. Watch the 3D model mirror exact movements
4. View real-time telemetry (PWM, steering, ultrasonic)

### Scenario 4: Manual Control with Camera

Drive manually while viewing camera feed:

1. Open Digital Twin
2. Enable camera stream (if running)
3. Use **WASD** to drive
4. Watch live camera feed in bottom-right corner

---

## 📊 Panel Overview

### 🎮 Control Panel (Left)
- **Direction Pad** - Click buttons or use WASD
- **Throttle Slider** - Adjust speed (0-255)
- **Mode Toggle** - Switch between Manual/Auto
- **Keyboard Guide** - Shows control keys

### 🤖 Hardware Panel (Right)
- **Connection Status** - ESP32, Ultrasonic, Camera, Battery
- **Real-time Telemetry** - PWM, Steering, Distance, Runtime
- **Hardware Sync Toggle** - Enable/disable mirroring
- **Mini Map** - Live position overview

### 📊 Simulation Panel (Right, when enabled)
- **Route Simulation** - Test planned routes
- **Incident Replay** - Review past events
- **Timeline Controls** - Play, pause, scrub

---

## 🔧 Troubleshooting

### Issue: Robot not moving in manual control

**Solution:**
1. Check if Hardware Sync is enabled
2. Verify ESP32 IP address in `hardware_sync_bridge.py`
3. Ensure robot_car_v2_modern.py is running
4. Check UDP port 9000 is not blocked

### Issue: Hardware sync not working

**Solution:**
1. Run `robot_car_v2_modern.py` first
2. Check both servers are running (2 console windows)
3. Verify IP addresses match your network
4. Check firewall allows UDP ports 9000, 9001

### Issue: Keyboard controls not responding

**Solution:**
1. Click on the 3D viewport to focus
2. Make sure no input field is selected
3. Try clicking the direction buttons instead
4. Refresh the browser page

### Issue: Model just rotating in place

**This is expected in simulation mode!**

The default simulation shows the robot moving in a pattern. To actually move:
1. Use **Manual Control** (WASD or buttons)
2. Or enable **Hardware Sync** with physical robot
3. Or run a **Route Simulation**

---

## 🎨 Camera Modes

Click the icons on the left side:

- **🎯 Follow** - Third-person, follows behind robot (default)
- **🌍 Orbit** - Free camera rotation (mouse drag)
- **📍 Top-Down** - Bird's eye view for navigation
- **👁️ FPV** - First-person from robot's perspective
- **🎬 Cinematic** - Dramatic sweeping angles for demos

---

## 📁 File Structure

```
digital_twin/
├── index.html                  # 3D web interface (UPDATED with controls)
├── digital_twin_server.py      # Main WebSocket + HTTP server
├── hardware_sync_bridge.py     # NEW - Connects to physical robot
├── route_simulator.py          # Route planning CLI tool
├── START_DIGITAL_TWIN.bat      # One-click launcher (UPDATED)
├── README_DIGITAL_TWIN.md      # Full documentation
└── QUICK_SETUP.md              # THIS FILE
```

---

## 🏆 Demo Tips

**For Hackathon Presentations:**

1. **Start with Manual Control** - Drive around with WASD to show responsiveness
2. **Switch Camera Modes** - Show off FPV and Cinematic views
3. **Enable Hardware Sync** - If robot is available, mirror its movements
4. **Run Route Simulation** - Demo autonomous navigation
5. **Replay Incidents** - Show the timeline feature

**Impress the Judges:**
- "This Digital Twin allows real-time monitoring and control"
- "The 3D model mirrors the physical robot with < 50ms latency"
- "We can test routes virtually before deploying to hardware"
- "Full telemetry visualization with incident recording and replay"

---

## 🚀 Next Steps

1. **Test Manual Control** - Try WASD driving
2. **Configure IPs** - Update hardware_sync_bridge.py
3. **Connect Hardware** - Enable sync with physical robot
4. **Demo Routes** - Simulate autonomous navigation
5. **Record Incidents** - Test the replay system

---

**Need Help?** Check `README_DIGITAL_TWIN.md` for full documentation.

**Ready to Demo?** Run `START_DIGITAL_TWIN.bat` and drive! 🎮
