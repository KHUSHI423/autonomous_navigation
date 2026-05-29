# 🔮 DIGITAL TWIN - COMPLETE STEP-BY-STEP SETUP GUIDE

## 📋 What This Does

The Digital Twin creates a **real-time 3D virtual replica** of your physical robot car that:
- Mirrors exact movements from hardware (throttle, steering, ultrasonic)
- Shows live sensor data with 500ms update rate
- Allows manual control via WASD or on-screen buttons
- Displays camera feed (when configured)
- Works even without hardware (manual control mode)

---

## 🔍 DATA FLOW VERIFIED

```
PHYSICAL ROBOT
    ↓ (ESP32 sends STATUS via UDP)
robot_car_v2_modern.py [Receives on UDP 9001]
    ↓ (Forwards STATUS via UDP)
hardware_sync_bridge.py [Receives on UDP 9002]
    ├→ HTTP Server (Port 8767) → index.html polls every 500ms
    └→ WebSocket (Port 8766) → index.html real-time updates
```

### Data Format
```
STATUS:AUTO:150:0.02:25.5
       │    │    │     └─ Ultrasonic: 25.5 cm
       │    │    └────── Steering: 0.02 (-1.0 to 1.0)
       │    └─────────── Throttle: 150 (0-255)
       └──────────────── Mode: AUTO or MANUAL
```

---

## ⚙️ STEP 1: FIND YOUR IP ADDRESSES

### Find Laptop IP
Open **Command Prompt** and run:
```cmd
ipconfig
```

Look for **IPv4 Address** under your WiFi adapter:
```
Wireless LAN adapter Wi-Fi:
   IPv4 Address. . . . . . . . . . . : 10.17.122.100  ← THIS IS YOUR LAPTOP_IP
```

### Find ESP32 IP
Open Arduino IDE Serial Monitor for ESP32 (115200 baud) and note the IP:
```
WiFi Connected!
IP Address: 10.17.122.207  ← THIS IS YOUR ESP32_IP
```

---

## 📝 STEP 2: CONFIGURE FILES

### File 1: hardware_sync_bridge_FIXED.py
**Location:** `hardware_setup\digital_twin\hardware_sync_bridge_FIXED.py`

**Line 24:** Update ESP32 IP
```python
ESP32_IP = "10.17.122.207"  # ← Change to YOUR ESP32 IP
```

**Save and rename** to `hardware_sync_bridge.py` (replace existing)

### File 2: robot_car_v2_modern.py
**Location:** `hardware_setup\robot_car_and_pi_v2_2esp\robot_car_v2_modern.py`

**Line 26:** Update ESP32 IP
```python
ESP32_IP = "10.17.122.207"  # ← Change to YOUR ESP32 IP
```

**Add forwarding code** (around line 78, after socket creation):
```python
# Add this after line 78 (after "Sockets ready!\n" print)
forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Forward socket ready\n")
```

**Modify status_receiver_thread** (around line 127) to forward data:
```python
def status_receiver_thread():
    global esp32_status, ultrasonic_dist
    last_ultra_print = 0
    while True:
        try:
            data, addr = status_sock.recvfrom(1024)
            message = data.decode('utf-8').strip()
            
            # ADD THIS LINE - Forward to hardware_sync_bridge
            forward_sock.sendto(data, ("127.0.0.1", 9002))
            
            if message.startswith("STATUS:"):
                # ... rest of existing code
```

---

## 🚀 STEP 3: START THE SYSTEM (IN ORDER)

### ⚠️ IMPORTANT: Start in this exact order!

### TERMINAL 1 - Hardware Sync Bridge
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python hardware_sync_bridge.py
```

**Expected Output:**
```
======================================================================
  HARDWARE SYNC BRIDGE - Digital Twin Data Server
======================================================================
UDP Forward: 127.0.0.1:9002 (from robot_car_v2_modern.py)
HTTP Sensors: http://localhost:8767/sensors
WebSocket: ws://localhost:8766
ESP32 Commands: 10.17.122.207:9000
======================================================================
📡 UDP forward receiver listening on 127.0.0.1:9002
🌐 HTTP sensor server started on port 8767
🔌 WebSocket started on ws://localhost:8766
```

**✅ CHECK:** Keep this terminal OPEN. You should see "UDP forward receiver listening"

---

### TERMINAL 2 - Robot Car Controller
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\robot_car_and_pi_v2_2esp
python robot_car_v2_modern.py
```

**Expected Output:**
```
======================================================================
  ROBOT CAR V2 - MODERN DASHBOARD
======================================================================
Video Port: 5000
ESP32 IP: 10.17.122.207:9000
Model: ../../yolov8n.pt
...
Video receiver started
Status receiver started
Forward socket ready

-> 150:0.02      | CRUISE (clear)            | Ultra: --     | Objects: 0
```

**✅ CHECK:** You should see "Forward socket ready" and STATUS messages being sent

---

### TERMINAL 3 - Digital Twin Web Server
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python -m http.server 8080
```

**Expected Output:**
```
Serving HTTP on :: port 8080 (http://[::]:8080/) ...
```

**✅ CHECK:** Keep this terminal OPEN

---

## 🌐 STEP 4: OPEN DIGITAL TWIN IN BROWSER

1. Open **Chrome** or **Edge** browser
2. Navigate to: **http://localhost:8080**

### What You Should See:

✅ **3D Robot Car** in center of neon grid  
✅ **Control Panel** at bottom with direction buttons  
✅ **Telemetry Panel** on right showing sensor data  
✅ **Connection Status** showing "Connecting to hardware..." or "Connected"  
✅ **Camera Viewer** in bottom-right (shows "Waiting for stream..." until camera configured)

---

## 🎮 STEP 5: TEST CONTROLS

### Manual Control (Works without hardware)

**Keyboard:**
- Press **W** - Robot moves forward
- Press **S** - Robot moves backward  
- Press **A** - Robot rotates left
- Press **D** - Robot rotates right
- **Release** - Robot stops

**On-screen Buttons:**
- Click **⬆️** - Move forward (hold to continue)
- Click **⬇️** - Move backward
- Click **⬅️** - Rotate left
- Click **➡️** - Rotate right
- Click **🛑** - Emergency stop

**Speed Slider:**
- Drag slider to change speed (50-255)

### What to Watch:
- **Position** in header updates as robot moves
- **Heading** changes when rotating
- **Speed** shows current velocity
- **Wheels animate** when moving

---

## 📡 STEP 6: VERIFY REAL-TIME SENSOR DATA

### Check Hardware Sync Bridge Terminal
You should see STATUS messages being received:
```
📊 Robot: Mode=AUTO, Throttle=150, Ultra=999cm, Speed=75
📊 Robot: Mode=AUTO, Throttle=120, Ultra=25cm, Speed=60
```

### Check Browser Telemetry Panel

**Connection Status** (top of right panel):
- 🟢 **Green dot** + "Connected to hardware" = ✅ Receiving data
- 🟠 **Orange dot** + "No hardware" = Manual control only

**Sensor Cards:**
- **Ultrasonic** - Shows distance in cm (updates every 500ms)
- **PWM** - Motor power (0-255)
- **Heading** - Current rotation in degrees
- **Position** - X, Z coordinates

**Vehicle State:**
- **Throttle** - Current motor power
- **Steering** - Current steering ratio
- **Speed** - Calculated speed
- **Distance** - Total distance traveled
- **Runtime** - Time since start
- **Mode** - AUTO or MANUAL

### Test Ultrasonic Sensor
Wave your hand in front of ultrasonic sensor:
- Value should change from `--` to actual distance (e.g., "25 cm")
- Color changes: Green (>50cm) → Orange (30-50cm) → Pink (<30cm)

---

## 🔍 STEP 7: VERIFY HTTP SENSOR ENDPOINT

Open new browser tab: **http://localhost:8767/sensors**

**Expected JSON Response:**
```json
{
  "ultrasonic": 999,
  "pwm": 150,
  "steering": 0.0,
  "mode": "AUTO",
  "speed": 75,
  "timestamp": 1711180800.123
}
```

**✅ CHECK:** Values should match what you see in telemetry panel

---

## 📷 STEP 8: ADD CAMERA STREAM (OPTIONAL)

### Option A: Add MJPEG Server to robot_car_v2_modern.py

**Add imports** (top of file):
```python
from flask import Flask, Response
```

**Add after model loading** (around line 60):
```python
# Flask camera stream server
app = Flask(__name__)

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5001, threaded=True)

threading.Thread(target=start_flask, daemon=True).start()
print("Camera stream: http://localhost:5001/video_feed\n")
```

**Update index.html** camera viewer (line ~230):
Replace:
```html
<div class="camera-feed" id="camera-feed">
    <div>Waiting for stream...</div>
</div>
```

With:
```html
<div class="camera-feed" id="camera-feed">
    <img src="http://localhost:5001/video_feed" alt="Camera Stream" />
</div>
```

**Restart robot_car_v2_modern.py** and camera feed should appear!

---

## ✅ VERIFICATION CHECKLIST

Before demo, verify ALL:

- [ ] LAPTOP_IP and ESP32_IP updated in Python files
- [ ] Forward socket code added to robot_car_v2_modern.py
- [ ] hardware_sync_bridge.py running (Terminal 1)
- [ ] robot_car_v2_modern.py running (Terminal 2)
- [ ] HTTP server running (Terminal 3)
- [ ] http://localhost:8080 shows 3D robot
- [ ] WASD/buttons move the robot
- [ ] Telemetry updates in real-time
- [ ] Connection status shows green (if hardware connected)
- [ ] Ultrasonic value changes when object detected
- [ ] http://localhost:8767/sensors returns valid JSON
- [ ] Camera stream visible (if configured)

---

## 🐛 TROUBLESHOOTING

### Problem: "Address already in use"
**Solution:** Another process is using the port
```cmd
# Find process using port 8766
netstat -ano | findstr :8766

# Kill the process (replace PID)
taskkill /PID 12345 /F
```

### Problem: No STATUS messages in hardware_sync_bridge
**Solution:** Check forwarding code in robot_car_v2_modern.py
- Verify `forward_sock.sendto(data, ("127.0.0.1", 9002))` is present
- Check Windows Firewall allows UDP port 9002

### Problem: Connection shows orange dot
**Solution:** Hardware not connected or wrong port
- Verify hardware_sync_bridge.py is running
- Check robot_car_v2_modern.py is forwarding to port 9002
- Try http://localhost:8767/sensors directly

### Problem: 3D robot doesn't move
**Solution:** JavaScript error or controls not initialized
- Press F12 in browser, check Console for errors
- Try different browser (Chrome recommended)
- Refresh page (Ctrl+F5)

### Problem: Camera shows "Waiting for stream..."
**Solution:** Camera integration not configured
- Add Flask server to robot_car_v2_modern.py (Step 8)
- Or ignore - doesn't affect other functionality

---

## 🎯 QUICK START COMMANDS

Copy-paste these in **THREE SEPARATE** terminals:

**TERMINAL 1:**
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python hardware_sync_bridge.py
```

**TERMINAL 2:**
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\robot_car_and_pi_v2_2esp
python robot_car_v2_modern.py
```

**TERMINAL 3:**
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python -m http.server 8080
```

**BROWSER:** http://localhost:8080

---

## 📊 EXPECTED BEHAVIOR

### When System is Running Correctly:

1. **hardware_sync_bridge.py terminal:**
   ```
   📡 UDP forward receiver listening on 127.0.0.1:9002
   🌐 HTTP sensor server started on port 8767
   🔌 WebSocket started on ws://localhost:8766
   📊 Robot: Mode=AUTO, Throttle=150, Ultra=999cm, Speed=75
   ```

2. **robot_car_v2_modern.py terminal:**
   ```
   -> 150:0.02      | CRUISE (clear)            | Ultra: --     | Objects: 0
   -> 150:0.02      | CRUISE (clear)            | Ultra: 45cm   | Objects: 0
   ```

3. **Browser (http://localhost:8080):**
   - 3D robot visible
   - Green connection dot (if hardware connected)
   - Telemetry updating every 500ms
   - WASD controls work

---

**🎉 You're ready to demo!**
