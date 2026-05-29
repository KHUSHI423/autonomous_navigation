# 🚀 DIGITAL TWIN - FINAL STARTUP GUIDE

## ⚠️ IMPORTANT - READ THIS FIRST!

The error you're seeing is because **an old Python process is still running** and holding port 9001.

---

## ✅ STEP-BY-STEP FIX

### STEP 1: KILL ALL PYTHON PROCESSES

**Option A: Run the batch file**
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
KILL_PYTHON_PROCESSES.bat
```

**Option B: Manual kill**
```cmd
taskkill /F /IM python.exe
```

**Option C: Task Manager**
1. Press Ctrl+Shift+Esc
2. Find all "Python" processes
3. Right-click → End Task

---

### STEP 2: START IN CORRECT ORDER (WITH DELAYS)

**⚠️ WAIT 2-3 SECONDS BETWEEN EACH TERMINAL!**

#### TERMINAL 1 - Hardware Sync Bridge
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python hardware_sync_bridge.py
```

**Expected Output:**
```
======================================================================
  HARDWARE SYNC BRIDGE - Digital Twin Data Server
======================================================================
UDP STATUS:   127.0.0.1:9002 (from robot_car_v2)
UDP DETECT:   127.0.0.1:9003 (from robot_car_v2)
HTTP Sensors: http://localhost:8767/sensors
WebSocket:    ws://localhost:8766
ESP32:        10.17.122.207:9000
======================================================================
📡 UDP status receiver listening on 127.0.0.1:9002
📡 UDP detection receiver listening on 127.0.0.1:9003
🌐 HTTP sensor server started on port 8767
🔌 WebSocket started on ws://localhost:8766
```

**✅ WAIT 2-3 SECONDS BEFORE CONTINUING!**

---

#### TERMINAL 2 - Robot Car with Flask
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\robot_car_and_pi_v2_2esp
python robot_car_v2_complete.py
```

**Expected Output:**
```
======================================================================
  ROBOT CAR V2 - COMPLETE
  Digital Twin + Flask Detection Server
======================================================================
Video Port: 5000
ESP32 IP: 10.17.122.207:9000
Model: ../../yolov8n.pt
Camera: 640x480
Ultrasonic Priority: < 30.0cm
Digital Twin Forward: 127.0.0.1:9002
Flask Detection API: http://localhost:5002/detections

Press 'q' quit | 'm' toggle auto/manual
======================================================================
Model loaded!

Sockets ready!

✅ Digital Twin forwarding enabled
🌐 Detection API: http://localhost:5002/detections

Video receiver started
Status receiver started
```

**If you see "OSError: [WinError 10048]":**
- CLOSE ALL TERMINALS
- Run KILL_PYTHON_PROCESSES.bat again
- WAIT 5 seconds
- Try again from Terminal 1

**✅ WAIT 2-3 SECONDS BEFORE CONTINUING!**

---

#### TERMINAL 3 - Web Server
```cmd
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python -m http.server 8080
```

**Expected Output:**
```
Serving HTTP on :: port 8080 (http://[::]:8080/) ...
```

---

### STEP 3: OPEN BROWSER

Navigate to: **http://localhost:8080**

**What you should see:**
- ✅ 3D robot car in center of grid
- ✅ Control panel at bottom with buttons
- ✅ Telemetry panel on right
- ✅ Connection status (may show orange if no hardware)

---

### STEP 4: TEST CONTROLS

#### Test Movement:
1. **Open browser console (F12)**
2. **Press W** - Should see:
   ```
   START MOVE: forward
   EXECUTE: forward rotation: 0.00
   FORWARD to: 0.01, 0.00
   ```
3. **Robot should move forward** (position updates in header)
4. **Press A** - Robot rotates left (heading changes)
5. **Release** - Robot stops

#### Test Object Spawning:
1. **Press 1** - Car spawns at (0, 20)
2. **Press 2** - Bike spawns at (0, 15)
3. **Press 3** - Person spawns at (0, 12)
4. **Press 4** - Tree spawns at (0, 25)
5. **Press 5** - All objects cleared

#### Test YOLO Detection:
1. **Check Terminal 2** - Should show "Objects: X"
2. **Navigate to** http://localhost:5002/detections
3. **Should see JSON** like:
   ```json
   {
     "detections": [
       {"class": "person", "distance": 8.5, "x_center": 320}
     ],
     "timestamp": 1711180800.123,
     "mode": "AUTO",
     "ultrasonic": 999
   }
   ```
4. **Digital Twin should show 3D models** of detected objects

---

## 🔍 TROUBLESHOOTING

### Problem: "OSError: [WinError 10048]" in Terminal 2

**Cause:** Another process is using port 9001

**Solution:**
1. Close ALL terminals
2. Run `KILL_PYTHON_PROCESSES.bat`
3. Wait 5 seconds
4. Start again from Terminal 1
5. **WAIT 2-3 seconds between terminals!**

---

### Problem: Robot doesn't move when pressing W

**Cause:** JavaScript error or event listener not attached

**Solution:**
1. Open browser console (F12)
2. Look for errors (red text)
3. Press W - check for "START MOVE: forward" message
4. If no message, refresh page (Ctrl+F5)
5. Try Chrome browser (recommended)

---

### Problem: No objects appearing

**Cause:** Flask server not running or wrong URL

**Solution:**
1. Check Terminal 2 shows "Flask Detection API: http://localhost:5002/detections"
2. Navigate to http://localhost:5002/detections
3. Should see JSON with detections
4. If 404 error, robot_car_v2_complete.py not running properly
5. Press 1-4 to test manual spawning

---

### Problem: Connection shows orange dot

**Cause:** hardware_sync_bridge.py not receiving data

**Solution:**
1. Check Terminal 1 shows "UDP status receiver listening on 127.0.0.1:9002"
2. Check Terminal 2 shows "Digital Twin Forward: 127.0.0.1:9002"
3. Check Terminal 1 shows "📊 Robot: Mode=..." messages
4. Try http://localhost:8767/sensors directly

---

## 📊 VERIFICATION CHECKLIST

Before declaring success, verify ALL:

**Processes:**
- [ ] Only 3 Python terminals running (no duplicates)
- [ ] No error messages in any terminal
- [ ] Terminal 1: "UDP status receiver listening on 127.0.0.1:9002"
- [ ] Terminal 2: "Flask Detection API: http://localhost:5002/detections"
- [ ] Terminal 3: "Serving HTTP on :: port 8080"

**Browser:**
- [ ] http://localhost:8080 loads with 3D robot
- [ ] http://localhost:8767/sensors returns JSON
- [ ] http://localhost:5002/detections returns JSON with detections
- [ ] Connection status visible in top-right of telemetry panel

**Movement:**
- [ ] Press W - Console shows "START MOVE: forward"
- [ ] Console shows "FORWARD to: X.XX, Z.ZZ"
- [ ] Robot 3D model moves
- [ ] Position display updates
- [ ] Heading changes when pressing A/D

**Objects:**
- [ ] Press 1 - Car appears in 3D scene
- [ ] Press 2 - Bike appears
- [ ] Press 3 - Person appears
- [ ] Press 4 - Tree appears
- [ ] Press 5 - All objects disappear
- [ ] Object count updates in header

**Sensors:**
- [ ] Terminal 2 shows "Objects: X" in console output
- [ ] http://localhost:5002/detections shows detection data
- [ ] 3D objects auto-spawn when YOLO detects something

---

## 🎯 QUICK START COMMANDS

**COPY-PASTE THESE IN ORDER (WITH 2-3 SECOND DELAYS):**

```cmd
:: TERMINAL 1
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python hardware_sync_bridge.py

:: WAIT 2-3 SECONDS!

:: TERMINAL 2
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\robot_car_and_pi_v2_2esp
python robot_car_v2_complete.py

:: WAIT 2-3 SECONDS!

:: TERMINAL 3
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\digital_twin
python -m http.server 8080

:: BROWSER
http://localhost:8080
```

---

## 🎉 SUCCESS CRITERIA

Your Digital Twin is working correctly when:

1. ✅ **3 Python terminals** running without errors
2. ✅ **3D robot visible** in browser at http://localhost:8080
3. ✅ **WASD moves the robot** (with console logs)
4. ✅ **1-4 spawns 3D objects** (car, bike, person, tree)
5. ✅ **YOLO detections appear** as 3D models automatically
6. ✅ **Sensor data updates** in telemetry panel
7. ✅ **No port conflict errors** in any terminal

---

**🚀 You're ready to demo!**
