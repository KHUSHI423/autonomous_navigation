# 🚀 360° Futuristic Real-Time Mapping Dashboard

A futuristic 3D visualization dashboard that integrates IMU, encoder, GPS, and webcam data to create a real-time 360° mapping system similar to Google Maps 3D view.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [ESP32 Sensor Connections](#esp32-sensor-connections)
- [Implementation Plan](#implementation-plan)
- [Phase 1: Backend Setup](#phase-1-backend-setup)
- [Phase 2: Sensor Fusion](#phase-2-sensor-fusion)
- [Phase 3: 3D Visualization](#phase-3-3d-visualization)
- [Phase 4: Real-Time Updates](#phase-4-real-time-updates)
- [Phase 5: Object Detection & Marking](#phase-5-object-detection--marking)
- [Phase 6: Futuristic UI](#phase-6-futuristic-ui)
- [Phase 7: Polish & Optimize](#phase-7-polish--optimize)

---

## 🎯 Overview

This project creates a **real-time 3D dashboard** that visualizes your surroundings using sensor fusion from multiple sources:

| Sensor | Purpose | Data Provided |
|--------|---------|---------------|
| **IMU (MPU6050)** | Orientation & Acceleration | Roll, Pitch, Yaw, Accelerometer |
| **Encoder (AS5600)** | Precise Heading | Absolute Angle (0-360°) |
| **GPS (NEO-6M/NEO-8M)** | Global Position | Latitude, Longitude, Altitude, Speed |
| **Webcam** | Visual Context & Object Detection | Live Video Feed, Detected Objects |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SENSORS HARDWARE                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   IMU    │  │ Encoder  │  │   GPS    │  │  Webcam  │    │
│  │(MPU6050) │  │ (AS5600) │  │(NEO-6M)  │  │  (USB)   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │              │          │
│       └─────────────┴──────┬──────┴──────────────┘          │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │     ESP32       │                        │
│                   │  (Microcontroller)                       │
│                   └────────┬────────┘                        │
└────────────────────────────┼────────────────────────────────┘
                             │ WiFi (WebSocket)
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   BACKEND SERVER (Node.js)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • WebSocket Server (Socket.IO)                      │   │
│  │  • Sensor data aggregation                           │   │
│  │  • Sensor fusion (Kalman/Madgwick Filter)            │   │
│  │  • Coordinate transformation (GPS → Local ENU)       │   │
│  │  • Object detection from webcam (OpenCV/YOLO)        │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │ WebSocket (real-time)
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  FRONTEND DASHBOARD (React + Three.js)       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Three.js 3D Scene                                   │   │
│  │  • 360° surround view                                │   │
│  │  • Real-time position/orientation                    │   │
│  │  • Path trail marking                                │   │
│  │  • Detected objects as 3D markers                    │   │
│  │  • Webcam feed with AR overlay                       │   │
│  │  • Futuristic HUD elements                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hardware Requirements

| Component | Model | Quantity | Purpose |
|-----------|-------|----------|---------|
| **Microcontroller** | ESP32 DevKit v1 | 1 | Main sensor hub |
| **IMU** | MPU6050 (GY-521) | 1 | Orientation & acceleration |
| **Encoder** | KY-040 Rotary Encoder | 1 | Rotation angle & direction |
| **GPS Module** | NEO-6M / NEO-8M | 1 | Global positioning |
| **Webcam** | Any USB Webcam | 1 | Visual context & object detection |
| **Jumper Wires** | Female-to-Female | 20+ | Connections |
| **Breadboard** | - | 1 | Prototyping |
| **USB Cable** | Micro-USB | 1 | ESP32 power & programming |
| **Resistors** | 10kΩ (optional) | 2 | Pull-ups for encoder |

---

## 🔌 ESP32 Sensor Connections

### Pinout Diagram

```
                        ┌─────────────────────────┐
                        │      ESP32 DevKit       │
                        │                         │
                        │  ┌─────────────────┐    │
                        │  │     ESP32       │    │
                        │  │                 │    │
                        │  │  GPIO Pins      │    │
                        │  └─────────────────┘    │
                        │                         │
                        └─────────────────────────┘
```

---

### 1. MPU6050 (IMU) - I2C Connection

| MPU6050 Pin | ESP32 Pin | GPIO | Description |
|-------------|-----------|------|-------------|
| **VCC** | 3.3V | 3.3V | Power (3.3V) |
| **GND** | GND | GND | Ground |
| **SCL** | GPIO 22 | D22 | I2C Clock |
| **SDA** | GPIO 21 | D21 | I2C Data |
| **INT** | GPIO 4 | D4 | Interrupt (optional) |

**Note:** MPU6050 uses I2C protocol. Default ESP32 I2C pins are GPIO 21 (SDA) and GPIO 22 (SCL).

---

### 2. Rotary Encoder (KY-040 / EC11) - Digital Input Connection

| Rotary Encoder Pin | ESP32 Pin | GPIO | Description |
|--------------------|-----------|------|-------------|
| **GND** | GND | GND | Ground |
| **+ (VCC)** | 3.3V | 3.3V | Power (3.3V) |
| **SW (Button)** | GPIO 13 | D13 | Switch press (optional) |
| **DT (Data)** | GPIO 12 | D12 | Pulse A (quadrature signal) |
| **CLK (Clock)** | GPIO 14 | D14 | Pulse B (quadrature signal) |

**How Rotary Encoder Works:**
- Produces two square waves (DT and CLK) with 90° phase difference
- ESP32 counts pulses to determine rotation angle
- Direction detected by which signal leads
- Typical resolution: 20-30 pulses per revolution (P/R)

**Wiring Notes:**
- Add 10kΩ pull-up resistors on DT and CLK lines if encoder doesn't have built-in pull-ups
- SW pin is for the push-button switch (optional)

---

### 3. NEO-6M/NEO-8M (GPS Module) - UART Connection

| GPS Module Pin | ESP32 Pin | GPIO | Description |
|----------------|-----------|------|-------------|
| **VCC** | 5V (VIN) | 5V | Power (3.3V-5V) |
| **GND** | GND | GND | Ground |
| **TX** | GPIO 16 | D16 | UART RX (connect to ESP32 RX) |
| **RX** | GPIO 17 | D17 | UART TX (connect to ESP32 TX) |
| **PPS** | GPIO 15 | D15 | Pulse Per Second (optional) |

**UART Configuration:**
- Baud Rate: 9600 (NEO-6M) or 38400 (NEO-8M)
- Serial Port: Serial2 (Hardware UART)

---

### 4. Complete Wiring Summary Table

| Component | VCC | GND | Pin 1 | Pin 2 | Pin 3 | Additional |
|-----------|-----|-----|-------|-------|-------|------------|
| **MPU6050** | 3.3V | GND | SDA → GPIO 21 | SCL → GPIO 22 | - | INT → GPIO 4 (opt) |
| **Rotary Encoder** | 3.3V | GND | DT → GPIO 12 | CLK → GPIO 14 | SW → GPIO 13 | 10kΩ pull-ups on DT/CLK |
| **NEO-6M GPS** | 5V | GND | TX → GPIO 16 | RX → GPIO 17 | - | PPS → GPIO 15 (opt) |

**Important Notes:**
- ⚠️ **I2C Bus:** MPU6050 uses GPIO 21 & 22. Add 4.7kΩ pull-up resistors if module doesn't have them.
- ⚠️ **Rotary Encoder:** Uses GPIO 12 & 14 for quadrature signals. These are NOT I2C pins.
- ⚠️ **Voltage Levels:** GPS modules often work at 3.3V-5V. Check your module specs.
- ⚠️ **UART Conflict:** Don't use GPIO 16/17 for other purposes as they're dedicated to GPS.
- ⚠️ **GPIO 12:** Can be strapping pin for boot mode. Ensure encoder doesn't interfere during boot.

---

### ESP32 Pin Reference Diagram

```
        ┌─────────────────────────────────────┐
        │           ESP32 DevKit V1           │
        │                                     │
        │  [USB]                              │
        │    ┌─────────────────────────┐      │
        │    │                         │      │
        │    │   ┌───────────────┐     │      │
        │    │   │    ESP32      │     │      │
        │    │   │   CHIP        │     │      │
        │    │   └───────────────┘     │      │
        │    │                         │      │
        │    └─────────────────────────┘      │
        │                                     │
        │  LEFT SIDE (GPIO):                  │
        │  ┌─────────────────────────────┐    │
        │  │ EN    3V3   GND   GND   23  │    │
        │  │ 22   GND    21    19   18   │    │
        │  │  5    17    16    15    4   │ ← I2C: 21,22 | GPS: 16,17
        │  │  2     0     2     3   13   │ ← Encoder SW: 13
        │  │ 14   3V3    25   26   27    │ ← Encoder CLK: 14
        │  │ 12   GND    26   25   33   │ ← Encoder DT: 12
        │  │ 13   GND    27   26   32   │
        │  │ 14    5V    28   27   33   │
        │  └─────────────────────────────┘    │
        │                                     │
        └─────────────────────────────────────┘
```

---

### Visual Wiring Diagram

```
                    ┌─────────────────┐
                    │    MPU6050      │
                    │                 │
                    │ VCC  SCL  SDA   │
                    └──┬────┬────┬────┘
                       │    │    │
              ┌────────┘    │    └────────┐
              │             │             │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   3.3V    │ │  GPIO 22  │ │  GPIO 21  │
        │  (Power)  │ │   (SCL)   │ │   (SDA)   │
        └───────────┘ └───────────┘ └───────────┘
              │             │             │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   3.3V    │ │  GPIO 14  │ │  GPIO 12  │
        │  (Power)  │ │  (CLK)    │ │  (DT)     │
        └───────────┘ └───────────┘ └───────────┘
              │             │             │
              │    ┌────────┘    ┌────────┘
              │    │             │
        ┌─────▼─────▼─────┐ ┌─────▼─────┐
        │  Rotary Encoder │ │  ESP32    │
        │    (KY-040)     │ │           │
        │  CLK   DT   SW  │ │ GPIO14 12 │
        └─────────────────┘ └───────────┘

        GPS Module (NEO-6M)
        ┌─────────────────┐
        │ VCC  TX   RX    │
        └──┬────┬────┬────┘
           │    │    │
     ┌─────┘    │    └────────┐
     │          │             │
   ┌─▼─┐    ┌───▼───┐   ┌────▼────┐
   │5V │    │GPIO 16│   │ GPIO 17 │
   │   │    │ (RX)  │   │  (TX)   │
   └───┘    └───────┘   └─────────┘
```

---

## 📅 Implementation Plan

| Phase | Duration | Description | Status |
|-------|----------|-------------|--------|
| **Phase 1** | 2 days | Backend Setup | 🟢 In Progress |
| **Phase 2** | 2 days | Sensor Fusion Algorithm | ⚪ Pending |
| **Phase 3** | 3 days | 3D Visualization with Three.js | ⚪ Pending |
| **Phase 4** | 2 days | Real-Time WebSocket Updates | ⚪ Pending |
| **Phase 5** | 3 days | Object Detection & Marking | ⚪ Pending |
| **Phase 6** | 2 days | Futuristic UI Dashboard | ⚪ Pending |
| **Phase 7** | 2 days | Polish & Optimization | ⚪ Pending |

**Total Estimated Time:** 16 days

---

## 🚀 Phase 1: Backend Setup

### Objectives
1. Set up Node.js server with Express
2. Implement WebSocket server using Socket.IO
3. Create serial communication for ESP32
4. Set up webcam capture with OpenCV
5. Create basic data structures for sensor fusion

### Project Structure

```
GPS_DASH/
├── backend/
│   ├── server.js              # Main server file
│   ├── websocket/
│   │   └── socketHandler.js   # WebSocket event handlers
│   ├── sensors/
│   │   ├── esp32Handler.js    # ESP32 serial communication
│   │   ├── imuParser.js       # MPU6050 data parser
│   │   ├── encoderParser.js   # AS5600 data parser
│   │   └── gpsParser.js       # NEO-6M NMEA parser
│   ├── camera/
│   │   └── webcamCapture.js   # Webcam capture & processing
│   ├── fusion/
│   │   └── sensorFusion.js    # Kalman/Madgwick filter
│   ├── utils/
│   │   └── coordinateTransform.js  # GPS to local ENU
│   ├── package.json
│   └── .env
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Scene3D.jsx
│   │   │   ├── HUD.jsx
│   │   │   └── CameraFeed.jsx
│   │   ├── hooks/
│   │   │   └── useSensorData.js
│   │   ├── utils/
│   │   │   └── threeHelpers.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── esp32_firmware/
│   └── main/
│       ├── main.ino         # ESP32 Arduino code
│       ├── imu_mpu6050.cpp
│       ├── encoder_as5600.cpp
│       ├── gps_neo6m.cpp
│       └── wifi_websocket.cpp
└── README.md
```

### Step 1: Initialize Backend

```bash
# Create project structure
mkdir -p backend websocket sensors camera fusion utils
mkdir -p frontend/src/components frontend/src/hooks frontend/src/utils
mkdir -p esp32_firmware

# Initialize backend
cd backend
npm init -y
npm install express socket.io serialport @serialport/parser-readline
npm install opencv4nodejs-rebuild
npm install dotenv
```

### Step 2: Backend Server Code

Create `backend/server.js`:

```javascript
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173", // Vite default
    methods: ["GET", "POST"]
  }
});

// Import handlers
const { setupESP32Handler } = require('./sensors/esp32Handler');
const { setupWebcamCapture } = require('./camera/webcamCapture');
const { sensorFusion } = require('./fusion/sensorFusion');

// Shared state
let sensorData = {
  imu: { roll: 0, pitch: 0, yaw: 0, ax: 0, ay: 0, az: 0 },
  encoder: { angle: 0 },
  gps: { lat: 0, lon: 0, alt: 0, speed: 0 },
  fused: { x: 0, y: 0, z: 0, heading: 0 },
  objects: []
};

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);
  
  // Send current sensor data on connect
  socket.emit('sensorData', sensorData);
  
  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
  });
});

// Setup ESP32 serial connection
setupESP32Handler(io, (newData) => {
  // Update sensor data
  sensorData = { ...sensorData, ...newData };
  
  // Apply sensor fusion
  const fused = sensorFusion(sensorData);
  sensorData.fused = fused;
  
  // Broadcast to all clients
  io.emit('sensorData', sensorData);
});

// Setup webcam capture
setupWebcamCapture(io, (objects) => {
  sensorData.objects = objects;
  io.emit('objects', objects);
});

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📡 WebSocket ready at ws://localhost:${PORT}`);
});
```

### Step 3: ESP32 Handler

Create `backend/sensors/esp32Handler.js`:

```javascript
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

function setupESP32Handler(io, onData) {
  const PORT = process.env.ESP32_PORT || 'COM3'; // Change for your system
  const BAUD = 115200;

  const port = new SerialPort({
    path: PORT,
    baudRate: BAUD,
    autoOpen: false
  });

  const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));

  parser.on('data', (data) => {
    try {
      const parsed = JSON.parse(data.trim());
      
      // Validate and forward data
      if (parsed.type && parsed.data) {
        onData({ [parsed.type]: parsed.data });
      }
    } catch (err) {
      console.error('Parse error:', err.message);
    }
  });

  port.open((err) => {
    if (err) {
      console.error('Error opening port:', err.message);
      return;
    }
    console.log(`✅ ESP32 connected on ${PORT}`);
  });

  return port;
}

module.exports = { setupESP32Handler };
```

### Step 4: ESP32 Firmware (Single File)

All ESP32 code is in **one file** for simplicity:

📁 `esp32_firmware/main/main.ino`

This single file includes:
- ✅ MPU6050 IMU driver (I2C)
- ✅ KY-040 Rotary Encoder (interrupts)
- ✅ NEO-6M GPS driver (UART)
- ✅ WiFi & WebSocket client
- ✅ JSON data packaging

**Required Arduino Libraries:**
Install via Arduino Library Manager or PlatformIO:
- `WebSockets` by Markus Sattler
- `ArduinoJson` by Benoit Blanchon  
- `TinyGPS++` by Mikal Hart

**Upload Options:**

**Option A: Arduino IDE**
1. Open `esp32_firmware/main/main.ino` in Arduino IDE
2. Install required libraries from Library Manager
3. Select Board: "ESP32 Dev Module"
4. Select Port: Your ESP32 COM port
5. Click Upload

**Option B: PlatformIO (Recommended)**
```bash
# Navigate to firmware folder
cd esp32_firmware

# Install dependencies
pio lib install

# Upload firmware
pio run --target upload

# Monitor serial output
pio device monitor
```

**Option C: Command Line (esptool)**
```bash
# Compile with Arduino CLI
arduino-cli compile --fqbn esp32:esp32:esp32 esp32_firmware

# Upload
arduino-cli upload -p COM3 --fqbn esp32:esp32:esp32 esp32_firmware
```

**Configuration:**
Before uploading, edit these lines in `main.ino`:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* WS_SERVER = "192.168.1.100";  // Your backend server IP
```

### Step 5: Environment Configuration

Create `backend/.env`:

```env
PORT=3001
ESP32_PORT=COM3
NODE_ENV=development
WEBSOCKET_URL=ws://localhost:3001
FRONTEND_URL=http://localhost:5173
```

### Step 6: Package.json Scripts

Update `backend/package.json`:

```json
{
  "name": "gps-dash-backend",
  "version": "1.0.0",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "socket.io": "^4.6.1",
    "serialport": "^12.0.0",
    "cors": "^2.8.5",
    "dotenv": "^16.0.3"
  }
}
```

---

## 📡 Phase 2: Sensor Fusion

### Objectives
- Implement Madgwick AHRS filter for IMU orientation
- Fuse IMU + encoder for accurate heading
- Convert GPS coordinates to local ENU frame
- Implement Kalman filter for position smoothing

### Key Algorithms

**Madgwick Filter:** Combines accelerometer + gyroscope data for stable orientation.

**Kalman Filter:** Smooths noisy sensor data and predicts position.

**ENU Conversion:** Converts GPS (lat/lon) to local Cartesian coordinates.

---

## 🎨 Phase 3: 3D Visualization

### Objectives
- Set up Three.js scene with 360° camera
- Create futuristic ground grid with glow effects
- Add 3D model for ego vehicle/robot
- Implement smooth camera controls (orbit, first-person)

### Features
- Neon grid lines
- Glowing path trail
- 3D markers for detected objects
- Real-time orientation update

---

## ⚡ Phase 4: Real-Time Updates

### Objectives
- Establish WebSocket connection from frontend
- Stream sensor data at 50-100Hz
- Implement interpolation for smooth motion
- Add latency compensation

---

## 🎯 Phase 5: Object Detection & Marking

### Objectives
- Capture webcam feed in browser
- Process frames with OpenCV.js or send to backend
- Detect objects using pre-trained YOLO model
- Place 3D markers at detected object locations
- Add distance estimation (if depth available)

---

## 🖥️ Phase 6: Futuristic UI

### Objectives
- Create HUD with sensor data displays
- Add neon/glow effects and dark theme
- Implement mini-map (top-down view)
- Add settings panel for calibration
- Create responsive layout

### UI Components
- Speedometer
- Compass/heading indicator
- GPS coordinates display
- IMU visualization (3D cube)
- Object counter
- Recording controls

---

## ✨ Phase 7: Polish & Optimize

### Objectives
- Optimize for 60 FPS rendering
- Add sound effects and animations
- Implement data logging/playback
- Add keyboard shortcuts
- Test with real sensor data
- Create build scripts

---

## 🛠️ Development Workflow

### Starting the Project

```bash
# Terminal 1: Backend
cd backend
npm install
npm run dev

# Terminal 2: Frontend
cd frontend
npm install
npm run dev

# Terminal 3: ESP32 (Arduino IDE or PlatformIO)
# Upload esp32_firmware/main/main.ino
```

### Accessing the Dashboard

Open browser: `http://localhost:5173`

---

## 📝 Notes

- **Webcam Integration:** For now, we're using browser-based webcam capture via `navigator.mediaDevices.getUserMedia()`
- **ESP32 Communication:** Uses WiFi WebSocket for wireless data transmission
- **Sensor Fusion:** Critical for accurate positioning - will be implemented in Phase 2
- **Coordinate System:** Using ENU (East-North-Up) for local mapping

---

## 🚦 Next Steps

1. ✅ Review ESP32 connections and gather hardware
2. ✅ Set up backend server (Phase 1)
3. ⏭️ Test ESP32 firmware with individual sensors
4. ⏭️ Implement sensor fusion (Phase 2)
5. ⏭️ Build 3D visualization (Phase 3)

---

## 📞 Support

For issues or questions, check:
- ESP32 documentation: https://docs.espressif.com/
- Three.js guides: https://threejs.org/docs/
- Socket.IO docs: https://socket.io/docs/

---

**Last Updated:** March 23, 2026
