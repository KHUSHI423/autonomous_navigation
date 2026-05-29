# ADAS Navigation Dashboard

A futuristic automotive-style GPS navigation dashboard with real-time visualization using Flask, MapLibre GL JS, and ESP32 GPS integration.

![Dashboard](https://img.shields.io/badge/Status-Ready-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)

---

## 🚀 Features

- **Real-time GPS Tracking** - Live location updates every second
- **3D Map Visualization** - MapLibre GL JS with tilt and bearing
- **Futuristic Dark UI** - Automotive dashboard aesthetic
- **Serial GPS Support** - ESP32 + GPS module via COM port
- **Simulation Mode** - Test without hardware
- **NMEA Parsing** - Standard GPS sentence support
- **Speed & Heading Display** - Animated gauge and compass
- **Route Tracking** - Visual path history
- **Responsive Design** - Adapts to different screen sizes

---

## 📁 Project Structure

```
FINAL_DASHBOARD/
├── server.py           # Flask backend server
├── gps_simulator.py    # GPS data simulator
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/
│   └── index.html     # Main dashboard HTML
└── static/
    ├── style.css      # Futuristic dark theme styles
    └── script.js      # Frontend JavaScript logic
```

---

## 🔧 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import flask, serial; print('Dependencies OK')"
```

---

## 🏃 How to Run

### Option A: With GPS Hardware (ESP32 + GPS Module)

1. **Connect Hardware:**
   - Connect GPS module to ESP32 (TX → RX, RX → TX)
   - Connect ESP32 to laptop via USB
   - Note the COM port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)

2. **Configure Serial Port:**
   Edit `server.py` and update:
   ```python
   SERIAL_PORT = 'COM3'  # Change to your ESP32 port
   BAUD_RATE = 9600      # Match your GPS module baud rate
   ```

3. **Start the Server:**
   ```bash
   python server.py
   ```

4. **Open Dashboard:**
   Navigate to `http://localhost:5000` in your browser

5. **Connect Serial:**
   - Click "CONNECT" button on the dashboard
   - Select your ESP32 COM port
   - GPS data will appear automatically

---

### Option B: Simulation Mode (No Hardware Required)

1. **Start the Flask Server:**
   ```bash
   python server.py
   ```

2. **Open Dashboard:**
   Navigate to `http://localhost:5000`

3. **Enable Simulation:**
   - Toggle "SIMULATION MODE" switch on the dashboard
   - GPS data will be simulated automatically

**OR** use the standalone simulator:

```bash
python gps_simulator.py
```
Select option 1 (API Mode) and enter desired speed.

---

## 🧪 Testing Without GPS

### Method 1: Dashboard Simulation Toggle

1. Open `http://localhost:5000`
2. Enable "SIMULATION MODE" toggle
3. Watch the car move on the map with varying speed

### Method 2: GPS Simulator Script

```bash
python gps_simulator.py
```

Choose from:
- **API Mode** - Sends data directly to Flask server
- **Serial Mode** - Sends NMEA to virtual COM port
- **List Ports** - Show available serial ports

### Method 3: Manual API Calls

Test the API directly:

```bash
# Get current GPS data
curl http://localhost:5000/gps

# Update GPS data manually
curl -X POST http://localhost:5000/gps/update \
  -H "Content-Type: application/json" \
  -d "{\"latitude\": 28.6139, \"longitude\": 77.2090, \"speed\": 50}"

# Reset to default
curl -X POST http://localhost:5000/gps/reset
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/gps` | GET | Current GPS data (JSON) |
| `/gps/update` | POST | Update GPS data (simulation) |
| `/gps/reset` | POST | Reset to default location |
| `/serial/ports` | GET | List available COM ports |
| `/serial/connect` | POST | Connect to serial port |
| `/serial/disconnect` | POST | Disconnect from serial |
| `/health` | GET | Server health check |

---

## 📡 GPS Module Integration

### Supported NMEA Sentences

- **$GPGGA** - GPS Fix Data
- **$GPRMC** - Recommended Minimum GPS Data
- **$GNGGA** - GNSS Fix Data
- **$GNRMC** - GNSS Recommended Minimum

### ESP32 Wiring

```
ESP32          GPS Module
─────          ──────────
3.3V    ───→   VCC
GND     ───→   GND
GPIO 16 ───→   TX
GPIO 17 ───→   RX
```

### ESP32 Code Example

```cpp
#include <TinyGPSPlus.h>
#include <HardwareSerial.h>

TinyGPSPlus gps;
HardwareSerial GPSSerial(1);

void setup() {
  Serial.begin(9600);
  GPSSerial.begin(9600, SERIAL_8N1, 16, 17);
}

void loop() {
  while (GPSSerial.available()) {
    gps.encode(GPSSerial.read());
  }
  
  if (gps.location.isUpdated()) {
    Serial.printf("$GPGGA,%s,%f,%c,%f,%c,1,%02d,1.0,%f,M,0.0,M,,\n",
      // ... format NMEA output
    );
  }
}
```

---

## 🎨 UI Components

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  [LOGO] ADAS NAV    [STATUS]    [TIME] [DATE]          │
├──────────┬──────────────────────────────┬──────────────┤
│          │                              │              │
│  GPS     │                              │   VEHICLE    │
│  DATA    │          3D MAP              │   INFO       │
│          │                              │              │
│  [COM    │    [Car Marker]              │   [Gauge]    │
│  PORT]   │                              │              │
│          │    [Route Line]              │   [Alerts]   │
├──────────┴──────────────────────────────┴──────────────┤
│  [MODE: NAVIGATION]    [GPS SIGNAL]    [v1.0.0]       │
└─────────────────────────────────────────────────────────┘
```

### Color Scheme

| Color | Hex | Usage |
|-------|-----|-------|
| Primary BG | `#0a0a0f` | Main background |
| Accent Cyan | `#00d4ff` | Primary accent |
| Accent Green | `#00ff88` | Speed/Active |
| Accent Orange | `#ffaa00` | Warnings |
| Accent Red | `#ff4466` | Errors |

---

## 🛠️ Troubleshooting

### Server won't start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Use different port
# Edit server.py: app.run(port=5001)
```

### Serial connection fails
1. Verify COM port in Device Manager
2. Check baud rate matches GPS module
3. Ensure no other app is using the port
4. Try different USB cable/port

### Map not loading
1. Check internet connection (tiles require online)
2. Open browser console for errors
3. Clear browser cache

### GPS data not updating
1. Verify GPS module has sky view
2. Check NMEA output from ESP32
3. Enable simulation mode to test UI

---

## 📝 Customization

### Change Default Location

Edit `server.py`:
```python
gps_data = {
    'latitude': 40.7128,  # New York
    'longitude': -74.0060,
    # ...
}
```

### Update Map Style

Edit `static/script.js` - `initMap()`:
```javascript
style: 'https://demotiles.maplibre.org/style.json'
```

### Adjust Update Rate

Edit `static/script.js`:
```javascript
const CONFIG = {
    GPS_POLL_INTERVAL: 500,  // milliseconds
    // ...
};
```

---

## 📄 License

MIT License - Free for personal and commercial use

---

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review browser console for errors
3. Verify serial port configuration

---

**Built with ❤️ for Automotive Innovation**
