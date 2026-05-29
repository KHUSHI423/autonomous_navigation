# FUTURISTIC ADAS 3D NAVIGATION DASHBOARD
## CIT Coimbatore Campus Simulation

A Tesla-inspired automotive navigation dashboard with real-time GPS visualization, 3D buildings, and dark theme UI.

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
cd cit_adas_dashboard
pip install -r requirements.txt
```

### Step 2: Run the Server
```bash
python server.py
```

### Step 3: Open Dashboard
Navigate to **http://localhost:5002** in your browser

### Step 4: Start Simulation
Click **"START NAVIGATION"** button to begin the GPS simulation

---

## 📍 Location Details

**Starting Point:** Coimbatore Institute of Technology (CIT)
- **Latitude:** 11.0283
- **Longitude:** 77.0270
- **Altitude:** ~420 meters

The simulation follows an elliptical route around the CIT campus with realistic speed variations.

---

## ✨ Features

### Map Features
| Feature | Description |
|---------|-------------|
| **3D Buildings** | Uniform height (40-60m) extruded buildings |
| **Dark Theme** | Tesla-style black/dark color scheme |
| **Tilted View** | 60° pitch for dramatic 3D perspective |
| **Route Line** | Glowing cyan path showing traveled route |
| **Smooth Camera** | Auto-follows car with bearing rotation |

### UI Components
| Component | Description |
|-----------|-------------|
| **Speed Display** | Large digital speedometer (0-180 km/h) |
| **Speed Graph** | Real-time speed history visualization |
| **Compass Ring** | Rotating compass with cardinal directions |
| **Heading Display** | Digital heading (0-360°) + cardinal direction |
| **GPS Coordinates** | Live latitude, longitude, altitude |
| **Trip Computer** | Elapsed time, distance, avg/max speed |
| **Signal Strength** | GPS satellite count & signal bars |
| **Map Controls** | Tilt, zoom, center, 3D toggle buttons |

### Dashboard Layout
```
┌─────────────────────────────────────────────────────────────┐
│  [LOGO] ADAS NAV    [STATUS]    [00:00:00]  [2024-01-01]   │
├──────────┬──────────────────────────────────┬──────────────┤
│          │                                  │              │
│  SPEED   │                                  │   TRIP       │
│  000     │         3D MAP                   │   COMPUTER   │
│  KM/H    │                                  │              │
│  [Graph] │      [Car Marker]                │   GPS        │
│          │      [Route Line]                │   COORDS     │
│  COMPASS │                                  │              │
│  [Ring]  │                                  │   MAP        │
│          │                                  │   CONTROLS   │
│  STATUS  │                                  │              │
│  SAT/ACC │                                  │              │
├──────────┴──────────────────────────────────┴──────────────┤
│  [▶ START] [⏸ STOP] [↺ RESET]   [3D MODE]   [📍 CIT]      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎮 Controls

### Main Controls
| Button | Action |
|--------|--------|
| **▶ START NAVIGATION** | Begin GPS simulation |
| **⏸ STOP** | Pause simulation |
| **↺ RESET** | Return to starting position |

### Map Controls
| Button | Action |
|--------|--------|
| **⌃** | Tilt view up (increase pitch) |
| **⌄** | Tilt view down (decrease pitch) |
| **+** | Zoom in |
| **−** | Zoom out |
| **⌖** | Center map on car |
| **◧** | Toggle 3D buildings on/off |

---

## 🛠️ Technical Details

### Tech Stack
- **Backend:** Python 3.x + Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Map Library:** MapLibre GL JS
- **Data Source:** OpenStreetMap tiles
- **Font:** Orbitron (display), Rajdhani (body)

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/gps` | GET | Current GPS data (JSON) |
| `/gps/start` | POST | Start simulation |
| `/gps/stop` | POST | Stop simulation |
| `/gps/reset` | POST | Reset to start position |
| `/route` | GET | Get route points |
| `/health` | GET | System health check |

### GPS Data Format
```json
{
    "latitude": 11.0283,
    "longitude": 77.0270,
    "altitude": 420.0,
    "speed": 35.5,
    "heading": 180.0,
    "satellites": 10,
    "accuracy": 2.5,
    "timestamp": "2024-01-01T12:00:00",
    "status": "simulated"
}
```

---

## 🎨 Color Scheme

| Element | Color | Usage |
|---------|-------|-------|
| Background | `#000000` | Pure black |
| Card BG | `rgba(10,10,15,0.85)` | Panels |
| Accent Cyan | `#00ffff` | Primary accent |
| Accent Blue | `#0066ff` | Secondary accent |
| Accent Green | `#00ff88` | Success/Active |
| Accent Orange | `#ff8800` | Warnings |
| Accent Red | `#ff3344` | Errors/Stop |

---

## 🔄 Simulation Behavior

### Route Pattern
- Elliptical path around CIT campus
- ~100 predefined waypoints
- Automatic looping

### Speed Profile
- Campus speed limit: 25-45 km/h
- Random variation: ±3 km/h
- Smooth transitions

### Update Rate
- GPS data: Every 1 second
- Map rendering: 60 FPS
- UI updates: Real-time

---

## 🔌 Future Hardware Integration

### Serial GPS (ESP32 + GPS Module)
```python
# Add to server.py
import serial

SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# Parse NMEA sentences
# Update gps_data via /gps/serial endpoint
```

### WiFi GPS (Network NMEA)
```python
# Connect to WiFi GPS module
# Stream NMEA over TCP/UDP
# Update gps_data in real-time
```

---

## 📝 Customization

### Change Location
Edit `server.py`:
```python
CIT_LAT = 11.0283  # New latitude
CIT_LON = 77.0270  # New longitude
```

### Adjust Building Height
Edit `script.js`:
```javascript
const CONFIG = {
    BUILDING_HEIGHT: 60  // meters
};
```

### Change Update Rate
Edit `script.js`:
```javascript
const CONFIG = {
    POLL_INTERVAL: 500  // milliseconds
};
```

### Modify Route
Edit `generate_cit_route()` in `server.py`:
```python
# Adjust ellipse parameters
lat_radius = 0.010  # Larger area
lon_radius = 0.012
```

---

## 🐛 Troubleshooting

### Map not loading
- Check internet connection (OSM tiles require online)
- Open browser console (F12) for errors
- Clear browser cache

### Simulation not starting
- Click START button (ensure not disabled)
- Check browser console for errors
- Restart server if needed

### Port 5002 already in use
Edit `server.py`:
```python
app.run(host='0.0.0.0', port=5003, ...)  # Change port
```

### Slow performance
- Reduce building density in `script.js`
- Lower map zoom level
- Use hardware acceleration in browser

---

## 📊 Project Structure

```
cit_adas_dashboard/
├── server.py              # Flask backend server
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Dashboard HTML
└── static/
    ├── style.css         # Futuristic dark theme
    └── script.js         # Map & GPS logic
```

---

## 🎯 Comparison with Other Dashboards

| Feature | This Dashboard | Original | 3D Nav |
|---------|---------------|----------|--------|
| Port | 5002 | 5000 | 5001 |
| Location | CIT Coimbatore | Delhi | Delhi |
| Map Style | Dark automotive | Standard OSM | Dark Tesla |
| Buildings | 3D uniform | 2D | 3D fallback |
| Route | Elliptical campus | Random | Road grid |
| UI Theme | Cyan/Blue | Blue/Purple | Blue/Green |
| Speed Graph | ✅ Real-time | ❌ | ❌ |
| Trip Computer | ✅ Full | Basic | Basic |

---

## 📄 License

MIT License - Free for educational and commercial use

---

## 🤝 Credits

- **Map Data:** © OpenStreetMap Contributors
- **Map Library:** MapLibre GL JS
- **Fonts:** Google Fonts (Orbitron, Rajdhani)
- **Design Inspiration:** Tesla Navigation UI

---

**Built for ADAS Visualization & Automotive HMI Development**
