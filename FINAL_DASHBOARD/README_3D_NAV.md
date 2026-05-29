# ADAS 3D Navigation Dashboard - Tesla Style

Enhanced 3D navigation with uniform building heights, dark map theme, and road-following simulation.

---

## 🚀 Quick Start

### Run the 3D Navigation Server

```bash
python server_3d_nav.py
```

**Server will start on: http://localhost:5001**

---

## 🎮 How to Use

### 1. Open Dashboard
Navigate to **http://localhost:5001** in your browser

### 2. Start Navigation Simulation
- Click **"START NAVIGATION"** button, OR
- Toggle **"AUTO SIMULATION"** switch

The car will automatically follow roads with realistic movement!

### 3. Map Controls
| Button | Function |
|--------|----------|
| ⌃ | Tilt view up |
| ⌄ | Tilt view down |
| + | Zoom in |
| − | Zoom out |
| ⌖ | Center on car |
| ◩ | Toggle 3D buildings |

---

## ✨ Features

### 3D Map Features
- ✅ **Uniform height 3D buildings** - Tesla-style visualization
- ✅ **Dark map theme** - Black/dark gray color scheme
- ✅ **Tilted perspective** - 60° pitch for 3D effect
- ✅ **Road-following simulation** - Car stays on roads
- ✅ **Smooth camera follow** - Automotive-grade transitions

### UI Features
- ✅ **Large speed display** - Tesla-inspired design
- ✅ **Animated speed bar** - Visual speed indicator
- ✅ **Compass ring** - Real-time heading display
- ✅ **Trip computer** - Time, distance, avg speed
- ✅ **GPS data panel** - Lat/Lon/Alt/Satellites
- ✅ **Signal strength** - GPS satellite bars

---

## 🆚 Comparison: Original vs 3D Nav

| Feature | Original (server.py) | 3D Nav (server_3d_nav.py) |
|---------|---------------------|---------------------------|
| Port | 5000 | 5001 |
| Map Style | Standard OSM | Dark Tesla-style |
| Buildings | 2D only | 3D extruded (uniform height) |
| Map Tilt | 45° | 60° (more dramatic) |
| Simulation | Random movement | Road-following |
| UI Theme | Cyberpunk blue | Dark automotive |
| Speed Display | Overlay | Large left panel |
| Best for | General GPS tracking | Automotive demo |

---

## 🎯 Simulation Behavior

The road-following simulation:
1. Generates a grid of roads around current position
2. Snaps car position to nearest road
3. Adjusts heading to align with road direction
4. Varies speed realistically (20-70 km/h)
5. Smooth acceleration/deceleration

---

## 🛠️ Customization

### Change Building Height
Edit `script_3d_nav.js`:
```javascript
const uniformHeight = 50; // Change this value (meters)
```

### Adjust Map Tilt
Edit `script_3d_nav.js`:
```javascript
const CONFIG = {
    MAP_TILT: 60,  // 0-75 degrees
    // ...
};
```

### Change Default Location
Edit `server_3d_nav.py`:
```python
gps_data = {
    'latitude': 40.7128,  # New York
    'longitude': -74.0060,
    # ...
}
```

---

## 📡 With Real GPS Hardware

1. Edit `server_3d_nav.py`:
```python
SERIAL_PORT = 'COM3'  # Your ESP32 port
BAUD_RATE = 9600      # Your GPS baud rate
```

2. Run server and connect via dashboard

---

## 🔧 Troubleshooting

### Map not loading
- Check internet connection (OSM tiles require online)
- Open browser console for errors

### Simulation not working
- Click "START NAVIGATION" first
- Check browser console for errors

### Port 5001 already in use
Edit `server_3d_nav.py`:
```python
app.run(host='0.0.0.0', port=5002, ...)  # Change port
```

---

## 🎨 Color Scheme

| Element | Color |
|---------|-------|
| Background | `#000000` (pure black) |
| Map land | `#0d0d0d` |
| Buildings | `#1a1a1a` |
| Accent blue | `#0066ff` |
| Accent cyan | `#00d4ff` |
| Speed text | `#ffffff` |

---

**Run both servers simultaneously to compare!**
- Original: http://localhost:5000
- 3D Nav: http://localhost:5001
