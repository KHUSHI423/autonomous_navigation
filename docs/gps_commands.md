# 📍 GPS Commands Reference Guide
## EdgeDrive3D - Complete GPS Integration Commands

---

## 🚀 Quick Start

### 1. Launch GPS Dashboard (Simulated Mode - No Hardware Required)

```bash
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder
python main.py gps-dashboard
```

**Access the dashboard at:** `http://localhost:8502`

---

### 2. Launch 3D GPS Dashboard (CIT Campus - Recommended!)

```bash
# Launch 3D dashboard with simulated GPS
python main.py 3d-dashboard

# Access at: http://localhost:8503
```

**Features:**
- 🏫 Full 3D visualization of Coimbatore Institute of Technology
- 🏢 10 campus landmarks with accurate heights
- 🎮 Interactive camera controls (rotate, zoom, pan)
- 🚗 Real-time vehicle tracking in 3D
- 🎯 Detected objects shown in 3D space
- 🌍 Multiple map styles (Dark, Light, Satellite)
- ✨ Beautiful visual effects and animations

---

### 3. Launch GPS Dashboard (Hardware Mode - Real GPS)

```bash
# Windows - Auto-detect GPS port
python main.py gps-dashboard --simulate=False

# Windows - Specify COM port
python main.py gps-dashboard --simulate=False --gps-port COM3

# Linux - Specify serial port
python main.py gps-dashboard --simulate=False --gps-port /dev/ttyUSB0

# Custom baud rate
python main.py gps-dashboard --simulate=False --gps-port COM3 --baudrate 115200
```

---

## 📦 Installation Commands

### Install All Dependencies

```bash
# Navigate to project folder
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder

# Install requirements
pip install -r requirements.txt
```

### Install GPS-Specific Dependencies

```bash
# GPS and mapping libraries
pip install folium pynmea2 pyserial branca geojson

# Streamlit dashboard
pip install streamlit streamlit-folium

# Additional utilities
pip install filterpy numpy opencv-python
```

---

## 🧪 Testing Commands

### Run GPS Integration Tests

```bash
# Run all GPS tests
python tests/test_gps_integration.py

# Expected output:
# - GPS Hardware Detection
# - Simulated GPS Reading
# - Coordinate Transformation
# - Kalman Filter
# - OpenStreetMap Visualization
# - Full GPS Perception Pipeline
```

### Test GPS Hardware Detection

```bash
# List all available serial devices
python -c "from hardware.gps_reader import list_gps_devices; devs = list_gps_devices(); [print(f'{d[\"device\"]}: {d[\"description\"]}') for d in devs]"

# Or run the GPS reader directly
python hardware/gps_reader.py
```

### Test Coordinate Transformation

```bash
# Run coordinate transform tests
python utils/coordinate_transform.py
```

### Test GPS Perception Engine

```bash
# Run GPS perception engine standalone test
python core/perception_engine_gps.py
```

### Test GPS Configuration

```bash
# Create and display default GPS config
python config/gps_settings.py
```

---

## 🔧 Configuration Commands

### Create Default GPS Configuration

```bash
# Generate default config file
python config/gps_settings.py
```

This creates `config/gps_settings.yaml` with default settings.

### Load Custom GPS Configuration (Python)

```python
from config.gps_settings import GPSSystemConfig

# Load from file
config = GPSSystemConfig.load("config/gps_settings.yaml")

# Or use presets
config = GPSSystemConfig.simulated_config()      # For testing
config = GPSSystemConfig.hardware_config()       # For USB GPS
config = GPSSystemConfig.usb_dongle_config()     # Optimized for USB
config = GPSSystemConfig.high_accuracy_config()  # For RTK GPS
```

### Edit GPS Configuration

Edit `config/gps_settings.yaml`:

```yaml
gps:
  port: auto              # or COM3, /dev/ttyUSB0
  baudrate: 9600
  simulate: true          # Set false for real GPS
  use_kalman_filter: true

map:
  default_zoom: 16
  show_trajectory: true
  show_fov: true

perception:
  dashboard_port: 8502
  yolo_model: yolov8m.pt
  confidence: 0.4
```

---

## 🎯 GPS Dashboard Commands

### Basic Commands

```bash
# Default (simulated GPS)
python main.py gps-dashboard

# Hardware GPS (auto-detect port)
python main.py gps-dashboard --simulate=False

# Specify port and baud rate
python main.py gps-dashboard --simulate=False --gps-port COM3 --baudrate 9600
```

### Advanced Options

```bash
# Custom dashboard port
python main.py gps-dashboard --port 8503

# Custom address (network access)
python main.py gps-dashboard --address 0.0.0.0 --port 8502

# Full example with all options
python main.py gps-dashboard \
  --simulate=False \
  --gps-port COM3 \
  --baudrate 115200 \
  --port 8502 \
  --address localhost
```

---

## 📍 GPS Hardware Commands

### Find GPS Device (Windows)

```bash
# Check Device Manager for COM ports
# Press Win+X → Device Manager → Ports (COM & LPT)

# Or use PowerShell
Get-PnpDevice -Class Ports | Select-Object Name, DeviceID, Status
```

### Find GPS Device (Linux)

```bash
# List USB serial devices
ls -l /dev/ttyUSB*

# List ACM devices
ls -l /dev/ttyACM*

# Check dmesg for GPS detection
dmesg | grep tty

# List all serial ports
python -c "import serial.tools.list_ports; ports = serial.tools.list_ports.comports(); [print(f'{p.device}: {p.description}') for p in ports]"
```

### Test GPS Connection

```bash
# Test GPS with auto-detection
python -c "
from hardware.gps_reader import GPSReceiver
gps = GPSReceiver(port='auto', simulate=False)
gps.start()
import time
time.sleep(5)
reading = gps.get_position()
print(reading)
gps.disconnect()
"
```

### GPS Reader Interactive Test

```bash
# Run interactive GPS test
python hardware/gps_reader.py
```

---

## 🗺️ Map Visualization Commands

### Generate Test Map (Python)

```python
from dashboard.components.osm_view import create_osm_map, create_vehicle_marker, add_legend_to_map

# Create map
m = create_osm_map(center_lat=12.9716, center_lon=77.5946, zoom=16)

# Add vehicle
vehicle = create_vehicle_marker(12.9716, 77.5946, heading=45, speed=5.0)
vehicle.add_to(m)

# Add legend
add_legend_to_map(m)

# Save
m.save("test_map.html")
```

### Run OSM Visualization Test

```bash
# Generate test map with markers
python tests/test_gps_integration.py
# Output: output/gps_tests/test_osm_map.html
```

---

## 📊 Coordinate Transformation Commands

### Test Camera to GPS Conversion

```python
from utils.coordinate_transform import CoordinateTransformer

# Initialize transformer
transformer = CoordinateTransformer(
    base_lat=12.9716,
    base_lon=77.5946,
    base_alt=920.0,
    base_heading=45.0
)

# Convert camera coords to GPS
lat, lon, alt = transformer.camera_to_gps(x=5.0, y=0.0, z=20.0)
print(f"GPS: {lat:.6f}, {lon:.6f}, {alt:.1f}m")

# Convert GPS to camera coords
x, y, z = transformer.gps_to_camera(lat, lon, alt)
print(f"Camera: {x:.2f}, {y:.2f}, {z:.2f}m")
```

### Calculate Distance Between GPS Points

```python
from utils.coordinate_transform import haversine_distance

lat1, lon1 = 12.9716, 77.5946
lat2, lon2 = 12.9726, 77.5956

distance = haversine_distance(lat1, lon1, lat2, lon2)
print(f"Distance: {distance:.2f}m")
```

### Calculate Bearing

```python
from utils.coordinate_transform import bearing

brg = bearing(lat1, lon1, lat2, lon2)
print(f"Bearing: {brg:.1f}°")
```

---

## 📼 GPS Logging Commands

### Start GPS Logging Session

```python
from hardware.gps_reader import GPSLogger

logger = GPSLogger(filepath="gps_trajectory.jsonl")
logger.start()

# Log data
from hardware.gps_reader import GPSReading
reading = GPSReading(latitude=12.9716, longitude=77.5946, altitude=920.0, is_valid=True)
logger.log(reading, extra_data={'objects': []})

# Stop logging
logger.stop()
```

### Replay GPS Log

```python
from hardware.gps_reader import GPSLogger

logger = GPSLogger()
for record in logger.replay("gps_trajectory.jsonl"):
    print(record)
```

### Save Trajectory from Engine

```python
from core.perception_engine_gps import GPSPerceptionEngine

engine = GPSPerceptionEngine({'gps_simulate': True})
engine.start_gps()

# ... process frames ...

# Save trajectory
engine.save_trajectory("output/gps_logs/trajectory.json")
engine.stop_gps()
```

---

## 🔍 Debugging Commands

### Check GPS Statistics

```python
from hardware.gps_reader import GPSReceiver

gps = GPSReceiver(simulate=True)
gps.start()

# Get statistics
stats = gps.get_statistics()
print(stats)

gps.disconnect()
```

### Monitor GPS Updates

```python
from hardware.gps_reader import GPSReceiver
import time

gps = GPSReceiver(simulate=True)
gps.start()

print("Monitoring GPS updates (Ctrl+C to stop)...")
try:
    while True:
        reading = gps.get_position()
        if reading and reading.is_valid:
            print(f"\r{reading}", end="")
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    gps.disconnect()
```

### Wait for GPS Fix

```python
from hardware.gps_reader import GPSReceiver

gps = GPSReceiver(simulate=False)
gps.start()

print("Waiting for GPS fix (30 seconds)...")
reading = gps.wait_for_fix(timeout=30.0)

if reading:
    print(f"✓ GPS Fix: {reading}")
else:
    print("✗ No GPS fix acquired")

gps.disconnect()
```

### Register GPS Update Callback

```python
from hardware.gps_reader import GPSReceiver

def on_gps_update(reading):
    print(f"GPS Update: {reading.latitude:.6f}, {reading.longitude:.6f}")

gps = GPSReceiver(simulate=True)
gps.on_update(on_gps_update)
gps.start()
```

---

## 🎓 Python API Quick Reference

### GPS Receiver

```python
from hardware.gps_reader import GPSReceiver, GPSReading

# Initialize
gps = GPSReceiver(port='auto', baudrate=9600, simulate=True)

# Start reading
gps.start()

# Get position
reading = gps.get_position()
if reading and reading.is_valid:
    print(f"Lat: {reading.latitude}")
    print(f"Lon: {reading.longitude}")
    print(f"Alt: {reading.altitude}")
    print(f"Speed: {reading.speed} m/s")
    print(f"Heading: {reading.heading}°")
    print(f"Satellites: {reading.satellites}")
    print(f"Accuracy: ±{reading.accuracy}m")

# Stop
gps.stop()
gps.disconnect()
```

### GPS Perception Engine

```python
from core.perception_engine_gps import GPSPerceptionEngine

# Initialize
engine = GPSPerceptionEngine({
    'yolo_model': 'yolov8m.pt',
    'confidence': 0.4,
    'max_depth': 50.0,
    'gps_simulate': True,
    'gps_port': 'auto',
    'baudrate': 9600
})

# Start GPS
engine.start_gps()

# Process frame
import cv2
frame = cv2.imread('image.jpg')
result = engine.process_frame_gps(frame)

# Access results
print(f"GPS: {result.gps_reading}")
print(f"Objects with GPS: {result.objects_with_gps}")
print(f"Trajectory: {result.trajectory_point}")
print(f"Decision: {result.decision}")

# Get statistics
stats = engine.get_statistics()

# Stop
engine.stop_gps()
```

### Coordinate Transformer

```python
from utils.coordinate_transform import CoordinateTransformer, haversine_distance, bearing

# Initialize
transformer = CoordinateTransformer(
    base_lat=12.9716,
    base_lon=77.5946,
    base_alt=920.0,
    base_heading=45.0
)

# Camera to GPS
lat, lon, alt = transformer.camera_to_gps(x=5.0, y=0.0, z=20.0)

# GPS to Camera
point = transformer.gps_to_camera(lat, lon, alt)
print(f"Camera: ({point.x}, {point.y}, {point.z})")

# Update position (for moving vehicle)
transformer.update_base_position(
    latitude=12.9720,
    longitude=77.5950,
    heading=50.0
)

# Distance between points
dist = haversine_distance(lat1, lon1, lat2, lon2)

# Bearing
brg = bearing(lat1, lon1, lat2, lon2)
```

---

## 🛠️ Troubleshooting Commands

### No GPS Device Found

```bash
# Windows - List all COM ports
python -c "import serial.tools.list_ports; ports = serial.tools.list_ports.comports(); [print(f'{p.device}: {p.description}') for p in ports]"

# Linux - List serial devices
ls -l /dev/ttyUSB* /dev/ttyACM*

# Use simulated mode for testing
python main.py gps-dashboard --simulate=True
```

### Map Not Loading

```bash
# Check internet connection
ping openstreetmap.org

# Try different tile provider in dashboard settings
# (Street, Satellite, Terrain, Dark)
```

### GPS Fix Not Acquiring

```bash
# Move GPS antenna outdoors with clear sky view
# Wait 30-60 seconds for cold start
# Check antenna connection

# Test with longer timeout
python -c "
from hardware.gps_reader import GPSReceiver
gps = GPSReceiver(simulate=False)
gps.start()
reading = gps.wait_for_fix(timeout=60.0)
print(reading)
gps.disconnect()
"
```

### High GPS Noise

```python
# Enable Kalman filter
from config.gps_settings import GPSSystemConfig
config = GPSSystemConfig()
config.gps.use_kalman_filter = True
config.gps.process_noise = 0.05
config.gps.measurement_noise = 1.0
config.save("config/gps_settings.yaml")
```

---

## 📈 Performance Monitoring

### Get Real-time Statistics

```python
from core.perception_engine_gps import GPSPerceptionEngine

engine = GPSPerceptionEngine({'gps_simulate': True})
engine.start_gps()

# Process frames and monitor
for i in range(100):
    result = engine.process_frame_gps(frame)
    
    if i % 10 == 0:
        stats = engine.get_statistics()
        print(f"GPS Updates: {stats['gps_updates']}")
        print(f"Objects Geolocated: {stats['objects_geolocated']}")
        print(f"Trajectory Points: {stats['trajectory_points']}")

engine.stop_gps()
```

### Export Session Data

```bash
# From dashboard sidebar:
# Click "Export Session Data" button

# Or programmatically:
python -c "
from core.perception_engine_gps import GPSPerceptionEngine
engine = GPSPerceptionEngine({'gps_simulate': True})
engine.start_gps()
# ... process frames ...
engine.save_trajectory('output/gps_logs/session_data.json')
engine.stop_gps()
"
```

---

## 🎯 Command Summary Table

| Command | Description |
|---------|-------------|
| `python main.py 3d-dashboard` | 🏫 Launch 3D GPS dashboard (CIT Campus) |
| `python main.py 3d-dashboard --simulate=False` | 3D dashboard with hardware GPS |
| `python main.py gps-dashboard` | Launch GPS dashboard (2D OpenStreetMap) |
| `python main.py gps-dashboard --simulate=False` | Launch with hardware GPS |
| `python main.py gps-dashboard --gps-port COM3` | Specify GPS port |
| `python tests/test_gps_integration.py` | Run GPS integration tests |
| `python hardware/gps_reader.py` | Test GPS reader |
| `python utils/coordinate_transform.py` | Test coordinate transforms |
| `python core/perception_engine_gps.py` | Test GPS perception engine |
| `python config/gps_settings.py` | Create GPS config |
| `pip install folium pynmea2 pyserial` | Install GPS dependencies |
| `python dashboard/components/map_3d.py` | Test 3D map component |

---

## 📞 Support

For more details, see:
- `GPS_QUICKSTART.md` - Quick start guide
- `new_plan_gps.md` - Full integration plan
- `DOCUMENTATION.md` - System documentation

**Built with ❤️ using OpenStreetMap © OpenStreetMap contributors**
