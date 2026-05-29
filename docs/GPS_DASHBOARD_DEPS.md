# GPS Dashboard - Dependencies & Setup

## 📦 Required Packages

All packages should be installed for **Python 3.13** (Streamlit's Python version).

### Core Dependencies
```bash
# Install all at once
py -3.13 -m pip install torch torchvision torchaudio opencv-python ultralytics numpy pandas streamlit folium branca plotly
```

### Individual Packages

| Package | Purpose | Status |
|---------|---------|--------|
| `torch` | PyTorch for MiDaS depth | ⏳ Installing... |
| `torchvision` | PyTorch vision utilities | ⏳ Installing... |
| `opencv-python` | Image processing | ✅ Installed |
| `ultralytics` | YOLOv8 object detection | ❓ Need to verify |
| `numpy` | Numerical operations | ✅ Installed |
| `pandas` | Data handling | ✅ Installed |
| `streamlit` | Dashboard framework | ✅ Installed |
| `folium` | OpenStreetMap visualization | ✅ Installed |
| `branca` | Folium dependency | ✅ Installed |
| `plotly` | Interactive charts | ✅ Installed |

---

## 🚀 Quick Start

### 1. Install All Dependencies
```bash
cd C:\Users\Khushi Tirkey\OneDrive\Documents\PROJECTS\MiniProject\combined_final_new_gps\combined_proj_folder

# Install PyTorch (CPU version - slower but works everywhere)
py -3.13 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
py -3.13 -m pip install opencv-python ultralytics
```

### 2. Run GPS Dashboard
```bash
python main.py gps-dashboard
```

### 3. Access Dashboard
Open your browser to: **http://localhost:8502**

---

## 📁 Dashboard Files

```
dashboard/
├── gps_dashboard.py          # Main GPS dashboard (OpenStreetMap)
├── gps_dashboard_3d.py       # 3D GPS dashboard (CIT Campus)
├── app.py                    # Basic perception dashboard
├── realtime_app.py           # Real-time processing app
└── components/
    ├── osm_view.py           # OpenStreetMap components
    └── map_3d.py             # 3D map components
```

---

## 🎯 Dashboard Features

### GPS Dashboard (OpenStreetMap)
- ✅ Real-time GPS tracking on OpenStreetMap
- ✅ Vehicle position marker
- ✅ Detected objects overlay
- ✅ Trajectory visualization
- ✅ Perception data display
- ✅ BEV (Bird's Eye View)
- ✅ Depth visualization

### 3D GPS Dashboard (CIT Campus)
- ✅ 3D campus visualization
- ✅ Interactive camera controls
- ✅ Real-time vehicle tracking
- ✅ Object visualization in 3D

---

## 🐛 Troubleshooting

### Error: No module named 'torch'
```bash
py -3.13 -m pip install torch torchvision
```

### Error: No module named 'cv2'
```bash
py -3.13 -m pip install opencv-python
```

### Error: No module named 'ultralytics'
```bash
py -3.13 -m pip install ultralytics
```

### Dashboard won't start
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart dashboard
python main.py gps-dashboard
```

---

## 📊 Performance Notes

### With CPU-only PyTorch:
- Depth estimation: ~2-3 seconds per frame
- Object detection: ~200-500ms per frame
- **Recommended for:** Testing, development

### With GPU PyTorch (if you have NVIDIA GPU):
```bash
# Uninstall CPU version first
py -3.13 -m pip uninstall torch torchvision torchaudio

# Install GPU version
py -3.13 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Depth estimation: ~25-35ms per frame
- Object detection: ~50-100ms per frame
- **Recommended for:** Real-time processing

---

## 🎓 Usage Examples

### Simulated GPS Mode (for testing)
```bash
python main.py gps-dashboard --simulate
```

### Hardware GPS Mode
```bash
python main.py gps-dashboard --simulate=False --gps-port COM3
```

### 3D Dashboard
```bash
python main.py 3d-dashboard
```

---

**Last Updated:** March 19, 2026  
**Python Version:** 3.13  
**Status:** ⏳ Installing PyTorch...
