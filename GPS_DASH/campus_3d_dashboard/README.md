# 🏛️ Campus 3D Digital Twin Dashboard

A futuristic, hackathon-ready 3D navigation dashboard for campus visualization with LiDAR point cloud aesthetics, glassmorphism UI, and animated pathfinding.

![Dashboard Preview](preview.png)

---

## ✨ Features

- **🔮 LiDAR Point Cloud Visualization** - Buildings rendered as glowing particles
- **🎨 Glassmorphism UI** - Modern frosted glass interface panels
- **🌟 Bloom & Glow Effects** - Neon cyan/blue color scheme with post-processing
- **🛤️ Animated Pathfinding** - Glowing paths between locations with particle effects
- **📊 Real-time Stats Dashboard** - Occupancy, active users, building info
- **🎮 Interactive 3D Controls** - Orbit, pan, zoom, rotate
- **📍 Location Markers** - Animated cones with hover labels

---

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ installed
- Python 3.8+ (for DXF conversion)

### 1. Install Dependencies

```bash
cd campus_3d_dashboard
npm install
```

### 2. Convert DXF Model (Optional)

If you have a DXF file of your campus:

```bash
# Install Python dependencies
pip install ezdxf numpy

# Run converter
python convert_dxf.py
```

This will generate `public/models/campus_model.json`

### 3. Run Development Server

```bash
npm run dev
```

The app will open at `http://localhost:3000`

---

## 📁 Project Structure

```
campus_3d_dashboard/
├── src/
│   ├── components/
│   │   ├── CampusPointCloud.jsx   # 3D point cloud renderer
│   │   ├── AnimatedPath.jsx        # Path animation with particles
│   │   ├── HUDOverlay.jsx          # Top heads-up display
│   │   ├── LocationPanel.jsx       # Left location list
│   │   ├── StatsPanel.jsx          # Right statistics panel
│   │   └── LoadingScreen.jsx       # Futuristic loading animation
│   ├── App.jsx                     # Main app component
│   ├── main.jsx                    # React entry point
│   └── index.css                   # Global styles + glassmorphism
├── public/
│   └── models/
│       └── campus_model.json       # Generated 3D model
├── convert_dxf.py                  # DXF to JSON converter
├── package.json
├── vite.config.js
├── tailwind.config.js
└── README.md
```

---

## 🎮 Controls

| Action | Mouse |
|--------|-------|
| Rotate | Left click + drag |
| Pan | Right click + drag |
| Zoom | Scroll wheel |
| Select Location | Click on marker |
| Create Path | Right-click between locations |

---

## 🎨 Customization

### Colors

Edit `tailwind.config.js`:

```js
colors: {
  'neon-cyan': '#00ffcc',    // Primary color
  'neon-blue': '#00ccff',    // Secondary color
  'neon-purple': '#cc00ff',  // Accent color
}
```

### Campus Locations

Edit `src/App.jsx`:

```js
const CAMPUS_LOCATIONS = [
  { id: 1, name: 'Main Gate', lat: 21.1458, lng: 81.8292, type: 'entrance' },
  // Add your locations here
]
```

### Path Routes

Edit paths in `src/App.jsx`:

```js
const PATHS = {
  '1-2': [
    { x: 0, y: 0, z: 0 },
    { x: 10, y: 0, z: 20 },
  ],
}
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **Three.js** | 3D rendering |
| **React Three Fiber** | React + Three.js integration |
| **React Three Drei** | Pre-built 3D components |
| **React Three Postprocessing** | Bloom, depth of field |
| **Framer Motion** | UI animations |
| **Tailwind CSS** | Styling |
| **Vite** | Build tool |

---

## 📦 Build for Production

```bash
npm run build
```

Output will be in `dist/` folder.

Preview production build:

```bash
npm run preview
```

---

## 🎯 Hackathon Tips

### To Win Judges Over:

1. **Demo the Pathfinding** - Right-click to create glowing paths
2. **Show Real-time Stats** - Occupancy changes dynamically
3. **Rotate the View** - Show off the 3D point cloud
4. **Highlight Accessibility** - Mention wheelchair route planning
5. **AR Integration Idea** - Pitch mobile AR navigation future feature

### Presentation Script:

> "Our Campus Digital Twin transforms navigation using real-time 3D visualization. Watch as I create a path from the Main Gate to the Engineering Block - the glowing cyan line shows the optimal route while particle effects indicate direction of travel."

---

## 🔧 Troubleshooting

### "Cannot find module 'three'"
```bash
npm install three
```

### DXF conversion fails
```bash
pip install ezdxf numpy
```

### Port 3000 already in use
Edit `vite.config.js`:
```js
server: { port: 3001, open: true }
```

---

## 📄 License

MIT License - Feel free to use for your hackathon!

---

## 🙏 Credits

Created for GPS DASH Mini Project 2024

**Technologies used:**
- Point cloud generation inspired by LiDAR visualization
- Glassmorphism UI design pattern
- Three.js reactive rendering with React Three Fiber

---

## 🚀 Next Steps

1. **Import Real DXF** - Run `python convert_dxf.py` with your campus file
2. **Add Backend** - Connect to real-time occupancy sensors
3. **Mobile AR** - Integrate WebXR for phone AR navigation
4. **Multi-user** - Add WebSocket for collaborative viewing

---

**Made with 💚 for hackathon success!**
