# 🚀 Quick Start Guide: Video-Based 3D Mapping System

## 📋 Project Title:
**"EdgeDrive3D Vision: AI-Powered 3D Mapping & Autonomous Navigation System"**

---

## 🎯 What This Does:

1. **Loads a video file** (no camera needed!)
2. **Processes with AI**:
   - YOLOv8 for object detection (cars, people, etc.)
   - MiDaS for depth estimation
3. **Generates 3D mapping**:
   - 3D point cloud from depth
   - 3D object positions
4. **Visualizes in dashboard**:
   - Interactive 3D viewer
   - Video player with detections
   - Analytics and export

---

## 📦 Installation (Already Done!):

✅ Python 3.13  
✅ PyTorch  
✅ YOLOv8 (ultralytics)  
✅ OpenCV  
✅ MiDaS (depth estimation)  
✅ Plotly (3D visualization)  
✅ Streamlit (dashboard)  

---

## 🎬 Step-by-Step Testing:

### Step 1: Process Video File

```bash
cd "C:\Users\Khushi Tirkey\OneDrive\Documents\PROJECTS\MiniProject\combined_final_new_gps\combined_proj_folder"

# Process demo video (first 100 frames)
py -3.13 video_3d_mapping.py demo_video1.mp4
```

**What happens:**
- Loads `demo_video1.mp4`
- Runs YOLO detection on each frame
- Estimates depth with MiDaS
- Generates 3D point cloud
- Saves results to `output/` folder

**Expected output:**
```
Processing video: demo_video1.mp4
Video: 1280x720 @ 30fps, 1049 frames
Processing first 100 frames...

  Frame 10/100 (2.5 FPS)
  Frame 20/100 (2.5 FPS)
  ...

✓ Processed 100 frames in 40.0s (2.5 FPS)
✓ Results saved to: output/video_results_20260319_...
```

---

### Step 2: View 3D Dashboard

```bash
# Launch 3D visualization dashboard
py -3.13 -m streamlit run dashboard/video_3d_dashboard.py --server.port 8505
```

**Open browser:** http://localhost:8505

**What you'll see:**

#### Tab 1: 🗺️ 3D Map
- **Interactive 3D point cloud** - Rotate, zoom, pan
- **3D object positions** - See where cars/people are in 3D space
- **Frame slider** - Navigate through video frames
- **Live stats** - FPS, object count

#### Tab 2: 📷 Video
- **Video player** - See frames with detection overlays
- **Depth map** - Color-coded depth visualization
- **Frame-by-frame navigation**

#### Tab 3: 📊 Analytics
- **Detection timeline** - Graph of objects over time
- **Class distribution** - Bar chart of detected objects
- **Performance metrics** - FPS, processing time

#### Tab 4: 💾 Data
- **Export JSON** - Download all detection data
- **Summary stats** - Total frames, objects, avg FPS

---

### Step 3: Use Your Own Video

**Option A: Upload in Dashboard**
1. Open dashboard (http://localhost:8505)
2. In sidebar, uncheck "Use Demo Video"
3. Click "Upload Video"
4. Select your video file (.mp4, .avi, .mov)
5. Adjust settings:
   - Max Frames: 100 (for testing)
   - Confidence: 0.4
6. Click "🚀 Process Video"

**Option B: Command Line**
```bash
# Process your video
py -3.13 video_3d_mapping.py "path/to/your/video.mp4"
```

---

## 🎮 Dashboard Controls:

### 3D Point Cloud Viewer:
- **Left-click + drag** - Rotate view (360°)
- **Right-click + drag** - Pan
- **Scroll wheel** - Zoom in/out
- **Shift + drag** - Tilt

### Video Viewer:
- **Slider** - Navigate frames
- **Depth map** - See distance (red=far, blue=close)

### Analytics:
- **Timeline** - See detection patterns
- **Bar chart** - Object class distribution

---

## 📊 Sample Output:

### Console Output:
```
============================================================
  EdgeDrive3D Video Processor v1.0
============================================================

Initializing components...
  Loading Depth Model: midas_hybrid...
  ✓ Depth model loaded on cpu
  Loading YOLO Model: yolov8n.pt...
  ✓ YOLO model loaded on cpu

✓ Video Processor Ready!

Processing video: demo_video1.mp4
Video: 1280x720 @ 30fps, 1049 frames
Processing first 100 frames...

Processing...
  Frame 10/100 (2.5 FPS)
  Frame 20/100 (2.5 FPS)
  ...

✓ Processed 100 frames in 40.0s (2.5 FPS)
✓ Results saved to: output/video_results_20260319_153045

✓ Processing complete! 100 frames processed
  Total objects detected: 245
  Check 'output/' folder for results
```

### Dashboard Visualization:
- **3D Point Cloud**: Thousands of colored points forming 3D scene
- **Objects**: Red dots for people, blue for vehicles
- **Depth Map**: Color gradient showing distances
- **Timeline**: Graph showing detection count per frame

---

## 🔧 Configuration Options:

### In Command Line:
```python
processor = Video3DProcessor({
    'yolo_model': 'yolov8n.pt',      # Model: yolov8n/s/m/l
    'confidence': 0.4,                # Detection threshold (0-1)
    'depth_model': 'midas_hybrid',    # Depth: midas_small/hybrid/large
    'camera_fx': 800.0,               # Camera focal length X
    'camera_cx': 320.0,               # Camera principal point X
})
```

### In Dashboard:
- **Max Frames**: Limit processing (faster testing)
- **Confidence**: Filter weak detections
- **Use Demo Video**: Toggle demo/upload

---

## 📁 File Structure:

```
combined_proj_folder/
├── video_3d_mapping.py          # Main processor
├── dashboard/
│   └── video_3d_dashboard.py    # 3D visualization
├── demo_video1.mp4              # Test video
├── output/
│   └── video_results_*/         # Processing results
│       ├── results.json         # All detections
│       ├── pointcloud_000.ply   # 3D point clouds
│       └── ...
└── QUICK_START_VIDEO.md         # This file
```

---

## 🎯 Testing Checklist:

- [ ] Install all dependencies (already done ✅)
- [ ] Run `video_3d_mapping.py demo_video1.mp4`
- [ ] Check output folder for results
- [ ] Launch dashboard: `streamlit run dashboard/video_3d_dashboard.py`
- [ ] Open http://localhost:8505
- [ ] Explore 3D point cloud (rotate, zoom)
- [ ] Navigate frames with slider
- [ ] View depth maps
- [ ] Check analytics tab
- [ ] Export JSON data

---

## 🐛 Troubleshooting:

### "MiDaS model not loading"
```bash
# Check internet connection (downloads on first run)
# Wait 1-2 minutes for initial download
```

### "Out of memory"
```bash
# Reduce max frames
py -3.13 video_3d_mapping.py video.mp4 --max-frames 50

# Or use smaller model
# Edit video_3d_mapping.py, change 'yolov8n.pt' to 'yolov8n.pt'
```

### "Dashboard not loading"
```bash
# Check if port 8505 is free
# Try different port:
py -3.13 -m streamlit run dashboard/video_3d_dashboard.py --server.port 8506
```

### "No objects detected"
```bash
# Lower confidence threshold
# Edit video_3d_mapping.py, change confidence=0.4 to 0.2
```

---

## 🚀 Next Steps (After Testing):

### Phase 1: ✅ Video Testing (NOW)
- [x] Create video processor
- [x] Create 3D dashboard
- [ ] Test with demo video
- [ ] Test with custom videos

### Phase 2: ML + Mapping Integration (NEXT)
- [ ] Add 3D bounding boxes
- [ ] Improve depth accuracy
- [ ] Add object tracking across frames
- [ ] Fuse detections over time

### Phase 3: Sensor Integration (LATER)
- [ ] Add camera input (live feed)
- [ ] Add GPS fusion
- [ ] Add IMU for orientation
- [ ] Real-time processing

### Phase 4: Decision Making (FUTURE)
- [ ] Path planning
- [ ] Obstacle avoidance
- [ ] Motor control
- [ ] Autonomous navigation

---

## 📞 Quick Commands:

```bash
# Process demo video
py -3.13 video_3d_mapping.py demo_video1.mp4

# Launch dashboard
py -3.13 -m streamlit run dashboard/video_3d_dashboard.py --server.port 8505

# Process custom video
py -3.13 video_3d_mapping.py "path/to/video.mp4"

# View results
cd output
dir
```

---

**Ready to test? Run the commands above and explore your 3D mapping system!** 🗺️🚀

---

**Last Updated:** March 19, 2026  
**Version:** 1.0.0 - Video Testing Mode  
**Status:** ✅ Ready to Test
