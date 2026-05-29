# Quick Start Guide - main_complete.py

## 🚀 Quick Commands

### Test with Video
```bash
python main_complete.py video your_video.mp4
```

### Real-time Webcam
```bash
python main_complete.py webcam
```

### Process Image
```bash
python main_complete.py image photo.jpg --show
```

## 🎮 Keyboard Controls (During Video/Webcam)

| Key | What it does |
|-----|--------------|
| **`q`** | Quit |
| **`1`** | Toggle Lanes ON/OFF |
| **`2`** | Toggle Signs ON/OFF |
| **`3`** | Toggle Depth ON/OFF |
| **`4`** | Toggle Drowsiness ON/OFF |
| **`p`** | Pause Video |
| **`s`** | Save Screenshot |

## 🎯 Lane Detection Presets

| Preset | When to use |
|--------|-------------|
| `default` | Normal roads, good markings |
| `highway` | High-speed roads, clear lanes |
| `city` | Urban areas, complex markings |
| `faded` | Worn out lane markings |
| `night` | Low light conditions |
| `indian_road` | Indian road conditions |

## ⚡ Performance Modes

### Fast Mode (30+ FPS)
```bash
python main_complete.py video input.mp4 --fast
```

### Balanced Mode (15-25 FPS)
```bash
python main_complete.py video input.mp4 --model yolov8n.pt
```

### Accuracy Mode (10-15 FPS)
```bash
python main_complete.py video input.mp4 --model yolov8m.pt --all-models
```

## 📊 What Each Model Does

1. **YOLO Object Detection** - Finds cars, people, bikes in 3D
2. **Lane Detection (HLS)** - Finds road lanes with color + edges
3. **Traffic Signs** - Reads stop signs, speed limits, etc.
4. **Depth (MiDaS)** - Calculates distance to objects
5. **Drowsiness** - Monitors driver alertness
6. **Hazards** - Warns about nearby obstacles
7. **BEV** - Bird's eye view map
8. **Decision** - Combines everything for navigation

## 🔧 Common Issues

**Problem**: Too slow
```bash
# Solution: Use fast mode
python main_complete.py video input.mp4 --fast
```

**Problem**: Lanes not detected
```bash
# Solution: Try different preset
python main_complete.py video input.mp4 --lane-preset highway
```

**Problem**: Missing models error
```bash
# Solution: Install dependencies
pip install ultralytics timm opencv-python
```

## 💡 Pro Tips

- Use `--depth-skip 5` to process depth less frequently (faster)
- Use `--confidence 0.6` for fewer false detections
- Press `s` during video to save interesting frames
- Start with `--model yolov8n.pt` for best speed

---

**Full documentation**: See `MAIN_COMPLETE_README.md`
