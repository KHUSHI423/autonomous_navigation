# 🚀 Run ALL Models Together - Quick Guide

## One Command - Everything Visible!

```bash
python run_all_models.py video your_video.mp4
```

That's it! ALL models will run together with everything visible on screen!

## What You'll See

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   [Your Video Playing]                                     │
│                                                            │
│   🟢 GREEN ROAD PATH (between lanes)                       │
│   📦 Object boxes with labels                              │
│   🚦 Traffic sign labels                                   │
│   📏 Distance measurements                                 │
│   ⚠️ Hazard warnings                                       │
│                                                            │
│   ┌──────────────────────────────────┐                    │
│   │  ALL-MODEL DETECTION SYSTEM      │                    │
│   │  [ON] Objects: 5                 │                    │
│   │  [ON] Lanes | Width: 3.5m       │                    │
│   │  [ON] Signs: 2                   │                    │
│   │  [ON] Depth                      │                    │
│   │  [OFF] Drowsy (press 5)          │                    │
│   │  [ON] Hazards                    │                    │
│   │  [OFF] BEV Map (press 7)         │                    │
│   │  [ON] Decision | FORWARD         │                    │
│   │                                  │                    │
│   │  PERFORMANCE                     │                    │
│   │  FPS: 25.3 | Time: 40ms          │                    │
│   │  Elapsed: 120s | Frame: 3600     │                    │
│   └──────────────────────────────────┘                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Keyboard Controls

| Key | Toggle This Model |
|-----|-------------------|
| **`1`** | Object Detection (YOLO) |
| **`2`** | Lane Detection (HLS + Path) |
| **`3`** | Traffic Sign Detection |
| **`4`** | Depth Estimation |
| **`5`** | Drowsiness Detection |
| **`6`** | Hazard Detection |
| **`7`** | BEV Mapping |
| **`P`** | Pause/Resume |
| **`S`** | Save Screenshot |
| **`Q`** | Quit |

## Quick Examples

### Video with all models
```bash
python run_all_models.py video demo.mp4
```

### Webcam real-time
```bash
python run_all_models.py webcam
```

### Different lane preset
```bash
python run_all_models.py video highway.mp4 --lane-preset highway
```

### Better accuracy
```bash
python run_all_models.py video demo.mp4 --model yolov8m.pt --confidence 0.6
```

### Fast mode
```bash
python run_all_models.py video demo.mp4 --model yolov8n.pt
```

## All Models Run Together

✅ **Object Detection** - Finds cars, people, bikes with 3D positions  
✅ **Lane Detection** - Green filled path on road (like LinkedIn post!)  
✅ **Sign Detection** - Reads stop signs, speed limits, etc.  
✅ **Depth Estimation** - Distance to all objects  
✅ **Hazard Detection** - Warns about nearby obstacles  
✅ **Decision Making** - Suggests actions (forward/slow/stop)  
⭕ **Drowsiness** - Press `5` to enable  
⭕ **BEV Map** - Press `7` to enable  

## Troubleshooting

**Problem**: "Could not open video"  
**Fix**: Check video file is in the folder

**Problem**: Slow FPS  
**Fix**: Use `--model yolov8n.pt` (fastest)

**Problem**: Lanes not detected  
**Fix**: Try `--lane-preset highway` or `--lane-preset faded`

---

**Ready? Just run:**
```bash
python run_all_models.py video your_video.mp4
```

🎬 **Everything visible together - one command!**
