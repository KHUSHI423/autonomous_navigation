# 🚀 Intelligent ADAS - All Models + Dynamic Lanes

## ✅ What's Implemented

### 1. **ALL Models Integrated**
- ✅ Object Detection (YOLOv8)
- ✅ **Dynamic Lane Avoidance** (HIGHLY VISIBLE shifts)
- ✅ Traffic Sign Detection
- ✅ Depth Estimation (MiDaS)
- ✅ Driver Drowsiness Detection
- ✅ Road Hazard Detection
- ✅ Keyboard toggles for each model

### 2. **Dynamic Lane Avoidance - MUCH MORE VISIBLE**
- **Short Lane View:** Only bottom 30% of frame (very focused)
- **High Shift Amounts:** Up to 120px when object is near
- **Responsive:** 5-frame buffer (quick reaction)
- **Exponential Perspective:** Much more shift at bottom (near), less at top (far)
- **Color-Coded:**
  - Green: Normal driving
  - Yellow: Avoiding obstacle
  - Red: STOP (too close)

### 3. **Keyboard Controls**

| Key | Toggle |
|-----|--------|
| `1` | Objects ON/OFF |
| `2` | Lanes ON/OFF |
| `3` | Signs ON/OFF |
| `4` | Depth ON/OFF |
| `5` | Drowsiness ON/OFF |
| `6` | Hazards ON/OFF |
| `7` | BEV ON/OFF |
| `P` | Pause/Resume |
| `S` | Screenshot |
| `Q` | Quit |

---

## 🎯 How Lane Avoidance Works

### Distance-Based Shifts:

```
Object Distance    Shift Amount    Decision
───────────────────────────────────────────────
< 2.5m            Full shift       STOP (Red)
2.5-5m            120px max        AVOID NEAR (Orange)
5-10m             80px max         AVOID MID (Yellow)
10-15m            40px max         AVOID FAR (Light Yellow)
> 15m             Minimal          FORWARD (Green)
```

### Perspective Scaling:

```
Bottom of frame (near): 100% of shift applied
Middle of lane view:    60% of shift applied
Top of lane view (far): 20% of shift applied

Result: Lanes visibly curve around obstacles!
```

---

## 🚀 Quick Start

```bash
# Video mode
python main_intelligent.py video your_video.mp4

# Webcam mode
python main_intelligent.py webcam

# With custom model
python main_intelligent.py video demo.mp4 --model yolov8m.pt
```

---

## 📊 What You'll See

### Normal Driving:
```
┌────────────────────────────┐
│                            │
│   [Clear Road]             │
│                            │
│   ┌──────────┐             │
│   │ GREEN    │ ← Lanes    │
│   │ PATH     │   (normal)  │
│   └──────────┘             │
│                            │
│ LANE: 3.5m                 │
│ OFFSET: +0.20m             │
│                            │
│ INTELLIGENT ADAS           │
│ Objects[ON]: 2             │
│ DECISION: FORWARD          │
└────────────────────────────┘
```

### Obstacle Detected (Near):
```
┌────────────────────────────┐
│      [Car] 🚗              │
│                            │
│    ┌──────────┐            │
│    │ YELLOW   │ ← Shifted │
│    │ PATH →   │   RIGHT   │
│    └──────────┘            │
│                            │
│ LANE: 3.5m                 │
│ OFFSET: +0.45m             │
│                            │
│ INTELLIGENT ADAS           │
│ Objects[ON]: 3             │
│ DECISION: RIGHT (car)      │
└────────────────────────────┘
```

### Critical (STOP):
```
┌────────────────────────────┐
│   [Very Close!] 🚗         │
│                            │
│    ┌──────────┐            │
│    │ RED      │ ← STOP    │
│    │ PATH     │   (crit)  │
│    └──────────┘            │
│                            │
│ LANE: 3.5m                 │
│ OFFSET: +0.60m             │
│                            │
│ INTELLIGENT ADAS           │
│ Objects[ON]: 3             │
│ DECISION: STOP             │
└────────────────────────────┘
```

---

## 🎨 Key Features

### 1. **HIGHLY VISIBLE Lane Shifts**
- Up to 120px movement when object near
- Exponential perspective (more at bottom)
- Smooth 5-frame averaging
- Very obvious and visible!

### 2. **Short Lane View**
- Only 30% of frame height (bottom)
- Focused on immediate path
- Realistic ADAS view
- Not cluttered

### 3. **All Models Together**
- YOLO detects objects
- Lanes avoid obstacles
- Signs detected
- Depth estimated
- Drowsiness monitored
- Hazards warned

### 4. **Clean UI**
- No background boxes
- Text directly on video
- Color-coded decisions
- Minimal, Tesla-style

---

## 🔧 Configuration

### Adjust Lane Shift Aggressiveness:

In `IntelligentLaneAvoidance.__init__()`:
```python
self.max_shift_near = 120   # Change this (higher = more visible)
self.max_shift_mid = 80     # Medium distance
self.max_shift_far = 40     # Far distance
```

### Adjust Lane View Height:

```python
self.lane_display_height = 0.30  # Change this
# 0.25 = even shorter (25%)
# 0.40 = taller (40%)
```

### Adjust Response Speed:

```python
self.left_shift_buffer = deque(maxlen=5)  # Change this
# 3 = very responsive (less smooth)
# 10 = very smooth (slower response)
```

---

## 🎯 Example Scenarios

### Scenario 1: Car on Left (8m away)
```
Detection: Car at x=200, lane_center=320, distance=8m
Decision: AVOID MID - LEFT (car)
Shift: -75px left, -50px right
Color: Yellow
Lane: Visibly shifts left!
```

### Scenario 2: Truck in Center (4m away)
```
Detection: Truck blocking path, distance=4m
Decision: AVOID NEAR - RIGHT (truck)
Shift: +110px right, +80px left
Color: Orange
Lane: Dramatically shifts right!
```

### Scenario 3: Pedestrian Very Close (1.8m)
```
Detection: Person in path, distance=1.8m
Decision: STOP
Shift: 0px (critical - no time)
Color: Red
Lane: Turns red, alerts driver!
```

---

## ✅ Summary

**This system:**
- ✅ Includes ALL models (YOLO, Lanes, Signs, Depth, Drowsy, Hazards)
- ✅ Lanes MOVE visibly around obstacles (up to 120px)
- ✅ Short lane view (30% bottom only)
- ✅ Responsive and smooth (5-frame buffer)
- ✅ Color-coded (Green/Yellow/Red)
- ✅ Clean, minimal UI
- ✅ Keyboard toggles for everything

**Run it:**
```bash
python main_intelligent.py video your_video.mp4
```

**Watch the lanes dynamically curve around obstacles!** 🚗✨
