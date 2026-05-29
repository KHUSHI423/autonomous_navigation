# EdgeDrive3D - Quick Start Guide

## 🚀 Installation (5 minutes)

### Step 1: Install Dependencies

```bash
cd combined_proj_folder

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download YOLO Model

The YOLO model will be auto-downloaded on first run. For manual download:

```bash
# Download yolov8m.pt (recommended)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -O yolov8m.pt

# Or use nano for faster inference
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt
```

### Step 3: Test Installation

```bash
# Run test
python main.py image test_image.jpg -o output/ --show
```

---

## 🎮 Quick Start Modes

### Mode 1: Image Processing

```bash
python main.py image path/to/image.jpg -o output/ --show
```

### Mode 2: Video Processing

```bash
python main.py video road_video.mp4 --save --show
```

### Mode 3: Webcam

```bash
python main.py webcam
```

### Mode 4: Raspberry Pi Stream

**On Raspberry Pi:**
```bash
python hardware/pi_sender.py 192.168.1.100 -p 5000
```

**On Laptop:**
```bash
python main.py pi-stream --port 5000 --model yolov8n.pt
```

### Mode 5: Dashboard Only

```bash
python main.py dashboard
# Open http://localhost:8501
```

### Mode 6: Full System with Auto Control

```bash
# Terminal 1: Start laptop receiver with ESP32 control
python main.py pi-stream --port 5000 --auto --esp32-ip 192.168.4.1

# Terminal 2: On Raspberry Pi
python hardware/pi_sender.py <LAPTOP_IP> -p 5000
```

---

## 📊 Dashboard Features

Access the dashboard at: **http://localhost:8501**

### Tabs:
1. **Dashboard** - Main control panel with metrics
2. **3D View** - Interactive point cloud visualization
3. **BEV Map** - Bird's eye view with object tracking
4. **Analytics** - Performance charts and statistics

### Controls:
- **Start/Stop** - Toggle system
- **Confidence** - Adjust detection threshold
- **Model** - Switch YOLO models
- **Auto Control** - Enable autonomous ESP32 control

---

## 🔌 Hardware Setup

### ESP32 Motor Controller

1. **Flash the Arduino code:**
   - Open `hardware/hardware.ino` in Arduino IDE
   - Select board: DOIT ESP32 DEVKIT V1
   - Upload

2. **Connect motors:**
   ```
   Motor A: AIN1=26, AIN2=27, PWMA=25
   Motor B: BIN1=14, BIN2=13, PWMB=33
   Standby: STBY=32
   ```

3. **Power on ESP32**

4. **Connect to WiFi:**
   - SSID: `EdgeDrive3D_Robot`
   - Password: `edgedrive123`

5. **Test control:**
   - Open browser: http://192.168.4.1
   - Use web interface to test motors

### Raspberry Pi Camera Stream

1. **Connect camera to Pi 3B+**

2. **Install dependencies on Pi:**
   ```bash
   pip install opencv-python numpy
   ```

3. **Run sender:**
   ```bash
   python hardware/pi_sender.py 192.168.1.100 -p 5000 -W 640 -H 480
   ```

---

## ⚙️ Configuration

Edit `config/settings.yaml` to customize:

```yaml
perception:
  yolo_model: yolov8m.pt      # Model size: n, s, m, l, x
  confidence_threshold: 0.4    # Detection confidence
  max_depth_meters: 50.0       # Maximum depth range

hardware:
  udp_port: 5000               # UDP port for streaming
  esp32_ip: 192.168.4.1        # ESP32 IP address
  auto_control: false          # Enable autonomous control
```

---

## 🎯 Performance Tips

### For Real-time (>30 FPS):
- Use `yolov8n.pt` (nano model)
- Reduce resolution: 640x480
- Enable GPU (CUDA)

### For Maximum Accuracy:
- Use `yolov8l.pt` (large model)
- Increase resolution: 1280x720
- Set confidence to 0.5+

### For Hardware Control:
- Enable auto_control mode
- Adjust safety distances in config
- Test in open area first

---

## 🐛 Troubleshooting

### "Could not load YOLO model"
```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics==8.0.100
```

### "CUDA not available"
```bash
# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "UDP connection failed"
- Check firewall settings
- Ensure same network
- Verify IP addresses

### "ESP32 not responding"
- Check power supply (5V 2A)
- Reset ESP32
- Re-flash firmware

---

## 📞 Support

For issues:
1. Check logs in `output/logs/`
2. Run with `--debug` flag
3. Create GitHub issue with logs

---

## 🎓 Next Steps

1. **Test basic functionality** - Start with image processing
2. **Configure hardware** - Set up ESP32 and Pi
3. **Calibrate camera** - Use camera calibration tool
4. **Tune parameters** - Adjust for your environment
5. **Deploy** - Run full autonomous system

---

**Happy Building! 🚗**
