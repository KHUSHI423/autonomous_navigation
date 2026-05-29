# 🚀 START HERE - EdgeDrive3D Telegram Bot

> **Your Robot Car Control System - Ready in 2 Minutes!**

---

## ⚡ QUICK START (Choose One)

### Option 1: Automated Setup (RECOMMENDED - 2 minutes)

```batch
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\telegram_bot

SETUP_WIZARD.bat
```

This wizard will:
1. Guide you through creating a Telegram bot
2. Help you get your user ID
3. Configure everything automatically
4. Install dependencies
5. Offer to start the bot

**← If you're new, START HERE!**

---

### Option 2: Manual Setup (3 minutes)

**Step 1: Create Telegram Bot**
1. Open Telegram → Search `@BotFather`
2. Send `/newbot`
3. Follow prompts
4. **Copy the token** (e.g., `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

**Step 2: Get Your User ID**
1. Search `@userinfobot`
2. Press Start
3. **Copy your User ID** (e.g., `123456789`)

**Step 3: Configure .env**
Create/edit `hardware_setup/telegram_bot/.env`:
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_ADMIN_IDS=123456789
```

**Step 4: Install & Run**
```batch
pip install -r requirements.txt
python telegram_bot_main.py
```

**Step 5: Open Telegram**
1. Find your bot by username
2. Press **START**
3. Done! 🎉

---

## 📁 Files Overview

### For Setup
| File | Use |
|------|-----|
| **SETUP_WIZARD.bat** | ← Automated setup (BEST) |
| **LAUNCHER.bat** | Launch after setup |
| **.env.example** | Configuration template |

### For Learning
| File | Use |
|------|-----|
| **SETUP.md** | Detailed setup guide |
| **QUICKSTART.md** | 5-minute quickstart |
| **telegram_bot_cmnds.md** | All commands |
| **INDEX.md** | Documentation hub |

### For Hackathon
| File | Use |
|------|-----|
| **HACKATHON_SUMMARY.md** | Judges' guide |
| **FEATURES.md** | Feature showcase |
| **README.md** | Complete documentation |

---

## 🎯 What You Get

### Control Methods (4 Ways!)
- 🎮 **Joystick** - Inline Telegram buttons
- 🎤 **Voice** - "Forward", "Stop", "Left"
- 🌐 **Web App** - Beautiful graphical interface
- ⌨️ **Text** - `/f`, `/b`, `/l`, `/r`, `/s`

### AI Features
- 🧠 Autonomous navigation
- 🔍 Object detection (YOLOv8)
- 📊 Decision explanations
- 📈 Analytics & charts

### GPS & Navigation
- 🗺️ Live location tracking
- 🎯 Waypoint missions
- 🏠 Return-to-home
- 🚧 Geofencing alerts

### Safety Systems
- 🛑 Emergency stop
- 🚨 SOS alerts
- 🔋 Low battery detection
- 📡 Signal loss protection

---

## ✅ Success Checklist

After setup, you should have:

- [ ] Bot created on Telegram
- [ ] Token saved in `.env`
- [ ] User ID configured
- [ ] Dependencies installed
- [ ] Bot running (no errors)
- [ ] Bot responds to `/start`

---

## 🐛 Troubleshooting

### "ImportError: cannot import name 'Update'"
```batch
pip uninstall telegram -y
pip install python-telegram-bot==20.7
```

### "No module named 'telegram'"
```batch
pip install -r requirements.txt
```

### "Bot doesn't respond"
1. Check bot is running (console shows messages)
2. Verify token is correct in `.env`
3. Check your user ID is in `TELEGRAM_ADMIN_IDS`

### "Controller disconnected"
This is normal without hardware. Bot works in simulated mode!

---

## 🎮 First Commands

Once running, try these:

```
/start          → Welcome menu
/help           → Tutorials
/control        → Joystick
/video          → Camera
/gps            → Location
/status         → System status
/autonomous     → AI mode
```

---

## 📞 Next Steps

### Beginner (15 min)
1. ✅ Complete setup with `SETUP_WIZARD.bat`
2. ✅ Test `/control` joystick
3. ✅ Try `/video` and `/gps`
4. ✅ Read `SETUP.md`

### Intermediate (30 min)
1. ✅ Read `telegram_bot_cmnds.md`
2. ✅ Create a GPS mission
3. ✅ Test autonomous mode
4. ✅ Review telemetry

### Advanced (1 hour)
1. ✅ Read `FEATURES.md`
2. ✅ Review source code
3. ✅ Customize settings
4. ✅ Connect hardware

---

## 🏆 Why This Is Special

| Feature | Others | EdgeDrive3D |
|---------|--------|-------------|
| Setup Time | 30+ min | **2 min** |
| Control Methods | 1 | **4** |
| Platform | App required | **Telegram** |
| AI Integration | None | **Full YOLOv8** |
| GPS | Basic | **Missions** |
| Documentation | Basic | **7 guides** |

---

## 📚 Documentation Map

```
START_HERE.md (YOU ARE HERE)
    ↓
SETUP_WIZARD.bat → Automated setup
    ↓
SETUP.md → Detailed guide
    ↓
telegram_bot_cmnds.md → All commands
    ↓
FEATURES.md → Feature showcase
    ↓
HACKATHON_SUMMARY.md → For judges
```

---

## 🎓 Quick Reference

### Essential Commands
```
/start          Initialize bot
/control        Joystick panel
/autonomous     Toggle AI mode
/gps            Share location
/video          Live camera
/emergency      Safety controls
```

### Text Shortcuts
```
/f    Forward
/b    Backward
/l    Left
/r    Right
/s    Stop
```

### Voice Commands
```
"Forward"    → Move forward
"Stop"       → Emergency stop
"Left"       → Turn left
"Right"      → Turn right
```

---

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Bot Configuration
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_ADMIN_IDS=your_user_id

# Robot Settings
ESP32_IP=10.17.122.207
ESP32_COMMAND_PORT=9000

# Features
AUTONOMOUS_ENABLED=true
GEOFENCE_ENABLED=true
GPS_SIMULATE=true
```

---

## 🌟 Features at a Glance

```
┌─────────────────────────────────────────┐
│  EDGEDRIVE3D TELEGRAM BOT               │
├─────────────────────────────────────────┤
│  🎮 4 Control Methods                   │
│  🧠 AI-Powered Autonomy                 │
│  🗺️ GPS Mission Planning               │
│  📹 Live Video Streaming                │
│  📊 Advanced Telemetry                  │
│  🚨 Comprehensive Safety                │
│  👥 Multi-User Support                  │
└─────────────────────────────────────────┘
```

---

## 🚀 Ready to Start?

### Fastest Path:
```batch
SETUP_WIZARD.bat
```

### Manual Path:
1. Create bot with @BotFather
2. Get user ID from @userinfobot
3. Edit `.env` with your details
4. Run `python telegram_bot_main.py`
5. Open Telegram, find your bot, press START

---

## 📞 Need Help?

1. **Setup issues?** → See `SETUP.md`
2. **Command questions?** → See `telegram_bot_cmnds.md`
3. **Want features?** → See `FEATURES.md`
4. **Hackathon demo?** → See `HACKATHON_SUMMARY.md`

---

**Let's get started! Run `SETUP_WIZARD.bat` now!** 🎉

---

*EdgeDrive3D Team - March 2026*
