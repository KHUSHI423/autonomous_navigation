# 🤖 EdgeDrive3D Telegram Bot - Complete Command Reference

> **Your Ultimate Guide to Controlling the Robot Car via Telegram**

---

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [Core Commands](#-core-commands)
3. [Movement Control](#-movement-control)
4. [Video & Vision](#-video--vision)
5. [GPS & Navigation](#-gps--navigation)
6. [Mission Planning](#-mission-planning)
7. [Telemetry & Analytics](#-telemetry--analytics)
8. [Emergency & Safety](#-emergency--safety)
9. [Voice Commands](#-voice-commands)
10. [Text Shortcuts](#-text-shortcuts)
11. [Web App Interface](#-web-app-interface)
12. [Advanced Features](#-advanced-features)
13. [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### Step 1: Setup (One-Time)

```bash
# Navigate to bot directory
cd hardware_setup/telegram_bot

# Set environment variables (Windows)
set TELEGRAM_BOT_TOKEN=your_token_here
set TELEGRAM_ADMIN_IDS=your_user_id

# Set environment variables (Linux/Mac)
export TELEGRAM_BOT_TOKEN=your_token_here
export TELEGRAM_ADMIN_IDS=your_user_id

# Install dependencies
pip install -r requirements.txt

# Launch the bot
python telegram_bot_main.py
```

### Step 2: Open Telegram

1. Search for your bot by name
2. Press **START** or send `/start`
3. You'll see the main menu with buttons

### Step 3: First Commands

```
/start          → Welcome message & main menu
/help           → Tutorials & guides
/control        → Open joystick control panel
/status         → Check system status
```

---

## 📋 Core Commands

### `/start` - Initialize Bot

**Purpose:** Start the bot and see the welcome menu

**Response:**
```
🤖 Welcome to EdgeDrive3D Robot Car!

Hello [Your Name]! 👋

I'm your autonomous robot car assistant...

[Control] [Video]
[GPS] [Auto]
[Telemetry] [Mission]
[Emergency] [Help]
```

**Usage:**
- Send when you first open the bot
- Resets your session state
- Shows main menu buttons

---

### `/help` - Help & Tutorials

**Purpose:** Access comprehensive help guides

**Response:**
```
📚 EdgeDrive3D Robot Car - Help Guide

━━━━━━━━━━━━━━━━━━━━━━
🎮 CONTROL MODES
━━━━━━━━━━━━━━━━━━━━━━

Manual Control:
• Use inline buttons for joystick
• Hold button for continuous movement
• Release to stop

Autonomous Mode:
• AI makes all driving decisions
• Obstacle avoidance enabled
• Voice commands supported

[🎮 Open Controls] [🤖 Try Autonomous]
```

**Sections Covered:**
- Control modes explanation
- Voice command list
- Mission planning steps
- Emergency features
- Telemetry overview
- Tips & best practices

---

### `/status` - System Status

**Purpose:** Quick overview of all systems

**Response:**
```
ℹ️ EdgeDrive3D System Status

🤖 Bot: ✅ Online
🎮 Controller: ✅ Connected
📹 Video: ✅ Active
🗺️ GPS: ✅ Active
🧠 Decision Engine: ✅ Enabled

👥 Active Users: 2
🎯 Waypoints: 5
🚨 Emergency: Normal

Uptime: Since 14:32:15
```

**When to Use:**
- Before starting operations
- When something seems wrong
- To check battery/connection

---

## 🎮 Movement Control

### `/control` - Control Panel

**Purpose:** Open the joystick control interface

**Response:**
```
🎮 Control Panel

Mode: MANUAL
Throttle: 0
Steering: 0.00

[⬆️ Forward]
[⬅️ Left] [⏹️ Stop] [➡️ Right]
[⬇️ Backward]

[🐢 Slow] [🐇 Fast]
[🔄 Toggle Mode]
```

**Button Actions:**

| Button | Action | Throttle | Steering |
|--------|--------|----------|----------|
| ⬆️ Forward | Move forward | 180 | 0.0 |
| ⬇️ Backward | Move backward | -180 | 0.0 |
| ⬅️ Left | Turn left | 150 | -0.5 |
| ➡️ Right | Turn right | 150 | +0.5 |
| ⏹️ Stop | Emergency stop | 0 | 0.0 |
| 🐢 Slow | Set slow speed | 100 | Current |
| 🐇 Fast | Set fast speed | 200 | Current |
| 🔄 Toggle Mode | Switch AUTO/MANUAL | - | - |

**Pro Tips:**
- **Tap** buttons for quick movements
- **Hold** (in web app) for continuous motion
- Use **Stop** for immediate halt
- Toggle mode to switch between manual/AI control

---

### Movement Button Callbacks

When you press control buttons, you'll see confirmation messages:

```
⬆️ Moving Forward
Throttle: 180

⬅️ Turning Left
Steering: -0.50

⏹️ STOPPED

🐢 Speed: SLOW (100)

🐇 Speed: FAST (200)

🔄 Mode: AUTO
```

---

## 📹 Video & Vision

### `/video` - Live Video Stream

**Purpose:** Get the latest camera frame from the robot

**Response:**
```
📹 Fetching latest frame...

[Photo: Live camera feed with detections]

📹 Live Video
📸 Frame #1247
🕐 Timestamp: 14:45:32
```

**Features:**
- Real-time camera feed
- Object detection overlays (if enabled)
- Frame counter for debugging
- Timestamp for synchronization

**Refresh:** Send `/video` again for updated frame

---

### `/analyze` - AI Image Analysis

**Purpose:** Get detailed AI analysis of current view

**Response:**
```
🔍 Vision Analysis Report

📊 Objects Detected: 3

🚗 Car: 1 detected (avg: 94%)
👤 Person: 2 detected (avg: 87%)

📐 Image Size: 640x480
🧠 Processing Time: 45ms
```

**When to Use:**
- Verify object detection is working
- Get detailed detection statistics
- Debug vision system

---

### Send Image to Bot

**Purpose:** Analyze any image you send

**How To:**
1. Take a photo in Telegram
2. Send it to the bot
3. Bot analyzes with AI

**Response:**
```
🔍 Vision Analysis Report

📊 Objects Detected: 2

🚗 Car: 1 detected (95%)
🌲 Tree: 1 detected (89%)

📐 Image Size: 1280x720
```

---

## 🗺️ GPS & Navigation

### `/gps` - Share Location

**Purpose:** Get current GPS coordinates and live location

**Response:**
```
[Live Location Pin - Updates for 60 seconds]

🗺️ GPS Position

📍 Lat: 12.971590
📍 Lon: 77.594560
📏 Alt: 920.5m
🚀 Speed: 2.3 m/s
🧭 Heading: 45.0°
📡 Satellites: 12
📊 Accuracy: ±3.2m
```

**Features:**
- **Live Location:** Updates in real-time for 60 seconds
- **Heading:** Direction robot is facing
- **Speed:** Current ground speed
- **Accuracy:** GPS precision estimate

**Pro Tips:**
- Tap the location pin to open in maps
- Live location auto-expires after 60s
- Send `/gps` again to refresh

---

### `/location` - Detailed Position

**Purpose:** Get detailed GPS data without live pin

**Response:**
```
📍 Position Report
━━━━━━━━━━━━━━━━━━━━━━

Coordinates:
  Latitude:  12.971590° N
  Longitude: 77.594560° E
  Altitude:  920.5m

Movement:
  Speed:     2.3 m/s (8.3 km/h)
  Heading:   45.0° (NE)
  Distance:  125.3m from home

GPS Quality:
  Satellites: 12
  HDOP:      0.9
  Fix:       3D

🟢 Excellent signal quality
```

---

## 🎯 Mission Planning

### `/mission` - Mission Interface

**Purpose:** Access mission planning and management

**Response:**
```
🎯 Mission Planning

Current Waypoints: 3

How to create a mission:

1️⃣ Send a location message
2️⃣ I'll add it as a waypoint
3️⃣ Send more locations
4️⃣ Use buttons below to manage

[▶️ Start Mission]
[🗑️ Clear All] [📋 List Waypoints]
[🏠 Return to Home]
```

**Creating a Mission:**

**Step 1:** Send location pins
- Tap attachment 📎 in Telegram
- Select "Location"
- Share your current location or pick on map

**Step 2:** Bot confirms each waypoint
```
✅ Waypoint added!
📍 12.971590, 77.594560
Total waypoints: 1

Send more locations or use /mission_start
```

**Step 3:** Review waypoints
```
📋 Mission Waypoints

1. 🏠 Start Position
   📍 12.971590, 77.594560
   ⏰ 14:30:15

2. 🎯 Waypoint 2
   📍 12.972100, 77.595200
   ⏰ 14:31:22

3. 🎯 Waypoint 3
   📍 12.972800, 77.594800
   ⏰ 14:32:45

━━━━━━━━━━
Total: 3 waypoints
Estimated distance: 250m
Estimated time: ~2 minutes
```

**Step 4:** Start mission
```
🚀 Mission Started!

📍 Current: Waypoint 1/3
🎯 Target: Waypoint 2
📏 Distance: 85.3m
🧭 Bearing: 45° NE

Status: En route...
```

---

### Mission Button Actions

| Button | Action | Response |
|--------|--------|----------|
| ▶️ Start Mission | Begin autonomous execution | "🚀 Mission started!" |
| 🗑️ Clear All | Delete all waypoints | "🗑️ Mission cleared" |
| 📋 List Waypoints | Show waypoint list | Detailed list |
| 🏠 Return to Home | Autonomous RTH | "🏠 Returning home..." |

---

### `/mission_start` - Begin Mission

**Purpose:** Start executing the planned mission

**Prerequisites:**
- At least 1 waypoint must be set
- Robot must be in AUTO mode

**Response:**
```
🚀 Mission Started!

━━━━━━━━━━━━━━━━━━━━━━
Mission ID: #20260323-001
Waypoints: 3
Status: ACTIVE
━━━━━━━━━━━━━━━━━━━━━━

📍 Current: Waypoint 1/3
🎯 Target: Waypoint 2
📏 Distance: 85.3m
🧭 Bearing: 45° NE

Progress: [=====>    ] 33%

I'll notify you at each waypoint!
```

**During Mission:**
```
✅ Waypoint 2 Reached!

📍 12.972100, 77.595200
⏰ 14:35:22

Next: Waypoint 3 (120m ahead)

Progress: [=========>] 66%
```

---

### `/mission_abort` - Abort Mission

**Purpose:** Stop mission execution immediately

**Response:**
```
⚠️ Mission Aborted

Mission ID: #20260323-001
Waypoints completed: 2/3
Reason: User requested

Robot is stopping...
Use /control to resume manual control
```

---

### `/mission_clear` - Clear Waypoints

**Purpose:** Delete all planned waypoints

**Response:**
```
🗑️ Mission Cleared

All waypoints have been removed.
Mission status: INACTIVE

Send new locations to create a mission.
```

---

### `/mission_rth` or `/return_home` - Return to Home

**Purpose:** Autonomous return to starting position

**Response:**
```
🏠 Return to Home Activated

📍 Home: 12.971590, 77.594560
📍 Current: 12.972800, 77.594800
📏 Distance: 150.2m
🧭 Bearing: 225° SW

Navigating home...
ETA: ~1 minute
```

---

## 📊 Telemetry & Analytics

### `/telemetry` - System Diagnostics

**Purpose:** Complete system health check

**Response:**
```
📊 System Telemetry

━━━━━━━━━━━━━━━━━━━━━━
🚗 Vehicle Status
━━━━━━━━━━━━━━━━━━━━━━
Throttle: 150/200
Steering: +0.50
Mode: AUTO

━━━━━━━━━━━━━━━━━━━━━━
🗺️ GPS Data
━━━━━━━━━━━━━━━━━━━━━━
Position: 12.9716° N, 77.5946° E
Speed: 2.3 m/s
Heading: 45.0°
Satellites: 12
Accuracy: ±3.2m

━━━━━━━━━━━━━━━━━━━━━━
🧠 Decision Engine
━━━━━━━━━━━━━━━━━━━━━━
Action: CRUISE
Reason: Clear path ahead
Confidence: 95%

━━━━━━━━━━━━━━━━━━━━━━
📹 Video Stream
━━━━━━━━━━━━━━━━━━━━━━
Frames: 1247
Status: ✅ Active

━━━━━━━━━━━━━━━━━━━━━━
🔋 System
━━━━━━━━━━━━━━━━━━━━━━
Uptime: 00:23:45
Emergency: ✅ Normal
```

**Sections Explained:**

| Section | Information |
|---------|-------------|
| Vehicle Status | Throttle, steering, mode |
| GPS Data | Position, speed, heading, satellites |
| Decision Engine | Current AI decision & reasoning |
| Video Stream | Frame count, connection status |
| System | Uptime, emergency state |

---

### `/analytics` - Session Statistics

**Purpose:** View session analytics and charts

**Response:**
```
📊 Session Analytics Report
━━━━━━━━━━━━━━━━━━━━━━

⏱️ Duration: 00:23:45
🎮 Commands: 127
🚀 Max Speed: 3.2 m/s
📍 Distance: 450.5m

🔍 Object Detections:
• Car: 15
• Person: 8
• Bicycle: 3

━━━━━━━━━━━━━━━━━━━━━━
🤖 EdgeDrive3D Analytics
Session ID: 7842
```

**Charts Generated:**

The bot can generate visual charts:

1. **Speed Chart** - Speed over time
2. **Throttle Chart** - Throttle distribution
3. **GPS Track** - Path visualization
4. **Decision Pie** - Decision distribution

**To request charts:**
```
/chart speed      → Speed over time
/chart throttle   → Throttle distribution
/chart track      → GPS track map
/chart decisions  → Decision pie chart
```

---

### `/alerts` - Alert History

**Purpose:** View system alert history

**Response:**
```
🚨 Alert History

🔋 Low Battery: 2
🚀 High Speed: 1
📡 GPS Signal Weak: 0
🛑 Emergency Stop: 0

━━━━━━━━━━
Total Alerts: 3
Last Alert: 5 minutes ago
```

---

## 🚨 Emergency & Safety

### `/emergency` - Emergency Controls

**Purpose:** Access emergency control panel

**Response:**
```
🚨 EMERGENCY CONTROLS

⚠️ Use these in emergency situations only!

Available Actions:
• 🛑 Emergency Stop - Immediate halt
• 🏠 Return to Home - Autonomous RTH
• 📍 Send SOS Alert - Notify all admins
• 🔓 Reset Emergency - Clear emergency state

Safety Features:
✅ Geofencing active
✅ Obstacle avoidance
✅ Low battery detection
✅ Signal loss protection

[🛑 EMERGENCY STOP]
[🏠 Return Home] [📍 SOS Alert]
[🔓 Reset]
```

**⚠️ WARNING:** Only use these in genuine emergencies!

---

### `/emergency_stop` - Immediate Halt

**Purpose:** Stop all robot movement instantly

**Response:**
```
🛑 EMERGENCY STOP EXECUTED!

All motors disabled.
Notifying all admins...

[Alert sent to 3 administrators]
```

**What Happens:**
1. Motors immediately cut power
2. All admins receive notification
3. Emergency state activated
4. Robot won't move until reset

**To Reset:**
```
/emergency_reset
```

---

### `/emergency_reset` - Clear Emergency

**Purpose:** Reset emergency state and resume control

**Response:**
```
🔓 Emergency State Cleared

Emergency stop has been reset.
You can now resume control.

Use /control to access joystick
```

---

### `/sos` - Broadcast SOS Alert

**Purpose:** Send emergency alert to all administrators

**Response:**
```
🚨 SOS ALERT SENT!

Emergency notification broadcast to:
• Admin 1 ✅
• Admin 2 ✅
• Admin 3 ✅

Current location shared.
Help is on the way!
```

**What Admins Receive:**
```
🚨 SOS EMERGENCY ALERT!

Robot: EdgeDrive3D #001
Time: 2026-03-23 14:45:32
Location: 12.971590, 77.594560

User [Your Name] has triggered
an emergency SOS alert!

[View Location] [Call User]
```

---

## 🗣️ Voice Commands

### How to Use Voice Commands

1. **Tap microphone** 🎤 in Telegram chat
2. **Speak clearly** one of the commands below
3. **Bot executes** the corresponding action

---

### Voice Command List

| Say This | Bot Does This |
|----------|---------------|
| "Forward" / "Go" | Moves forward |
| "Backward" / "Reverse" | Moves backward |
| "Left" / "Turn left" | Turns left |
| "Right" / "Turn right" | Turns right |
| "Stop" / "Emergency stop" | Emergency halt |
| "Take photo" | Captures image |
| "Show location" | Shares GPS coordinates |
| "Status report" | Sends telemetry |
| "Start mission" | Begins mission execution |
| "Return home" | Returns to base |
| "Enable autonomous mode" | Switches to AI control |
| "Disable autonomous mode" | Switches to manual |

---

### Voice Command Responses

```
🎤 Voice message received!

Executing: FORWARD

⬆️ Moving Forward
Throttle: 180
```

```
🎤 Voice message received!

Executing: EMERGENCY STOP

🛑 EMERGENCY STOP EXECUTED!
```

**Note:** Speech recognition requires additional setup. See configuration section.

---

## ⌨️ Text Shortcuts

### Quick Text Commands

For faster control, use these single-letter shortcuts:

| Send This | Does This |
|-----------|-----------|
| `/f` or `f` | Forward |
| `/b` or `b` | Backward |
| `/l` or `l` | Left |
| `/r` or `r` | Right |
| `/s` or `s` | Stop |
| `/a` | Toggle autonomous mode |
| `/v` | Send video frame |
| `/g` | Send GPS location |
| `/t` | Send telemetry |

---

### Example Usage

```
You: /f
Bot: ⬆️ Moving Forward

You: /r
Bot: ➡️ Turning Right

You: /s
Bot: ⏹️ STOPPED

You: /a
Bot: 🔄 Mode: AUTO
```

---

## 🌐 Web App Interface

### Accessing the Web App

**Method 1:** Menu Button
- Open bot in Telegram
- Tap "Launch Web App" button (if configured)

**Method 2:** Inline Button
- Send `/control`
- Tap "Open Web App" button

**Method 3:** Direct Link
- Visit: `https://t.me/YOUR_BOT_NAME/app`

---

### Web App Features

```
┌─────────────────────────────────────┐
│   🤖 EdgeDrive3D                    │
│   Advanced Robot Control Interface  │
├─────────────────────────────────────┤
│  Mode: MANUAL  🟢  12.4V           │
├─────────────────────────────────────┤
│         [⬆️]                        │
│    [⬅️] [⏹️] [➡️]                  │
│         [⬇️]                        │
├─────────────────────────────────────┤
│  [👤 Manual] [🤖 Auto]             │
├─────────────────────────────────────┤
│  Throttle: 150    Steering: +0.50  │
│  Speed: 2.3 m/s   Heading: 45°     │
├─────────────────────────────────────┤
│  [📹 Video] [🗺️ GPS]              │
│  [🎯 Mission] [🚨 SOS]             │
├─────────────────────────────────────┤
│  [14:32:15] System initialized...  │
│  [14:32:20] Moving forward         │
│  [14:32:25] Stopped                │
└─────────────────────────────────────┘
```

---

### Web App Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `W` or `↑` | Forward |
| `S` or `↓` | Backward |
| `A` or `←` | Left |
| `D` or `→` | Right |
| `Space` | Stop |
| `M` | Toggle mode |
| `V` | Toggle video |
| `G` | Show GPS |

---

### Web App Advantages

✅ **Real-time telemetry** - Live updates without refreshing
✅ **Smooth controls** - Hold buttons for continuous movement
✅ **Visual feedback** - See throttle/steering changes instantly
✅ **Log console** - Track all commands and responses
✅ **Modern UI** - Beautiful gradient design
✅ **Mobile optimized** - Works great on phones
✅ **Desktop support** - Keyboard controls for PC

---

## 🔧 Advanced Features

### `/autonomous` - Toggle AI Mode

**Purpose:** Switch between manual and autonomous control

**Response (Activating):**
```
🤖 Autonomous Mode ACTIVATED

✅ AI is now in control
✅ Obstacle avoidance enabled
✅ Decision engine active
✅ Safety systems monitoring

The robot will:
• Navigate autonomously
• Avoid obstacles
• Make intelligent decisions

Send /control to switch back to manual
```

**Response (Deactivating):**
```
👤 Manual Mode ACTIVATED

✅ You are now in control
✅ Use joystick buttons
✅ Voice commands enabled

Send /autonomous to enable AI
```

---

### `/decision` - Get AI Decision

**Purpose:** See current AI decision with explanation

**Response:**
```
🧠 Decision Engine Report

Action: CRUISE
Speed: 150/200

🟢 Maintaining speed - Safe distance from obstacles

📝 Reason: Clear path ahead, no obstacles within 15m
📊 Confidence: 95%
⏱️ Decision Time: 5ms

Objects Detected: 2
• Car: 12.5m ahead
• Person: 8.3m to left
```

---

### `/config` - View Configuration

**Purpose:** Check current bot configuration

**Response:**
```
⚙️ Bot Configuration

━━━━━━━━━━━━━━━━━━━━━━
Network:
  ESP32 IP:    10.17.122.207
  Command Port: 9000
  Video Port:  5000

Modes:
  Autonomous:  ✅ Enabled
  Geofencing:  ✅ Enabled
  Radius:      100m

Intervals:
  Telemetry:   5 seconds
  Video:       2 seconds
  GPS:         3 seconds

Users:
  Admins:      2
  Operators:   1
  Viewers:     0
```

---

## 🐛 Troubleshooting

### Bot Not Responding

**Symptoms:**
- Bot doesn't reply to commands
- Buttons don't work

**Solutions:**
```
1. Check if bot is running:
   /status

2. Verify your user ID is authorized:
   Contact admin to add you

3. Restart the bot:
   Stop bot process (Ctrl+C)
   python telegram_bot_main.py

4. Check internet connection
```

---

### No Video Signal

**Symptoms:**
- `/video` returns "No video signal"
- Black/missing camera feed

**Solutions:**
```
1. Verify camera is connected

2. Check video stream is running:
   /status → Video should show "✅ Active"

3. Verify UDP port is open:
   Port 5000 must be accessible

4. Restart video stream on robot
```

---

### GPS Not Working

**Symptoms:**
- `/gps` shows "No signal"
- Location not updating

**Solutions:**
```
1. Check GPS hardware connection

2. Wait for GPS fix (30-60 seconds)

3. Move to open area with sky view

4. Check satellite count:
   Need 4+ satellites for fix

5. Try simulated mode for testing:
   Set GPS_SIMULATE=true
```

---

### Robot Not Moving

**Symptoms:**
- Commands sent but no movement
- Throttle shows 0

**Solutions:**
```
1. Check ESP32 connection:
   /status → Controller should be "✅ Connected"

2. Verify ESP32 IP is correct:
   Check environment variable

3. Test web interface:
   Visit http://ESP32_IP:8080

4. Check battery level:
   Low battery prevents movement

5. Ensure not in emergency state:
   /emergency_reset
```

---

### Connection Lost

**Symptoms:**
- "Controller: ⚠️ Disconnected"
- Commands not executing

**Solutions:**
```
1. Check robot is powered on

2. Verify WiFi connection:
   Robot and laptop on same network

3. Ping ESP32:
   ping 10.17.122.207

4. Restart ESP32

5. Update ESP32 IP if changed:
   set ESP32_IP=new_ip_address
```

---

## 📞 Quick Reference Card

### Essential Commands

```
┌─────────────────────────────────────┐
│ START HERE                          │
├─────────────────────────────────────┤
│ /start       Initialize bot         │
│ /help        Get help               │
│ /control     Open joystick          │
│ /status      Check systems          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ MOVEMENT                            │
├─────────────────────────────────────┤
│ /f or f      Forward                │
│ /b or b      Backward               │
│ /l or l      Left                   │
│ /r or r      Right                  │
│ /s or s      Stop                   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ MONITORING                          │
├─────────────────────────────────────┤
│ /video       Live camera            │
│ /gps         Location               │
│ /telemetry   Diagnostics            │
│ /analytics   Statistics             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ EMERGENCY                           │
├─────────────────────────────────────┤
│ /emergency   Emergency panel        │
│ /emergency_stop  IMMEDIATE HALT     │
│ /sos         Alert admins           │
│ /emergency_reset  Reset             │
└─────────────────────────────────────┘
```

---

## 🎓 Pro Tips

### Efficiency Tips

1. **Use text shortcuts** for quick control (`/f`, `/b`, `/l`, `/r`, `/s`)
2. **Pin important messages** for quick access
3. **Use voice commands** when hands are busy
4. **Enable notifications** for alerts
5. **Check `/status`** before each session

### Safety Tips

1. **Always know** where `/emergency_stop` is
2. **Test in open area** before complex missions
3. **Monitor battery** during operations
4. **Keep GPS home** updated
5. **Set geofence** appropriately

### Mission Tips

1. **Scout locations** before sending robot
2. **Add extra waypoints** for flexibility
3. **Monitor progress** via telemetry
4. **Be ready to abort** if needed
5. **Review analytics** after missions

---

**Happy Controlling! 🚀**

*For more help, contact your administrator or check the README.md*
