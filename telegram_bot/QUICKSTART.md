# 🚀 EdgeDrive3D Telegram Bot - Quick Start Guide

> **Get your robot car running on Telegram in 5 minutes!**

---

## ⚡ 5-Minute Setup

### Step 1: Create Your Bot (2 minutes)

1. **Open Telegram** and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g., "My Robot Car")
4. Choose a username (e.g., "edge_drive_bot")
5. **COPY THE TOKEN** (save it for Step 3)

```
Example:
Use this token to access the HTTP API:
1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

---

### Step 2: Get Your User ID (30 seconds)

1. Search for **@userinfobot**
2. Press **Start**
3. **COPY YOUR USER ID** (it's a number like 123456789)

---

### Step 3: Configure the Bot (1 minute)

**Windows:**
```batch
cd hardware_setup\telegram_bot

REM Create .env file
copy .env.example .env

REM Edit .env and set:
REM TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
REM TELEGRAM_ADMIN_IDS=123456789
```

**Edit .env file:**
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_ADMIN_IDS=123456789
ESP32_IP=10.17.122.207
```

---

### Step 4: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

---

### Step 5: Launch! (30 seconds)

**Option A: Use Launcher (Easiest)**
```batch
START_BOT.bat
```

**Option B: Run Directly**
```bash
python telegram_bot_main.py
```

---

### Step 6: Open Telegram

1. Find your bot by the username you created
2. Press **START**
3. You'll see the control panel!

```
🤖 Welcome to EdgeDrive3D Robot Car!

[🎮 Control] [📹 Video]
[🗺️ GPS] [🤖 Auto]
[📊 Telemetry] [🎯 Mission]
[🚨 Emergency] [❓ Help]
```

---

## 🎮 First Control Test

### Test Manual Control

1. Send `/control` to bot
2. Tap **⬆️ Forward** button
3. Robot should move forward!
4. Tap **⏹️ Stop** to halt

### Test Video Stream

1. Send `/video` to bot
2. You'll receive a photo from the camera
3. Shows live view from robot!

### Test GPS

1. Send `/gps` to bot
2. Bot shares its location
3. Tap the location pin to view on map

---

## ✅ Success Checklist

- [ ] Bot created with @BotFather
- [ ] Token saved to .env file
- [ ] User ID added to TELEGRAM_ADMIN_IDS
- [ ] Dependencies installed
- [ ] Bot running (no errors in console)
- [ ] Bot responds to `/start` command
- [ ] Control panel buttons work
- [ ] Video stream shows images
- [ ] GPS location appears

---

## 🐛 Quick Troubleshooting

### "Bot doesn't respond"
- Check bot is running (console should show "Starting Telegram Bot...")
- Verify token is correct in .env
- Make sure you're using the correct bot username

### "Unauthorized access"
- Check your user ID in TELEGRAM_ADMIN_IDS
- User ID must be a number (e.g., 123456789)
- Multiple IDs: 123456789,987654321

### "No video signal"
- Verify robot camera is connected
- Check video stream is running on robot
- UDP port 5000 must be accessible

### "Controller disconnected"
- Verify ESP32_IP is correct
- Check robot is powered on
- Ensure laptop and robot on same network

---

## 📱 Essential Commands

Once running, these are the commands you'll use most:

```
/start          → Main menu
/control        → Joystick controls
/video          → Live camera
/gps            → Location
/telemetry      → System status
/emergency      → Emergency controls
/help           → Tutorials
```

---

## 🎯 Next Steps

### Beginner
1. ✅ Test basic movement controls
2. ✅ Try video streaming
3. ✅ Check GPS location
4. ✅ Read `/help` tutorials

### Intermediate
1. Create a mission with location pins
2. Enable autonomous mode
3. Review telemetry charts
4. Set up geofencing

### Advanced
1. Configure voice commands
2. Customize alert thresholds
3. Review analytics dashboard
4. Set up multi-user access

---

## 📞 Need Help?

1. **Check logs:** Look in `logs/bot.log`
2. **Run diagnostics:** Send `/status` to bot
3. **Review docs:** See `README.md` and `telegram_bot_cmnds.md`
4. **Restart bot:** Stop (Ctrl+C) and run again

---

## 🔧 Configuration Quick Reference

### Change Bot Token
Edit `.env`:
```bash
TELEGRAM_BOT_TOKEN=new_token_here
```

### Add More Admins
Edit `.env`:
```bash
TELEGRAM_ADMIN_IDS=123456789,987654321,111222333
```

### Change ESP32 IP
Edit `.env`:
```bash
ESP32_IP=192.168.1.100
```

### Enable GPS Hardware
Edit `.env`:
```bash
GPS_SIMULATE=false
GPS_PORT=COM3
```

---

## 🎓 Learning Resources

| Document | Purpose |
|----------|---------|
| `README.md` | Complete feature overview |
| `telegram_bot_cmnds.md` | All commands reference |
| `.env.example` | Configuration options |
| `requirements.txt` | Python dependencies |

---

## 🏆 You're Ready!

Your robot car is now controllable from anywhere in the world via Telegram!

**Try this:**
1. Walk away from robot
2. Control it remotely via Telegram
3. Watch the video stream
4. Navigate using GPS
5. Create autonomous missions

**Happy controlling! 🚀**

---

*For detailed documentation, see README.md*
*For all commands, see telegram_bot_cmnds.md*
