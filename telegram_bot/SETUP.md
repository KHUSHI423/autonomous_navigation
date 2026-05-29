# ⚡ Setup Guide - EdgeDrive3D Telegram Bot

> **Get your bot running in 3 minutes!**

---

## 🎯 Step-by-Step Setup

### Step 1: Create Telegram Bot (2 minutes)

1. **Open Telegram** on your phone or desktop
2. **Search for** `@BotFather`
3. **Send** `/newbot`
4. **Follow prompts:**
   ```
   Choose a name for your bot:
   → My Robot Car
   
   Choose a username for your bot:
   → edge_drive_bot
   ```
5. **COPY THE TOKEN** - It looks like:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

---

### Step 2: Get Your User ID (30 seconds)

1. **Search for** `@userinfobot` on Telegram
2. **Press Start**
3. **Copy your User ID** (e.g., `123456789`)

---

### Step 3: Configure .env File (1 minute)

**Option A: Edit .env file directly**

1. Open `hardware_setup/telegram_bot/.env` in a text editor
2. Find these lines:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_ADMIN_IDS=your_user_id_here
   ```
3. Replace with your values:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_ADMIN_IDS=123456789
   ```
4. **Save the file**

**Option B: Use Windows command prompt**

```batch
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\telegram_bot

set TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
set TELEGRAM_ADMIN_IDS=123456789
```

---

### Step 4: Run the Bot (10 seconds)

**Option A: Use the quick launcher**
```batch
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\telegram_bot
RUN_QUICK.bat
```

**Option B: Run directly with environment variables**

**Windows:**
```batch
cd C:\SANJEEVI\PROJECTS\mini_project\combined_final_new_gps\combined_proj_folder\hardware_setup\telegram_bot

set TELEGRAM_BOT_TOKEN=your_token_here
set TELEGRAM_ADMIN_IDS=your_user_id
python telegram_bot_main.py
```

**Linux/Mac:**
```bash
cd hardware_setup/telegram_bot

export TELEGRAM_BOT_TOKEN=your_token_here
export TELEGRAM_ADMIN_IDS=your_user_id
python telegram_bot_main.py
```

---

### Step 5: Open Telegram (5 seconds)

1. **Search for your bot** by the username you created (e.g., `edge_drive_bot`)
2. **Press START** or send `/start`
3. **You'll see the welcome menu!**

```
🤖 Welcome to EdgeDrive3D Robot Car!

Hello [Your Name]! 👋

[🎮 Control] [📹 Video]
[🗺️ GPS] [🤖 Auto]
[📊 Telemetry] [🎯 Mission]
[🚨 Emergency] [❓ Help]
```

---

## ✅ Success Checklist

- [ ] Bot created with @BotFather
- [ ] Token copied
- [ ] User ID obtained
- [ ] .env file edited with token and ID
- [ ] Bot runs without errors
- [ ] Bot responds to `/start` in Telegram

---

## 🐛 Troubleshooting

### "ImportError: cannot import name 'Update' from 'telegram'"

**Fix:**
```batch
pip uninstall telegram -y
pip install python-telegram-bot==20.7
```

### "No module named 'telegram'"

**Fix:**
```batch
pip install -r requirements.txt
```

### "Bot doesn't respond"

**Check:**
1. Bot is running (console shows "Starting Telegram Bot...")
2. You're using the correct bot username
3. Your user ID is in TELEGRAM_ADMIN_IDS

### "Unauthorized access"

**Fix:**
- Check your user ID in `.env`
- Make sure it's just the number (e.g., `123456789`)
- Multiple IDs: `123456789,987654321`

### "Controller disconnected"

**This is normal if ESP32 is not connected.** The bot will still work in simulated mode. To connect real hardware:

1. Power on your robot car
2. Ensure ESP32 is connected to same network
3. Update ESP32_IP in .env if needed

---

## 🎮 First Commands to Try

Once your bot is running:

```
/start          → Welcome menu
/help           → Tutorials
/control        → Joystick controls
/video          → Camera (will show "no signal" without hardware)
/gps            → Location (simulated)
/status         → System status
```

---

## 📝 .env Configuration Reference

```bash
# REQUIRED: Your bot token from @BotFather
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# REQUIRED: Your Telegram user ID
TELEGRAM_ADMIN_IDS=123456789

# Optional: ESP32 IP (for hardware control)
ESP32_IP=10.17.122.207

# Optional: Ports (default values work fine)
ESP32_COMMAND_PORT=9000
ESP32_STATUS_PORT=9001
UDP_VIDEO_PORT=5000

# Optional: Features
AUTONOMOUS_ENABLED=true
GEOFENCE_ENABLED=true
GPS_SIMULATE=true
```

---

## 🚀 Next Steps

After setup:

1. **Test basic control** - `/control`, tap Forward
2. **Read commands guide** - See `telegram_bot_cmnds.md`
3. **Try autonomous mode** - `/autonomous`
4. **Create a mission** - Send location pins, then `/mission_start`
5. **Review features** - See `FEATURES.md`

---

## 📞 Quick Reference

| File | Purpose |
|------|---------|
| `.env` | Your configuration |
| `telegram_bot_main.py` | Main bot code |
| `RUN_QUICK.bat` | Quick launcher |
| `telegram_bot_cmnds.md` | All commands |
| `QUICKSTART.md` | 5-minute guide |

---

## 🎓 Need More Help?

1. **Check logs** - Bot console shows errors
2. **Read docs** - See `INDEX.md` for navigation
3. **Review troubleshooting** - `telegram_bot_cmnds.md` has detailed section

---

**Happy Controlling! 🤖**

*For complete documentation, see INDEX.md*
