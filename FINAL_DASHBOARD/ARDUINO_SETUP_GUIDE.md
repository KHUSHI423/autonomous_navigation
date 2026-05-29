# ESP32 WiFi GPS - Arduino IDE Setup Guide

## Problem Fixed ✓

The error occurred because:
1. **Wrong Port**: `test_gps_imu.py` was trying to connect to port 80, but the Flask server runs on port 5000
2. **Wrong Protocol**: The script used raw sockets instead of HTTP POST requests
3. **Server Not Running**: The connection timeout means the server wasn't running or IP was wrong

---

## Solution Overview

### Files Created/Updated:
1. **`esp32_wifi_gps.ino`** - Arduino code for ESP32 to send GPS data via WiFi
2. **`test_gps_imu.py`** - Fixed Python test script to verify server connection
3. **This guide** - Step-by-step setup instructions

---

## Step-by-Step Setup

### Step 1: Find Your Laptop's IP Address

Open Command Prompt and run:
```cmd
ipconfig
```

Look for **IPv4 Address** under your active network adapter (WiFi or Ethernet):
```
Wireless LAN adapter Wi-Fi:
   IPv4 Address. . . . . . . . . . . : 192.168.1.100  <-- THIS IS YOUR IP
```

**Write down this IP address** - you'll need it in multiple places.

---

### Step 2: Start the Flask Server

Open Command Prompt in your project folder and run:
```cmd
python server.py
```

You should see:
```
============================================================
ADAS Visualization Dashboard
============================================================
Server: http://0.0.0.0:5000
Serial Port: COM3 | Baud: 9600
============================================================
GPS Sources:
  - Serial: ESP32 via USB
  - WiFi: ESP32 via HTTP POST
============================================================
```

**Keep this window open!** The server must be running for ESP32 to connect.

---

### Step 3: Test Server Connection (Optional)

Open a **new** Command Prompt and run:
```cmd
python test_gps_imu.py
```

**Before running**, edit `test_gps_imu.py` and update:
```python
SERVER_IP = "192.168.1.100"  # Change to YOUR laptop IP from Step 1
```

If successful, you'll see:
```
✓ Server is healthy: {...}
✓ Success: {...}
```

If it fails:
- Make sure `server.py` is running
- Check firewall allows port 5000
- Verify IP address is correct

---

### Step 4: Install Arduino IDE

If not already installed:
1. Download from: https://www.arduino.cc/en/software
2. Install Arduino IDE (latest version)
3. Open Arduino IDE

---

### Step 5: Add ESP32 Board Support to Arduino IDE

1. Open Arduino IDE
2. Go to **File** → **Preferences**
3. In "Additional Board Manager URLs", add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Click **OK**
5. Go to **Tools** → **Board** → **Boards Manager**
6. Search for "ESP32"
7. Install **ESP32 by Espressif Systems**
8. Go to **Tools** → **Board** → **ESP32 Arduino** → Select your board (e.g., "DOIT ESP32 DEVKIT V1")

---

### Step 6: Configure ESP32 Code

Open **`esp32_wifi_gps.ino`** in Arduino IDE.

**Update these values:**
```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";        // Your WiFi name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"; // Your WiFi password

const char* SERVER_IP = "192.168.1.100";  // YOUR laptop IP from Step 1
```

**GPS Module Connections (if using hardware GPS):**
```cpp
#define GPS_RX_PIN 16  // ESP32 pin connected to GPS TX
#define GPS_TX_PIN 17  // ESP32 pin connected to GPS RX
```

**Common GPS Modules:**
- **NEO-6M**: VCC → 3.3V or 5V, GND → GND, TX → GPIO 16, RX → GPIO 17
- **NEO-7M**: Same as NEO-6M
- **NEO-M8N**: Same as NEO-6M

---

### Step 7: Upload Code to ESP32

1. Connect ESP32 to laptop via USB cable
2. In Arduino IDE:
   - **Tools** → **Port** → Select COM port (e.g., COM3)
   - Click **Upload** button (→ arrow)
3. Wait for upload to complete (may take 1-2 minutes)
4. Open **Serial Monitor** (magnifying glass icon)
5. Set baud rate to **115200**

---

### Step 8: Verify ESP32 Output

In Serial Monitor, you should see:
```
========================================
   ESP32 WiFi GPS Data Sender
========================================
GPS Serial initialized
Connecting to WiFi: YOUR_WIFI_SSID
✓ WiFi connected successfully!
IP Address: 192.168.1.105
Signal Strength (RSSI): -65 dBm
========================================
Setup complete!
========================================

Sending GPS data to: http://192.168.1.100:5000/gps/update
JSON Payload: {"latitude":28.613900,"longitude":77.209000,...}
✓ HTTP Response Code: 200
Response: {"success":true,"status":"updated",...}
```

---

### Step 9: Verify Server Receives Data

In the **server.py** console, you should see:
```
✓ GPS UPDATE (WiFi): {'latitude': 28.6139, 'longitude': 77.209, ...}
  ✓ Status: active
  ✓ Source: wifi
  ✓ Position: 28.6139, 77.209
```

---

### Step 10: Open Dashboard

Open your web browser and go to:
```
http://localhost:5000
```

You should see:
- ✓ **Green "ACTIVE" status**
- ✓ Moving coordinates (if GPS has fix)
- ✓ Updated speed, altitude, satellites

---

## Troubleshooting

### Error: "A connection attempt failed because the connected party did not properly respond"

**Causes:**
1. Server not running
2. Wrong IP address
3. Firewall blocking port 5000
4. ESP32 and laptop on different networks

**Fix:**
```cmd
# 1. Start server
python server.py

# 2. Verify IP (should match SERVER_IP in ESP32 code)
ipconfig

# 3. Allow port 5000 in Windows Firewall
# Control Panel → Windows Firewall → Advanced Settings → 
# Inbound Rules → New Rule → Port → TCP → 5000 → Allow
```

---

### Error: "WiFi connection failed"

**Fix:**
1. Check WiFi SSID and password are correct
2. Ensure ESP32 is within WiFi range
3. Try 2.4GHz WiFi (ESP32 doesn't support 5GHz)
4. Check for special characters in password

---

### Error: "HTTP Error: -1" or "Connection failed"

**Causes:**
1. Server not running
2. Wrong server IP in ESP32 code
3. Port 5000 blocked

**Fix:**
```cpp
// Verify these match your setup
const char* SERVER_IP = "192.168.1.100";  // Your laptop IP
const int SERVER_PORT = 5000;
```

---

### Dashboard shows "DISCONNECTED"

**Check:**
1. Server console shows "✓ GPS UPDATE (WiFi)"
2. ESP32 serial shows "✓ HTTP Response Code: 200"
3. IP addresses match everywhere
4. Firewall allows port 5000

---

### GPS shows "NO FIX" or coordinates don't update

**If using hardware GPS module:**
1. Check wiring (TX→RX, RX→TX)
2. GPS needs clear sky view (test outdoors)
3. Wait 30-60 seconds for GPS lock
4. Check GPS module has power

**If using simulated data:**
- This is normal - ESP32 code sends simulated data when no GPS hardware is connected

---

## Quick Reference

### Server Commands
```cmd
# Start server
python server.py

# Test connection
python test_gps_imu.py
```

### ESP32 Code Locations to Edit
```cpp
// Line 15: WiFi SSID
const char* WIFI_SSID = "YOUR_WIFI_SSID";

// Line 16: WiFi Password
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Line 19: Server IP (your laptop IP)
const char* SERVER_IP = "192.168.1.100";
```

### Dashboard URL
```
http://localhost:5000
```

### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/gps/update` | POST | Update GPS from ESP32 |
| `/gps` | GET | Get current GPS data |
| `/health` | GET | Server health check |

---

## Hardware Connections (if using GPS module)

### ESP32 ↔ NEO-6M GPS
```
ESP32          NEO-6M
-----          ------
3.3V or 5V  →  VCC
GND         →  GND
GPIO 16     →  TX
GPIO 17     →  RX
```

### ESP32 Pin Options
You can change GPS pins in code:
```cpp
#define GPS_RX_PIN 16  // Any ESP32 RX-capable pin
#define GPS_TX_PIN 17  // Any ESP32 TX-capable pin
```

---

## Summary

1. ✓ Find laptop IP using `ipconfig`
2. ✓ Run `python server.py`
3. ✓ Update `esp32_wifi_gps.ino` with WiFi credentials and server IP
4. ✓ Upload code to ESP32 via Arduino IDE
5. ✓ Open Serial Monitor to verify
6. ✓ Open browser to `http://localhost:5000`
7. ✓ Dashboard shows live GPS data!

---

**Status: READY TO USE** 🚀
