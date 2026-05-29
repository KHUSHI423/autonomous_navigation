# ESP32 WiFi GPS Integration Guide

## Problem Fixed ✓

The server now properly handles WiFi GPS updates from ESP32 without being overwritten by serial disconnect.

---

## How It Works

### ESP32 Sends:
```cpp
HTTP POST http://192.168.164.5:5000/gps/update
Content-Type: application/json

{
  "latitude": 11.0286,
  "longitude": 77.0269,
  "speed": 25.5,
  "altitude": 420.0,
  "satellites": 10,
  "heading": 45.0
}
```

### Server Response:
```json
{
  "success": true,
  "status": "updated",
  "gps": {
    "latitude": 11.0286,
    "longitude": 77.0269,
    "status": "active",
    "timestamp": "2024-03-21T10:30:00"
  }
}
```

---

## Server Changes Made

### 1. Fixed `/gps/update` Route
- ✓ Properly updates `gps_data` from JSON
- ✓ Sets `status = "active"`
- ✓ Updates timestamp
- ✓ Tracks GPS source (wifi vs serial)

### 2. Fixed Serial Thread
- ✓ No longer resets `gps_data` when disconnected
- ✓ Only updates status if no WiFi updates for 5 seconds
- ✓ Prevents overwriting WiFi data

### 3. Network Configuration
- ✓ Listens on `0.0.0.0` (all network interfaces)
- ✓ CORS enabled for cross-origin requests
- ✓ Threaded mode for concurrent requests

---

## Setup Instructions

### Step 1: Find Laptop IP

**Windows:**
```cmd
ipconfig
```

Look for **IPv4 Address** (e.g., `192.168.164.5`)

**Linux/Mac:**
```bash
ifconfig
# or
ip addr show
```

### Step 2: Update ESP32 Code

```cpp
const char* serverIP = "192.168.164.5";  // Your laptop IP
const int serverPort = 5000;
```

### Step 3: Start Server

```bash
python server.py
```

Server will show:
```
Server: http://0.0.0.0:5000
WiFi GPS Update Endpoint:
  POST http://<server-ip>:5000/gps/update
```

### Step 4: Test with Curl

**Windows PowerShell:**
```powershell
curl -X POST http://192.168.164.5:5000/gps/update `
  -H "Content-Type: application/json" `
  -Body '{"latitude":11.02,"longitude":77.02,"speed":25}'
```

**Linux/Mac:**
```bash
curl -X POST http://192.168.164.5:5000/gps/update \
  -H "Content-Type: application/json" \
  -d '{"latitude":11.02,"longitude":77.02,"speed":25}'
```

### Step 5: Check GPS Data

```bash
curl http://192.168.164.5:5000/gps
```

Should show:
```json
{
  "latitude": 11.02,
  "longitude": 77.02,
  "status": "active",
  ...
}
```

### Step 6: Open Dashboard

Browser: `http://192.168.164.5:5000`

Dashboard should show:
- ✓ Status: **ACTIVE** (green)
- ✓ Updated coordinates
- ✓ Non-zero speed

---

## ESP32 Example Code

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverIP = "192.168.164.5";  // Laptop IP
const int serverPort = 5000;

void sendGPSData(float lat, float lon, float speed, float alt, int sats) {
  if(WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    
    String url = "http://" + String(serverIP) + ":" + String(serverPort) + "/gps/update";
    
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON payload
    String jsonPayload = "{";
    jsonPayload += "\"latitude\":" + String(lat, 6) + ",";
    jsonPayload += "\"longitude\":" + String(lon, 6) + ",";
    jsonPayload += "\"speed\":" + String(speed) + ",";
    jsonPayload += "\"altitude\":" + String(alt) + ",";
    jsonPayload += "\"satellites\":" + String(sats);
    jsonPayload += "}";
    
    // Send POST request
    int httpResponseCode = http.POST(jsonPayload);
    
    if(httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("HTTP Response: " + String(httpResponseCode));
      Serial.println("Response: " + response);
    } else {
      Serial.println("HTTP Error: " + String(httpResponseCode));
    }
    
    http.end();
  }
}

void setup() {
  Serial.begin(115200);
  
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Simulate GPS data
  float lat = 11.0286 + random(100) / 10000.0;
  float lon = 77.0269 + random(100) / 10000.0;
  float speed = random(10, 50);
  float alt = 420.0;
  int sats = random(8, 12);
  
  sendGPSData(lat, lon, speed, alt, sats);
  
  delay(1000);  // Send every 1 second
}
```

---

## Troubleshooting

### Dashboard shows "DISCONNECTED"

**Check:**
1. Server is running: `python server.py`
2. ESP32 sending to correct IP
3. Firewall allows port 5000
4. Laptop and ESP32 on same network

### HTTP Response: 400

**Check:**
1. JSON format is correct
2. Content-Type header set
3. Using POST method (not GET)

### HTTP Response: 0 / Connection Failed

**Check:**
1. Server IP is correct
2. Port 5000 not blocked
3. WiFi connection stable
4. Server is running

### Server shows "No JSON data received"

**Check:**
1. ESP32 code sets Content-Type header
2. JSON payload is valid
3. Using `http.POST(jsonPayload)` not `http.GET()`

---

## Test Without ESP32

Use the test script:

```bash
# Update SERVER_IP in test_wifi_gps.py first
python test_wifi_gps.py
```

This simulates ESP32 GPS updates.

---

## Server Console Output

When working correctly:

```
✓ GPS UPDATE (WiFi): {'latitude': 11.0286, 'longitude': 77.0269, ...}
  Status: active
  Position: 11.0286, 77.0269
```

Dashboard shows:
- Green "ACTIVE" status
- Moving coordinates
- Updated speed/altitude

---

## Key Server Fixes

1. **GPS Source Tracking**: Server tracks if data is from 'wifi' or 'serial'
2. **No Serial Override**: Serial thread doesn't reset WiFi data
3. **Proper Status Update**: WiFi updates set `status = "active"`
4. **Network Binding**: Server listens on `0.0.0.0` (all interfaces)
5. **Debug Printing**: Server prints received GPS data

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/gps/update` | POST | Update GPS from ESP32 WiFi |
| `/gps` | GET | Get current GPS data |
| `/health` | GET | Server health check |
| `/gps/reset` | POST | Reset to default location |

---

**Status: FIXED ✓**

ESP32 WiFi GPS updates now work correctly without being overwritten by serial disconnect logic.
