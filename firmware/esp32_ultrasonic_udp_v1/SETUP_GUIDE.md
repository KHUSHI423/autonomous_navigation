# ESP32 Ultrasonic UDP V1 - Setup Guide

## Overview
This code runs on **ESP32 #2** (Ultrasonic Sensor) and sends distance data to **ESP32 #1** (Motor Controller) via UDP.

## What's New in V1
✅ Improved error handling and logging  
✅ Automatic WiFi reconnection  
✅ Packet statistics tracking  
✅ Input validation (2-400cm range)  
✅ 10 Hz update rate (100ms interval)  
✅ Clear serial output with emojis  
✅ Watchdog timer safe  

## Wiring

### HC-SR04 Ultrasonic Sensor
```
HC-SR04    ESP32 #2
VCC   ---> 5V
GND   ---> GND
TRIG  ---> GPIO 5
ECHO  ---> GPIO 18
```

## Configuration Steps

### Step 1: Upload Motor Controller Code (ESP32 #1)
1. Open `esp32_motor_controller/esp32_motor_controller.ino`
2. Upload to ESP32 #1
3. Open Serial Monitor (115200 baud)
4. **Note the IP address** (e.g., `192.168.1.100` or `10.17.x.x`)

### Step 2: Configure Ultrasonic Code (ESP32 #2)
1. Open `esp32_ultrasonic_udp_v1/esp32_ultrasonic_udp_v1.ino`
2. **Change line 32:** Update `MOTOR_ESP_IP` to ESP32 #1's IP
   ```cpp
   const char* MOTOR_ESP_IP = "192.168.1.100";  // ← Use ESP32 #1 IP
   ```
3. Upload to ESP32 #2

### Step 3: Verify Connection
1. Open Serial Monitor for ESP32 #2 (115200 baud)
2. You should see:
   ```
   📡 Robot Car V2 - Ultrasonic Sensor ESP32
   ============================================
   Version: UDP V1
   
   📶 Connecting to WiFi: SANJEEVI
   ✅ WiFi Connected!
      IP Address: 192.168.1.xxx
      Signal Strength: -65 dBm
      Sending to: 192.168.1.100:9002
   ✅ UDP socket initialized
   
   🚀 Starting ultrasonic readings...
   
   📏 Distance: 25.50 cm
   ✅ Sent: ULTRA:25.50 cm | Packets: 1
   ```

### Step 4: Test with Motor Controller
1. Open Serial Monitor for ESP32 #1
2. You should see:
   ```
   📡 UDP Ultrasonic: 25.50 cm
   ```

## Protocol

### UDP Message Format
- **Format:** `ULTRA:25.50`
- **Port:** 9002
- **Frequency:** 10 Hz (every 100ms)
- **Example:** `ULTRA:25.50` means 25.50 cm

### Motor Controller Parser
The motor controller (ESP32 #1) expects this exact format:
```cpp
if (packet.startsWith("ULTRA:")) {
    ultrasonicDistance = packet.substring(6).toFloat();
}
```

## Troubleshooting

### WiFi Won't Connect
- ✅ Verify phone hotspot "SANJEEVI" is active
- ✅ Check password is "YOUR_WIFI_PASSWORD"
- ✅ Ensure ESP32 is within range
- ✅ Check serial monitor for error messages

### UDP Packets Fail to Send
- ✅ Verify `MOTOR_ESP_IP` matches ESP32 #1's IP
- ✅ Both ESP32s must be on the same WiFi network
- ✅ Check WiFi RSSI (should be > -80 dBm)
- ✅ Look for "❌ Send failed" in serial monitor

### Distance Always Shows 999.0
- ✅ Check HC-SR04 wiring (VCC, GND, TRIG, ECHO)
- ✅ Verify TRIG = GPIO 5, ECHO = GPIO 18
- ✅ Ensure sensor has clear path (no obstacles)
- ✅ Test with different distances (2-400cm range)

### Motor Controller Not Receiving Data
- ✅ Check ESP32 #1 serial monitor for "📡 UDP Ultrasonic"
- ✅ Verify both ESP32s on same network
- ✅ Test with web interface: `http://ESP32_1_IP:8080/distance?value=50`
- ✅ Check firewall settings on router

## Statistics

Every 10 seconds, the serial monitor shows:
```
📊 === STATISTICS ===
   Total packets sent: 150
   WiFi RSSI: -65 dBm
   WiFi Status: Connected
   ====================
```

## Performance

- **Update Rate:** 10 Hz (100ms interval)
- **Range:** 2-400 cm
- **Accuracy:** ±1 cm
- **WiFi Reconnect:** Automatic every 5 seconds if disconnected

## Files Created
- `esp32_ultrasonic_udp_v1/esp32_ultrasonic_udp_v1.ino` - Main code

## Next Steps
1. Upload code to ESP32 #2
2. Verify serial output shows distance readings
3. Test with ESP32 #1 motor controller
4. Integrate with laptop controller for obstacle avoidance
