"""
ESP32 Serial Data Reader - Test
Reads data from ESP32 via USB Serial
"""

import serial
import time

# Configuration
PORT = "COM9"
BAUD_RATE = 115200

print("\n🔗 ESP32 Serial Test")
print("=" * 50)
print(f"Connecting to {PORT} at {BAUD_RATE} baud...")

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for ESP32 reset
    print("✅ Connected!\n")
    print("Receiving data:\n")
    print("-" * 50)
    
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line:
                parts = line.split(',')
                if len(parts) == 4:
                    lat, lon, count, value = parts
                    print(f"Lat: {lat:>8} | Lon: {lon:>9} | Count: {count:>5} | Value: {value:>5}")
                else:
                    print(f"[INFO] {line}")
                    
except serial.SerialException as e:
    print(f"❌ Error: {e}")
    print("   - Close Arduino Serial Monitor")
    print("   - Check if ESP32 is connected")
    print("   - Verify COM port number")
except KeyboardInterrupt:
    print("\n\n👋 Stopped by user")
finally:
    if 'ser' in locals():
        ser.close()
        print("\n✅ Serial port closed")
