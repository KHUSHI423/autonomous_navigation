"""
ESP32 Sensor Data Reader
Reads MPU6050 + Encoder data from ESP32 via USB Serial
"""

import serial
import serial.tools.list_ports
import time
import json
from datetime import datetime

# ==================== Configuration ====================
BAUD_RATE = 115200
TIMEOUT = 1
ESP32_PORT = "COM9"  # Fixed COM port

def parse_sensor_data(line):
    """Parse CSV sensor data from ESP32"""
    try:
        parts = line.strip().split(',')
        
        if parts[0] == 'DATA':
            return {
                'type': 'MPU+ENC',
                'roll': float(parts[1]),
                'pitch': float(parts[2]),
                'yaw': float(parts[3]),
                'ax': float(parts[4]),
                'ay': float(parts[5]),
                'az': float(parts[6]),
                'gx': float(parts[7]),
                'gy': float(parts[8]),
                'gz': float(parts[9]),
                'encoder': int(parts[10]),
                'timestamp': int(parts[11])
            }
        elif parts[0] == 'ENC':
            return {
                'type': 'ENC_ONLY',
                'encoder': int(parts[1]),
                'timestamp': int(parts[2])
            }
        return None
    except (IndexError, ValueError) as e:
        return None

def main():
    print("\n🔍 Connecting to ESP32 on COM9...")
    
    try:
        ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for ESP32 to initialize
        print(f"✅ Connected at {BAUD_RATE} baud\n")
        print("=" * 70)
    except serial.SerialException as e:
        print(f"❌ Failed to open port: {e}")
        print("   Make sure ESP32 is connected and code is uploaded!")
        return

    try:
        print("⏳ Waiting for data... (rotate encoder or tilt MPU6050)")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Debug: print raw data
                print(f"[RAW] {line}")
                
                if line.startswith('✅') or line.startswith('🔧') or line.startswith('📡') or line.startswith('⚠️') or line.startswith('   '):
                    print(line)
                elif line.startswith('DATA') or line.startswith('ENC'):
                    data = parse_sensor_data(line)
                    if data:
                        if data['type'] == 'MPU+ENC':
                            print(f"\033[KRoll: {data['roll']:6.2f}° | Pitch: {data['pitch']:6.2f}° | Yaw: {data['yaw']:6.2f}° | Encoder: {data['encoder']:3d}°", end="\r")
                        else:
                            print(f"\033[KEncoder: {data['encoder']:3d}° (MPU6050 not connected)", end="\r")
                        
                        # Optional: Save to file
                        # save_to_file(data)
                        
    except KeyboardInterrupt:
        print("\n\n👋 Stopping...")
    finally:
        ser.close()
        print("\n✅ Serial port closed")

def save_to_file(data):
    """Save sensor data to JSON file (optional)"""
    data['datetime'] = datetime.now().isoformat()
    with open('sensor_data.json', 'a') as f:
        f.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    main()
