"""
Test script to verify hardware_sync_bridge.py is sending data correctly
"""

import requests
import time

print("="*70)
print("  TESTING HARDWARE SYNC BRIDGE - Sensor Data")
print("="*70)
print()

url = "http://localhost:8767/sensors"

print(f"Polling {url} every 1 second...")
print("Press Ctrl+C to stop")
print()

try:
    count = 0
    while count < 20:  # Poll 20 times
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"[{count+1}] Ultra: {data.get('ultrasonic', 'N/A'):>6}cm | "
                      f"PWM: {data.get('pwm', 0):>3} | "
                      f"Steer: {data.get('steering', 0):>5} | "
                      f"Speed: {data.get('speed', 0):>3} | "
                      f"Pos: ({data.get('position', {}).get('x', 0):>6.1f}, {data.get('position', {}).get('z', 0):>6.1f}) | "
                      f"Rot: {data.get('rotation', 0):>6.2f}")
            else:
                print(f"[{count+1}] Error: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[{count+1}] Connection error - Is hardware_sync_bridge.py running?")
        except Exception as e:
            print(f"[{count+1}] Error: {e}")
        
        count += 1
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\nStopped by user")

print()
print("="*70)
print("  TEST COMPLETE")
print("="*70)
