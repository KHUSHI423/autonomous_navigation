"""
Test script to simulate ESP32 WiFi GPS updates
Use this to test the server without ESP32 hardware
"""

import requests
import time
import random

# Server configuration
SERVER_IP = '192.168.164.5'  # Change to your laptop IP
SERVER_PORT = 5000
BASE_URL = f'http://{SERVER_IP}:{SERVER_PORT}'

# Test GPS data (simulating movement)
START_LAT = 11.0286
START_LON = 77.0269

def test_gps_update():
    """Send GPS update to server"""
    
    print("=" * 60)
    print("ESP32 WiFi GPS Simulator")
    print("=" * 60)
    print(f"Server: {BASE_URL}")
    print("=" * 60)
    
    # Test data
    test_points = [
        {'latitude': 11.0286, 'longitude': 77.0269, 'speed': 0, 'altitude': 420, 'satellites': 10},
        {'latitude': 11.0287, 'longitude': 77.0270, 'speed': 25, 'altitude': 421, 'satellites': 11},
        {'latitude': 11.0288, 'longitude': 77.0272, 'speed': 30, 'altitude': 422, 'satellites': 10},
        {'latitude': 11.0290, 'longitude': 77.0275, 'speed': 35, 'altitude': 423, 'satellites': 12},
        {'latitude': 11.0292, 'longitude': 77.0278, 'speed': 40, 'altitude': 424, 'satellites': 11},
    ]
    
    try:
        # First, check server health
        print("\n[1] Checking server health...")
        response = requests.get(f'{BASE_URL}/health', timeout=2)
        print(f"    Server Status: {response.json()}")
        
        # Send GPS updates
        print("\n[2] Sending GPS updates...")
        
        for i, gps_point in enumerate(test_points):
            print(f"\n--- Update {i+1} ---")
            print(f"    Sending: {gps_point}")
            
            # Send POST request
            response = requests.post(
                f'{BASE_URL}/gps/update',
                json=gps_point,
                headers={'Content-Type': 'application/json'},
                timeout=2
            )
            
            print(f"    HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"    Response: {result['status']}")
                if 'gps' in result:
                    gps = result['gps']
                    print(f"    Server GPS Status: {gps['status']}")
                    print(f"    Position: {gps['latitude']}, {gps['longitude']}")
            else:
                print(f"    Error: {response.text}")
            
            time.sleep(1)
        
        # Final check
        print("\n[3] Final GPS data check...")
        response = requests.get(f'{BASE_URL}/gps', timeout=2)
        print(f"    Current GPS: {response.json()}")
        
        print("\n" + "=" * 60)
        print("✓ Test completed!")
        print("=" * 60)
        print("\nNow open dashboard in browser:")
        print(f"    http://{SERVER_IP}:{SERVER_PORT}")
        print("\nDashboard should show:")
        print("    - Status: ACTIVE (green)")
        print("    - Updated coordinates")
        print("    - Non-zero speed")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ Connection Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if server is running: python server.py")
        print("  2. Check SERVER_IP matches your laptop IP")
        print("  3. Check firewall allows port 5000")
        print("  4. Make sure laptop and ESP32 on same network")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == '__main__':
    test_gps_update()
