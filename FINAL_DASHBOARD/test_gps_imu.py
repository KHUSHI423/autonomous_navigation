"""
GPS WiFi Test Client - Tests connection to Flask GPS Server
Uses HTTP POST to send GPS data to the server

IMPORTANT: 
1. First run: python server.py
2. Then run this test script
3. Update SERVER_IP to match your laptop's IP address
"""

import requests
import time
import random

# ================== CONFIGURATION ==================
# Find your laptop IP using: ipconfig (in cmd)
# Look for IPv4 Address under your active network adapter
SERVER_IP = "192.168.1.100"  # CHANGE THIS to your laptop IP
SERVER_PORT = 5000

# ================== TEST CONFIGURATION ==================
TEST_INTERVAL = 1.0  # Send data every 1 second
USE_SIMULATED_DATA = True  # True = generate fake GPS, False = use fixed values

# ================== SERVER URL ==================
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"


def check_server_health():
    """Check if server is running and healthy"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Server is healthy: {response.json()}")
            return True
        else:
            print(f"✗ Server returned status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {BASE_URL}")
        print("  Make sure server.py is running!")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Connection timed out to {BASE_URL}")
        return False


def send_gps_data(latitude, longitude, speed, altitude, satellites, heading=0.0):
    """Send GPS data to server via HTTP POST"""
    url = f"{BASE_URL}/gps/update"
    
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "speed": speed,
        "altitude": altitude,
        "satellites": satellites,
        "heading": heading
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success: {result}")
            return True
        else:
            print(f"✗ Server error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection failed to {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out")
        return False


def get_simulated_gps_data():
    """Generate simulated GPS data (simulates moving vehicle)"""
    # Base location (can be changed to any location)
    base_lat = 28.6139  # New Delhi
    base_lon = 77.2090
    
    # Add small random movement
    latitude = base_lat + random.uniform(-0.001, 0.001)
    longitude = base_lon + random.uniform(-0.001, 0.001)
    speed = random.uniform(20.0, 60.0)  # km/h
    altitude = random.uniform(200.0, 250.0)  # meters
    satellites = random.randint(8, 12)
    heading = random.uniform(0.0, 360.0)
    
    return latitude, longitude, speed, altitude, satellites, heading


def get_current_gps_data():
    """Get current GPS data from server"""
    try:
        response = requests.get(f"{BASE_URL}/gps", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def main():
    """Main test function"""
    print("=" * 60)
    print("   GPS WiFi Test Client")
    print("=" * 60)
    print(f"Server URL: {BASE_URL}")
    print("=" * 60)
    
    # Step 1: Check server health
    print("\n[Step 1] Checking server health...")
    if not check_server_health():
        print("\nERROR: Server is not running or not reachable!")
        print("\nTo fix this:")
        print("  1. Run: python server.py")
        print("  2. Make sure SERVER_IP matches your laptop IP")
        print("  3. Check firewall allows port 5000")
        return
    
    # Step 2: Send test GPS data
    print("\n[Step 2] Sending test GPS data...")
    print("Sending 10 updates (press Ctrl+C to stop early)...")
    print("-" * 60)
    
    count = 0
    success_count = 0
    
    try:
        while count < 10:
            count += 1
            
            if USE_SIMULATED_DATA:
                lat, lon, speed, alt, sats, heading = get_simulated_gps_data()
            else:
                # Fixed test values
                lat, lon, speed, alt, sats, heading = 28.6139, 77.2090, 25.0, 216.0, 10, 45.0
            
            print(f"\n[Test {count}/10]")
            print(f"  Sending: lat={lat:.6f}, lon={lon:.6f}, speed={speed:.1f} km/h")
            
            if send_gps_data(lat, lon, speed, alt, sats, heading):
                success_count += 1
            
            time.sleep(TEST_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    # Step 3: Show summary
    print("\n" + "=" * 60)
    print("   Test Summary")
    print("=" * 60)
    print(f"Tests sent: {count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {count - success_count}")
    
    # Step 4: Get final GPS data from server
    print("\n[Step 3] Final GPS data from server:")
    gps_data = get_current_gps_data()
    if gps_data:
        print("-" * 60)
        print(f"  Latitude:   {gps_data.get('latitude', 'N/A')}")
        print(f"  Longitude:  {gps_data.get('longitude', 'N/A')}")
        print(f"  Speed:      {gps_data.get('speed', 'N/A')} km/h")
        print(f"  Altitude:   {gps_data.get('altitude', 'N/A')} m")
        print(f"  Satellites: {gps_data.get('satellites', 'N/A')}")
        print(f"  Heading:    {gps_data.get('heading', 'N/A')}°")
        print(f"  Status:     {gps_data.get('status', 'N/A')}")
        print("-" * 60)
    
    print("\n✓ Test complete!")
    print("\nNext steps:")
    print("  1. Open browser: http://localhost:5000")
    print("  2. Check dashboard shows your GPS data")
    print("  3. Upload esp32_wifi_gps.ino to ESP32")
    print("=" * 60)


if __name__ == "__main__":
    main()
