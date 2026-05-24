import socket

# ESP32 IP address (check from Serial Monitor after upload)
ESP32_IP = "10.25.202.208"  # ← CHANGE THIS to your ESP32 IP
PORT = 80

print("\n🔗 ESP32 WiFi Test Client")
print("=" * 40)
print(f"Connecting to {ESP32_IP}:{PORT}...")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(10)

try:
    s.connect((ESP32_IP, PORT))
    print("✅ Connected to ESP32!\n")
    print("Receiving data:\n")
    print("-" * 40)
    
    while True:
        data = s.recv(1024).decode().strip()
        if data:
            parts = data.split(',')
            if len(parts) == 4:
                lat, lon, count, val = parts
                print(f"Lat: {lat} | Lon: {lon} | Count: {count} | Value: {val}")
            else:
                print(data)
except socket.timeout:
    print("❌ Connection timed out!")
    print("   Make sure ESP32 is running and IP is correct.")
except ConnectionRefusedError:
    print("❌ Connection refused!")
    print("   ESP32 server may not be running.")
except KeyboardInterrupt:
    print("\n\n👋 Stopped by user")
finally:
    s.close()
    print("\n✅ Connection closed")
