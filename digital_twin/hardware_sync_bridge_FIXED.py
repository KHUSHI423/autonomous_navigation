"""
=============================================================================
HARDWARE SYNC BRIDGE - FIXED VERSION
=============================================================================
Receives data forwarded from robot_car_v2_modern.py and serves to Digital Twin
Port configuration:
  - UDP 9002: Receive STATUS from robot_car_v2_modern.py
  - HTTP 8767: Sensor data for HTML polling
  - WebSocket 8766: Real-time state broadcast
=============================================================================
"""

import asyncio
import websockets
import json
import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import math

# ============ CONFIGURATION ============
WEBSOCKET_PORT = 8766
HTTP_SENSOR_PORT = 8767
UDP_FORWARD_PORT = 9002  # Receive from robot_car_v2_modern.py
ESP32_IP = "10.17.122.207"
ESP32_COMMAND_PORT = 9000

print("="*70)
print("  HARDWARE SYNC BRIDGE - Digital Twin Data Server")
print("="*70)
print(f"UDP Forward: 127.0.0.1:{UDP_FORWARD_PORT} (from robot_car_v2_modern.py)")
print(f"HTTP Sensors: http://localhost:{HTTP_SENSOR_PORT}/sensors")
print(f"WebSocket: ws://localhost:{WEBSOCKET_PORT}")
print(f"ESP32 Commands: {ESP32_IP}:{ESP32_COMMAND_PORT}")
print("="*70)

# ============ GLOBAL STATE ============
class HardwareState:
    def __init__(self):
        self.connected_clients = set()
        self.robot_state = {
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": 0,
            "throttle": 0,
            "steering": 0.0,
            "speed": 0,
            "ultrasonic": 999,
            "mode": "MANUAL",
            "pwm": 0,
            "runtime": 0,
            "distance": 0,
            "timestamp": time.time()
        }
        self.is_running = True
        self.last_update = time.time()
        self.start_time = time.time()

state = HardwareState()

# ============ HTTP SENSOR SERVER ============
class SensorHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/sensors':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            sensor_data = {
                "ultrasonic": state.robot_state.get("ultrasonic", 999),
                "pwm": state.robot_state.get("throttle", 0),
                "steering": state.robot_state.get("steering", 0.0),
                "mode": state.robot_state.get("mode", "MANUAL"),
                "speed": state.robot_state.get("speed", 0),
                "timestamp": time.time()
            }
            
            self.wfile.write(json.dumps(sensor_data).encode())
            print(f"📡 HTTP sensor request: ultra={sensor_data['ultrasonic']}cm, pwm={sensor_data['pwm']}")
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        pass

# ============ UDP FORWARD RECEIVER ============
def udp_forward_receiver():
    """Receive STATUS data forwarded from robot_car_v2_modern.py"""
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(("127.0.0.1", UDP_FORWARD_PORT))
    udp_sock.settimeout(1.0)
    
    print(f"📡 UDP forward receiver listening on 127.0.0.1:{UDP_FORWARD_PORT}")
    
    while state.is_running:
        try:
            data, addr = udp_sock.recvfrom(4096)
            message = data.decode('utf-8').strip()
            
            if message.startswith("STATUS:"):
                # Parse: STATUS:AUTO:150:0.02:25.5
                parts = message.split(":")
                if len(parts) >= 5:
                    mode = parts[1]
                    throttle = int(parts[2]) if parts[2].isdigit() else 0
                    steering = float(parts[3]) if parts[3] else 0.0
                    ultrasonic = float(parts[4]) if parts[4] not in ['--', '', '999'] else 999
                    
                    # Calculate delta time
                    now = time.time()
                    dt = now - state.last_update
                    
                    # Update state
                    state.robot_state["mode"] = mode
                    state.robot_state["throttle"] = throttle
                    state.robot_state["steering"] = steering
                    state.robot_state["ultrasonic"] = ultrasonic
                    state.robot_state["speed"] = abs(throttle) * 0.5
                    state.robot_state["pwm"] = throttle
                    state.robot_state["runtime"] = now - state.start_time
                    
                    # Simulate position based on throttle/steering
                    if dt > 0 and throttle > 0:
                        current_rotation = state.robot_state["rotation"]
                        
                        # Adjust rotation based on steering
                        if steering != 0:
                            state.robot_state["rotation"] += steering * dt * 2
                        
                        # Move forward/backward
                        speed_mps = (throttle / 255) * 2.0  # Approximate m/s
                        state.robot_state["position"]["x"] += math.sin(current_rotation) * speed_mps * dt
                        state.robot_state["position"]["z"] += math.cos(current_rotation) * speed_mps * dt
                        state.robot_state["distance"] += speed_mps * dt
                    
                    state.last_update = now
                    
                    # Print update
                    ultra_str = f"{ultrasonic:.0f}cm" if ultrasonic < 999 else "--"
                    print(f"📊 Robot: Mode={mode}, Throttle={throttle}, Ultra={ultra_str}, Speed={state.robot_state['speed']:.0f}")
                    
        except socket.timeout:
            pass
        except Exception as e:
            print(f"UDP error: {e}")
    
    udp_sock.close()

def send_robot_command(throttle: int, steering: float):
    """Send command to ESP32 via UDP"""
    try:
        command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd = f"{throttle}:{steering:.2f}"
        command_sock.sendto(cmd.encode(), (ESP32_IP, ESP32_COMMAND_PORT))
        print(f"-> ESP32 Command: {cmd}")
    except Exception as e:
        print(f"Command send error: {e}")

# ============ WEBSOCKET SERVER ============
async def handler(websocket):
    """Handle WebSocket connections from Digital Twin"""
    state.connected_clients.add(websocket)
    print(f"✅ Digital Twin connected. Total: {len(state.connected_clients)}")
    
    try:
        await websocket.send(json.dumps({
            "type": "init",
            "state": state.robot_state
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "command":
                    command = data.get("command")
                    throttle = data.get("throttle", 150)
                    steering = data.get("steering", 0.0)
                    
                    if command == "forward":
                        send_robot_command(throttle, 0.0)
                    elif command == "backward":
                        send_robot_command(-throttle, 0.0)
                    elif command == "left":
                        send_robot_command(int(throttle * 0.5), -0.5)
                    elif command == "right":
                        send_robot_command(int(throttle * 0.5), 0.5)
                    elif command == "stop":
                        send_robot_command(0, 0.0)
                        
            except json.JSONDecodeError:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        state.connected_clients.discard(websocket)
        print(f"❌ Digital Twin disconnected. Total: {len(state.connected_clients)}")

async def broadcast_loop():
    """Broadcast robot state at 30 FPS"""
    while state.is_running:
        if state.connected_clients:
            message = json.dumps({
                "type": "hardware_state",
                "state": state.robot_state,
                "pwm": state.robot_state["pwm"],
                "steering": state.robot_state["steering"],
                "ultrasonic": state.robot_state["ultrasonic"],
                "speed": state.robot_state["speed"]
            })
            
            disconnected = set()
            for client in state.connected_clients:
                try:
                    await client.send(message)
                except:
                    disconnected.add(client)
            
            state.connected_clients -= disconnected
        
        await asyncio.sleep(1/30)

async def main():
    ws_server = await websockets.serve(handler, "localhost", WEBSOCKET_PORT)
    print(f"🔌 WebSocket started on ws://localhost:{WEBSOCKET_PORT}")
    
    broadcast_task = asyncio.create_task(broadcast_loop())
    
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        broadcast_task.cancel()
        ws_server.close()

def run_async_server():
    asyncio.run(main())

if __name__ == "__main__":
    # Start UDP forward receiver
    udp_thread = threading.Thread(target=udp_forward_receiver, daemon=True)
    udp_thread.start()
    
    # Start HTTP sensor server
    http_server = HTTPServer(('localhost', HTTP_SENSOR_PORT), SensorHTTPHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    print(f"🌐 HTTP sensor server started on port {HTTP_SENSOR_PORT}")
    
    # Run WebSocket server
    try:
        run_async_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        state.is_running = False
        time.sleep(1)
        print("✅ Hardware sync bridge stopped")
