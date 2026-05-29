"""
=============================================================================
DIGITAL TWIN SERVER - Real-time Robot Car Visualization
=============================================================================
WebSocket server that streams robot car telemetry to the 3D digital twin
Receives data from robot_car_v2_modern.py and broadcasts to web clients

Features:
- WebSocket server on port 8765
- Receives robot state via UDP/TCP
- Broadcasts to all connected web clients
- Records incidents for replay
- Route simulation engine

Run: python digital_twin_server.py
Then open: http://localhost:8080
=============================================================================
"""

import asyncio
import websockets
import json
import socket
import threading
import time
import argparse
from datetime import datetime
from collections import deque
from typing import Dict, Set, Optional, Any
import struct
import numpy as np

# ============ CONFIGURATION ============
WEBSOCKET_PORT = 8765
HTTP_PORT = 8081  # Changed from 8080 to avoid conflict
UDP_ROBOT_PORT = 9001  # Receive status from robot car
UDP_CAMERA_PORT = 5000  # Receive camera frames

# Simulation defaults
DEFAULT_STATE = {
    "position": {"x": 0, "y": 0, "z": 0},
    "rotation": 0,
    "throttle": 0,
    "steering": 0.0,
    "speed": 0,
    "ultrasonic": 999,
    "mode": "MANUAL",
    "detections": [],
    "timestamp": 0
}

print("="*70)
print("  DIGITAL TWIN SERVER - Robot Car Visualization")
print("="*70)
print(f"WebSocket: ws://localhost:{WEBSOCKET_PORT}")
print(f"HTTP:      http://localhost:{HTTP_PORT}")
print(f"UDP Robot: localhost:{UDP_ROBOT_PORT}")
print(f"UDP Camera: localhost:{UDP_CAMERA_PORT}")
print(f"\nOpen browser: http://localhost:{HTTP_PORT}")
print("\nPress Ctrl+C to stop")
print("="*70)

# ============ GLOBAL STATE ============
class DigitalTwinState:
    def __init__(self):
        self.robot_state = DEFAULT_STATE.copy()
        self.robot_state["timestamp"] = time.time()
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.incident_history: deque = deque(maxlen=100)
        self.is_running = True
        self.last_update = time.time()
        
    def update(self, new_data: dict):
        """Update robot state with new data"""
        for key, value in new_data.items():
            if key in self.robot_state:
                self.robot_state[key] = value
        self.robot_state["timestamp"] = time.time()
        self.last_update = time.time()
        
    def add_incident(self, incident_type: str, details: dict):
        """Record an incident for replay"""
        incident = {
            "type": incident_type,
            "details": details,
            "timestamp": time.time(),
            "state_snapshot": self.robot_state.copy()
        }
        self.incident_history.append(incident)
        print(f"📝 INCIDENT RECORDED: {incident_type}")
        
state = DigitalTwinState()

# ============ WEBSOCKET SERVER ============
async def handler(websocket: websockets.WebSocketServerProtocol, path: str):
    """Handle WebSocket connections from web clients"""
    state.connected_clients.add(websocket)
    print(f"✅ Client connected. Total: {len(state.connected_clients)}")
    
    # Send current state immediately
    try:
        await websocket.send(json.dumps({
            "type": "init",
            "state": state.robot_state
        }))
    except:
        pass
    
    try:
        async for message in websocket:
            # Handle messages from client (commands, etc.)
            try:
                data = json.loads(message)
                if data.get("type") == "command":
                    print(f"📩 Command from client: {data.get('command')}")
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        state.connected_clients.discard(websocket)
        print(f"❌ Client disconnected. Total: {len(state.connected_clients)}")

async def broadcast_state():
    """Broadcast robot state to all connected clients"""
    if state.connected_clients:
        message = json.dumps({
            "type": "state_update",
            "state": state.robot_state
        })
        
        # Send to all clients
        disconnected = set()
        for client in state.connected_clients:
            try:
                await client.send(message)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        state.connected_clients -= disconnected

async def broadcast_loop():
    """Main broadcast loop - send updates at 30 FPS"""
    while state.is_running:
        await broadcast_state()
        await asyncio.sleep(1/30)  # 30 FPS

# ============ UDP RECEIVER (Robot Status) ============
def udp_receiver_thread():
    """Receive robot status via UDP from robot_car_v2_modern.py"""
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(("0.0.0.0", UDP_ROBOT_PORT))
    udp_sock.settimeout(1.0)
    
    print(f"📡 UDP receiver listening on port {UDP_ROBOT_PORT}")
    
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
                    ultrasonic = float(parts[4]) if parts[4] not in ['--', ''] else 999
                    
                    state.update({
                        "mode": mode,
                        "throttle": throttle,
                        "steering": steering,
                        "ultrasonic": ultrasonic,
                        "speed": throttle  # Approximate
                    })
                    
                    # Check for incidents
                    if ultrasonic < 999 and ultrasonic < 30:
                        state.add_incident("ULTRASONIC_CLOSE", {
                            "distance": ultrasonic,
                            "throttle": throttle,
                            "action": "emergency" if ultrasonic < 15 else "avoidance"
                        })
                    
                    print(f"📊 Robot update: Mode={mode}, Throttle={throttle}, Ultra={ultrasonic}cm")
                    
        except socket.timeout:
            pass
        except Exception as e:
            print(f"UDP error: {e}")
    
    udp_sock.close()

# ============ SIMULATION MODE ============
def simulation_thread():
    """Generate simulated robot data for demo purposes"""
    t = 0
    
    while state.is_running:
        t += 0.05
        
        # Simulate circular motion with variations
        x = np.sin(t * 0.5) * 30
        z = np.cos(t * 0.5) * 30
        rotation = np.arctan2(np.cos(t * 0.5), np.sin(t * 0.5))
        
        # Simulate throttle and steering
        throttle = 150 + np.sin(t) * 30
        steering = np.sin(t * 0.3) * 0.3
        
        # Simulate ultrasonic (occasional obstacles)
        if 10 < (t % 20) < 12:
            ultrasonic = 15 + np.random.random() * 10  # Close obstacle
            if ultrasonic < 20 and not hasattr(simulation_thread, 'incident_recorded'):
                state.add_incident("SIMULATED_OBSTACLE", {
                    "distance": ultrasonic,
                    "time": t
                })
                simulation_thread.incident_recorded = True
        else:
            ultrasonic = 80 + np.random.random() * 50
            if ultrasonic > 100:
                simulation_thread.incident_recorded = False
        
        speed = throttle * 0.5
        
        state.update({
            "position": {"x": x, "y": 0, "z": z},
            "rotation": rotation,
            "throttle": throttle,
            "steering": steering,
            "speed": speed,
            "ultrasonic": ultrasonic,
            "mode": "AUTO"
        })
        
        time.sleep(0.1)  # 10 Hz update rate

# ============ HTTP SERVER (Serve HTML) ============
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

def http_server_thread():
    """Serve the HTML interface"""
    server = HTTPServer(('localhost', HTTP_PORT), CustomHandler)
    print(f"🌐 HTTP server running at http://localhost:{HTTP_PORT}")
    server.serve_forever()

# ============ INCIDENT REPLAY API ============
async def incident_handler(websocket: websockets.WebSocketServerProtocol, path: str):
    """Handle incident replay requests"""
    if path == "/incidents":
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("action") == "get_all":
                    incidents = list(state.incident_history)
                    for inc in incidents:
                        inc["timestamp_str"] = datetime.fromtimestamp(inc["timestamp"]).strftime("%H:%M:%S")
                    await websocket.send(json.dumps({"incidents": incidents}))
                
                elif data.get("action") == "replay":
                    idx = data.get("index", 0)
                    if 0 <= idx < len(state.incident_history):
                        incident = state.incident_history[idx]
                        await websocket.send(json.dumps({
                            "type": "replay_data",
                            "incident": incident
                        }))
        except Exception as e:
            print(f"Incident handler error: {e}")

# ============ MAIN ============
async def main():
    """Main async entry point"""
    # Start WebSocket server
    ws_server = await websockets.serve(handler, "localhost", WEBSOCKET_PORT)
    print(f"🔌 WebSocket server started on ws://localhost:{WEBSOCKET_PORT}")
    
    # Start broadcast loop
    broadcast_task = asyncio.create_task(broadcast_loop())
    
    # Keep running
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        broadcast_task.cancel()
        ws_server.close()
        await ws_server.wait_closed()

def run_async_server():
    """Run the async WebSocket server"""
    asyncio.run(main())

if __name__ == "__main__":
    # Start UDP receiver thread
    udp_thread = threading.Thread(target=udp_receiver_thread, daemon=True)
    udp_thread.start()
    
    # Start simulation thread (for demo without physical robot)
    sim_thread = threading.Thread(target=simulation_thread, daemon=True)
    sim_thread.start()
    
    # Start HTTP server thread
    http_thread = threading.Thread(target=http_server_thread, daemon=True)
    http_thread.start()
    
    # Run async WebSocket server
    try:
        run_async_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        state.is_running = False
        time.sleep(1)
        print("✅ Digital Twin server stopped")
