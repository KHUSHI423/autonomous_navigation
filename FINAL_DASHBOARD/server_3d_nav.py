"""
ADAS 3D Navigation Dashboard - Tesla-style Dark Map
Enhanced 3D visualization with uniform building heights and road-following navigation
"""

import serial
import serial.tools.list_ports
import threading
import json
import time
import math
import random
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global GPS data storage
gps_data = {
    'latitude': 28.6139,
    'longitude': 77.2090,
    'altitude': 216.0,
    'speed': 0.0,
    'heading': 0.0,
    'satellites': 8,
    'accuracy': 2.5,
    'timestamp': datetime.now().isoformat(),
    'status': 'disconnected'
}

# Serial configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
serial_conn = None
serial_lock = threading.Lock()
read_thread = None
stop_thread = threading.Event()

# Road network for simulation (simplified road grid)
ROAD_NETWORK = []
SIMULATION_ACTIVE = False


def generate_road_network(center_lat, center_lon, radius=0.02):
    """Generate a grid of roads around the center point"""
    roads = []
    # Create a grid of roads
    for i in range(-5, 6):
        # Horizontal roads
        roads.append({
            'type': 'horizontal',
            'lat': center_lat + i * 0.001,
            'lon_start': center_lon - radius,
            'lon_end': center_lon + radius
        })
        # Vertical roads
        roads.append({
            'type': 'vertical',
            'lon': center_lon + i * 0.001,
            'lat_start': center_lat - radius,
            'lat_end': center_lat + radius
        })
    return roads


def snap_to_road(lat, lon, roads):
    """Snap position to nearest road"""
    min_distance = float('inf')
    snapped_lat, snapped_lon = lat, lon
    
    for road in roads:
        if road['type'] == 'horizontal':
            # Distance to horizontal road
            distance = abs(lat - road['lat'])
            if road['lon_start'] <= lon <= road['lon_end']:
                if distance < min_distance:
                    min_distance = distance
                    snapped_lat = road['lat']
                    snapped_lon = lon
        else:  # vertical
            # Distance to vertical road
            distance = abs(lon - road['lon'])
            if road['lat_start'] <= lat <= road['lat_end']:
                if distance < min_distance:
                    min_distance = distance
                    snapped_lat = lat
                    snapped_lon = road['lon']
    
    return snapped_lat, snapped_lon


def list_available_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def parse_nmea_line(line):
    """Parse NMEA GPS sentences"""
    global gps_data
    
    line = line.strip()
    if not line.startswith('$'):
        return
    
    parts = line.split(',')
    
    if parts[0] in ['$GPGGA', '$GNGGA']:
        try:
            if parts[1]:
                gps_data['timestamp'] = datetime.now().isoformat()
            if parts[2]:
                lat = float(parts[2])
                if parts[3] == 'S':
                    lat = -lat
                gps_data['latitude'] = lat
            if parts[4]:
                lon = float(parts[4])
                if parts[5] == 'W':
                    lon = -lon
                gps_data['longitude'] = lon
            if parts[6]:
                fix = int(parts[6])
                gps_data['status'] = 'active' if fix > 0 else 'no_fix'
            if parts[7]:
                gps_data['satellites'] = int(parts[7])
            if parts[8]:
                gps_data['accuracy'] = float(parts[8])
            if parts[9]:
                gps_data['altitude'] = float(parts[9])
        except (ValueError, IndexError):
            pass
    
    elif parts[0] in ['$GPRMC', '$GNRMC']:
        try:
            if parts[7]:
                speed_knots = float(parts[7])
                gps_data['speed'] = round(speed_knots * 1.852, 2)
            if parts[8]:
                gps_data['heading'] = float(parts[8])
        except (ValueError, IndexError):
            pass
    
    elif line.startswith('{'):
        try:
            data = json.loads(line)
            gps_data.update(data)
            gps_data['timestamp'] = datetime.now().isoformat()
            gps_data['status'] = 'active'
        except json.JSONDecodeError:
            pass


def read_serial_data():
    """Background thread to read GPS data from serial port"""
    global serial_conn, gps_data
    
    while not stop_thread.is_set():
        try:
            with serial_lock:
                if serial_conn and serial_conn.is_open:
                    if serial_conn.in_waiting > 0:
                        line = serial_conn.readline().decode('utf-8', errors='ignore')
                        parse_nmea_line(line)
                else:
                    gps_data['status'] = 'disconnected'
        except Exception as e:
            gps_data['status'] = f'error: {str(e)}'
        
        time.sleep(0.1)


def connect_serial(port):
    """Connect to serial port"""
    global serial_conn, gps_data
    
    try:
        with serial_lock:
            if serial_conn and serial_conn.is_open:
                serial_conn.close()
            
            serial_conn = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)
            gps_data['status'] = 'connected'
            print(f"Connected to {port}")
            return True
    except Exception as e:
        gps_data['status'] = f'connection_error: {str(e)}'
        print(f"Connection failed: {e}")
        return False


def disconnect_serial():
    """Disconnect from serial port"""
    global serial_conn
    
    with serial_lock:
        if serial_conn and serial_conn.is_open:
            serial_conn.close()
            gps_data['status'] = 'disconnected'
            print("Disconnected from serial")


@app.route('/')
def index():
    """Serve the 3D navigation dashboard"""
    return render_template('index_3d_nav.html')


@app.route('/gps')
def get_gps():
    """API endpoint to get current GPS data"""
    return jsonify(gps_data)


@app.route('/gps/update', methods=['POST'])
def update_gps():
    """Manual GPS update (for simulation/testing)"""
    global gps_data, SIMULATION_ACTIVE
    
    data = request.json
    if data:
        with serial_lock:
            gps_data.update(data)
            gps_data['timestamp'] = datetime.now().isoformat()
            gps_data['status'] = 'simulated'
            SIMULATION_ACTIVE = True
    return jsonify(gps_data)


@app.route('/gps/reset', methods=['POST'])
def reset_gps():
    """Reset GPS to default location"""
    global gps_data, SIMULATION_ACTIVE
    
    with serial_lock:
        gps_data = {
            'latitude': 28.6139,
            'longitude': 77.2090,
            'altitude': 216.0,
            'speed': 0.0,
            'heading': 0.0,
            'satellites': 8,
            'accuracy': 2.5,
            'timestamp': datetime.now().isoformat(),
            'status': 'reset'
        }
        SIMULATION_ACTIVE = False
    return jsonify(gps_data)


@app.route('/gps/start_simulation', methods=['POST'])
def start_simulation():
    """Start road-following GPS simulation"""
    global SIMULATION_ACTIVE, gps_data, ROAD_NETWORK
    
    SIMULATION_ACTIVE = True
    ROAD_NETWORK = generate_road_network(gps_data['latitude'], gps_data['longitude'])
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    
    return jsonify({'status': 'simulation_started'})


@app.route('/gps/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop simulation"""
    global SIMULATION_ACTIVE
    SIMULATION_ACTIVE = False
    return jsonify({'status': 'simulation_stopped'})


def simulation_loop():
    """Simulate GPS movement along roads"""
    global gps_data, SIMULATION_ACTIVE, ROAD_NETWORK
    
    sim_lat = gps_data['latitude']
    sim_lon = gps_data['longitude']
    sim_heading = random.uniform(0, 360)
    sim_speed = 0
    
    # Target speed range
    target_speed = random.uniform(30, 60)
    
    while SIMULATION_ACTIVE:
        # Smooth acceleration
        if sim_speed < target_speed:
            sim_speed += 0.5
        elif sim_speed > target_speed:
            sim_speed -= 0.5
        
        # Randomly change target speed
        if random.random() < 0.02:
            target_speed = random.uniform(20, 70)
        
        # Calculate movement
        speed_deg_per_sec = sim_speed / 111.0 / 3600.0
        
        # Move in current heading
        new_lat = sim_lat + speed_deg_per_sec * math.cos(math.radians(sim_heading))
        new_lon = sim_lon + speed_deg_per_sec * math.sin(math.radians(sim_heading))
        
        # Snap to road
        snapped_lat, snapped_lon = snap_to_road(new_lat, new_lon, ROAD_NETWORK)
        
        # If not on road, find nearest road and adjust heading
        if abs(snapped_lat - new_lat) > 0.0001 or abs(snapped_lon - new_lon) > 0.0001:
            # Find nearest road and align with it
            for road in ROAD_NETWORK:
                if road['type'] == 'horizontal':
                    if abs(new_lat - road['lat']) < 0.002:
                        sim_heading = 90 if random.random() > 0.5 else 270
                        snapped_lat = road['lat']
                        break
                else:
                    if abs(new_lon - road['lon']) < 0.002:
                        sim_heading = 0 if random.random() > 0.5 else 180
                        snapped_lon = road['lon']
                        break
        
        sim_lat = snapped_lat
        sim_lon = snapped_lon
        
        # Add slight heading variation for realism
        sim_heading += random.uniform(-2, 2)
        sim_heading = sim_heading % 360
        
        # Update GPS data
        with serial_lock:
            gps_data['latitude'] = sim_lat
            gps_data['longitude'] = sim_lon
            gps_data['altitude'] = 216.0 + random.uniform(-3, 3)
            gps_data['speed'] = round(sim_speed, 1)
            gps_data['heading'] = round(sim_heading, 1)
            gps_data['satellites'] = random.randint(7, 12)
            gps_data['accuracy'] = round(random.uniform(1.5, 4.0), 1)
            gps_data['timestamp'] = datetime.now().isoformat()
            gps_data['status'] = 'simulated'
        
        time.sleep(0.5)


@app.route('/serial/connect', methods=['POST'])
def connect():
    """Connect to specified serial port"""
    data = request.json
    port = data.get('port', SERIAL_PORT)
    success = connect_serial(port)
    return jsonify({'success': success, 'port': port})


@app.route('/serial/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from serial port"""
    disconnect_serial()
    return jsonify({'success': True})


@app.route('/serial/ports')
def get_ports():
    """Get list of available serial ports"""
    ports = list_available_ports()
    return jsonify({'ports': ports})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gps_status': gps_data['status'],
        'timestamp': datetime.now().isoformat()
    })


def start_server():
    """Start the GPS reading thread and Flask server"""
    global read_thread
    
    read_thread = threading.Thread(target=read_serial_data, daemon=True)
    read_thread.start()
    
    print("=" * 60)
    print("ADAS 3D NAVIGATION DASHBOARD - Tesla Style")
    print("=" * 60)
    print(f"Server: http://localhost:5001")
    print(f"Serial Port: {SERIAL_PORT} | Baud: {BAUD_RATE}")
    print("=" * 60)
    print("Features:")
    print("  ✓ 3D buildings with uniform height")
    print("  ✓ Dark Tesla-style map theme")
    print("  ✓ Road-following simulation")
    print("  ✓ Real-time GPS tracking")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_thread.set()
        disconnect_serial()
        time.sleep(1)
