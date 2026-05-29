"""
ADAS Visualization Dashboard - Flask Server
Reads GPS data from ESP32 via Serial OR WiFi
Fixed: Properly handles WiFi GPS updates without serial override
"""

import serial
import serial.tools.list_ports
import threading
import json
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global GPS data storage
gps_data = {
    'latitude': 28.6139,  # Default: New Delhi
    'longitude': 77.2090,
    'altitude': 216.0,
    'speed': 0.0,
    'heading': 0.0,
    'satellites': 0,
    'accuracy': 0.0,
    'timestamp': datetime.now().isoformat(),
    'status': 'disconnected'
}

# Serial configuration
SERIAL_PORT = 'COM3'  # Change to your ESP32 port
BAUD_RATE = 9600
serial_conn = None
serial_lock = threading.Lock()
read_thread = None
stop_thread = threading.Event()

# GPS source tracking
gps_source = 'none'  # 'none', 'serial', 'wifi'
last_update_time = time.time()


def list_available_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def parse_nmea_line(line):
    """Parse NMEA GPS sentences"""
    global gps_data, gps_source

    line = line.strip()
    if not line.startswith('$'):
        return

    parts = line.split(',')

    # Parse $GPGGA sentence
    if parts[0] in ['$GPGGA', '$GNGGA']:
        try:
            if parts[1]:  # Time
                gps_data['timestamp'] = datetime.now().isoformat()
            if parts[2]:  # Latitude
                lat = float(parts[2])
                if parts[3] == 'S':
                    lat = -lat
                gps_data['latitude'] = lat
            if parts[4]:  # Longitude
                lon = float(parts[4])
                if parts[5] == 'W':
                    lon = -lon
                gps_data['longitude'] = lon
            if parts[6]:  # Fix quality
                fix = int(parts[6])
                gps_data['status'] = 'active' if fix > 0 else 'no_fix'
            if parts[7]:  # Satellites
                gps_data['satellites'] = int(parts[7])
            if parts[8]:  # HDOP
                gps_data['accuracy'] = float(parts[8])
            if parts[9]:  # Altitude
                gps_data['altitude'] = float(parts[9])
            gps_source = 'serial'
            last_update_time = time.time()
        except (ValueError, IndexError):
            pass

    # Parse $GPRMC sentence
    elif parts[0] in ['$GPRMC', '$GNRMC']:
        try:
            if parts[7]:  # Speed in knots
                speed_knots = float(parts[7])
                gps_data['speed'] = round(speed_knots * 1.852, 2)  # Convert to km/h
            if parts[8]:  # Course/heading
                gps_data['heading'] = float(parts[8])
            gps_source = 'serial'
            last_update_time = time.time()
        except (ValueError, IndexError):
            pass

    # Parse custom JSON format from ESP32 (if used)
    elif line.startswith('{'):
        try:
            data = json.loads(line)
            gps_data.update(data)
            gps_data['timestamp'] = datetime.now().isoformat()
            gps_data['status'] = 'active'
            gps_source = 'serial'
            last_update_time = time.time()
        except json.JSONDecodeError:
            pass


def read_serial_data():
    """Background thread to read GPS data from serial port"""
    global serial_conn, gps_data, gps_source

    while not stop_thread.is_set():
        try:
            with serial_lock:
                if serial_conn and serial_conn.is_open:
                    if serial_conn.in_waiting > 0:
                        line = serial_conn.readline().decode('utf-8', errors='ignore')
                        parse_nmea_line(line)
                # FIXED: NEVER reset gps_data status when WiFi is the source
                # Only show disconnected if no updates from any source for 10 seconds
                elif gps_source == 'serial' and time.time() - last_update_time > 10:
                    gps_data['status'] = 'disconnected'
        except Exception as e:
            # FIXED: Don't overwrite status if WiFi is active
            if gps_source != 'wifi' and gps_source != 'serial':
                gps_data['status'] = f'error: {str(e)}'

        time.sleep(0.1)  # Small delay to prevent CPU hogging


def connect_serial(port):
    """Connect to serial port"""
    global serial_conn, gps_data, gps_source

    try:
        with serial_lock:
            if serial_conn and serial_conn.is_open:
                serial_conn.close()

            serial_conn = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for ESP32 to initialize
            gps_data['status'] = 'connected'
            gps_source = 'serial'
            print(f"✓ Connected to {port}")
            return True
    except Exception as e:
        gps_data['status'] = f'connection_error: {str(e)}'
        print(f"✗ Connection failed: {e}")
        return False


def disconnect_serial():
    """Disconnect from serial port"""
    global serial_conn, gps_data

    with serial_lock:
        if serial_conn and serial_conn.is_open:
            serial_conn.close()
            if gps_source != 'wifi':
                gps_data['status'] = 'disconnected'
            print("Disconnected from serial")


@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')


@app.route('/gps')
def get_gps():
    """API endpoint to get current GPS data"""
    return jsonify(gps_data)


@app.route('/gps/update', methods=['POST'])
def update_gps():
    """
    Update GPS data from ESP32 WiFi
    FIXED: Properly updates gps_data and sets status to active
    """
    global gps_data, gps_source, last_update_time

    # Check if request has JSON data
    if not request.is_json:
        print("✗ GPS Update: No JSON data received")
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    
    if not data:
        print("✗ GPS Update: Empty JSON")
        return jsonify({'error': 'Empty JSON'}), 400

    # FIXED: Print received data for debugging
    print(f"✓ GPS UPDATE (WiFi): {data}")

    # FIXED: Update gps_data with received data
    with serial_lock:
        # Update all fields from ESP32
        if 'latitude' in data:
            gps_data['latitude'] = float(data['latitude'])
        if 'longitude' in data:
            gps_data['longitude'] = float(data['longitude'])
        if 'speed' in data:
            gps_data['speed'] = float(data['speed'])
        if 'altitude' in data:
            gps_data['altitude'] = float(data['altitude'])
        if 'satellites' in data:
            gps_data['satellites'] = int(data['satellites'])
        if 'heading' in data:
            gps_data['heading'] = float(data['heading'])
        if 'accuracy' in data:
            gps_data['accuracy'] = float(data['accuracy'])

        # FIXED: Set status to active and update timestamp
        gps_data['status'] = 'active'
        gps_data['timestamp'] = datetime.now().isoformat()
        gps_source = 'wifi'
        last_update_time = time.time()

    print(f"  ✓ Status: {gps_data['status']}")
    print(f"  ✓ Source: {gps_source}")
    print(f"  ✓ Position: {gps_data['latitude']}, {gps_data['longitude']}")

    return jsonify({
        'success': True,
        'status': 'updated',
        'gps': gps_data
    })


@app.route('/gps/reset', methods=['POST'])
def reset_gps():
    """Reset GPS to default location"""
    global gps_data, gps_source

    with serial_lock:
        gps_data = {
            'latitude': 28.6139,
            'longitude': 77.2090,
            'altitude': 216.0,
            'speed': 0.0,
            'heading': 0.0,
            'satellites': 0,
            'accuracy': 0.0,
            'timestamp': datetime.now().isoformat(),
            'status': 'reset'
        }
        gps_source = 'none'
    return jsonify(gps_data)


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
        'gps_source': gps_source,
        'last_update_seconds_ago': time.time() - last_update_time,
        'timestamp': datetime.now().isoformat()
    })


def start_server():
    """Start the GPS reading thread and Flask server"""
    global read_thread

    # Start serial reading thread
    read_thread = threading.Thread(target=read_serial_data, daemon=True)
    read_thread.start()

    print("=" * 60)
    print("ADAS Visualization Dashboard")
    print("=" * 60)
    print(f"Server: http://0.0.0.0:5000")
    print(f"Serial Port: {SERIAL_PORT} | Baud: {BAUD_RATE}")
    print("=" * 60)
    print("GPS Sources:")
    print("  - Serial: ESP32 via USB")
    print("  - WiFi: ESP32 via HTTP POST")
    print("=" * 60)
    print("WiFi GPS Update Endpoint:")
    print("  POST http://<server-ip>:5000/gps/update")
    print("  JSON: {latitude, longitude, speed, altitude, satellites}")
    print("=" * 60)
    print("Available endpoints:")
    print("  GET  /           - Dashboard UI")
    print("  GET  /gps        - Current GPS data")
    print("  POST /gps/update - Update GPS (WiFi)")
    print("  GET  /serial/ports - List serial ports")
    print("  POST /serial/connect - Connect to serial")
    print("  POST /serial/disconnect - Disconnect serial")
    print("=" * 60)

    # FIXED: Run on all network interfaces (0.0.0.0)
    # This allows ESP32 WiFi requests from any IP
    app.run(
        host='0.0.0.0',  # Listen on all network interfaces
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_thread.set()
        disconnect_serial()
        time.sleep(1)
