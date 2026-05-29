"""
Automotive Navigation Dashboard
Real Map Data (Mapbox/OpenStreetMap)
CIT Coimbatore Navigation
"""

import threading
import time
import math
import random
import json
import requests
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# CIT Coimbatore - Main Gate
CIT_LAT = 11.0286
CIT_LON = 77.0269

# Destination: Coimbatore Airport
AIRPORT_LAT = 11.0300
AIRPORT_LON = 77.0440

# Global GPS data
gps_data = {
    'latitude': CIT_LAT,
    'longitude': CIT_LON,
    'altitude': 420.0,
    'speed': 0.0,
    'heading': 0.0,
    'satellites': 10,
    'accuracy': 2.0,
    'timestamp': datetime.now().isoformat(),
    'status': 'ready'
}

# Route data
ROUTE_POINTS = []
ROUTE_INDEX = 0
SIMULATION_RUNNING = False
ROUTE_COMPLETED = False

# Map tile cache
MAP_TILES = {}
TILE_RADIUS = 3  # Load tiles within 3 tiles radius


def generate_realistic_route():
    """
    Generate realistic road route from CIT Main Gate to Coimbatore Airport
    Uses interpolated waypoints along actual roads
    """
    points = []
    
    # Real road waypoints from CIT to Airport (via Avanashi Road)
    waypoints = [
        # Start: CIT Main Gate
        {'lat': 11.0286, 'lon': 77.0269, 'speed': 0, 'heading': 90},
        
        # Exit CIT campus onto Avanashi Road
        {'lat': 11.0286, 'lon': 77.0280, 'speed': 25, 'heading': 90},
        {'lat': 11.0286, 'lon': 77.0295, 'speed': 35, 'heading': 90},
        
        # Continue on Avanashi Road (East)
        {'lat': 11.0286, 'lon': 77.0320, 'speed': 40, 'heading': 90},
        {'lat': 11.0286, 'lon': 77.0350, 'speed': 45, 'heading': 90},
        {'lat': 11.0286, 'lon': 77.0380, 'speed': 45, 'heading': 90},
        {'lat': 11.0286, 'lon': 77.0410, 'speed': 40, 'heading': 90},
        
        # Approach airport
        {'lat': 11.0290, 'lon': 77.0425, 'speed': 35, 'heading': 60},
        {'lat': 11.0295, 'lon': 77.0435, 'speed': 30, 'heading': 45},
        
        # End: Coimbatore Airport entrance
        {'lat': 11.0300, 'lon': 77.0440, 'speed': 15, 'heading': 45},
        {'lat': 11.0300, 'lon': 77.0440, 'speed': 0, 'heading': 45}
    ]
    
    # Interpolate between waypoints for smooth movement
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        # Calculate distance
        dist = math.sqrt(
            (end['lat'] - start['lat'])**2 + 
            (end['lon'] - start['lon'])**2
        )
        
        # Number of points based on distance (more points = smoother)
        num_points = max(10, int(dist * 8000))
        
        for j in range(num_points):
            t = j / num_points
            lat = start['lat'] + t * (end['lat'] - start['lat'])
            lon = start['lon'] + t * (end['lon'] - start['lon'])
            speed = start['speed'] + t * (end['speed'] - start['speed'])
            heading = start['heading'] + t * (end['heading'] - start['heading'])
            
            # Add slight variation for realism
            points.append({
                'lat': lat + random.uniform(-0.00002, 0.00002),
                'lon': lon + random.uniform(-0.00002, 0.00002),
                'speed': max(0, speed + random.uniform(-2, 2)),
                'heading': (heading + random.uniform(-2, 2)) % 360
            })
    
    return points


def simulation_loop():
    """Background thread for GPS simulation"""
    global gps_data, ROUTE_INDEX, SIMULATION_RUNNING, ROUTE_POINTS, ROUTE_COMPLETED

    while SIMULATION_RUNNING:
        if ROUTE_POINTS and len(ROUTE_POINTS) > 0:
            point = ROUTE_POINTS[ROUTE_INDEX]

            # Update GPS data
            gps_data['latitude'] = point['lat']
            gps_data['longitude'] = point['lon']
            gps_data['speed'] = point['speed']
            gps_data['heading'] = point['heading']
            gps_data['altitude'] = 420.0 + random.uniform(-2, 2)
            gps_data['satellites'] = random.randint(8, 12)
            gps_data['accuracy'] = random.uniform(1.5, 3.0)
            gps_data['timestamp'] = datetime.now().isoformat()
            gps_data['status'] = 'simulated'

            # Move to next point
            ROUTE_INDEX += 1
            
            # Check if route completed
            if ROUTE_INDEX >= len(ROUTE_POINTS):
                ROUTE_COMPLETED = True
                SIMULATION_RUNNING = False
                gps_data['speed'] = 0
                gps_data['status'] = 'completed'
                print("✓ Route completed! Reached Coimbatore Airport.")
                break

        time.sleep(1.0)  # Update every 1 second


def start_simulation():
    """Start the simulation thread"""
    global SIMULATION_RUNNING, ROUTE_POINTS, ROUTE_INDEX, ROUTE_COMPLETED

    if not SIMULATION_RUNNING:
        ROUTE_POINTS = generate_realistic_route()
        ROUTE_INDEX = 0
        ROUTE_COMPLETED = False
        SIMULATION_RUNNING = True

        sim_thread = threading.Thread(target=simulation_loop, daemon=True)
        sim_thread.start()
        
        print(f"✓ Navigation started: CIT → Airport ({len(ROUTE_POINTS)} points)")
        return True
    return False


def stop_simulation():
    """Stop the simulation"""
    global SIMULATION_RUNNING
    SIMULATION_RUNNING = False
    gps_data['status'] = 'stopped'
    print("⏸ Navigation paused")


@app.route('/')
def index():
    """Serve the navigation dashboard"""
    return render_template('index.html')


@app.route('/gps')
def get_gps():
    """Get current GPS data"""
    return jsonify(gps_data)


@app.route('/gps/start', methods=['POST'])
def start_gps_simulation():
    """Start GPS navigation"""
    started = start_simulation()
    return jsonify({
        'success': started,
        'status': 'started' if started else 'already_running',
        'route_length': len(ROUTE_POINTS)
    })


@app.route('/gps/stop', methods=['POST'])
def stop_gps_simulation():
    """Stop GPS navigation"""
    stop_simulation()
    return jsonify({
        'success': True,
        'status': 'stopped'
    })


@app.route('/gps/reset', methods=['POST'])
def reset_gps():
    """Reset GPS to starting position"""
    global gps_data, ROUTE_INDEX, ROUTE_COMPLETED, SIMULATION_RUNNING

    SIMULATION_RUNNING = False
    ROUTE_INDEX = 0
    ROUTE_COMPLETED = False

    gps_data = {
        'latitude': CIT_LAT,
        'longitude': CIT_LON,
        'altitude': 420.0,
        'speed': 0.0,
        'heading': 0.0,
        'satellites': 0,
        'accuracy': 0.0,
        'timestamp': datetime.now().isoformat(),
        'status': 'reset'
    }
    
    print("↺ Reset to CIT Main Gate")
    return jsonify(gps_data)


@app.route('/gps/serial', methods=['POST'])
def update_serial_gps():
    """Update GPS from serial sensor"""
    global gps_data

    data = request.json
    if data:
        gps_data.update(data)
        gps_data['timestamp'] = datetime.now().isoformat()
        gps_data['status'] = 'serial'

    return jsonify(gps_data)


@app.route('/route')
def get_route():
    """Get the route data"""
    return jsonify({
        'route': ROUTE_POINTS,
        'start': {'lat': CIT_LAT, 'lon': CIT_LON, 'name': 'CIT Main Gate'},
        'end': {'lat': AIRPORT_LAT, 'lon': AIRPORT_LON, 'name': 'Coimbatore Airport'},
        'total_points': len(ROUTE_POINTS)
    })


@app.route('/map/tiles')
def get_map_info():
    """Get map tile configuration"""
    return jsonify({
        'provider': 'OpenStreetMap',
        'style': 'dark',
        'center': {'lat': CIT_LAT, 'lon': CIT_LON},
        'zoom': 17,
        'tile_servers': [
            'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
            'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',
            'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png'
        ]
    })


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'simulation': 'running' if SIMULATION_RUNNING else 'stopped',
        'route_completed': ROUTE_COMPLETED,
        'gps_status': gps_data['status'],
        'timestamp': datetime.now().isoformat()
    })


def init_server():
    """Initialize and start server"""
    print("=" * 60)
    print("  AUTOMOTIVE NAVIGATION DASHBOARD")
    print("  Real Map Data (OpenStreetMap)")
    print("=" * 60)
    print(f"  Route: CIT Main Gate → Coimbatore Airport")
    print(f"  Start: ({CIT_LAT}, {CIT_LON})")
    print(f"  End: ({AIRPORT_LAT}, {AIRPORT_LON})")
    print(f"  Server: http://localhost:5002")
    print("=" * 60)
    print("  Features:")
    print("    ✓ Real OpenStreetMap tiles")
    print("    ✓ Road-based navigation")
    print("    ✓ Low-angle dashboard camera")
    print("    ✓ Wide glowing route path")
    print("    ✓ 3D vehicle arrow")
    print("    ✓ Dynamic map loading")
    print("    ✓ GPS sensor support ready")
    print("=" * 60)
    print("  Controls:")
    print("    - START: Begin navigation to airport")
    print("    - STOP: Pause")
    print("    - RESET: Return to CIT gate")
    print("=" * 60)


if __name__ == '__main__':
    init_server()
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
