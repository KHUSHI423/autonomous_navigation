"""
=============================================================================
EDGE DRIVE 3D - GPS INTEGRATION TEST SCRIPT
=============================================================================
Comprehensive test script for GPS + OpenStreetMap integration

Tests:
1. GPS hardware detection
2. GPS data reading (simulated and hardware)
3. Coordinate transformations
4. OpenStreetMap visualization
5. Full pipeline integration

Usage:
    python tests/test_gps_integration.py

Author: EdgeDrive3D Team
=============================================================================
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.gps_reader import GPSReceiver, GPSReading, list_gps_devices
from utils.coordinate_transform import (
    CoordinateTransformer, ObjectGPSProjector,
    haversine_distance, bearing, destination_point,
    GPSKalmanFilter
)
from dashboard.components.osm_view import (
    create_osm_map, create_vehicle_marker, create_object_icon_marker,
    create_trajectory_polyline, create_fov_sector, add_legend_to_map
)


# ============================================================================
# TEST COLORS
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


# ============================================================================
# TEST 1: GPS HARDWARE DETECTION
# ============================================================================

def test_gps_hardware_detection():
    """Test GPS hardware detection"""
    print_header("TEST 1: GPS Hardware Detection")
    
    devices = list_gps_devices()
    
    if not devices:
        print_warning("No serial devices found")
        return False
    
    print_info(f"Found {len(devices)} serial device(s):")
    
    gps_devices = []
    for dev in devices:
        marker = "📍" if dev['is_gps'] else "  "
        print(f"  {marker} {dev['device']}: {dev['description']}")
        if dev['is_gps']:
            gps_devices.append(dev)
    
    if gps_devices:
        print_success(f"Found {len(gps_devices)} GPS device(s)")
        return True
    else:
        print_warning("No GPS devices detected (this is OK for simulated mode)")
        return None  # Neutral result


# ============================================================================
# TEST 2: SIMULATED GPS READING
# ============================================================================

def test_simulated_gps():
    """Test simulated GPS reading"""
    print_header("TEST 2: Simulated GPS Reading")
    
    gps = GPSReceiver(simulate=True)
    gps.start()
    
    print_info("Waiting for simulated GPS data...")
    time.sleep(1)
    
    reading = gps.get_position()
    
    if reading and reading.is_valid:
        print_success("Simulated GPS working correctly")
        print(f"\n  Position: {reading.latitude:.6f}, {reading.longitude:.6f}")
        print(f"  Altitude: {reading.altitude:.1f}m")
        print(f"  Speed: {reading.speed:.2f} m/s")
        print(f"  Heading: {reading.heading:.1f}°")
        print(f"  Satellites: {reading.satellites}")
        print(f"  Accuracy: ±{reading.accuracy:.1f}m")
        
        gps.disconnect()
        return True
    else:
        print_error("Failed to get simulated GPS data")
        gps.disconnect()
        return False


# ============================================================================
# TEST 3: COORDINATE TRANSFORMATION
# ============================================================================

def test_coordinate_transformation():
    """Test coordinate transformations"""
    print_header("TEST 3: Coordinate Transformation")
    
    # Create transformer (Bangalore coordinates)
    transformer = CoordinateTransformer(
        base_lat=12.9716,
        base_lon=77.5946,
        base_alt=920.0,
        base_heading=45.0
    )
    
    print_info("Reference point:")
    print(f"  Lat/Lon: {transformer.base_lat*180/3.14159:.6f}, {transformer.base_lon*180/3.14159:.6f}")
    print(f"  Heading: {transformer.base_heading*180/3.14159:.1f}°")
    
    # Test camera to GPS conversion
    print("\nCamera to GPS conversion:")
    test_points = [
        (0, 0, 10),    # 10m straight ahead
        (5, 0, 10),    # 5m right, 10m forward
        (-5, 0, 10),   # 5m left, 10m forward
        (0, 0, 50),    # 50m straight ahead
    ]
    
    all_passed = True
    for x, y, z in test_points:
        lat, lon, alt = transformer.camera_to_gps(x, y, z)
        print(f"  Camera({x:5.1f}, {y:5.1f}, {z:5.1f}) -> GPS({lat:.6f}, {lon:.6f}, {alt:.1f})")
    
    # Test round-trip
    print("\nRound-trip test (Camera -> GPS -> Camera):")
    original = (5.0, 0.0, 20.0)
    lat, lon, alt = transformer.camera_to_gps(*original)
    back = transformer.gps_to_camera(lat, lon, alt)
    
    error = sum(abs(a - b) for a, b in zip(original, (back.x, back.y, back.z)))
    print(f"  Original: {original}")
    print(f"  -> GPS: ({lat:.6f}, {lon:.6f}, {alt:.1f})")
    print(f"  -> Camera: ({back.x:.3f}, {back.y:.3f}, {back.z:.3f})")
    print(f"  Error: {error:.6f}m")
    
    if error < 0.01:
        print_success("Round-trip transformation accurate")
    else:
        print_warning(f"Round-trip error: {error:.6f}m")
        all_passed = False
    
    # Test distance calculation
    print("\nHaversine distance test:")
    lat1, lon1 = 12.9716, 77.5946
    lat2, lon2 = 12.9726, 77.5956
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"  Distance between test points: {dist:.2f}m")
    
    if 140 < dist < 160:  # Should be ~150m
        print_success("Distance calculation correct")
    else:
        print_warning(f"Distance seems incorrect (expected ~150m)")
        all_passed = False
    
    return all_passed


# ============================================================================
# TEST 4: KALMAN FILTER
# ============================================================================

def test_kalman_filter():
    """Test GPS Kalman filter"""
    print_header("TEST 4: Kalman Filter for GPS Smoothing")
    
    kf = GPSKalmanFilter(process_noise=0.1, measurement_noise=5.0)
    
    # True position (static)
    true_lat, true_lon = 12.9716, 77.5946
    
    print_info("Filtering noisy GPS readings (10 iterations):")
    print(f"  True position: {true_lat:.6f}, {true_lon:.6f}\n")
    
    errors_before = []
    errors_after = []
    
    for i in range(10):
        # Simulate noisy measurements
        import random
        noisy_lat = true_lat + random.gauss(0, 0.00005)
        noisy_lon = true_lon + random.gauss(0, 0.00005)
        
        filtered_lat, filtered_lon = kf.filter(noisy_lat, noisy_lon)
        
        error_before = haversine_distance(true_lat, true_lon, noisy_lat, noisy_lon)
        error_after = haversine_distance(true_lat, true_lon, filtered_lat, filtered_lon)
        
        errors_before.append(error_before)
        errors_after.append(error_after)
        
        print(f"  Step {i+1:2d}: Raw error: {error_before:6.2f}m -> Filtered: {error_after:6.2f}m")
    
    avg_before = sum(errors_before) / len(errors_before)
    avg_after = sum(errors_after) / len(errors_after)
    
    print(f"\n  Average error before: {avg_before:.2f}m")
    print(f"  Average error after:  {avg_after:.2f}m")
    print(f"  Improvement: {(1 - avg_after/avg_before)*100:.1f}%")
    
    if avg_after < avg_before:
        print_success("Kalman filter reducing GPS noise")
        return True
    else:
        print_warning("Kalman filter not improving accuracy")
        return False


# ============================================================================
# TEST 5: OPENSTREETMAP VISUALIZATION
# ============================================================================

def test_osm_visualization():
    """Test OpenStreetMap visualization"""
    print_header("TEST 5: OpenStreetMap Visualization")
    
    output_dir = Path("output/gps_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create map
    print_info("Creating OpenStreetMap...")
    m = create_osm_map(
        center_lat=12.9716,
        center_lon=77.5946,
        zoom=16,
        height="600"
    )
    
    # Add vehicle marker
    print_info("Adding vehicle marker...")
    vehicle = create_vehicle_marker(
        12.9716, 77.5946,
        heading=45.0,
        speed=5.0
    )
    vehicle.add_to(m)
    
    # Add object markers
    print_info("Adding object markers...")
    test_objects = [
        {'class': 'car', 'lat': 12.9720, 'lon': 77.5950, 'dist': 15.0, 'conf': 0.92},
        {'class': 'person', 'lat': 12.9718, 'lon': 77.5948, 'dist': 8.0, 'conf': 0.88},
        {'class': 'bicycle', 'lat': 12.9714, 'lon': 77.5944, 'dist': 12.0, 'conf': 0.75},
        {'class': 'bus', 'lat': 12.9722, 'lon': 77.5955, 'dist': 25.0, 'conf': 0.95},
    ]
    
    for obj in test_objects:
        marker = create_object_icon_marker(
            obj['lat'], obj['lon'],
            obj['class'], obj['dist'], obj['conf'],
            heading=45.0
        )
        marker.add_to(m)
    
    # Add trajectory
    print_info("Adding trajectory...")
    trajectory = [
        (12.9710, 77.5940),
        (12.9712, 77.5942),
        (12.9714, 77.5944),
        (12.9716, 77.5946),
    ]
    traj_line = create_trajectory_polyline(trajectory)
    traj_line.add_to(m)
    
    # Add FOV sector
    print_info("Adding FOV sector...")
    fov = create_fov_sector(12.9716, 77.5946, heading=45.0)
    fov.add_to(m)
    
    # Add legend
    print_info("Adding legend...")
    add_legend_to_map(m)
    
    # Save map
    output_path = output_dir / "test_osm_map.html"
    m.save(str(output_path))
    
    print_success(f"Map saved to: {output_path}")
    print_info("Open this file in a web browser to view the interactive map")
    
    return True


# ============================================================================
# TEST 6: FULL GPS PERCEPTION PIPELINE
# ============================================================================

def test_gps_perception_pipeline():
    """Test full GPS perception pipeline"""
    print_header("TEST 6: Full GPS Perception Pipeline")
    
    try:
        from core.perception_engine_gps import GPSPerceptionEngine
    except ImportError as e:
        print_error(f"Could not import GPS perception engine: {e}")
        return False
    
    # Create engine with simulated GPS
    print_info("Initializing GPS Perception Engine...")
    engine = GPSPerceptionEngine({'gps_simulate': True})
    
    # Start GPS
    engine.start_gps()
    
    # Create test frame
    import numpy as np
    import cv2
    
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.putText(test_frame, "GPS Test", (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Process frames
    print_info("Processing test frames...")
    results = []
    
    for i in range(5):
        result = engine.process_frame_gps(test_frame)
        results.append(result)
        
        gps_status = "GPS" if result.gps_reading and result.gps_reading.is_valid else "No GPS"
        obj_count = len(result.objects_with_gps)
        
        print(f"  Frame {i+1}: {gps_status} | Objects: {obj_count} | FPS: {result.fps:.1f}")
        
        time.sleep(0.1)
    
    # Get statistics
    stats = engine.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save trajectory
    output_dir = Path("output/gps_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_path = output_dir / "test_trajectory.json"
    engine.save_trajectory(str(trajectory_path))
    
    # Cleanup
    engine.stop_gps()
    
    print_success("GPS perception pipeline test complete")
    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all GPS integration tests"""
    print_header("EDGE DRIVE 3D - GPS INTEGRATION TESTS")
    
    results = {
        'GPS Hardware Detection': test_gps_hardware_detection(),
        'Simulated GPS': test_simulated_gps(),
        'Coordinate Transformation': test_coordinate_transformation(),
        'Kalman Filter': test_kalman_filter(),
        'OpenStreetMap Visualization': test_osm_visualization(),
        'GPS Perception Pipeline': test_gps_perception_pipeline(),
    }
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = 0
    failed = 0
    neutral = 0
    
    for test_name, result in results.items():
        if result is True:
            print_success(f"{test_name}: PASSED")
            passed += 1
        elif result is False:
            print_error(f"{test_name}: FAILED")
            failed += 1
        else:
            print_warning(f"{test_name}: NEUTRAL (no hardware)")
            neutral += 1
    
    print(f"\n{'='*40}")
    print(f"Total: {passed + failed + neutral} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Neutral: {neutral}")
    print(f"{'='*40}")
    
    if failed == 0:
        print_success("\nAll critical tests passed! GPS integration ready.")
        return 0
    else:
        print_error(f"\n{failed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
