"""
=============================================================================
EDGE DRIVE 3D - COORDINATE TRANSFORMATION UTILITIES
=============================================================================
Convert between camera-relative coordinates and GPS (WGS84) coordinates

Supports:
- ENU (East-North-Up) local tangent plane
- WGS84 GPS coordinates
- Camera frame to world frame transformation
- Bearing and distance calculations

Author: EdgeDrive3D Team
=============================================================================
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


# ============================================================================
# CONSTANTS
# ============================================================================

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis (equatorial radius) in meters
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis (polar radius)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ENUPoint:
    """Point in East-North-Up local coordinate system"""
    east: float  # X (meters east)
    north: float  # Y (meters north)
    up: float  # Z (meters up)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.east, self.north, self.up])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ENUPoint':
        return cls(east=arr[0], north=arr[1], up=arr[2])


@dataclass
class CameraPoint:
    """Point in camera coordinate system"""
    x: float  # Right (meters)
    y: float  # Down (meters)
    z: float  # Forward (meters)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CameraPoint':
        return cls(x=arr[0], y=arr[1], z=arr[2])


# ============================================================================
# COORDINATE TRANSFORMER
# ============================================================================

class CoordinateTransformer:
    """
    Transform between camera-relative, ENU, and GPS coordinates
    
    Coordinate Systems:
    1. Camera Frame: X=right, Y=down, Z=forward (right-handed)
    2. ENU Frame: X=east, Y=north, Z=up (right-handed)
    3. GPS Frame: latitude, longitude, altitude (WGS84)
    
    Usage:
        transformer = CoordinateTransformer(
            base_lat=12.9716,
            base_lon=77.5946,
            base_alt=920.0,
            base_heading=45.0
        )
        
        # Camera-relative to GPS
        lat, lon, alt = transformer.camera_to_gps(x=2.0, y=0.0, z=10.0)
        
        # GPS to camera-relative
        x, y, z = transformer.gps_to_camera(lat, lon, alt)
    """
    
    def __init__(
        self,
        base_lat: float,
        base_lon: float,
        base_alt: float = 0.0,
        base_heading: float = 0.0,
        base_pitch: float = 0.0,
        base_roll: float = 0.0
    ):
        """
        Initialize transformer with reference point
        
        Args:
            base_lat: Reference latitude (degrees)
            base_lon: Reference longitude (degrees)
            base_alt: Reference altitude (meters)
            base_heading: Vehicle heading (degrees, 0=North, clockwise)
            base_pitch: Vehicle pitch (degrees, 0=level, positive=up)
            base_roll: Vehicle roll (degrees, 0=level, positive=right)
        """
        self.base_lat = math.radians(base_lat)
        self.base_lon = math.radians(base_lon)
        self.base_alt = base_alt
        self.base_heading = math.radians(base_heading)
        self.base_pitch = math.radians(base_pitch)
        self.base_roll = math.radians(base_roll)
        
        # Pre-compute rotation matrices
        self._compute_rotation_matrices()
    
    def _compute_rotation_matrices(self):
        """Pre-compute rotation matrices for efficiency"""
        # Heading rotation (around Y axis, in ENU plane)
        ch, sh = math.cos(self.base_heading), math.sin(self.base_heading)
        self.R_heading = np.array([
            [ch, -sh, 0],
            [sh, ch, 0],
            [0, 0, 1]
        ])
        
        self.R_heading_inv = self.R_heading.T
        
        # Pitch rotation (around X axis)
        cp, sp = math.cos(self.base_pitch), math.sin(self.base_pitch)
        self.R_pitch = np.array([
            [1, 0, 0],
            [0, cp, -sp],
            [0, sp, cp]
        ])
        
        # Roll rotation (around Z axis)
        cr, sr = math.cos(self.base_roll), math.sin(self.base_roll)
        self.R_roll = np.array([
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1]
        ])
        
        # Combined vehicle rotation
        self.R_vehicle = self.R_heading @ self.R_pitch @ self.R_roll
        self.R_vehicle_inv = self.R_vehicle.T
    
    def camera_to_enu(self, x: float, y: float, z: float) -> ENUPoint:
        """
        Convert camera-relative coordinates to ENU
        
        Camera frame: X=right, Y=down, Z=forward
        ENU frame: X=east, Y=north, Z=up
        
        Transformation:
          1. Camera to vehicle (account for mounting)
          2. Vehicle to ENU (account for heading)
        """
        # Camera point
        p_camera = np.array([x, y, z])
        
        # Camera to vehicle (camera is typically mounted facing forward)
        # Camera Y (down) -> Vehicle Z (up)
        # Camera Z (forward) -> Vehicle Y (north/forward)
        # Camera X (right) -> Vehicle X (east/right)
        p_vehicle = np.array([
            p_camera[0],  # X stays X
            p_camera[2],  # Z becomes Y (forward)
            -p_camera[1]  # -Y becomes Z (up)
        ])
        
        # Vehicle to ENU (rotate by heading)
        p_enu = self.R_heading @ p_vehicle
        
        return ENUPoint.from_array(p_enu)
    
    def enu_to_camera(self, enu: ENUPoint) -> CameraPoint:
        """Convert ENU coordinates to camera-relative"""
        # ENU to vehicle
        p_vehicle = self.R_heading_inv @ enu.to_array()
        
        # Vehicle to camera (inverse of above)
        p_camera = np.array([
            p_vehicle[0],  # X stays X
            -p_vehicle[2],  # -Z becomes Y (down)
            p_vehicle[1]  # Y becomes Z (forward)
        ])
        
        return CameraPoint.from_array(p_camera)
    
    def enu_to_gps(
        self,
        east: float,
        north: float,
        up: float
    ) -> Tuple[float, float, float]:
        """
        Convert ENU coordinates to GPS (WGS84)
        
        Uses the equirectangular approximation (good for < 100km)
        
        Args:
            east: Distance east from reference (meters)
            north: Distance north from reference (meters)
            up: Height above reference (meters)
        
        Returns:
            (latitude, longitude, altitude) in degrees/meters
        """
        # Earth radius at reference latitude
        R = WGS84_A
        
        # Convert meters to degrees
        lat_offset = north / (R * math.cos(math.radians(0)))  # ~111,320 m/degree
        lon_offset = east / (R * math.cos(self.base_lat))
        
        # Apply offsets
        new_lat = self.base_lat + lat_offset
        new_lon = self.base_lon + lon_offset
        new_alt = self.base_alt + up
        
        return math.degrees(new_lat), math.degrees(new_lon), new_alt
    
    def gps_to_enu(
        self,
        latitude: float,
        longitude: float,
        altitude: float
    ) -> ENUPoint:
        """
        Convert GPS coordinates to ENU
        
        Args:
            latitude: GPS latitude (degrees)
            longitude: GPS longitude (degrees)
            altitude: GPS altitude (meters)
        
        Returns:
            ENUPoint (east, north, up in meters)
        """
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        
        # Differences from reference
        d_lat = lat_rad - self.base_lat
        d_lon = lon_rad - self.base_lon
        
        # Convert to meters
        R = WGS84_A
        north = d_lat * R
        east = d_lon * R * math.cos(self.base_lat)
        up = altitude - self.base_alt
        
        return ENUPoint(east=east, north=north, up=up)
    
    def camera_to_gps(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[float, float, float]:
        """
        Convert camera-relative coordinates to GPS
        
        Args:
            x: Right offset (meters)
            y: Down offset (meters)
            z: Forward offset (meters)
        
        Returns:
            (latitude, longitude, altitude)
        """
        # Camera -> ENU
        enu = self.camera_to_enu(x, y, z)
        
        # ENU -> GPS
        return self.enu_to_gps(enu.east, enu.north, enu.up)
    
    def gps_to_camera(
        self,
        latitude: float,
        longitude: float,
        altitude: float
    ) -> CameraPoint:
        """
        Convert GPS coordinates to camera-relative
        
        Args:
            latitude: GPS latitude (degrees)
            longitude: GPS longitude (degrees)
            altitude: GPS altitude (meters)
        
        Returns:
            CameraPoint (x=right, y=down, z=forward in meters)
        """
        # GPS -> ENU
        enu = self.gps_to_enu(latitude, longitude, altitude)
        
        # ENU -> Camera
        return self.enu_to_camera(enu)
    
    def update_base_position(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        heading: float = None
    ):
        """
        Update the reference position (for moving vehicles)
        
        Args:
            latitude: New reference latitude
            longitude: New reference longitude
            altitude: New reference altitude
            heading: New heading (optional, keeps old if None)
        """
        self.base_lat = math.radians(latitude)
        self.base_lon = math.radians(longitude)
        self.base_alt = altitude
        
        if heading is not None:
            self.base_heading = math.radians(heading)
            self._compute_rotation_matrices()


# ============================================================================
# OBJECT GPS PROJECTOR
# ============================================================================

class ObjectGPSProjector:
    """
    Project detected objects from camera frame to GPS coordinates
    
    This handles the full pipeline:
    1. 2D bbox + depth -> 3D camera coordinates
    2. 3D camera coordinates -> ENU
    3. ENU -> GPS
    """
    
    def __init__(self, transformer: CoordinateTransformer):
        self.transformer = transformer
    
    def project_object(
        self,
        bbox_2d: Tuple[int, int, int, int],
        depth: float,
        camera_intrinsics: dict,
        image_size: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Project a detected object to GPS coordinates
        
        Args:
            bbox_2d: (x1, y1, x2, y2) bounding box in pixels
            depth: Estimated distance to object (meters)
            camera_intrinsics: {fx, fy, cx, cy} camera parameters
            image_size: (width, height) of image
        
        Returns:
            (latitude, longitude, altitude) of object
        """
        # Get bbox center
        x1, y1, x2, y2 = bbox_2d
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Camera intrinsics
        fx = camera_intrinsics.get('fx', image_size[0])
        fy = camera_intrinsics.get('fy', image_size[1])
        cx = camera_intrinsics.get('cx', image_size[0] / 2)
        cy = camera_intrinsics.get('cy', image_size[1] / 2)
        
        # Convert to camera coordinates
        # Z = depth (forward)
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        x_cam = (center_u - cx) * depth / fx
        y_cam = (center_v - cy) * depth / fy
        z_cam = depth
        
        # Project to GPS
        return self.transformer.camera_to_gps(x_cam, y_cam, z_cam)
    
    def project_multiple_objects(
        self,
        objects: List[dict],
        camera_intrinsics: dict,
        image_size: Tuple[int, int]
    ) -> List[dict]:
        """
        Project multiple detected objects to GPS
        
        Args:
            objects: List of detected objects with bbox_2d and distance
            camera_intrinsics: Camera parameters
            image_size: Image dimensions
        
        Returns:
            Objects with added gps_coordinates field
        """
        for obj in objects:
            if 'bbox_2d' in obj and 'distance' in obj:
                lat, lon, alt = self.project_object(
                    obj['bbox_2d'],
                    obj['distance'],
                    camera_intrinsics,
                    image_size
                )
                obj['gps_coordinates'] = {
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt
                }
        
        return objects


# ============================================================================
# GEODETIC UTILITIES
# ============================================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate great-circle distance between two GPS points
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
    
    Returns:
        Distance in meters
    """
    R = WGS84_A
    
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate initial bearing from point 1 to point 2
    
    Args:
        lat1, lon1: Start point (degrees)
        lat2, lon2: End point (degrees)
    
    Returns:
        Bearing in degrees (0-360, North=0, clockwise)
    """
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    
    θ = math.atan2(x, y)
    
    return (math.degrees(θ) + 360) % 360


def destination_point(
    lat: float, lon: float,
    bearing_deg: float, distance: float
) -> Tuple[float, float]:
    """
    Calculate destination point given start, bearing, and distance
    
    Args:
        lat, lon: Start point (degrees)
        bearing_deg: Bearing (degrees, 0=North)
        distance: Distance (meters)
    
    Returns:
        (latitude, longitude) of destination
    """
    R = WGS84_A
    φ = math.radians(lat)
    λ = math.radians(lon)
    θ = math.radians(bearing_deg)
    δ = distance / R
    
    φ2 = math.asin(math.sin(φ) * math.cos(δ) + math.cos(φ) * math.sin(δ) * math.cos(θ))
    λ2 = λ + math.atan2(
        math.sin(θ) * math.sin(δ) * math.cos(φ),
        math.cos(δ) - math.sin(φ) * math.sin(φ2)
    )
    
    return math.degrees(φ2), math.degrees(λ2)


def interpolate_gps_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int
) -> List[Tuple[float, float]]:
    """
    Interpolate between two GPS points
    
    Args:
        start: (lat, lon) start point
        end: (lat, lon) end point
        num_points: Number of interpolated points
    
    Returns:
        List of (lat, lon) points
    """
    points = [start]
    
    for i in range(1, num_points):
        t = i / num_points
        lat = start[0] + t * (end[0] - start[0])
        lon = start[1] + t * (end[1] - start[1])
        points.append((lat, lon))
    
    points.append(end)
    return points


# ============================================================================
# KALMAN FILTER FOR GPS SMOOTHING
# ============================================================================

class GPSKalmanFilter:
    """
    Simple Kalman filter for smoothing GPS readings
    
    State: [lat, lon, lat_vel, lon_vel]
    Measurement: [lat, lon]
    """
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 5.0):
        """
        Initialize Kalman filter
        
        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance (meters^2)
        """
        # State vector [lat, lon, lat_vel, lon_vel]
        self.x = np.zeros(4)
        
        # State transition matrix (constant velocity model)
        dt = 0.1  # 10 Hz
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Covariance matrix
        self.P = np.eye(4) * 1000
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with new measurement
        
        Args:
            measurement: [latitude, longitude]
        
        Returns:
            Updated state estimate [latitude, longitude]
        """
        if not self.initialized:
            self.x[:2] = measurement
            self.initialized = True
            return measurement
        
        # Innovation
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:2]
    
    def filter(self, lat: float, lon: float) -> Tuple[float, float]:
        """Filter a GPS reading"""
        self.predict()
        filtered = self.update(np.array([lat, lon]))
        return filtered[0], filtered[1]


# ============================================================================
# MAIN (Test)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Coordinate Transformation Test")
    print("=" * 60)
    
    # Create transformer (Bangalore coordinates as example)
    transformer = CoordinateTransformer(
        base_lat=12.9716,
        base_lon=77.5946,
        base_alt=920.0,
        base_heading=45.0  # Northeast
    )
    
    print("\nReference point:")
    print(f"  Lat/Lon: {math.degrees(transformer.base_lat):.6f}, {math.degrees(transformer.base_lon):.6f}")
    print(f"  Heading: {math.degrees(transformer.base_heading):.1f}°")
    
    # Test camera to GPS
    print("\nCamera to GPS conversion:")
    test_points = [
        (0, 0, 10),    # 10m straight ahead
        (5, 0, 10),    # 5m right, 10m forward
        (-5, 0, 10),   # 5m left, 10m forward
        (0, 0, 50),    # 50m straight ahead
    ]
    
    for x, y, z in test_points:
        lat, lon, alt = transformer.camera_to_gps(x, y, z)
        print(f"  Camera({x:5.1f}, {y:5.1f}, {z:5.1f}) -> GPS({lat:.6f}, {lon:.6f}, {alt:.1f})")
    
    # Test distance calculation
    print("\nHaversine distance test:")
    lat1, lon1 = 12.9716, 77.5946
    lat2, lon2 = 12.9726, 77.5956  # ~150m northeast
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"  Distance: {dist:.2f}m")
    
    # Test bearing calculation
    print("\nBearing test:")
    brg = bearing(lat1, lon1, lat2, lon2)
    print(f"  Bearing: {brg:.1f}°")
    
    # Test Kalman filter
    print("\nKalman filter test:")
    kf = GPSKalmanFilter()
    true_lat, true_lon = 12.9716, 77.5946
    
    for i in range(10):
        # Simulate noisy measurements
        noisy_lat = true_lat + np.random.randn() * 0.00005
        noisy_lon = true_lon + np.random.randn() * 0.00005
        
        filtered_lat, filtered_lon = kf.filter(noisy_lat, noisy_lon)
        
        error_before = haversine_distance(true_lat, true_lon, noisy_lat, noisy_lon)
        error_after = haversine_distance(true_lat, true_lon, filtered_lat, filtered_lon)
        
        print(f"  Step {i+1}: Error {error_before:.2f}m -> {error_after:.2f}m")
