"""
3D Utilities Module
Helper functions for 3D transformations, coordinate systems, and geometry
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import yaml
from loguru import logger

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not installed. Some 3D utilities unavailable.")


# ============================================================================
# Coordinate Transformations
# ============================================================================

def image_to_world_coordinates(
    image_point: Tuple[int, int],
    homography_matrix: np.ndarray,
    ground_height: float = 0.0
) -> Tuple[float, float, float]:
    """
    Transform image coordinates to world coordinates using homography
    
    Args:
        image_point: (x, y) in image coordinates
        homography_matrix: 3x3 homography matrix
        ground_height: Y coordinate for ground plane
        
    Returns:
        (x, y, z) world coordinates
    """
    point = np.array([image_point], dtype=np.float32).reshape(-1, 1, 2)
    world_point = cv2.perspectiveTransform(point, homography_matrix)
    
    return (float(world_point[0, 0, 0]), ground_height, float(world_point[0, 0, 1]))


def world_to_image_coordinates(
    world_point: Tuple[float, float, float],
    inverse_homography: np.ndarray
) -> Tuple[int, int]:
    """
    Transform world coordinates to image coordinates
    
    Args:
        world_point: (x, y, z) world coordinates
        inverse_homography: Inverse of homography matrix
        
    Returns:
        (x, y) in image coordinates
    """
    point = np.array([[world_point[0], world_point[2]]], dtype=np.float32).reshape(-1, 1, 2)
    image_point = cv2.perspectiveTransform(point, inverse_homography)
    
    return (int(image_point[0, 0, 0]), int(image_point[0, 0, 1]))


def create_homography_from_points(
    src_points: List[Tuple[int, int]],
    dst_points: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create homography matrix from source and destination points
    
    Args:
        src_points: 4 points in source (image) coordinates
        dst_points: 4 points in destination (world) coordinates
        
    Returns:
        Tuple of (homography_matrix, inverse_homography)
    """
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    
    homography, _ = cv2.findHomography(src, dst)
    inverse = np.linalg.inv(homography)
    
    return homography, inverse


def create_default_homography(
    image_width: int,
    image_height: int,
    ground_width: float = 50.0,
    ground_depth: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a default homography for typical camera view
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        ground_width: Real-world ground width in meters
        ground_depth: Real-world ground depth in meters
        
    Returns:
        Tuple of (homography_matrix, inverse_homography)
    """
    # Define source points (image corners + perspective)
    margin = 50
    vanishing_y = int(image_height * 0.3)
    
    src_points = [
        (margin, image_height - margin),                    # Bottom left
        (image_width - margin, image_height - margin),      # Bottom right
        (int(image_width * 0.6), vanishing_y),              # Top right
        (int(image_width * 0.4), vanishing_y)               # Top left
    ]
    
    # Define destination points (rectangular ground plane)
    dst_points = [
        (-ground_width / 2, 0),
        (ground_width / 2, 0),
        (ground_width / 2, ground_depth),
        (-ground_width / 2, ground_depth)
    ]
    
    return create_homography_from_points(src_points, dst_points)


# ============================================================================
# Rotation and Orientation
# ============================================================================

def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Create rotation matrix around Y-axis
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Create rotation matrix around X-axis
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Create rotation matrix around Z-axis
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix
    
    Args:
        roll: Rotation around X-axis
        pitch: Rotation around Y-axis
        yaw: Rotation around Z-axis
        
    Returns:
        3x3 rotation matrix
    """
    Rx = rotation_matrix_x(roll)
    Ry = rotation_matrix_y(pitch)
    Rz = rotation_matrix_z(yaw)
    
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return roll, pitch, yaw


def estimate_orientation_from_velocity(
    velocity: Tuple[float, float, float]
) -> float:
    """
    Estimate object orientation from velocity vector
    
    Args:
        velocity: (vx, vy, vz) velocity vector
        
    Returns:
        Yaw angle in radians
    """
    vx, _, vz = velocity
    
    if abs(vx) < 0.01 and abs(vz) < 0.01:
        return 0.0
    
    return np.arctan2(vx, vz)


# ============================================================================
# Geometry Utilities
# ============================================================================

def bounding_box_to_dimensions(
    bbox: Tuple[int, int, int, int],
    distance: float,
    focal_length: float
) -> Tuple[float, float, float]:
    """
    Estimate real-world dimensions from bounding box
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box
        distance: Distance to object
        focal_length: Camera focal length in pixels
        
    Returns:
        (width, height, depth) in meters
    """
    x1, y1, x2, y2 = bbox
    
    pixel_width = x2 - x1
    pixel_height = y2 - y1
    
    # Pinhole camera model
    width = (pixel_width * distance) / focal_length
    height = (pixel_height * distance) / focal_length
    
    # Estimate depth based on typical aspect ratios
    depth = width * 0.5  # Simplified assumption
    
    return width, height, depth


def calculate_distance_3d(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float]
) -> float:
    """
    Calculate 3D Euclidean distance between two points
    
    Args:
        point1: (x1, y1, z1)
        point2: (x2, y2, z2)
        
    Returns:
        Distance
    """
    return np.sqrt(
        (point1[0] - point2[0])**2 +
        (point1[1] - point2[1])**2 +
        (point1[2] - point2[2])**2
    )


def calculate_speed(
    position1: Tuple[float, float, float],
    position2: Tuple[float, float, float],
    time_delta: float
) -> float:
    """
    Calculate speed from two positions and time delta
    
    Args:
        position1: First position (x, y, z)
        position2: Second position (x, y, z)
        time_delta: Time between positions in seconds
        
    Returns:
        Speed in meters per second
    """
    if time_delta <= 0:
        return 0.0
    
    distance = calculate_distance_3d(position1, position2)
    return distance / time_delta


def smooth_position(
    current: Tuple[float, float, float],
    previous: Tuple[float, float, float],
    alpha: float = 0.8
) -> Tuple[float, float, float]:
    """
    Apply exponential smoothing to position
    
    Args:
        current: Current position
        previous: Previous position
        alpha: Smoothing factor (0-1, higher = less smoothing)
        
    Returns:
        Smoothed position
    """
    return (
        alpha * current[0] + (1 - alpha) * previous[0],
        alpha * current[1] + (1 - alpha) * previous[1],
        alpha * current[2] + (1 - alpha) * previous[2]
    )


# ============================================================================
# Model Loading and Manipulation
# ============================================================================

def load_glb_model(filepath: str) -> Optional[Any]:
    """
    Load GLB model file
    
    Args:
        filepath: Path to GLB file
        
    Returns:
        trimesh.Scene or None
    """
    if not TRIMESH_AVAILABLE:
        logger.error("trimesh not available")
        return None
    
    try:
        mesh = trimesh.load(filepath)
        logger.info(f"Loaded model: {filepath}")
        return mesh
    except Exception as e:
        logger.error(f"Error loading model {filepath}: {e}")
        return None


def normalize_model_scale(
    model: Any,
    target_size: float = 2.0
) -> Any:
    """
    Normalize model to target size
    
    Args:
        model: trimesh.Scene or trimesh.Trimesh
        target_size: Target maximum dimension
        
    Returns:
        Scaled model
    """
    if not TRIMESH_AVAILABLE:
        return model
    
    try:
        bounds = model.bounds
        if bounds is None:
            return model
        
        dimensions = bounds[1] - bounds[0]
        max_dim = max(dimensions)
        
        if max_dim > 0:
            scale = target_size / max_dim
            model.apply_scale(scale)
        
        return model
    except Exception as e:
        logger.error(f"Error normalizing model scale: {e}")
        return model


def center_model(model: Any) -> Any:
    """
    Center model at origin
    
    Args:
        model: trimesh.Scene or trimesh.Trimesh
        
    Returns:
        Centered model
    """
    if not TRIMESH_AVAILABLE:
        return model
    
    try:
        model.rezero()
        return model
    except Exception as e:
        logger.error(f"Error centering model: {e}")
        return model


def get_model_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a 3D model
    
    Args:
        filepath: Path to model file
        
    Returns:
        Dictionary with model information
    """
    info = {
        'path': filepath,
        'exists': Path(filepath).exists(),
        'size': 0,
        'dimensions': None,
        'vertices': 0,
        'faces': 0
    }
    
    if not info['exists']:
        return info
    
    info['size'] = Path(filepath).stat().st_size
    
    if TRIMESH_AVAILABLE:
        try:
            model = trimesh.load(filepath)
            
            if hasattr(model, 'bounds') and model.bounds is not None:
                dimensions = model.bounds[1] - model.bounds[0]
                info['dimensions'] = {
                    'width': float(dimensions[0]),
                    'height': float(dimensions[1]),
                    'depth': float(dimensions[2])
                }
            
            if hasattr(model, 'vertices'):
                info['vertices'] = len(model.vertices)
            
            if hasattr(model, 'faces'):
                info['faces'] = len(model.faces)
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
    
    return info


# ============================================================================
# Camera Utilities
# ============================================================================

def create_perspective_camera(
    fov: float = 60.0,
    aspect: float = 16/9,
    near: float = 0.1,
    far: float = 1000.0
) -> Dict[str, float]:
    """
    Create perspective camera parameters
    
    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio (width/height)
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        Dictionary with camera parameters
    """
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2)
    
    return {
        'fov': fov,
        'aspect': aspect,
        'near': near,
        'far': far,
        'focal_length': f
    }


def create_orthographic_camera(
    width: float = 50.0,
    height: float = 50.0,
    near: float = -100.0,
    far: float = 100.0
) -> Dict[str, float]:
    """
    Create orthographic camera parameters
    
    Args:
        width: View volume width
        height: View volume height
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        Dictionary with camera parameters
    """
    return {
        'left': -width / 2,
        'right': width / 2,
        'bottom': -height / 2,
        'top': height / 2,
        'near': near,
        'far': far
    }


def look_at_matrix(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0, 1, 0)
) -> np.ndarray:
    """
    Create look-at view matrix
    
    Args:
        eye: Camera position
        target: Target to look at
        up: Up vector
        
    Returns:
        4x4 view matrix
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    
    # Forward vector
    f = (target - eye)
    f = f / np.linalg.norm(f)
    
    # Right vector
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    
    # Up vector (corrected)
    u = np.cross(r, f)
    
    # View matrix
    view = np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ])
    
    return view


# ============================================================================
# Visualization Helpers
# ============================================================================

def create_arrow_mesh(
    start: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    length: float = 1.0,
    radius: float = 0.1
) -> Optional[Any]:
    """
    Create arrow mesh for visualization
    
    Args:
        start: Start position
        direction: Direction vector
        length: Arrow length
        radius: Arrow radius
        
    Returns:
        trimesh mesh or None
    """
    if not TRIMESH_AVAILABLE:
        return None
    
    try:
        # Normalize direction
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        # Create cylinder for arrow shaft
        shaft_height = length * 0.7
        shaft = trimesh.creation.cylinder(
            radius=radius,
            height=shaft_height
        )
        
        # Create cone for arrow head
        head_height = length * 0.3
        head = trimesh.creation.cone(
            radius=radius * 2,
            height=head_height
        )
        head.apply_translation([0, shaft_height / 2 + head_height / 2, 0])
        
        # Combine
        arrow = trimesh.util.concatenate([shaft, head])
        
        # Rotate to match direction
        up = np.array([0, 1, 0])
        axis = np.cross(up, direction)
        angle = np.arccos(np.dot(up, direction))
        
        if np.linalg.norm(axis) > 0.001:
            rotation = trimesh.transformations.rotation_matrix(
                angle, axis
            )
            arrow.apply_transform(rotation)
        
        # Translate to start position
        arrow.apply_translation(start)
        
        return arrow
        
    except Exception as e:
        logger.error(f"Error creating arrow mesh: {e}")
        return None


def create_trajectory_mesh(
    positions: List[Tuple[float, float, float]],
    radius: float = 0.1,
    color: Tuple[float, float, float] = (1, 0, 0)
) -> Optional[Any]:
    """
    Create trajectory line mesh
    
    Args:
        positions: List of 3D positions
        radius: Line radius
        color: Line color
        
    Returns:
        trimesh mesh or None
    """
    if not TRIMESH_AVAILABLE or len(positions) < 2:
        return None
    
    try:
        # Create cylinders between consecutive points
        cylinders = []
        
        for i in range(len(positions) - 1):
            start = np.array(positions[i])
            end = np.array(positions[i + 1])
            
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length < 0.001:
                continue
            
            cylinder = trimesh.creation.cylinder(
                radius=radius,
                height=length
            )
            
            # Position and rotate
            midpoint = (start + end) / 2
            cylinder.apply_translation(midpoint)
            
            # Rotate to align with direction
            up = np.array([0, 1, 0])
            direction_norm = direction / length
            
            axis = np.cross(up, direction_norm)
            if np.linalg.norm(axis) > 0.001:
                angle = np.arccos(np.dot(up, direction_norm))
                rotation = trimesh.transformations.rotation_matrix(
                    angle, axis
                )
                cylinder.apply_transform(rotation)
            
            # Set color
            cylinder.visual.face_colors = np.tile(
                np.array(color + [1.0]),
                (len(cylinder.faces), 1)
            )
            
            cylinders.append(cylinder)
        
        if cylinders:
            return trimesh.util.concatenate(cylinders)
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating trajectory mesh: {e}")
        return None


# Import cv2 for perspective transform
try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not available. Some functions limited.")


if __name__ == "__main__":
    # Test utilities
    print("Testing 3D Utilities...")
    
    # Test homography
    H, H_inv = create_default_homography(1280, 720)
    print(f"Homography matrix:\n{H}")
    
    # Test rotation
    R = rotation_matrix_y(np.pi / 4)
    print(f"Rotation matrix (45 deg Y):\n{R}")
    
    # Test distance
    d = calculate_distance_3d((0, 0, 0), (1, 1, 1))
    print(f"Distance: {d:.3f}")
    
    # Test speed
    s = calculate_speed((0, 0, 0), (1, 0, 0), 0.1)
    print(f"Speed: {s:.1f} m/s")
