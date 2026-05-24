"""
Scene Mapper Module
Maps 2D detections from video to 3D world coordinates
Handles perspective transformation and ground plane projection
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
import cv2
from loguru import logger


class SceneObject:
    """Represents an object in 3D scene"""
    
    def __init__(self, track_id: int, class_name: str, position_3d: Tuple[float, float, float],
                 rotation: float = 0.0, scale: float = 1.0, model_path: str = ""):
        self.track_id = track_id
        self.class_name = class_name
        self.position_3d = position_3d  # (x, y, z) in world coordinates
        self.rotation = rotation  # Y-axis rotation in radians
        self.scale = scale
        self.model_path = model_path
        self.velocity = (0.0, 0.0, 0.0)
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'position': {
                'x': float(self.position_3d[0]),
                'y': float(self.position_3d[1]),
                'z': float(self.position_3d[2])
            },
            'rotation': float(self.rotation),
            'scale': float(self.scale),
            'model_path': self.model_path,
            'velocity': {
                'x': float(self.velocity[0]),
                'y': float(self.velocity[1]),
                'z': float(self.velocity[2])
            },
            'is_active': self.is_active
        }


class SceneMapper:
    """
    Maps 2D image coordinates to 3D world coordinates
    
    Uses homography transformation for ground plane mapping
    and perspective projection for height estimation
    """
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize scene mapper
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        
        # Scene settings
        scene_config = self.config['scene']
        self.ground_width = scene_config['ground']['width']
        self.ground_depth = scene_config['ground']['depth']
        
        # Camera calibration (homography matrix)
        self.homography_matrix: Optional[np.ndarray] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        
        # Image dimensions (set during calibration)
        self.image_width = 1280
        self.image_height = 720
        
        # Scale factor (pixels to meters)
        self.pixels_per_meter = 50.0
        
        # Origin point in image (bottom center typically)
        self.origin_image = (self.image_width // 2, self.image_height - 50)
        
        logger.info("SceneMapper initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'scene': {
                'ground': {
                    'width': 50,
                    'depth': 100,
                    'grid_size': 1,
                    'grid_color': '#444444'
                },
                'view': {
                    'default': 'perspective',
                    'perspective': {
                        'position': [20, 15, 20],
                        'look_at': [0, 0, 0],
                        'fov': 60
                    },
                    'top_down': {
                        'position': [0, 50, 0],
                        'look_at': [0, 0, 0],
                        'orthographic': True
                    }
                }
            },
            'analytics': {
                'speed': {
                    'pixels_per_meter': 50
                }
            }
        }
    
    def set_homography(self, src_points: List[Tuple[int, int]], 
                       dst_points: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Set homography matrix from source points
        
        Args:
            src_points: 4 points in image coordinates (corners of ground plane)
            dst_points: 4 points in world coordinates (optional, creates rectangular ground plane)
        """
        src = np.array(src_points, dtype=np.float32)
        
        if dst_points is None:
            # Create default destination (rectangular ground plane)
            dst = np.array([
                [-self.ground_width / 2, 0],
                [self.ground_width / 2, 0],
                [self.ground_width / 2, self.ground_depth],
                [-self.ground_width / 2, self.ground_depth]
            ], dtype=np.float32)
        else:
            dst = np.array(dst_points, dtype=np.float32)
        
        # Compute homography matrix
        self.homography_matrix, _ = cv2.findHomography(src, dst)
        
        logger.info(f"Homography matrix set: {self.homography_matrix is not None}")
    
    def set_camera_calibration(self, camera_matrix: np.ndarray, 
                               dist_coeffs: np.ndarray) -> None:
        """
        Set camera intrinsic parameters
        
        Args:
            camera_matrix: 3x3 camera matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def auto_calibrate_homography(self, frame_width: int, frame_height: int,
                                   vanishing_point: Optional[Tuple[int, int]] = None) -> None:
        """
        Automatically estimate homography based on typical camera view
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            vanishing_point: Vanishing point in image (optional)
        """
        self.image_width = frame_width
        self.image_height = frame_height
        
        # Default vanishing point (center top)
        if vanishing_point is None:
            vp = (frame_width // 2, int(frame_height * 0.3))
        else:
            vp = vanishing_point
        
        # Define ground plane quadrilateral in image
        # Bottom corners are full width, top corners converge toward vanishing point
        margin = 50
        src_points = [
            (margin, frame_height - margin),           # Bottom left
            (frame_width - margin, frame_height - margin),  # Bottom right
            (int(vp[0] + (frame_width / 2 - margin) * 0.3), vp[1] + int(frame_height * 0.4)),  # Top right
            (int(vp[0] - (frame_width / 2 - margin) * 0.3), vp[1] + int(frame_height * 0.4))   # Top left
        ]
        
        self.set_homography(src_points)
        logger.info(f"Auto-calibrated homography for {frame_width}x{frame_height} frame")
    
    def image_to_world(self, image_point: Tuple[int, int]) -> Tuple[float, float]:
        """
        Transform image point to world coordinates (ground plane)
        
        Args:
            image_point: (x, y) in image coordinates
            
        Returns:
            (x, z) in world coordinates (y=0 for ground plane)
        """
        if self.homography_matrix is None:
            self.auto_calibrate_homography(self.image_width, self.image_height)
        
        point = np.array([image_point], dtype=np.float32).reshape(-1, 1, 2)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return (float(world_point[0, 0, 0]), float(world_point[0, 0, 1]))
    
    def bbox_to_world_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """
        Convert bounding box to 3D world position
        
        Uses bottom center of bbox as ground contact point
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (x, y, z) world position (y is height, typically 0 for ground objects)
        """
        x1, y1, x2, y2 = bbox
        
        # Bottom center is the ground contact point
        bottom_center = ((x1 + x2) // 2, y2)
        
        # Transform to world coordinates
        world_x, world_z = self.image_to_world(bottom_center)
        
        # Estimate height from bbox height
        bbox_height = y2 - y1
        estimated_height = bbox_height / self.pixels_per_meter
        
        # Y coordinate is half the estimated height (center of object)
        world_y = estimated_height / 2
        
        return (world_x, world_y, world_z)
    
    def map_detection_to_scene(self, detection: Any, model_path: str = "") -> SceneObject:
        """
        Map a detection to 3D scene object
        
        Args:
            detection: Detection object with bbox
            model_path: Path to 3D model GLB file
            
        Returns:
            SceneObject with 3D position
        """
        # Get 3D position
        position_3d = self.bbox_to_world_position(detection.bbox)
        
        # Estimate scale from distance (objects farther away are smaller in image)
        distance = np.sqrt(position_3d[0]**2 + position_3d[2]**2)
        scale = 1.0 + distance * 0.01  # Slight scale adjustment
        
        # Estimate rotation from velocity direction (if available)
        rotation = 0.0
        if hasattr(detection, 'velocity'):
            vx, vz = detection.velocity
            if abs(vx) > 0.1 or abs(vz) > 0.1:
                rotation = np.arctan2(vx, vz)
        
        # Create scene object
        scene_obj = SceneObject(
            track_id=getattr(detection, 'track_id', 0),
            class_name=detection.class_name,
            position_3d=position_3d,
            rotation=rotation,
            scale=scale,
            model_path=model_path
        )
        
        # Set velocity
        if hasattr(detection, 'velocity'):
            vx, vy = detection.velocity
            # Convert pixel velocity to world velocity
            world_vx, world_vz = vx / self.pixels_per_meter, vy / self.pixels_per_meter
            scene_obj.velocity = (world_vx, 0, world_vz)
        
        return scene_obj
    
    def map_track_to_scene(self, track: Any, model_path: str = "") -> SceneObject:
        """
        Map a tracked object to 3D scene object
        
        Args:
            track: TrackedObject from tracker
            model_path: Path to 3D model GLB file
            
        Returns:
            SceneObject with 3D position and velocity
        """
        # Get 3D position from bottom center
        position_3d = self.bbox_to_world_position(track.bbox)
        
        # Estimate scale based on object class and distance
        scale = self._estimate_scale(track.class_name, position_3d)
        
        # Estimate rotation from movement direction
        rotation = 0.0
        if len(track.center_history) > 1:
            positions = list(track.center_history)
            if len(positions) >= 2:
                dx = positions[-1][0] - positions[-2][0]
                dy = positions[-1][1] - positions[-2][1]
                if abs(dx) > 1 or abs(dy) > 1:
                    rotation = np.arctan2(dx, dy)
        
        # Convert velocity to world coordinates
        velocity_world = (0.0, 0.0, 0.0)
        if hasattr(track, 'velocity'):
            vx, vy = track.velocity
            velocity_world = (
                vx / self.pixels_per_meter,
                0,
                vy / self.pixels_per_meter
            )
        
        # Create scene object
        scene_obj = SceneObject(
            track_id=track.track_id,
            class_name=track.class_name,
            position_3d=position_3d,
            rotation=rotation,
            scale=scale,
            model_path=model_path
        )
        scene_obj.velocity = velocity_world
        
        return scene_obj
    
    def _estimate_scale(self, class_name: str, position_3d: Tuple[float, float, float]) -> float:
        """
        Estimate object scale based on class and position
        
        Args:
            class_name: Object class name
            position_3d: (x, y, z) position
            
        Returns:
            Scale factor
        """
        # Base scales for different classes
        base_scales = {
            'person': 1.0,
            'bicycle': 1.2,
            'motorcycle': 1.5,
            'car': 2.0,
            'bus': 3.5,
            'truck': 3.0,
            'bicycle': 1.0
        }
        
        base = base_scales.get(class_name, 1.0)
        
        # Distance-based adjustment
        distance = np.sqrt(position_3d[0]**2 + position_3d[2]**2)
        distance_factor = 1.0 + distance * 0.005
        
        return base * distance_factor
    
    def create_scene_from_tracks(self, tracks: List[Any], 
                                  model_paths: Dict[str, str]) -> List[SceneObject]:
        """
        Create list of scene objects from all tracks
        
        Args:
            tracks: List of TrackedObject
            model_paths: Mapping of class names to model paths
            
        Returns:
            List of SceneObject
        """
        scene_objects = []
        
        for track in tracks:
            model_path = model_paths.get(track.class_name, "")
            scene_obj = self.map_track_to_scene(track, model_path)
            scene_objects.append(scene_obj)
        
        return scene_objects
    
    def get_scene_bounds(self) -> Dict[str, float]:
        """
        Get scene boundary information
        
        Returns:
            Dictionary with scene bounds
        """
        return {
            'width': self.ground_width,
            'depth': self.ground_depth,
            'x_min': -self.ground_width / 2,
            'x_max': self.ground_width / 2,
            'z_min': 0,
            'z_max': self.ground_depth
        }
    
    def draw_ground_plane(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw ground plane grid on frame for visualization
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with grid overlay
        """
        if self.homography_matrix is None:
            return frame
        
        # Create grid lines in world coordinates
        grid_size = 5.0
        color = (0, 255, 0)
        
        # Longitudinal lines
        for x in np.arange(-self.ground_width/2, self.ground_width/2 + 1, grid_size):
            world_points = np.array([
                [[x, 0]],
                [[x, self.ground_depth]]
            ], dtype=np.float32)
            
            # Transform to image coordinates
            inv_h = np.linalg.inv(self.homography_matrix)
            image_points = cv2.perspectiveTransform(world_points, inv_h)
            
            pt1 = (int(image_points[0, 0, 0]), int(image_points[0, 0, 1]))
            pt2 = (int(image_points[1, 0, 0]), int(image_points[1, 0, 1]))
            
            cv2.line(frame, pt1, pt2, color, 1)
        
        # Latitudinal lines
        for z in np.arange(0, self.ground_depth + 1, grid_size):
            world_points = np.array([
                [[-self.ground_width/2, z]],
                [[self.ground_width/2, z]]
            ], dtype=np.float32)
            
            image_points = cv2.perspectiveTransform(world_points, 
                                                    np.linalg.inv(self.homography_matrix))
            
            pt1 = (int(image_points[0, 0, 0]), int(image_points[0, 0, 1]))
            pt2 = (int(image_points[1, 0, 0]), int(image_points[1, 0, 1]))
            
            cv2.line(frame, pt1, pt2, color, 1)
        
        return frame


def test_mapper(config_path: str = "config_realtime.yaml"):
    """Test scene mapper"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from video_processor import VideoProcessor
    from object_detector import ObjectDetector
    from tracker import ByteTrack
    
    processor = VideoProcessor(config_path)
    detector = ObjectDetector(config_path)
    tracker = ByteTrack(config_path)
    mapper = SceneMapper(config_path)
    
    if not processor.initialize():
        return
    if not detector.initialize():
        return
    
    logger.info("Press 'q' to exit, 'c' to calibrate")
    
    # Auto-calibrate on first frame
    calibrated = False
    
    try:
        for frame in processor.frames_generator():
            if not calibrated:
                mapper.auto_calibrate_homography(frame.shape[1], frame.shape[0])
                calibrated = True
                logger.info("Homography calibrated")
            
            # Detect and track
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            
            # Map to 3D
            model_paths = {
                'person': 'models/humans/hum_pedestrian_blue.glb',
                'car': 'models/vehicles/veh_car_sedan_blue.glb',
                'bus': 'models/vehicles/veh_bus_city_red.glb'
            }
            scene_objects = mapper.create_scene_from_tracks(tracks, model_paths)
            
            # Draw ground plane
            frame = mapper.draw_ground_plane(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, detections)
            
            # Print 3D positions
            for obj in scene_objects[:3]:  # Show first 3
                logger.debug(f"{obj.class_name}: pos={obj.position_3d}, vel={obj.velocity}")
            
            # FPS
            fps = processor.get_fps()
            frame = processor.draw_fps(frame, fps)
            
            cv2.imshow("Scene Mapper Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        processor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import cv2
    
    config_file = "config_realtime.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    test_mapper(config_file)
