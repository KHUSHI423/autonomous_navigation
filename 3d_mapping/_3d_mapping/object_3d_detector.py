"""
Object Detection with 3D Position and Dimension Estimation
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from ultralytics import YOLO
import torch

from camera_config import CameraIntrinsics, get_object_dimensions


@dataclass
class Object3D:
    """3D Object representation"""
    # 2D detection info
    class_id: int
    class_name: str
    confidence: float
    bbox_2d: List[int]  # [x1, y1, x2, y2]
    
    # 3D position (in camera coordinates, meters)
    position_3d: np.ndarray = None  # [x, y, z]
    
    # Estimated dimensions (meters)
    dimensions: Dict[str, float] = None  # {'length', 'width', 'height'}
    estimated_dimensions: Dict[str, float] = None  # Calculated from depth
    
    # Depth info
    depth_mean: float = 0.0
    depth_min: float = 0.0
    depth_max: float = 0.0
    
    # 3D bounding box corners (8 points)
    bbox_3d: np.ndarray = None
    
    # Distance from camera
    distance: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox_2d': self.bbox_2d,
            'position_3d': self.position_3d.tolist() if self.position_3d is not None else None,
            'dimensions': self.dimensions,
            'estimated_dimensions': self.estimated_dimensions,
            'depth_mean': self.depth_mean,
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'distance': self.distance,
            'bbox_3d': self.bbox_3d.tolist() if self.bbox_3d is not None else None,
        }


class Object3DDetector:
    """
    Detect objects and estimate their 3D positions and dimensions
    """
    
    TRAFFIC_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        5: 'bus', 6: 'train', 7: 'truck', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench'
    }
    
    COLORS = {
        'person': (0, 255, 255),
        'bicycle': (255, 165, 0),
        'car': (0, 255, 0),
        'motorcycle': (255, 0, 255),
        'bus': (255, 0, 0),
        'train': (128, 0, 128),
        'truck': (0, 165, 255),
        'traffic light': (0, 0, 255),
        'fire hydrant': (255, 255, 0),
        'stop sign': (0, 0, 200),
        'parking meter': (200, 200, 0),
        'bench': (139, 69, 19),
    }
    
    def __init__(
        self,
        model_size: str = 'yolov8m.pt',
        confidence_threshold: float = 0.5,
        device: str = None
    ):
        """Initialize 3D object detector"""
        print(f"Loading YOLOv8 model: {model_size}...")
        self.model = YOLO(model_size)
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"  Device: {self.device}")
        print("  ✓ 3D Object detector ready!")
    
    def detect_3d(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        camera: CameraIntrinsics,
        filter_traffic_only: bool = True
    ) -> List[Object3D]:
        """
        Detect objects and estimate their 3D properties
        
        Args:
            image: RGB/BGR image
            depth_map: Depth map in meters
            camera: Camera intrinsic parameters
            filter_traffic_only: Only detect traffic-related objects
            
        Returns:
            List of Object3D instances
        """
        # Run 2D detection
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        objects_3d = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            
            # Filter for traffic classes
            if filter_traffic_only and class_id not in self.TRAFFIC_CLASSES:
                continue
            
            # Get 2D bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Create 3D object
            obj = Object3D(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox_2d=[x1, y1, x2, y2]
            )
            
            # Get depth statistics for this object
            obj = self._estimate_3d_properties(obj, depth_map, camera)
            
            objects_3d.append(obj)
        
        return objects_3d
    
    def _estimate_3d_properties(
        self,
        obj: Object3D,
        depth_map: np.ndarray,
        camera: CameraIntrinsics
    ) -> Object3D:
        """Estimate 3D properties for a detected object"""
        x1, y1, x2, y2 = obj.bbox_2d
        
        # Get depth values within bounding box
        depth_roi = depth_map[y1:y2, x1:x2]
        
        # Filter valid depth values
        valid_depths = depth_roi[(depth_roi > 0.1) & (depth_roi < 100)]
        
        if len(valid_depths) == 0:
            return obj
        
        # Depth statistics
        obj.depth_mean = float(np.median(valid_depths))  # Use median for robustness
        obj.depth_min = float(np.min(valid_depths))
        obj.depth_max = float(np.max(valid_depths))
        obj.distance = obj.depth_mean
        
        # Calculate 3D center position
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Project to 3D
        x_3d = (center_u - camera.cx) * obj.depth_mean / camera.fx
        y_3d = (center_v - camera.cy) * obj.depth_mean / camera.fy
        z_3d = obj.depth_mean
        
        obj.position_3d = np.array([x_3d, y_3d, z_3d])
        
        # Get standard dimensions for this object type
        obj.dimensions = get_object_dimensions(obj.class_name)
        
        # Estimate dimensions from image and depth
        obj.estimated_dimensions = self._estimate_dimensions(
            obj, depth_map, camera
        )
        
        # Calculate 3D bounding box
        obj.bbox_3d = self._calculate_3d_bbox(obj)
        
        return obj
    
    def _estimate_dimensions(
        self,
        obj: Object3D,
        depth_map: np.ndarray,
        camera: CameraIntrinsics
    ) -> Dict[str, float]:
        """
        Estimate object dimensions from 2D bbox and depth
        """
        x1, y1, x2, y2 = obj.bbox_2d
        depth = obj.depth_mean
        
        if depth <= 0:
            return obj.dimensions.copy()
        
        # Calculate apparent width and height in meters
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        
        # Convert to meters using depth and focal length
        width_meters = width_pixels * depth / camera.fx
        height_meters = height_pixels * depth / camera.fy
        
        # Estimate depth extent from depth variation
        depth_extent = obj.depth_max - obj.depth_min
        length_meters = max(depth_extent, width_meters * 0.5)  # Approximate
        
        return {
            'width': round(width_meters, 2),
            'height': round(height_meters, 2),
            'length': round(length_meters, 2)
        }
    
    def _calculate_3d_bbox(self, obj: Object3D) -> np.ndarray:
        """
        Calculate 8 corners of 3D bounding box
        
        Returns array of shape (8, 3) with corners:
        Front face: [0,1,2,3], Back face: [4,5,6,7]
        """
        if obj.position_3d is None or obj.estimated_dimensions is None:
            return None
        
        dims = obj.estimated_dimensions
        center = obj.position_3d
        
        # Half dimensions
        hw = dims['width'] / 2
        hh = dims['height'] / 2
        hl = dims['length'] / 2
        
        # 8 corners relative to center
        corners = np.array([
            [-hw, -hh, -hl],  # 0: front-bottom-left
            [+hw, -hh, -hl],  # 1: front-bottom-right
            [+hw, +hh, -hl],  # 2: front-top-right
            [-hw, +hh, -hl],  # 3: front-top-left
            [-hw, -hh, +hl],  # 4: back-bottom-left
            [+hw, -hh, +hl],  # 5: back-bottom-right
            [+hw, +hh, +hl],  # 6: back-top-right
            [-hw, +hh, +hl],  # 7: back-top-left
        ])
        
        # Translate to center position
        corners = corners + center
        
        return corners
    
    def draw_detections_2d(
        self,
        image: np.ndarray,
        objects: List[Object3D],
        show_3d_info: bool = True
    ) -> np.ndarray:
        """Draw 2D detections with 3D information"""
        result = image.copy()
        
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_2d
            color = self.COLORS.get(obj.class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Prepare label with 3D info
            if show_3d_info and obj.distance > 0:
                label = f"{obj.class_name} {obj.distance:.1f}m"
                
                # Add dimensions
                if obj.estimated_dimensions:
                    dims = obj.estimated_dimensions
                    label += f" [{dims['width']:.1f}x{dims['height']:.1f}m]"
            else:
                label = f"{obj.class_name} {obj.confidence:.0%}"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1, cv2.LINE_AA)
            cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0,0,0), 1, cv2.LINE_AA)
            
            # Draw text
            cv2.putText(result, label, (x1 + 4, y1 - 4), font, font_scale, 
                       (255, 255, 255), thickness, cv2.LINE_AA)
        
        return result
    
    def draw_3d_boxes(
        self,
        image: np.ndarray,
        objects: List[Object3D],
        camera: CameraIntrinsics
    ) -> np.ndarray:
        """Draw 3D bounding boxes projected onto 2D image"""
        result = image.copy()
        
        for obj in objects:
            if obj.bbox_3d is None:
                continue
            
            color = self.COLORS.get(obj.class_name, (0, 255, 0))
            
            # Project 3D points to 2D
            points_2d = self._project_3d_to_2d(obj.bbox_3d, camera)
            
            if points_2d is None:
                continue
            
            # Draw 3D box edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
                (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
            ]
            
            for i, j in edges:
                pt1 = tuple(points_2d[i].astype(int))
                pt2 = tuple(points_2d[j].astype(int))
                cv2.line(result, pt1, pt2, color, 2, cv2.LINE_AA)
        
        return result
    
    def _project_3d_to_2d(
        self,
        points_3d: np.ndarray,
        camera: CameraIntrinsics
    ) -> np.ndarray:
        """Project 3D points to 2D image coordinates"""
        # Filter points behind camera
        if np.any(points_3d[:, 2] <= 0):
            return None
        
        # Project using camera intrinsics
        x = points_3d[:, 0] * camera.fx / points_3d[:, 2] + camera.cx
        y = points_3d[:, 1] * camera.fy / points_3d[:, 2] + camera.cy
        
        points_2d = np.stack([x, y], axis=-1)
        
        return points_2d


def create_scene_summary(objects: List[Object3D]) -> Dict:
    """Create a summary of all detected objects in the scene"""
    summary = {
        'total_objects': len(objects),
        'by_class': {},
        'closest_object': None,
        'farthest_object': None,
        'objects': []
    }
    
    if not objects:
        return summary
    
    # Count by class
    for obj in objects:
        if obj.class_name not in summary['by_class']:
            summary['by_class'][obj.class_name] = {
                'count': 0,
                'avg_distance': 0,
                'distances': []
            }
        summary['by_class'][obj.class_name]['count'] += 1
        if obj.distance > 0:
            summary['by_class'][obj.class_name]['distances'].append(obj.distance)
    
    # Calculate average distances
    for class_name, data in summary['by_class'].items():
        if data['distances']:
            data['avg_distance'] = round(np.mean(data['distances']), 2)
        del data['distances']
    
    # Find closest and farthest
    valid_objects = [o for o in objects if o.distance > 0]
    if valid_objects:
        closest = min(valid_objects, key=lambda x: x.distance)
        farthest = max(valid_objects, key=lambda x: x.distance)
        
        summary['closest_object'] = {
            'class': closest.class_name,
            'distance': round(closest.distance, 2)
        }
        summary['farthest_object'] = {
            'class': farthest.class_name,
            'distance': round(farthest.distance, 2)
        }
    
    # Add all objects
    summary['objects'] = [obj.to_dict() for obj in objects]
    
    return summary
