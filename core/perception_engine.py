"""
=============================================================================
EDGE DRIVE 3D - UNIFIED PERCEPTION ENGINE (ENHANCED)
=============================================================================
Main perception pipeline combining:
- Monocular Depth Estimation (MiDaS/Depth Anything)
- 3D Object Detection (YOLOv8/v11)
- Bird's Eye View Mapping
- Lane Detection (OpenCV)
- Traffic Sign Detection (OpenCV/YOLO)
- Point Cloud Generation
- Intelligent Decision Making for Hardware Control

Author: EdgeDrive3D Team
Version: 2.0.0 - Enhanced with Lane & Sign Detection
=============================================================================
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import time
import json

from ultralytics import YOLO

# Import new modules
from .lane_detector import LaneDetector, LaneResult
from .sign_detector import SignDetector, SignDetectionResult, TrafficSign


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fov_degrees: float = 70.0

    @classmethod
    def from_image_size(cls, width: int, height: int, fov: float = 70.0):
        fov_rad = np.radians(fov)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx
        return cls(fx=fx, fy=fy, cx=width/2, cy=height/2, 
                   width=width, height=height, fov_degrees=fov)


@dataclass
class Object3D:
    """3D Object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox_2d: Tuple[int, int, int, int]
    position_3d: Optional[np.ndarray] = None
    distance: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    bbox_3d: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'class_name': self.class_name,
            'confidence': round(self.confidence, 3),
            'distance_m': round(self.distance, 2),
            'position_3d': self.position_3d.tolist() if self.position_3d is not None else None,
            'dimensions': self.dimensions,
        }


@dataclass
class LaneDetection:
    """Lane detection result"""
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    lane_width: float = 0.0
    curvature: float = 0.0
    vehicle_offset: float = 0.0
    confidence: float = 0.0


@dataclass
class PerceptionResult:
    """Complete perception pipeline output"""
    timestamp: float
    fps: float

    # Depth
    depth_map: Optional[np.ndarray] = None

    # 3D Objects
    objects_3d: List[Object3D] = field(default_factory=list)

    # Lanes
    lanes: Optional[LaneResult] = None

    # Traffic Signs
    traffic_signs: List[TrafficSign] = field(default_factory=list)

    # Visualizations
    depth_colored: Optional[np.ndarray] = None
    detections_overlay: Optional[np.ndarray] = None
    bev_image: Optional[np.ndarray] = None

    # Point cloud
    point_cloud: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (points, colors)

    # Decision
    decision: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'fps': round(self.fps, 2),
            'num_objects': len(self.objects_3d),
            'objects': [obj.to_dict() for obj in self.objects_3d],
            'lanes': self.lanes.to_dict() if self.lanes else None,
            'traffic_signs': [sign.to_dict() for sign in self.traffic_signs],
            'decision': self.decision,
        }


# ============================================================================
# DEPTH ESTIMATOR
# ============================================================================

class DepthEstimator:
    """Monocular depth estimation using MiDaS"""
    
    def __init__(self, model_type: str = 'midas_hybrid', device: str = None):
        self.model_type = model_type
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model from torch hub"""
        print(f"  Loading Depth Model: {self.model_type}...")
        
        try:
            if self.model_type == 'midas_small':
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform
            elif self.model_type == 'midas_large':
                self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).dpt_transform
            else:  # hybrid (default)
                self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).dpt_transform
            
            self.model.to(self.device)
            self.model.eval()
            print(f"  ✓ Depth model loaded on {self.device}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load depth model: {e}")
            self.model = None
    
    def estimate_depth(self, image: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
        """Estimate depth from image"""
        if self.model is None:
            return np.zeros_like(image[:, :, 0], dtype=np.float32)
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        depth_relative = prediction.cpu().numpy()
        
        # Normalize and convert to metric
        depth_min = depth_relative.min()
        depth_max = depth_relative.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_relative - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_relative)
        
        # Invert so higher = farther
        depth_inverted = 1.0 - depth_normalized
        depth_meters = depth_inverted * max_depth
        
        return np.clip(depth_meters, 0.1, max_depth).astype(np.float32)
    
    def create_colormap(self, depth: np.ndarray) -> np.ndarray:
        """Create colored depth visualization"""
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)


# ============================================================================
# 3D OBJECT DETECTOR
# ============================================================================

class ObjectDetector3D:
    """YOLO-based 3D object detection"""
    
    TRAFFIC_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        5: 'bus', 6: 'train', 7: 'truck', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench'
    }
    
    COLORS = {
        'person': (0, 255, 255), 'bicycle': (255, 165, 0), 'car': (0, 255, 0),
        'motorcycle': (255, 0, 255), 'bus': (255, 0, 0), 'train': (128, 0, 128),
        'truck': (0, 165, 255), 'traffic light': (0, 0, 255),
        'fire hydrant': (255, 255, 0), 'stop sign': (0, 0, 200),
    }
    
    OBJECT_DIMENSIONS = {
        'car': {'length': 4.5, 'width': 1.8, 'height': 1.5},
        'truck': {'length': 8.0, 'width': 2.5, 'height': 3.5},
        'bus': {'length': 12.0, 'width': 2.5, 'height': 3.5},
        'motorcycle': {'length': 2.2, 'width': 0.8, 'height': 1.2},
        'bicycle': {'length': 1.8, 'width': 0.5, 'height': 1.1},
        'person': {'length': 0.5, 'width': 0.5, 'height': 1.7},
    }
    
    def __init__(self, model_path: str = 'yolov8m.pt', confidence: float = 0.5):
        print(f"  Loading YOLO Model: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  ✓ YOLO model loaded on {self.device}")
    
    def detect_3d(self, image: np.ndarray, depth_map: np.ndarray, 
                  camera: CameraIntrinsics) -> List[Object3D]:
        """Detect objects and estimate 3D properties"""
        results = self.model(image, conf=self.confidence, device=self.device, verbose=False)[0]
        
        objects = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.TRAFFIC_CLASSES:
                continue
            
            class_name = self.TRAFFIC_CLASSES[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            obj = Object3D(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox_2d=(x1, y1, x2, y2)
            )
            
            # Estimate 3D properties
            self._estimate_3d_properties(obj, depth_map, camera)
            objects.append(obj)
        
        # Objects with unknown distance (0.0 sentinel) sort to the end, not the front.
        return sorted(objects, key=lambda x: x.distance if x.distance > 0 else float('inf'))
    
    # Real-world heights (metres) for COCO classes the YOLO model detects.
    # Used as a final distance fallback when depth + ground-plane estimates fail.
    _CLASS_HEIGHT_PRIORS = {
        'person': 1.70, 'bicycle': 1.10, 'car': 1.50, 'motorcycle': 1.30,
        'motorbike': 1.30, 'bus': 3.20, 'truck': 3.00, 'train': 3.80,
        'traffic light': 0.80, 'stop sign': 0.60, 'dog': 0.50, 'cow': 1.40,
        'horse': 1.60, 'sheep': 0.90, 'bird': 0.20,
    }

    def _estimate_3d_properties(self, obj: Object3D, depth_map: np.ndarray,
                                camera: CameraIntrinsics):
        """Estimate 3D position and dimensions.

        Distance chain — each stage only runs if the previous one was invalid,
        so we almost always produce a usable number:
          1. Median of valid MiDaS depths inside the bbox (most accurate).
          2. Ground-plane geometry from the bbox bottom, assuming a 1.5 m camera
             mount: d = H * fy / (v_bottom - cy).
          3. Class-prior pinhole: d = real_height_m * fy / bbox_height_px,
             using typical heights for known COCO classes.
          4. As a last resort, leave distance = 0.0 so downstream code can skip
             the label rather than display a wrong number.
        """
        x1, y1, x2, y2 = obj.bbox_2d

        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[(depth_roi > 0.1) & (depth_roi < 150)]

        distance = 0.0
        if len(valid_depths) > 0:
            distance = float(np.median(valid_depths))

        if distance <= 0:
            H_camera = 1.5
            v_bottom = y2
            if v_bottom > camera.cy + 5:
                distance = min(H_camera * camera.fy / (v_bottom - camera.cy), 120.0)

        if distance <= 0:
            real_h = self._CLASS_HEIGHT_PRIORS.get(obj.class_name)
            bbox_h_px = y2 - y1
            if real_h is not None and bbox_h_px > 2:
                distance = min(real_h * camera.fy / bbox_h_px, 120.0)

        obj.distance = distance

        if obj.distance > 0:
            center_u = (x1 + x2) / 2
            center_v = (y1 + y2) / 2
            x_3d = (center_u - camera.cx) * obj.distance / camera.fx
            y_3d = (center_v - camera.cy) * obj.distance / camera.fy
            obj.position_3d = np.array([x_3d, y_3d, obj.distance])

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            obj.dimensions = {
                'width': bbox_w * obj.distance / camera.fx,
                'height': bbox_h * obj.distance / camera.fy,
                'length': obj.dimensions.get('length', 2.0)
            }
        else:
            obj.position_3d = None
            obj.dimensions = {'width': 0.0, 'height': 0.0,
                              'length': obj.dimensions.get('length', 2.0)}
        
        # 3D bbox corners
        obj.bbox_3d = self._calculate_3d_bbox(obj)
    
    def _calculate_3d_bbox(self, obj: Object3D) -> np.ndarray:
        """Calculate 8 corners of 3D bounding box"""
        if obj.position_3d is None or not obj.dimensions:
            return None
        
        dims = obj.dimensions
        hw, hh, hl = dims['width']/2, dims['height']/2, dims.get('length', 2.0)/2
        c = obj.position_3d
        
        corners = np.array([
            [-hw, -hh, -hl], [+hw, -hh, -hl], [+hw, +hh, -hl], [-hw, +hh, -hl],
            [-hw, -hh, +hl], [+hw, -hh, +hl], [+hw, +hh, +hl], [-hw, +hh, +hl],
        ]) + c
        
        return corners
    
    def draw_detections(self, image: np.ndarray, objects: List[Object3D]) -> np.ndarray:
        """
        Production-quality object drawing:
        - Shows class + confidence + DISTANCE in meters
        - Clean label format: 'Truck: 0.87 (45m)'
        """
        result = image.copy()

        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_2d
            color = self.COLORS.get(obj.class_name, (0, 255, 0))
            
            # Get distance
            distance = obj.distance if hasattr(obj, 'distance') and obj.distance > 0 else 0

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Format label with distance
            if distance > 0 and distance < 200:
                label = f"{obj.class_name.capitalize()}: {obj.confidence:.2f} ({distance:.0f}m)"
            else:
                label = f"{obj.class_name.capitalize()}: {obj.confidence:.2f}"
            
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - 28), (x1 + tw + 12, y1), color, -1)
            
            # Label text (white for contrast)
            cv2.putText(result, label, (x1 + 6, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return result


# ============================================================================
# BIRD'S EYE VIEW MAPPER
# ============================================================================

class BEVMapper:
    """Generate Bird's Eye View visualizations"""
    
    def __init__(self):
        self.colors = {
            'person': (0, 255, 255), 'car': (0, 255, 0), 'truck': (0, 165, 255),
            'bus': (255, 0, 0), 'motorcycle': (255, 0, 255), 'bicycle': (255, 165, 0),
        }
    
    def create_bev(self, objects: List[Object3D], 
                   range_x: Tuple[float, float] = (-20, 20),
                   range_z: Tuple[float, float] = (0, 50),
                   size: Tuple[int, int] = (600, 800)) -> np.ndarray:
        """Create bird's eye view image"""
        width, height = size
        bev = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw grid
        self._draw_grid(bev, range_x, range_z)
        
        # Draw ego vehicle
        ego_x, ego_y = width // 2, height - 60
        cv2.circle(bev, (ego_x, ego_y), 15, (0, 200, 0), -1)
        cv2.putText(bev, "EGO", (ego_x-15, ego_y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        
        # Draw objects
        for obj in objects:
            if obj.position_3d is None:
                continue
            
            x_3d, z_3d = obj.position_3d[0], obj.position_3d[2]
            
            x_bev = int(((x_3d - range_x[0]) / (range_x[1] - range_x[0])) * width)
            y_bev = int((1 - (z_3d - range_z[0]) / (range_z[1] - range_z[0])) * (height - 90)) + 30
            
            x_bev = np.clip(x_bev, 20, width-20)
            y_bev = np.clip(y_bev, 30, height-90)
            
            color = self.colors.get(obj.class_name, (255, 255, 255))
            
            # Draw object
            obj_w = max(8, int(obj.dimensions.get('width', 1.5) * width / (range_x[1] - range_x[0])))
            obj_l = max(8, int(obj.dimensions.get('length', 3.0) * (height-90) / (range_z[1] - range_z[0])))
            
            cv2.rectangle(bev, (x_bev-obj_w//2, y_bev-obj_l//2), 
                         (x_bev+obj_w//2, y_bev+obj_l//2), color, -1)
            cv2.rectangle(bev, (x_bev-obj_w//2, y_bev-obj_l//2), 
                         (x_bev+obj_w//2, y_bev+obj_l//2), (255, 255, 255), 1)
            
            label = f"{obj.class_name[:4]} {obj.distance:.0f}m"
            cv2.putText(bev, label, (x_bev-20, y_bev-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        cv2.putText(bev, "BIRD'S EYE VIEW", (10, 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return bev
    
    def _draw_grid(self, image: np.ndarray, range_x: Tuple[float, float], 
                   range_z: Tuple[float, float]):
        """Draw grid on BEV"""
        height, width = image.shape[:2]
        
        for z in range(0, int(range_z[1])+1, 10):
            y = int((1 - (z - range_z[0]) / (range_z[1] - range_z[0])) * (height-90)) + 30
            cv2.line(image, (0, y), (width, y), (30, 30, 30), 1)
            cv2.putText(image, f"{z}m", (5, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        cv2.line(image, (width//2, 30), (width//2, height-60), (40, 40, 40), 1)


# ============================================================================
# DECISION MAKER
# ============================================================================

class DecisionMaker:
    """Make intelligent decisions based on perception"""

    def __init__(self):
        self.safety_distance = 2.0
        self.warning_distance = 5.0

    def make_decision(self, objects: List[Object3D], 
                      lanes: Optional[LaneResult],
                      traffic_signs: List[TrafficSign]) -> Dict[str, Any]:
        """Generate control decisions based on perception"""
        decision = {
            'action': 'forward',
            'speed': 180,
            'reason': 'clear_path',
            'warnings': [],
            'timestamp': time.time()
        }

        # Check for traffic signs first (highest priority)
        for sign in traffic_signs:
            if sign.sign_type == 'stop':
                decision['action'] = 'stop'
                decision['speed'] = 0
                decision['reason'] = 'stop_sign_detected'
                decision['warnings'].append(f"STOP SIGN at {sign.distance_estimate:.1f}m")
                return decision
            
            elif sign.sign_type.startswith('speed_limit'):
                # Extract speed limit number
                try:
                    speed_limit = int(sign.sign_type.split('_')[-1])
                    decision['speed_limit'] = speed_limit
                    decision['warnings'].append(f"Speed limit: {speed_limit} km/h")
                except:
                    pass
        
        # Check for obstacles
        known = [o for o in objects if o.distance > 0]
        if known:
            # Find closest object (ignore unknown-distance detections)
            closest = min(known, key=lambda x: x.distance)
            decision['closest_object'] = {
                'class': closest.class_name,
                'distance': closest.distance,
                'position': closest.position_3d.tolist() if closest.position_3d is not None else None
            }

            # Decision logic based on distance
            if closest.distance < self.safety_distance:
                decision['action'] = 'stop'
                decision['speed'] = 0
                decision['reason'] = f'obstacle_too_close: {closest.class_name}'
                decision['warnings'].append(f"CRITICAL: {closest.class_name} at {closest.distance:.1f}m")

            elif closest.distance < self.warning_distance:
                if closest.position_3d is not None:
                    if closest.position_3d[0] < -1.0:
                        decision['action'] = 'right'
                        decision['reason'] = 'avoid_left_obstacle'
                    elif closest.position_3d[0] > 1.0:
                        decision['action'] = 'left'
                        decision['reason'] = 'avoid_right_obstacle'
                    else:
                        decision['action'] = 'slow'
                        decision['reason'] = 'obstacle_ahead'

                decision['speed'] = int(100 * (closest.distance / self.warning_distance))
                decision['warnings'].append(f"WARNING: {closest.class_name} at {closest.distance:.1f}m")
        
        # Check lane position
        if lanes and lanes.vehicle_offset != 0:
            if abs(lanes.vehicle_offset) > 0.5:
                if lanes.vehicle_offset < 0:
                    decision['lane_correction'] = 'right'
                else:
                    decision['lane_correction'] = 'left'
                decision['warnings'].append(f"Lane offset: {lanes.vehicle_offset:+.2f}m")
        
        # Check lane curvature
        if lanes and lanes.curvature > 0.01:
            decision['curve_warning'] = True
            if decision['speed'] > 100:
                decision['speed'] = 100
            decision['warnings'].append(f"Curve ahead (curvature: {lanes.curvature:.4f})")

        return decision


# ============================================================================
# MAIN PERCEPTION ENGINE
# ============================================================================

class PerceptionEngine:
    """Unified perception pipeline"""

    def __init__(self, config: Dict = None):
        config = config or {}

        self.config = {
            'yolo_model': config.get('yolo_model', 'yolov8n.pt'),  # Faster default
            'depth_model': config.get('depth_model', 'midas_hybrid'),
            'confidence': config.get('confidence', 0.4),
            'max_depth': config.get('max_depth', 50.0),
            'fov': config.get('fov', 70.0),
            'lane_preset': config.get('lane_preset', 'default'),
            'enable_lane': config.get('enable_lane', True),
            'enable_signs': config.get('enable_signs', True),
            'enable_depth': config.get('enable_depth', True),  # NEW: option to disable depth
            'depth_skip_frames': config.get('depth_skip_frames', 2),  # Process depth every N frames (0 = every frame)
            'sign_mode': config.get('sign_mode', 'opencv'),
        }

        print("\n" + "="*60)
        print("  EdgeDrive3D Perception Engine v2.0 (ENHANCED)")
        print("="*60)
        print("\n  Modules:")
        print("  - Depth Estimation (MiDaS)")
        print("  - 3D Object Detection (YOLOv8)")
        print("  - Lane Detection (OpenCV)")
        print("  - Traffic Sign Detection")
        print("  - BEV Mapping")
        print("  - Decision Making")
        print("="*60)

        # Initialize components
        print("\nInitializing components...")
        self.camera = None
        
        # Depth Estimator
        self.depth_estimator = DepthEstimator(self.config['depth_model'])
        
        # 3D Object Detector
        self.object_detector = ObjectDetector3D(self.config['yolo_model'], self.config['confidence'])
        
        # Lane Detector
        if self.config['enable_lane']:
            self.lane_detector = LaneDetector(
                preset=self.config['lane_preset'],
                debug=False
            )
            print("  ✓ Lane Detector initialized")
        else:
            self.lane_detector = None
            print("  ⚠ Lane Detector disabled")
        
        # Sign Detector
        if self.config['enable_signs']:
            self.sign_detector = SignDetector(
                mode=self.config['sign_mode'],
                confidence_threshold=self.config['confidence']
            )
            print("  ✓ Sign Detector initialized")
        else:
            self.sign_detector = None
            print("  ⚠ Sign Detector disabled")
        
        # BEV Mapper
        self.bev_mapper = BEVMapper()
        
        # Decision Maker
        self.decision_maker = DecisionMaker()

        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0.0
        self.depth_frame_counter = 0  # For frame skipping

        print("\n✓ Enhanced Perception Engine Ready!\n")
    
    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        """Process single frame through complete pipeline"""
        if self.start_time is None:
            self.start_time = time.time()

        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0.0

        # Set camera intrinsics
        if self.camera is None:
            self.camera = CameraIntrinsics.from_image_size(
                frame.shape[1], frame.shape[0], self.config['fov']
            )

        # Initialize visualization with original frame
        detections_overlay = frame.copy()

        # Initialize variables
        depth_map = None
        depth_colored = None
        lanes = None

        # ========== STAGE 1: Lane Detection FIRST (needed for depth masking) ==========
        if self.lane_detector:
            lanes = self.lane_detector.detect(frame)
            detections_overlay = self.lane_detector.draw_lanes(detections_overlay, lanes)

        # ========== STAGE 2: Depth Estimation (ONLY on lane region) ==========
        skip_frames = self.config.get('depth_skip_frames', 2)
        should_process_depth = (self.config.get('enable_depth', True) and
                               (skip_frames == 0 or self.depth_frame_counter % (skip_frames + 1) == 0))

        if should_process_depth:
            # Create lane mask to apply depth ONLY on road region
            if lanes and lanes.left_points is not None and lanes.right_points is not None:
                # Create mask from lane polygon
                lane_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                height, width = frame.shape[:2]
                roi_y_start = int(height * 0.60)
                
                left_pts = lanes.left_points
                right_pts = lanes.right_points
                left_filtered = left_pts[left_pts[:, 1] >= roi_y_start]
                right_filtered = right_pts[right_pts[:, 1] >= roi_y_start]
                
                if len(left_filtered) > 2 and len(right_filtered) > 2:
                    pts = np.vstack([left_filtered, right_filtered[::-1]]).astype(np.int32)
                    cv2.fillPoly(lane_mask, [pts], 255)
                    
                    # Apply depth ONLY on masked lane region
                    lane_region = cv2.bitwise_and(frame, frame, mask=lane_mask)
                    depth_map = self.depth_estimator.estimate_depth(lane_region, self.config['max_depth'])
                    depth_colored = self.depth_estimator.create_colormap(depth_map)
                else:
                    # Fallback: full frame if no lane detected
                    depth_map = self.depth_estimator.estimate_depth(frame, self.config['max_depth'])
                    depth_colored = self.depth_estimator.create_colormap(depth_map)
            else:
                # No lanes detected: apply to full frame
                depth_map = self.depth_estimator.estimate_depth(frame, self.config['max_depth'])
                depth_colored = self.depth_estimator.create_colormap(depth_map)
            
            self.depth_frame_counter += 1
        else:
            # Create empty depth map for object detection
            depth_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        # ========== STAGE 3: 3D Object Detection (with higher confidence) ==========
        objects_3d = self.object_detector.detect_3d(frame, depth_map, self.camera)
        detections_overlay = self.object_detector.draw_detections(detections_overlay, objects_3d)

        # ========== STAGE 4: Traffic Sign Detection (with ROI restriction) ==========
        traffic_signs = []
        if self.sign_detector:
            # Restrict sign detection to upper 60% of frame (signs are above road)
            sign_roi_end = int(frame.shape[0] * 0.60)
            sign_region = frame[:sign_roi_end, :]
            
            sign_result = self.sign_detector.detect(sign_region)
            traffic_signs = sign_result.signs

            # Adjust coordinates back to full frame
            for sign in traffic_signs:
                if hasattr(sign, 'bbox') and sign.bbox is not None:
                    sign.bbox = (sign.bbox[0], sign.bbox[1] + sign_roi_end, 
                                sign.bbox[2], sign.bbox[3] + sign_roi_end)
                self.sign_detector.estimate_distance(sign, frame.shape[0])

            detections_overlay = self.sign_detector.draw_signs(detections_overlay, sign_result)

        # ========== STAGE 5: BEV Generation ==========
        bev_image = self.bev_mapper.create_bev(objects_3d)

        # ========== STAGE 6: Decision Making ==========
        decision = self.decision_maker.make_decision(objects_3d, lanes, traffic_signs)

        # ========== STAGE 7: Point Cloud Generation ==========
        points, colors = self._generate_pointcloud(frame, depth_map)

        return PerceptionResult(
            timestamp=time.time(),
            fps=self.current_fps,
            depth_map=depth_map,
            objects_3d=objects_3d,
            lanes=lanes,
            traffic_signs=traffic_signs,
            depth_colored=depth_colored,
            detections_overlay=detections_overlay,
            bev_image=bev_image,
            point_cloud=(points, colors),
            decision=decision
        )
    
    def _generate_pointcloud(self, image: np.ndarray, 
                             depth_map: np.ndarray, 
                             downsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud from depth"""
        h, w = depth_map.shape
        u = np.arange(0, w, downsample)
        v = np.arange(0, h, downsample)
        u, v = np.meshgrid(u, v)
        
        z = depth_map[::downsample, ::downsample]
        valid = (z > 0.1) & (z < self.config['max_depth'])
        
        x = (u - self.camera.cx) * z / self.camera.fx
        y = (v - self.camera.cy) * z / self.camera.fy
        
        points = np.stack([x, y, z], axis=-1)[valid].reshape(-1, 3)
        
        if len(image.shape) == 3:
            colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            colors = colors[::downsample, ::downsample][valid].reshape(-1, 3).astype(np.float32) / 255.0
        else:
            gray = image[::downsample, ::downsample][valid].reshape(-1)
            colors = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        
        return points, colors
    
    def save_results(self, result: PerceptionResult, output_dir: str = "output"):
        """Save perception results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images
        if result.detections_overlay is not None:
            cv2.imwrite(str(output_path / f"{timestamp}_detections.jpg"), result.detections_overlay)
        
        if result.bev_image is not None:
            cv2.imwrite(str(output_path / f"{timestamp}_bev.jpg"), result.bev_image)
        
        if result.depth_colored is not None:
            cv2.imwrite(str(output_path / f"{timestamp}_depth.jpg"), result.depth_colored)
        
        # Save point cloud
        if result.point_cloud is not None:
            self._save_pointcloud(result.point_cloud, output_path / f"{timestamp}.ply")
        
        # Save JSON
        with open(output_path / f"{timestamp}_results.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_pointcloud(self, pointcloud: Tuple[np.ndarray, np.ndarray], filepath: Path):
        """Save point cloud as PLY"""
        points, colors = pointcloud
        
        with open(filepath, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r*255)} {int(g*255)} {int(b*255)}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test perception engine
    engine = PerceptionEngine()
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    
    result = engine.process_frame(test_image)
    print(f"Processed frame: {len(result.objects_3d)} objects detected")
    print(f"Decision: {result.decision['action']}")
