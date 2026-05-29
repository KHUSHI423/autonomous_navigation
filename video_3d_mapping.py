"""
=============================================================================
EDGE DRIVE 3D - VIDEO-BASED 3D MAPPING SYSTEM
=============================================================================
Test system for 3D mapping using video files (no camera/GPS required)

Features:
- Load video file
- Process with YOLO detection
- Generate 3D point cloud from depth
- Visualize in interactive 3D dashboard
- Export mapping data

Author: EdgeDrive3D Team
Version: 1.0.0 - Video Testing Mode
=============================================================================
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

from ultralytics import YOLO


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection3D:
    """3D Detection result"""
    class_name: str
    confidence: float
    bbox_2d: Tuple[int, int, int, int]
    position_3d: Tuple[float, float, float]  # (x, y, z) in meters
    distance: float
    
    def to_dict(self) -> dict:
        return {
            'class_name': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox_2d': self.bbox_2d,
            'position_3d': [round(p, 3) for p in self.position_3d],
            'distance_m': round(self.distance, 2)
        }


@dataclass
class FrameResult:
    """Processing result for one frame"""
    frame_number: int
    timestamp: float
    fps: float
    
    # Detections
    objects: List[Detection3D] = field(default_factory=list)
    
    # Point cloud
    point_cloud: Optional[np.ndarray] = None
    point_colors: Optional[np.ndarray] = None
    
    # Visualizations
    annotated_frame: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'fps': round(self.fps, 2),
            'num_objects': len(self.objects),
            'objects': [obj.to_dict() for obj in self.objects]
        }


# ============================================================================
# DEPTH ESTIMATOR (MiDaS)
# ============================================================================

class DepthEstimator:
    """Monocular depth estimation using MiDaS"""
    
    def __init__(self, model_type: str = 'midas_hybrid'):
        print(f"  Loading Depth Model: {model_type}...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load MiDaS
        self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).dpt_transform
        
        print(f"  ✓ Depth model loaded on {self.device}")
    
    def estimate_depth(self, image: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
        """Estimate depth from image"""
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        # Convert to numpy
        depth_relative = prediction.cpu().numpy()
        
        # Normalize to 0-1
        depth_min = depth_relative.min()
        depth_max = depth_relative.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_relative - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_relative)
        
        # Invert (higher = farther)
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
        5: 'bus', 6: 'train', 7: 'truck', 9: 'traffic light'
    }
    
    COLORS = {
        'person': (0, 255, 255), 'bicycle': (255, 165, 0), 'car': (0, 255, 0),
        'motorcycle': (255, 0, 255), 'bus': (255, 0, 0), 'truck': (0, 165, 255)
    }
    
    OBJECT_HEIGHTS = {
        'person': 1.7, 'car': 1.5, 'truck': 3.5,
        'bus': 3.5, 'motorcycle': 1.2, 'bicycle': 1.1
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.4):
        print(f"  Loading YOLO Model: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  ✓ YOLO model loaded on {self.device}")
    
    def detect_3d(self, image: np.ndarray, depth_map: np.ndarray, 
                  camera_fx: float = 800.0, camera_cx: float = 320.0,
                  camera_cy: float = 240.0) -> List[Detection3D]:
        """Detect objects and estimate 3D positions"""
        
        # Run YOLO detection
        results = self.model(image, conf=self.confidence, device=self.device, verbose=False)[0]
        
        objects = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.TRAFFIC_CLASSES:
                continue
            
            class_name = self.TRAFFIC_CLASSES[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Estimate distance from depth
            depth_roi = depth_map[y1:y2, x1:x2]
            valid_depths = depth_roi[(depth_roi > 0.5) & (depth_roi < 100)]
            
            if len(valid_depths) > 0:
                distance = float(np.median(valid_depths))
            else:
                # Fallback: estimate from bbox size
                bbox_height = y2 - y1
                real_height = self.OBJECT_HEIGHTS.get(class_name, 2.0)
                distance = (real_height * camera_fx) / bbox_height
                distance = np.clip(distance, 1.0, 50.0)
            
            # Calculate 3D position
            center_u = (x1 + x2) / 2
            center_v = (y1 + y2) / 2
            
            x_3d = (center_u - camera_cx) * distance / camera_fx
            y_3d = (center_v - camera_cy) * distance / camera_fx
            
            obj = Detection3D(
                class_name=class_name,
                confidence=confidence,
                bbox_2d=(x1, y1, x2, y2),
                position_3d=(x_3d, y_3d, distance),
                distance=distance
            )
            objects.append(obj)
        
        return sorted(objects, key=lambda x: x.distance)
    
    def draw_detections(self, image: np.ndarray, objects: List[Detection3D]) -> np.ndarray:
        """Draw detections on image"""
        result = image.copy()
        
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_2d
            color = self.COLORS.get(obj.class_name, (0, 255, 0))
            
            # Draw bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Draw label
            label = f"{obj.class_name} {obj.distance:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(result, label, (x1+3, y1-4), cv2.FONT_HERSHEY_DUPLEX,
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return result


# ============================================================================
# POINT CLOUD GENERATOR
# ============================================================================

class PointCloudGenerator:
    """Generate 3D point cloud from depth map"""
    
    def __init__(self, camera_fx: float = 800.0, camera_fy: float = 800.0,
                 camera_cx: float = 320.0, camera_cy: float = 240.0):
        self.fx = camera_fx
        self.fy = camera_fy
        self.cx = camera_cx
        self.cy = camera_cy
    
    def generate(self, depth_map: np.ndarray, rgb_image: np.ndarray, 
                 downsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud from depth + RGB"""
        
        h, w = depth_map.shape
        
        # Create meshgrid
        u = np.arange(0, w, downsample)
        v = np.arange(0, h, downsample)
        u, v = np.meshgrid(u, v)
        
        # Get depth values
        z = depth_map[::downsample, ::downsample]
        valid = (z > 0.1) & (z < 100)
        
        # Convert to 3D
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Stack points
        points = np.stack([x, y, z], axis=-1)[valid].reshape(-1, 3)
        
        # Get colors
        if len(rgb_image.shape) == 3:
            colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            colors = colors[::downsample, ::downsample][valid].reshape(-1, 3).astype(np.float32) / 255.0
        else:
            gray = rgb_image[::downsample, ::downsample][valid].reshape(-1)
            colors = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        
        return points, colors


# ============================================================================
# MAIN VIDEO PROCESSOR
# ============================================================================

class Video3DProcessor:
    """Process video file and generate 3D mapping data"""
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        print("\n" + "="*60)
        print("  EdgeDrive3D Video Processor v1.0")
        print("="*60)
        print("\nInitializing components...")
        
        # Initialize modules
        self.depth_estimator = DepthEstimator(config.get('depth_model', 'midas_hybrid'))
        self.object_detector = ObjectDetector3D(
            config.get('yolo_model', 'yolov8n.pt'),
            config.get('confidence', 0.4)
        )
        self.point_cloud_gen = PointCloudGenerator(
            config.get('camera_fx', 800.0),
            config.get('camera_fy', 800.0),
            config.get('camera_cx', 320.0),
            config.get('camera_cy', 240.0)
        )
        
        print("\n✓ Video Processor Ready!\n")
    
    def process_video(self, video_path: str, max_frames: int = None,
                     save_results: bool = True) -> List[FrameResult]:
        """Process video file"""
        
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return []
        
        # Get video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total} frames")
        
        if max_frames:
            total = min(total, max_frames)
            print(f"Processing first {max_frames} frames...")
        
        results = []
        frame_count = 0
        start_time = time.time()
        
        print("\nProcessing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame, frame_count)
            results.append(result)
            
            frame_count += 1
            
            # Progress
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_count}/{total} ({current_fps:.1f} FPS)")
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n✓ Processed {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
        
        # Save results
        if save_results:
            self.save_results(results, f"output/video_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return results
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> FrameResult:
        """Process single frame"""
        
        start_time = time.time()
        
        # 1. Depth estimation
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        # 2. 3D object detection
        objects = self.object_detector.detect_3d(frame, depth_map)
        
        # 3. Draw detections
        annotated = self.object_detector.draw_detections(frame, objects)
        
        # 4. Generate point cloud
        points, colors = self.point_cloud_gen.generate(depth_map, frame, downsample=8)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        # Create result
        result = FrameResult(
            frame_number=frame_number,
            timestamp=time.time(),
            fps=fps,
            objects=objects,
            point_cloud=points,
            point_colors=colors,
            annotated_frame=annotated,
            depth_map=depth_map
        )
        
        return result
    
    def save_results(self, results: List[FrameResult], output_dir: str):
        """Save results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        data = {
            'total_frames': len(results),
            'total_objects': sum(len(r.objects) for r in results),
            'frames': [r.to_dict() for r in results]
        }
        
        with open(output_path / 'results.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save point clouds (first 10 frames)
        for i, result in enumerate(results[:10]):
            if result.point_cloud is not None:
                self._save_point_cloud(
                    result.point_cloud,
                    result.point_colors,
                    output_path / f'pointcloud_{i:03d}.ply'
                )
        
        print(f"✓ Results saved to: {output_path}")
    
    def _save_point_cloud(self, points: np.ndarray, colors: np.ndarray, filepath: Path):
        """Save point cloud as PLY file"""
        
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
    import sys
    
    # Create processor
    processor = Video3DProcessor({
        'yolo_model': 'yolov8n.pt',
        'confidence': 0.4,
        'depth_model': 'midas_hybrid'
    })
    
    # Get video path from command line or use demo
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'demo_video1.mp4'
    
    # Process video
    results = processor.process_video(
        video_path,
        max_frames=100,  # Process first 100 frames for testing
        save_results=True
    )
    
    print(f"\n✓ Processing complete! {len(results)} frames processed")
    print(f"  Total objects detected: {sum(len(r.objects) for r in results)}")
    print("  Check 'output/' folder for results")
