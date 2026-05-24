"""
Object Detector Module
YOLOv8-based real-time object detection for traffic scenes
Optimized for Indian urban environments with vehicles and pedestrians
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
from loguru import logger
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    YOLO = None


class Detection:
    """Represents a single object detection"""
    
    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: Tuple[int, int, int, int], mask: Optional[np.ndarray] = None):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.mask = mask
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.bottom_center = (self.center[0], bbox[3])  # Contact point with ground
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'bottom_center': self.bottom_center,
            'width': self.bbox[2] - self.bbox[0],
            'height': self.bbox[3] - self.bbox[1]
        }
    
    def __repr__(self) -> str:
        return f"Detection({self.class_name}, {self.confidence:.2f}, {self.bbox})"


class ObjectDetector:
    """
    YOLOv8-based object detector
    
    Features:
    - Real-time detection
    - Multiple model sizes (nano to extra-large)
    - GPU/CPU inference
    - Class filtering
    - Confidence thresholding
    """
    
    # COCO class names (relevant for traffic)
    COCO_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        # ... more classes
    }
    
    # Traffic-relevant classes
    TRAFFIC_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize object detector
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.is_initialized = False
        
        # Detection settings
        det_config = self.config['detection']
        self.model_name = det_config['model']
        self.confidence_threshold = det_config['confidence']
        self.iou_threshold = det_config['iou']
        self.max_detections = det_config['max_detections']
        self.classes_filter = det_config.get('classes', None)
        self.device = det_config.get('device', 'cpu')
        
        logger.info(f"ObjectDetector initialized with model: {self.model_name}")
    
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
            'detection': {
                'model': 'yolov8n.pt',
                'confidence': 0.5,
                'iou': 0.7,
                'max_detections': 100,
                'classes': None,
                'device': 'cpu'
            }
        }
    
    def initialize(self) -> bool:
        """
        Initialize YOLO model
        
        Returns:
            True if successful, False otherwise
        """
        if YOLO is None:
            logger.error("ultralytics library not installed")
            return False
        
        try:
            # Load YOLO model
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Set device
            if self.device != 'cpu':
                try:
                    self.model.to(self.device)
                    logger.info(f"Using device: {self.device}")
                except Exception as e:
                    logger.warning(f"Failed to use {self.device}, falling back to CPU: {e}")
                    self.device = 'cpu'
            
            self.is_initialized = True
            logger.info("YOLO model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform object detection on frame
        
        Args:
            frame: Input frame (BGR or RGB)
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                classes=self.classes_filter,
                verbose=False,
                device=self.device
            )
            
            # Parse results
            detections = []
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes
                num_detections = len(boxes)
                
                for i in range(num_detections):
                    # Extract box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # Extract confidence and class
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Create detection object
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    )
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def detect_async(self, frame: np.ndarray) -> Any:
        """
        Start async detection (for pipelined processing)
        
        Args:
            frame: Input frame
            
        Returns:
            Async result object
        """
        if not self.is_initialized:
            if not self.initialize():
                return None
        
        try:
            # Run async inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                classes=self.classes_filter,
                verbose=False,
                device=self.device,
                stream=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Async detection error: {e}")
            return None
    
    def get_detections_from_stream(self, results: Any) -> List[Detection]:
        """
        Get detections from async stream
        
        Args:
            results: Async result object
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        try:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    num_detections = len(boxes)
                    
                    for i in range(num_detections):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2)
                        )
                        detections.append(detection)
        except Exception as e:
            logger.error(f"Error parsing stream results: {e}")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                       show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame (BGR)
            detections: List of detections
            show_labels: Show class labels
            show_confidence: Show confidence scores
            
        Returns:
            Frame with drawn detections
        """
        # Color map for different classes
        colors = {
            'person': (0, 255, 0),      # Green
            'bicycle': (255, 128, 0),   # Orange
            'car': (0, 128, 255),       # Blue
            'motorcycle': (0, 255, 255), # Yellow
            'bus': (255, 0, 128),       # Pink
            'truck': (128, 0, 255),     # Purple
        }
        
        for det in detections:
            # Get color for class
            color = colors.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = det.class_name
                if show_confidence:
                    label += f" {det.confidence:.2f}"
                
                # Label background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(frame, (x1, y1 - label_h - 10),
                             (x1 + label_w, y1), color, -1)
                
                # Label text
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw center point
            cx, cy = det.center
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            
            # Draw bottom center (ground contact point)
            bx, by = det.bottom_center
            cv2.circle(frame, (bx, by), 3, (0, 0, 255), -1)
        
        return frame
    
    def get_statistics(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Get detection statistics
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total': len(detections),
            'by_class': {},
            'avg_confidence': 0.0
        }
        
        if len(detections) == 0:
            return stats
        
        # Count by class
        for det in detections:
            class_name = det.class_name
            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = 0
            stats['by_class'][class_name] += 1
        
        # Average confidence
        stats['avg_confidence'] = sum(d.confidence for d in detections) / len(detections)
        
        return stats


def test_detector(config_path: str = "config_realtime.yaml", source: int = 0):
    """
    Test object detector with video feed
    
    Args:
        config_path: Path to configuration file
        source: Video source (0 for webcam)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from video_processor import VideoProcessor
    
    detector = ObjectDetector(config_path)
    processor = VideoProcessor(config_path)
    
    if not detector.initialize():
        logger.error("Failed to initialize detector")
        return
    
    if not processor.initialize():
        logger.error("Failed to initialize video processor")
        return
    
    logger.info("Press 'q' to exit")
    
    frame_count = 0
    total_detections = 0
    
    try:
        for frame in processor.frames_generator():
            # Detect objects
            detections = detector.detect(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, detections)
            
            # Draw statistics
            stats = detector.get_statistics(detections)
            info_text = f"Objects: {stats['total']} | Avg: {stats['avg_confidence']:.2f}"
            cv2.putText(frame, info_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw FPS
            fps = processor.get_fps()
            frame = processor.draw_fps(frame, fps)
            
            # Display
            cv2.imshow("Object Detector Test", frame)
            
            frame_count += 1
            total_detections += stats['total']
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        processor.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            logger.info(f"Processed {frame_count} frames, {total_detections} total detections")
            logger.info(f"Average detections per frame: {total_detections / frame_count:.2f}")


if __name__ == "__main__":
    import sys
    
    config_file = "config_realtime.yaml"
    source = 0
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    if len(sys.argv) > 2:
        source = int(sys.argv[2])
    
    test_detector(config_file, source)
