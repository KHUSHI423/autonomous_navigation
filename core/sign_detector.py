"""
=============================================================================
TRAFFIC SIGN DETECTION MODULE
=============================================================================
Traffic sign detection using color and shape-based detection.
Supports Indian traffic signs including:
- Speed limit signs
- Stop signs
- Yield/Give way signs
- Warning signs
- Regulatory signs

Two modes:
1. OpenCV-based (color + shape detection) - Fast, no model needed
2. YOLO-based (if model available) - More accurate

Author: EdgeDrive3D Team
=============================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrafficSign:
    """Traffic sign detection result"""
    sign_type: str
    sign_class: str  # Category: regulatory, warning, informational
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int] = (0, 0)
    distance_estimate: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)
    
    def to_dict(self) -> dict:
        return {
            'sign_type': self.sign_type,
            'sign_class': self.sign_class,
            'confidence': round(self.confidence, 3),
            'bbox': self.bbox,
            'center': self.center,
            'distance_estimate_m': round(self.distance_estimate, 2),
        }


@dataclass
class SignDetectionResult:
    """Complete sign detection result"""
    signs: List[TrafficSign] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'num_signs': len(self.signs),
            'signs': [sign.to_dict() for sign in self.signs],
            'processing_time_ms': round(self.processing_time_ms, 2),
        }


# ============================================================================
# TRAFFIC SIGN DETECTOR
# ============================================================================

class SignDetector:
    """
    Traffic sign detection using OpenCV color and shape analysis
    
    Usage:
        detector = SignDetector()
        result = detector.detect(frame)
    """
    
    # Traffic sign classes for Indian roads
    SIGN_TYPES = {
        # Regulatory signs (red/white)
        'stop': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'octagon'},
        'yield': {'class': 'regulatory', 'color': (0, 255, 255), 'shape': 'triangle'},
        'no_entry': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'no_parking': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'no_horn': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        
        # Speed limit signs
        'speed_limit_20': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'speed_limit_30': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'speed_limit_40': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'speed_limit_50': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'speed_limit_60': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'speed_limit_80': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        'speed_limit': {'class': 'regulatory', 'color': (0, 0, 255), 'shape': 'circle'},
        
        # Warning signs (yellow/orange triangle)
        'pedestrian_crossing': {'class': 'warning', 'color': (0, 255, 255), 'shape': 'triangle'},
        'school_zone': {'class': 'warning', 'color': (0, 255, 255), 'shape': 'triangle'},
        'curve_left': {'class': 'warning', 'color': (0, 255, 255), 'shape': 'triangle'},
        'curve_right': {'class': 'warning', 'color': (0, 255, 255), 'shape': 'triangle'},
        'merge': {'class': 'warning', 'color': (0, 255, 255), 'shape': 'triangle'},
        'intersection': {'class': 'warning', 'color': (0, 255, 255), 'shape': 'triangle'},
        
        # Informational signs (blue)
        'parking': {'class': 'informational', 'color': (255, 0, 0), 'shape': 'square'},
        'hospital': {'class': 'informational', 'color': (255, 0, 0), 'shape': 'square'},
        'fuel_station': {'class': 'informational', 'color': (255, 0, 0), 'shape': 'square'},
    }
    
    # Color ranges in HSV
    COLOR_RANGES = {
        'red': [
            (0, 70, 50), (10, 255, 255),    # Lower red
            (170, 70, 50), (180, 255, 255)  # Upper red
        ],
        'yellow': [
            (20, 70, 50), (35, 255, 255)
        ],
        'blue': [
            (100, 70, 50), (130, 255, 255)
        ],
        'white': [
            (0, 0, 200), (180, 20, 255)
        ],
    }
    
    def __init__(self, mode: str = 'opencv', confidence_threshold: float = 0.65):
        """
        Initialize sign detector
        
        Args:
            mode: Detection mode ('opencv' or 'yolo')
            confidence_threshold: Minimum confidence for detection
        """
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        
        # YOLO model (optional, for enhanced detection)
        self.yolo_model = None
        if mode == 'yolo':
            self._load_yolo_model()
        
        print(f"  ✓ Sign Detector initialized (mode: {mode})")
    
    def _load_yolo_model(self):
        """Load YOLO model for sign detection if available"""
        try:
            from ultralytics import YOLO
            
            # Try to load custom traffic sign model
            model_paths = [
                'traffic_sign_model.pt',
                'models/traffic_sign_model.pt',
                '../models/traffic_sign_model.pt',
            ]
            
            for path in model_paths:
                if Path(path).exists():
                    self.yolo_model = YOLO(path)
                    print(f"  ✓ YOLO traffic sign model loaded: {path}")
                    return
            
            print(f"  ⚠ No YOLO traffic sign model found, using OpenCV mode")
            self.mode = 'opencv'
            
        except Exception as e:
            print(f"  ⚠ Could not load YOLO model: {e}, using OpenCV mode")
            self.mode = 'opencv'
    
    def detect(self, image: np.ndarray) -> SignDetectionResult:
        """
        Detect traffic signs in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            SignDetectionResult with detected signs
        """
        import time
        start_time = time.time()
        
        if self.mode == 'yolo' and self.yolo_model:
            signs = self._detect_yolo(image)
        else:
            signs = self._detect_opencv(image)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SignDetectionResult(
            signs=signs,
            processing_time_ms=processing_time
        )
    
    def _detect_opencv(self, image: np.ndarray) -> List[TrafficSign]:
        """Detect signs using OpenCV color and shape analysis"""
        signs = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect red regions (stop signs, speed limits)
        red_masks = self._create_color_masks(hsv, 'red')
        for mask in red_masks:
            signs.extend(self._find_signs_in_mask(image, mask, 'red'))
        
        # Detect yellow regions (warning signs)
        yellow_masks = self._create_color_masks(hsv, 'yellow')
        for mask in yellow_masks:
            signs.extend(self._find_signs_in_mask(image, mask, 'yellow'))
        
        # Detect blue regions (informational signs)
        blue_masks = self._create_color_masks(hsv, 'blue')
        for mask in blue_masks:
            signs.extend(self._find_signs_in_mask(image, mask, 'blue'))
        
        # Apply NMS to remove duplicate detections
        signs = self._non_max_suppression(signs)
        
        # Filter by confidence
        signs = [s for s in signs if s.confidence >= self.confidence_threshold]
        
        return signs
    
    def _detect_yolo(self, image: np.ndarray) -> List[TrafficSign]:
        """Detect signs using YOLO model"""
        if self.yolo_model is None:
            return self._detect_opencv(image)
        
        results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)[0]
        signs = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            sign_info = self.SIGN_TYPES.get(class_name, {
                'class': 'unknown',
                'color': (255, 255, 255),
                'shape': 'unknown'
            })
            
            sign = TrafficSign(
                sign_type=class_name,
                sign_class=sign_info['class'],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                color=sign_info['color']
            )
            signs.append(sign)
        
        return signs
    
    def _create_color_masks(self, hsv: np.ndarray, color: str) -> List[np.ndarray]:
        """Create binary masks for color segmentation"""
        masks = []
        
        if color in self.COLOR_RANGES:
            ranges = self.COLOR_RANGES[color]
            
            # Handle split ranges (like red)
            for i in range(0, len(ranges), 2):
                lower = np.array(ranges[i], dtype=np.uint8)
                upper = np.array(ranges[i + 1], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                
                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                masks.append(mask)
        
        return masks
    
    def _find_signs_in_mask(self, image: np.ndarray, mask: np.ndarray, 
                            color_name: str) -> List[TrafficSign]:
        """Find sign candidates in a binary mask"""
        signs = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area (realistic sign sizes: 30x30px to 300x300px)
            if area < 900:  # Too small - noise, road markings
                continue
            if area > 90000:  # Too large - entire image region
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio filter (signs are roughly square)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Determine sign type based on shape and color
            sign_type, confidence = self._classify_sign(contour, color_name, image)
            
            if sign_type and confidence >= self.confidence_threshold:
                sign_info = self.SIGN_TYPES.get(sign_type, {
                    'class': 'unknown',
                    'color': (0, 0, 255)
                })
                
                sign = TrafficSign(
                    sign_type=sign_type,
                    sign_class=sign_info['class'],
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h),
                    center=(x + w // 2, y + h // 2),
                    color=sign_info['color']
                )
                signs.append(sign)
        
        return signs
    
    def _classify_sign(self, contour: np.ndarray, color_name: str, 
                       image: np.ndarray) -> Tuple[Optional[str], float]:
        """Classify sign based on shape and color"""
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        area = cv2.contourArea(contour)
        
        # Shape analysis
        shape = 'unknown'
        shape_confidence = 0.0
        
        if vertices == 3:
            shape = 'triangle'
            shape_confidence = 0.8
        elif vertices == 4:
            # Check if square
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2:
                shape = 'square'
                shape_confidence = 0.8
            else:
                shape = 'rectangle'
                shape_confidence = 0.7
        elif vertices == 5:
            shape = 'pentagon'
            shape_confidence = 0.7
        elif vertices == 8:
            shape = 'octagon'
            shape_confidence = 0.9
        elif vertices > 8:
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.7:
                    shape = 'circle'
                    shape_confidence = min(circularity, 0.95)
        
        # Classify based on color and shape
        sign_type = None
        confidence = shape_confidence * 0.7  # Base confidence from shape
        
        if color_name == 'red':
            if shape == 'octagon':
                sign_type = 'stop'
                confidence *= 1.2
            elif shape == 'circle':
                sign_type = 'speed_limit'
                confidence *= 1.0
            elif shape == 'triangle':
                sign_type = 'yield'
                confidence *= 0.9
        
        elif color_name == 'yellow':
            if shape == 'triangle':
                sign_type = 'pedestrian_crossing'
                confidence *= 1.0
        
        elif color_name == 'blue':
            if shape == 'square':
                sign_type = 'parking'
                confidence *= 0.9
        
        return sign_type, min(confidence, 1.0)
    
    def _non_max_suppression(self, signs: List[TrafficSign], 
                             iou_threshold: float = 0.5) -> List[TrafficSign]:
        """Remove duplicate detections using non-maximum suppression"""
        if not signs:
            return []
        
        # Sort by confidence
        signs = sorted(signs, key=lambda s: s.confidence, reverse=True)
        
        keep = []
        while signs:
            best = signs.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            signs = [s for s in signs if self._iou(best.bbox, s.bbox) < iou_threshold]
        
        return keep
    
    def _iou(self, box1: Tuple[int, int, int, int], 
             box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def draw_signs(self, image: np.ndarray, result: SignDetectionResult) -> np.ndarray:
        """Draw detected signs on image"""
        overlay = image.copy()
        
        for sign in result.signs:
            x1, y1, x2, y2 = sign.bbox
            color = sign.color
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            
            # Draw label background
            label = f"{sign.sign_type.replace('_', ' ').title()}"
            conf_label = f"{sign.confidence:.0%}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
            
            # Background rectangle
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 10, y1), color, -1)
            
            # Text
            cv2.putText(overlay, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence (smaller, bottom right)
            cv2.putText(overlay, conf_label, (x2 - text_w - 5, y1 + text_h - 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
        
        # Draw summary
        if result.signs:
            summary = f"Signs: {len(result.signs)}"
            cv2.putText(overlay, summary, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return overlay
    
    def estimate_distance(self, sign: TrafficSign, image_height: int) -> float:
        """
        Estimate distance to sign based on apparent size
        
        Args:
            sign: Detected traffic sign
            image_height: Image height in pixels
            
        Returns:
            Estimated distance in meters
        """
        # Average real-world sign size (approximately 60cm diameter)
        REAL_SIGN_SIZE = 0.6  # meters
        
        # Get sign dimensions
        w = sign.bbox[2] - sign.bbox[0]
        h = sign.bbox[3] - sign.bbox[1]
        apparent_size = max(w, h)
        
        if apparent_size > 0:
            # Simple pinhole camera model
            # distance = (real_size * focal_length) / apparent_size
            # Assuming focal length ~ image height
            focal_length = image_height
            distance = (REAL_SIGN_SIZE * focal_length) / apparent_size
            sign.distance_estimate = distance
        
        return sign.distance_estimate


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sign_detector(mode: str = 'opencv', 
                         confidence: float = 0.5) -> SignDetector:
    """Factory function to create sign detector"""
    return SignDetector(mode=mode, confidence_threshold=confidence)


if __name__ == "__main__":
    # Test sign detector
    detector = SignDetector(mode='opencv', confidence_threshold=0.5)
    
    # Create test image with simulated stop sign (red octagon)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw red octagon (approximate)
    center = (320, 240)
    radius = 50
    pts = cv2.regularPolygon2PolyPoints(center, 8, radius, 0)[0]
    pts = pts.astype(np.int32)
    cv2.fillPoly(test_image, [pts], (0, 0, 255))
    cv2.putText(test_image, "STOP", (center[0] - 30, center[0] + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    result = detector.detect(test_image)
    
    print(f"\nSign Detection Test:")
    print(f"  Signs detected: {len(result.signs)}")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")
    
    for sign in result.signs:
        print(f"    - {sign.sign_type} ({sign.confidence:.0%}) at {sign.bbox}")
    
    # Draw results
    output = detector.draw_signs(test_image, result)
    cv2.imwrite("test_sign_output.jpg", output)
    print(f"\n✓ Test output saved to: test_sign_output.jpg")
