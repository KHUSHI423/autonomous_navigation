"""
=============================================================================
EDGE DRIVE 3D - INTELLIGENT ADAS WITH LANE AVOIDANCE
=============================================================================
Advanced driver assistance system with:
- Dynamic lane shifting based on obstacles
- Intelligent collision avoidance
- Smooth steering transitions
- Clean, minimal Tesla-style HUD
- Real-time obstacle detection & path planning

Author: EdgeDrive3D Team
=============================================================================
"""

import cv2
import numpy as np
import sys
import os
import time
import argparse
from collections import deque

# Add core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class IntelligentLaneAvoidance:
    """
    Highly dynamic lane avoidance system - lanes MOVE noticeably around obstacles
    Short lane view (only 30% of frame bottom) for focused, realistic visualization
    """
    
    def __init__(self, frame_width, frame_height):
        """Initialize lane avoidance system"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Original lane points (from detection)
        self.original_left_pts = None
        self.original_right_pts = None
        
        # Adjusted lane points (after obstacle avoidance)
        self.adjusted_left_pts = None
        self.adjusted_right_pts = None
        
        # Smoothing buffers - shorter for more responsive movement
        self.left_shift_buffer = deque(maxlen=5)  # Only 5 frames = more responsive
        self.right_shift_buffer = deque(maxlen=5)
        
        # Current shift amounts (pixels)
        self.current_left_shift = 0
        self.current_right_shift = 0
        
        # Avoidance state
        self.avoidance_active = False
        self.avoidance_direction = 0  # -1=left, 0=none, 1=right
        self.avoidance_severity = 0.0  # 0.0=none, 1.0=critical
        
        # DANGER ZONE - focused on immediate path (bottom 30%)
        self.lane_display_height = 0.30  # Only show 30% of frame (very short)
        self.danger_zone_y_start = int(frame_height * (1.0 - self.lane_display_height))
        self.danger_zone_y_end = frame_height
        self.danger_zone_x_center = frame_width // 2
        self.danger_zone_width = int(frame_width * 0.5)  # Wider danger zone
        
        # Decision state
        self.decision = "FORWARD"
        self.decision_confidence = 0.0
        
        # Shift multipliers - make avoidance VERY visible
        self.max_shift_near = 120   # Max pixels to shift when object is NEAR
        self.max_shift_mid = 80    # Medium distance shift
        self.max_shift_far = 40    # Far object shift
        
        print("  ✓ Dynamic Lane Avoidance initialized (SHORT view, HIGHLY responsive)")
    
    def update_original_lanes(self, left_pts, right_pts):
        """
        Update original lane detection points
        
        Args:
            left_pts: Detected left lane points (N, 2)
            right_pts: Detected right lane points (N, 2)
        """
        self.original_left_pts = left_pts
        self.original_right_pts = right_pts
    
    def calculate_avoidance(self, objects_3d):
        """
        Calculate HIGHLY DYNAMIC lane avoidance based on detected objects
        Lanes MOVE significantly when objects are near
        """
        if self.original_left_pts is None or self.original_right_pts is None:
            return 0, 0, "FORWARD", 0.0
        
        # Find most critical object in danger zone
        critical_object = self._find_critical_object(objects_3d)
        
        if critical_object is None:
            # No obstacle, gradually return to original
            self.avoidance_active = False
            self.avoidance_severity *= 0.7  # Faster decay
            return 0, 0, "FORWARD", self.avoidance_severity
        
        # Object detected in danger zone
        self.avoidance_active = True
        
        # Calculate object position relative to lane
        obj_x = critical_object.get('center_x', self.danger_zone_x_center)
        obj_y = critical_object.get('center_y', self.danger_zone_y_end)
        obj_distance = critical_object.get('distance', 10.0)
        
        # Calculate lane center at object's Y position
        lane_center_x = self._get_lane_center_at_y(obj_y)
        
        # Determine offset from lane center
        offset_from_center = obj_x - lane_center_x
        
        # Determine avoidance direction and shift
        left_shift = 0
        right_shift = 0
        decision = "FORWARD"
        
        # Check if object is very close (critical)
        if obj_distance < 2.5:  # Slightly larger critical zone
            # STOP - too close to safely navigate
            decision = "STOP"
            self.avoidance_severity = 1.0
            # Full shift to show avoidance
            self.avoidance_direction = -1 if offset_from_center < 0 else 1
            return left_shift, right_shift, decision, self.avoidance_severity
        
        # Calculate distance-based shift multiplier
        if obj_distance < 5.0:
            max_shift = self.max_shift_near  # 120px - very visible
            decision = "AVOID NEAR"
        elif obj_distance < 10.0:
            max_shift = self.max_shift_mid   # 80px - moderate
            decision = "AVOID MID"
        else:
            max_shift = self.max_shift_far   # 40px - slight
            decision = "AVOID FAR"
        
        # Object is in lane, calculate avoidance
        if abs(offset_from_center) < 60:  # Object near lane center
            # Object directly in path, shift based on available space
            left_space = lane_center_x - self._get_left_lane_at_y(obj_y)
            right_space = self._get_right_lane_at_y(obj_y) - lane_center_x
            
            if left_space > right_space:
                # More space on left, shift left
                left_shift = -int(min(max_shift, left_space * 0.5))
                right_shift = -int(min(max_shift * 0.7, right_space * 0.4))
                decision = f"LEFT ({critical_object.get('class_name', 'obj')})"
                self.avoidance_direction = -1
            else:
                # More space on right, shift right
                left_shift = int(min(max_shift * 0.7, left_space * 0.4))
                right_shift = int(min(max_shift, right_space * 0.5))
                decision = f"RIGHT ({critical_object.get('class_name', 'obj')})"
                self.avoidance_direction = 1
        elif offset_from_center < 0:
            # Object on left side of lane
            shift_amount = int(min(max_shift, abs(offset_from_center) * 0.6))
            left_shift = -shift_amount
            right_shift = -int(shift_amount * 0.6)
            decision = f"LEFT ({critical_object.get('class_name', 'obj')})"
            self.avoidance_direction = -1
        else:
            # Object on right side of lane
            shift_amount = int(min(max_shift, offset_from_center * 0.6))
            left_shift = int(shift_amount * 0.6)
            right_shift = shift_amount
            decision = f"RIGHT ({critical_object.get('class_name', 'obj')})"
            self.avoidance_direction = 1
        
        # Calculate severity based on distance
        self.avoidance_severity = max(0.3, min(1.0, 8.0 / max(obj_distance, 1.0)))
        
        return left_shift, right_shift, decision, self.avoidance_severity
    
    def _find_critical_object(self, objects_3d):
        """Find the most critical object in the danger zone"""
        critical_obj = None
        min_distance = float('inf')
        
        for obj in objects_3d:
            # Get object bounding box center
            bbox = obj.bbox if hasattr(obj, 'bbox') else None
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if object is in danger zone
            if (self.danger_zone_y_start <= center_y <= self.danger_zone_y_end and
                self.danger_zone_x_center - self.danger_zone_width//2 <= center_x <= 
                self.danger_zone_x_center + self.danger_zone_width//2):
                
                # Get distance
                distance = obj.distance_3d if hasattr(obj, 'distance_3d') else 10.0
                
                # Find closest object
                if distance < min_distance:
                    min_distance = distance
                    critical_obj = {
                        'center_x': center_x,
                        'center_y': center_y,
                        'distance': distance,
                        'class_name': obj.class_name if hasattr(obj, 'class_name') else 'unknown'
                    }
        
        return critical_obj
    
    def _get_lane_center_at_y(self, y_pos):
        """Get lane center X position at given Y"""
        if self.original_left_pts is None or self.original_right_pts is None:
            return self.danger_zone_x_center
        
        # Find closest points to y_pos
        left_idx = np.argmin(np.abs(self.original_left_pts[:, 1] - y_pos))
        right_idx = np.argmin(np.abs(self.original_right_pts[:, 1] - y_pos))
        
        left_x = self.original_left_pts[left_idx, 0]
        right_x = self.original_right_pts[right_idx, 0]
        
        return (left_x + right_x) // 2
    
    def _get_left_lane_at_y(self, y_pos):
        """Get left lane X position at given Y"""
        if self.original_left_pts is None:
            return self.danger_zone_x_center - 100
        
        idx = np.argmin(np.abs(self.original_left_pts[:, 1] - y_pos))
        return self.original_left_pts[idx, 0]
    
    def _get_right_lane_at_y(self, y_pos):
        """Get right lane X position at given Y"""
        if self.original_right_pts is None:
            return self.danger_zone_x_center + 100
        
        idx = np.argmin(np.abs(self.original_right_pts[:, 1] - y_pos))
        return self.original_right_pts[idx, 0]
    
    def get_adjusted_lanes(self):
        """
        Get adjusted lane points with HIGHLY DYNAMIC obstacle avoidance
        Lanes VISIBLELY MOVE around obstacles
        """
        if self.original_left_pts is None or self.original_right_pts is None:
            return None, None
        
        # Calculate new shift targets
        left_shift = int(self.current_left_shift)
        right_shift = int(self.current_right_shift)
        
        # Apply smoothing (5 frames = responsive but smooth)
        self.left_shift_buffer.append(left_shift)
        self.right_shift_buffer.append(right_shift)
        
        # Use average of recent shifts
        smoothed_left_shift = int(np.mean(self.left_shift_buffer))
        smoothed_right_shift = int(np.mean(self.right_shift_buffer))
        
        # Update current shifts
        self.current_left_shift = smoothed_left_shift
        self.current_right_shift = smoothed_right_shift
        
        # Apply shifts to lane points
        adjusted_left = self.original_left_pts.copy()
        adjusted_right = self.original_right_pts.copy()
        
        # STRONGER perspective shift - much more at bottom (near)
        height = self.frame_height
        lane_top_y = int(height * (1.0 - self.lane_display_height))  # 70% height
        lane_bottom_y = height
        
        for i in range(len(adjusted_left)):
            y_pos = adjusted_left[i, 1]
            # Calculate position ratio (0 at top, 1 at bottom)
            position_ratio = max(0.0, min(1.0, (y_pos - lane_top_y) / max(lane_bottom_y - lane_top_y, 1)))
            # Aggressive perspective: 100% at bottom, 20% at top
            shift_factor = 0.2 + 0.8 * (position_ratio ** 1.5)  # Exponential for more near shift
            
            adjusted_left[i, 0] += int(smoothed_left_shift * shift_factor)
            adjusted_right[i, 0] += int(smoothed_right_shift * shift_factor)
        
        self.adjusted_left_pts = adjusted_left
        self.adjusted_right_pts = adjusted_right
        
        return adjusted_left, adjusted_right
    
    def get_lane_color(self):
        """Get lane color based on avoidance state"""
        if self.decision == "STOP":
            return (0, 0, 255)  # Red - critical
        elif self.avoidance_active:
            return (0, 255, 255)  # Yellow - avoiding
        else:
            return (0, 255, 0)  # Green - normal


class IntelligentADAS:
    """
    Complete intelligent ADAS system with ALL models integrated:
    - Object Detection (YOLOv8)
    - Dynamic Lane Avoidance (HIGHLY VISIBLE shifts)
    - Traffic Sign Detection
    - Depth Estimation (MiDaS)
    - Driver Drowsiness Detection
    - Road Hazard Detection
    - BEV Mapping (optional)
    """
    
    def __init__(self, config=None):
        """Initialize intelligent ADAS system with ALL models"""
        self.config = config or {}
        
        print("\n" + "="*70)
        print("INITIALIZING INTELLIGENT ADAS - ALL MODELS")
        print("="*70)
        
        # Initialize all components
        self._init_perception_engine()
        self._init_drowsiness_detector()
        self._init_hazard_detector()
        
        # Lane avoidance (initialized when frame size known)
        self.lane_avoidance = None
        
        # Model toggle states
        self.enable_objects = True
        self.enable_lanes = True
        self.enable_signs = True
        self.enable_depth = True
        self.enable_drowsiness = False  # Press 5 to enable
        self.enable_hazards = True
        self.enable_bev = False  # Press 7 to enable
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.start_time = time.time()
        self.paused = False
        
        print("\n" + "="*70)
        print("ALL MODELS READY - INTELLIGENT ADAS ACTIVE")
        print("="*70)
        print("\nKeyboard Shortcuts:")
        print("  1: Toggle Objects  2: Toggle Lanes  3: Toggle Signs")
        print("  4: Toggle Depth    5: Toggle Drowsy 6: Toggle Hazards")
        print("  7: Toggle BEV      P: Pause         S: Screenshot")
        print("  Q/ESC: Quit")
        print("="*70)
    
    def _init_perception_engine(self):
        """Initialize perception engine with all models"""
        from core.perception_engine import PerceptionEngine
        
        try:
            print("\n[1/4] Initializing Perception Engine (YOLO + Lanes + Signs + Depth)...")
            self.engine = PerceptionEngine({
                'yolo_model': self.config.get('yolo_model', 'yolov8n.pt'),
                'confidence': self.config.get('confidence', 0.5),
                'max_depth': self.config.get('max_depth', 50.0),
                'enable_lane': True,
                'lane_preset': self.config.get('lane_preset', 'default'),
                'enable_signs': True,
                'sign_mode': self.config.get('sign_mode', 'opencv'),
                'enable_depth': True,
                'depth_skip_frames': self.config.get('depth_skip', 2),
            })
            print("      ✓ Perception Engine ready")
        except Exception as e:
            print(f"      ⚠ Perception Engine failed: {e}")
            self.engine = None
    
    def _init_drowsiness_detector(self):
        """Initialize drowsiness detector"""
        try:
            print("\n[2/4] Initializing Drowsiness Detector...")
            self.drowsiness_detector = DrowsinessDetector()
            print("      ✓ Drowsiness Detector ready (press 5 to enable)")
        except Exception as e:
            print(f"      ⚠ Drowsiness Detector failed: {e}")
            self.drowsiness_detector = None
    
    def _init_hazard_detector(self):
        """Initialize hazard detector"""
        try:
            print("\n[3/4] Initializing Hazard Detector...")
            self.hazard_detector = HazardDetector()
            print("      ✓ Hazard Detector ready")
        except Exception as e:
            print(f"      ⚠ Hazard Detector failed: {e}")
            self.hazard_detector = None
    
    def _init_lane_avoidance(self, width, height):
        """Initialize lane avoidance when frame size is known"""
        if self.lane_avoidance is None:
            self.lane_avoidance = IntelligentLaneAvoidance(width, height)
    
    def process_frame(self, frame):
        """
        Process frame with ALL models and intelligent lane avoidance
        
        Args:
            frame: Input BGR image
            
        Returns:
            Processed frame with all intelligent overlays
        """
        self.frame_count += 1
        frame_start = time.time()
        
        # Initialize lane avoidance on first frame
        height, width = frame.shape[:2]
        self._init_lane_avoidance(width, height)
        
        # ===== RUN ALL ENABLED MODELS =====
        
        # 1. Perception Engine (YOLO + Lanes + Signs + Depth)
        perception_result = None
        if self.engine and (self.enable_objects or self.enable_signs or self.enable_depth):
            try:
                perception_result = self.engine.process_frame(frame)
            except Exception as e:
                if self.frame_count % 60 == 0:
                    print(f"  ⚠ Perception error: {e}")
        
        # 2. Update lane avoidance with detected objects
        if perception_result and hasattr(perception_result, 'objects_3d'):
            objects = perception_result.objects_3d
            
            # Update original lanes from perception
            if perception_result.lanes and perception_result.lanes.left_points is not None:
                self.lane_avoidance.update_original_lanes(
                    perception_result.lanes.left_points,
                    perception_result.lanes.right_points
                )
            
            # Calculate avoidance if lanes enabled
            if self.enable_lanes:
                left_shift, right_shift, decision, severity = self.lane_avoidance.calculate_avoidance(objects)
                self.lane_avoidance.decision = decision
                self.lane_avoidance.decision_confidence = severity
        
        # 3. Get adjusted lanes (dynamic avoidance)
        if self.enable_lanes and self.lane_avoidance:
            adjusted_left, adjusted_right = self.lane_avoidance.get_adjusted_lanes()
        
        # 4. Drowsiness Detection
        drowsiness_info = {}
        if self.enable_drowsiness and self.drowsiness_detector:
            try:
                drowsiness_info = self.drowsiness_detector.detect(frame)
            except Exception as e:
                if self.frame_count % 60 == 0:
                    print(f"  ⚠ Drowsiness error: {e}")
        
        # 5. Hazard Detection
        if self.enable_hazards and self.hazard_detector and perception_result:
            try:
                if hasattr(perception_result, 'objects_3d') and perception_result.objects_3d:
                    hazards = self.hazard_detector.detect(perception_result.objects_3d)
                else:
                    hazards = []
            except Exception as e:
                if self.frame_count % 60 == 0:
                    print(f"  ⚠ Hazard error: {e}")
        
        # ===== DRAW EVERYTHING =====
        output = self._draw_intelligent_hud(frame, perception_result, drowsiness_info)
        
        # Track performance
        process_time = time.time() - frame_start
        self.processing_times.append(process_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        return output
    
    def _draw_intelligent_hud(self, frame, perception_result, drowsiness_info):
        """Draw intelligent HUD with ALL models and SHORT lane view"""
        output = frame.copy()
        height, width = frame.shape[:2]
        
        # ===== 1. DRAW INTELLIGENT LANES (SHORT VIEW - 30% only) =====
        if (self.enable_lanes and self.lane_avoidance and
            self.lane_avoidance.adjusted_left_pts is not None and
            self.lane_avoidance.adjusted_right_pts is not None):
            
            left_pts = self.lane_avoidance.adjusted_left_pts
            right_pts = self.lane_avoidance.adjusted_right_pts
            lane_color = self.lane_avoidance.get_lane_color()
            
            # SHORT lane view - only bottom 30%
            max_y = int(height * 0.70)  # Start at 70% height (very short)
            left_filtered = left_pts[left_pts[:, 1] >= max_y]
            right_filtered = right_pts[right_pts[:, 1] >= max_y]
            
            if len(left_filtered) > 1 and len(right_filtered) > 1:
                # Create polygon
                pts = np.vstack([
                    left_filtered,
                    right_filtered[::-1]
                ]).astype(np.int32)
                
                # Draw with dynamic alpha (more visible when avoiding)
                alpha = 0.20 if not self.lane_avoidance.avoidance_active else 0.40
                
                lane_layer = output.copy()
                cv2.fillPoly(lane_layer, [pts], lane_color)
                
                mask = np.zeros(output.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                
                blended = cv2.addWeighted(output, 1.0 - alpha, lane_layer, alpha, 0)
                output = np.where(mask[:,:,np.newaxis] > 0, blended, output)
                
                # Draw thicker boundary lines (more visible)
                line_thickness = 4 if self.lane_avoidance.avoidance_active else 3
                for i in range(len(left_filtered) - 1):
                    cv2.line(output, tuple(left_filtered[i]), tuple(left_filtered[i+1]),
                           lane_color, line_thickness, cv2.LINE_AA)
                
                for i in range(len(right_filtered) - 1):
                    cv2.line(output, tuple(right_filtered[i]), tuple(right_filtered[i+1]),
                           lane_color, line_thickness, cv2.LINE_AA)
        
        # ===== 2. DRAW OBJECT DETECTIONS (Bounding boxes with labels & confidence) =====
        if self.enable_objects and perception_result and hasattr(perception_result, 'objects_3d'):
            for obj in perception_result.objects_3d:
                bbox = obj.bbox if hasattr(obj, 'bbox') else None
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    class_name = obj.class_name if hasattr(obj, 'class_name') else 'unknown'
                    confidence = obj.confidence if hasattr(obj, 'confidence') else 0.0
                    distance = obj.distance_3d if hasattr(obj, 'distance_3d') else 0.0

                    # Color based on threat level (distance)
                    if distance < 3.0:
                        box_color = (0, 0, 255)  # Red - critical
                    elif distance < 7.0:
                        box_color = (0, 165, 255)  # Orange - warning
                    else:
                        box_color = (0, 255, 0)  # Green - safe

                    # Draw bounding box (thick lines for visibility)
                    cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 3)
                    
                    # Create label with class name and confidence
                    label = f"{class_name.capitalize()}: {confidence:.2f}"
                    
                    # Get label size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw filled rectangle for label background
                    cv2.rectangle(output, 
                                 (x1, y1 - 30), 
                                 (x1 + text_width + 10, y1), 
                                 box_color, -1)
                    
                    # Draw white text on colored background
                    cv2.putText(output, label, 
                               (x1 + 5, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw distance below label if available
                    if distance > 0 and distance < 50:
                        dist_label = f"{distance:.1f}m"
                        (dist_w, dist_h), _ = cv2.getTextSize(
                            dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(output,
                                     (x1, y2 + 5),
                                     (x1 + dist_w + 8, y2 + 25),
                                     box_color, -1)
                        cv2.putText(output, dist_label,
                                   (x1 + 4, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ===== 3. DRAW DROWSINESS (if enabled) =====
        if self.enable_drowsiness and drowsiness_info and drowsiness_info.get('available'):
            output = self.drowsiness_detector.draw_status(output, drowsiness_info)
        
        # ===== 4. DRAW HAZARDS (if enabled) =====
        if self.enable_hazards and perception_result:
            try:
                if hasattr(perception_result, 'objects_3d') and perception_result.objects_3d:
                    hazards = self.hazard_detector.detect(perception_result.objects_3d)
                    if hazards:
                        output = self.hazard_detector.draw_hazards(output, hazards)
            except:
                pass
        
        # ===== 5. DRAW MINIMAL HUD =====
        
        # Bottom-left: Lane info
        x_hud = 15
        y_hud = height - 80
        line_height = 22
        
        # Lane width (if available)
        if perception_result and perception_result.lanes:
            lane_w = perception_result.lanes.lane_width_meters
            lane_color = self.lane_avoidance.get_lane_color() if self.lane_avoidance else (0, 255, 0)
            cv2.putText(output, f"LANE: {lane_w:.1f}m",
                       (x_hud, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lane_color, 2)
            
            # Offset
            offset = perception_result.lanes.vehicle_offset
            offset_color = (0, 255, 0) if abs(offset) < 0.3 else (0, 165, 255)
            cv2.putText(output, f"OFFSET: {offset:+.2f}m",
                       (x_hud, y_hud + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 1)
        
        # Top-right: System status (compact)
        x_status = width - 300
        y_status = 35
        
        cv2.putText(output, "INTELLIGENT ADAS", (x_status, y_status),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_pos = y_status + line_height
        
        # Objects
        obj_count = 0
        if perception_result and hasattr(perception_result, 'objects_3d'):
            obj_count = len(perception_result.objects_3d)
        obj_status = "ON" if self.enable_objects else "OFF"
        cv2.putText(output, f"Objects[{obj_status}]: {obj_count}",
                   (x_status, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_pos += line_height
        
        # Signs
        if self.enable_signs and perception_result and hasattr(perception_result, 'traffic_signs'):
            sign_count = len(perception_result.traffic_signs)
            if sign_count > 0:
                cv2.putText(output, f"Signs: {sign_count}",
                           (x_status, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
                y_pos += line_height
        
        # Decision (prominent, color-coded)
        decision = "FORWARD"
        decision_color = (0, 255, 0)
        
        if self.lane_avoidance and self.enable_lanes:
            decision = self.lane_avoidance.decision
            # Extract base decision (remove object name)
            base_decision = decision.split('(')[0].strip()
            decision_color = {
                'FORWARD': (0, 255, 0),
                'AVOID NEAR': (0, 165, 255),
                'AVOID MID': (0, 255, 255),
                'AVOID FAR': (255, 255, 0),
                'STOP': (0, 0, 255)
            }.get(base_decision, (255, 255, 255))
        elif not self.enable_lanes:
            decision = "LANES OFF"
            decision_color = (100, 100, 100)
        
        cv2.putText(output, f"DECISION: {decision}",
                   (x_status, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, decision_color, 2)
        y_pos += line_height
        
        # Drowsiness status
        if self.enable_drowsiness:
            drowsy_status = "ON" if self.enable_drowsiness else "OFF"
            cv2.putText(output, f"Drowsy[{drowsy_status}]",
                       (x_status, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += line_height
        
        # Model toggles
        lane_status = "ON" if self.enable_lanes else "OFF"
        cv2.putText(output, f"Models: Lane[{lane_status}]",
                   (x_status, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return output


class DrowsinessDetector:
    """Simple driver drowsiness detection"""
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.available = True
        except:
            self.available = False
    
    def detect(self, frame):
        if not self.available:
            return {'is_drowsy': False, 'error_rate': 0.0, 'available': False}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        is_drowsy = False
        error_rate = 0.0
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            if len(eyes) >= 2:
                eye_areas = [ew * eh for (ex, ey, ew, eh) in eyes]
                avg_eye_area = np.mean(eye_areas)
                face_area = w * h
                eye_to_face_ratio = avg_eye_area / face_area if face_area > 0 else 0
                if eye_to_face_ratio < 0.25:
                    is_drowsy = True
                error_rate = max(0, (0.25 - eye_to_face_ratio) / 0.25)
        return {'is_drowsy': is_drowsy, 'error_rate': error_rate, 'available': True}
    
    def draw_status(self, frame, info):
        if not info.get('available'):
            return frame
        x, y = 15, 100
        if info.get('is_drowsy'):
            color, text = (0, 0, 255), "DROWSY!"
        elif info.get('error_rate', 0) > 0.3:
            color, text = (0, 255, 255), "GETTING DROWSY"
        else:
            color, text = (0, 255, 0), "DRIVER: ALERT"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Fatigue: {info.get('error_rate', 0):.0%}", (x, y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return frame


class HazardDetector:
    """Road hazard detection"""
    def __init__(self):
        self.hazard_classes = {'person', 'bicycle', 'motorcycle', 'traffic light'}
        self.min_distance = 5.0
    
    def detect(self, objects_3d):
        hazards = []
        for obj in objects_3d:
            if hasattr(obj, 'class_name') and obj.class_name in self.hazard_classes:
                distance = obj.distance_3d if hasattr(obj, 'distance_3d') else 10.0
                if 0 < distance < self.min_distance:
                    hazards.append({
                        'class': obj.class_name,
                        'distance': distance,
                        'severity': "critical" if distance < 2.0 else "warning"
                    })
        return hazards
    
    def draw_hazards(self, frame, hazards):
        x, y = 15, 160
        for h in hazards:
            color = (0, 0, 255) if h['severity'] == 'critical' else (0, 255, 255)
            text = f"CRITICAL: {h['class']} at {h['distance']:.1f}m" if h['severity'] == 'critical' else f"WARNING: {h['class']} at {h['distance']:.1f}m"
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        return frame


def run_video(video_path, config=None):
    """Run intelligent ADAS on video with ALL models"""
    print("\n" + "="*70)
    print("RUNNING INTELLIGENT ADAS - ALL MODELS")
    print("="*70)
    
    # Initialize system
    system = IntelligentADAS(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    print(f"\n{'='*70}")
    print("INTELLIGENT BEHAVIORS:")
    print("  ✓ Dynamic lane shifting around obstacles (HIGHLY VISIBLE)")
    print("  ✓ Short lane view (bottom 30% only)")
    print("  ✓ All models integrated (YOLO+Lanes+Signs+Depth+Drowsy+Hazards)")
    print("  ✓ Color-coded feedback (Green/Yellow/Red)")
    print("="*70)
    print("Keyboard: 1-7 toggle models | P: Pause | S: Screenshot | Q: Quit\n")
    
    while True:
        if not system.paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ Video ended")
                break
            
            # Process frame
            result_frame = system.process_frame(frame)
            
            # Show
            cv2.imshow("Intelligent ADAS - All Models", result_frame)
            
            # Progress
            if system.frame_count % 30 == 0:
                avg_time = np.mean(system.processing_times[-10:]) if system.processing_times else 0
                decision = system.lane_avoidance.decision if system.lane_avoidance else 'N/A'
                print(f"Frame {system.frame_count}/{total_frames} | "
                      f"Avg: {avg_time*1000:.0f}ms | Decision: {decision}")
        else:
            # Paused
            cv2.imshow("Intelligent ADAS - All Models", result_frame if 'result_frame' in locals() else frame)
        
        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            print("\nQuitting...")
            break
        elif key == ord('p'):
            system.paused = not system.paused
            print(f"\n{'⏸ PAUSED' if system.paused else '▶ RESUMED'}")
        elif key == ord('s'):
            filename = f"screenshot_{system.frame_count}.jpg"
            cv2.imwrite(filename, result_frame if 'result_frame' in locals() else frame)
            print(f"\n✓ Screenshot saved: {filename}")
        elif key == ord('1'):
            system.enable_objects = not system.enable_objects
            print(f"\n  Objects: {'ON ✓' if system.enable_objects else 'OFF ✗'}")
        elif key == ord('2'):
            system.enable_lanes = not system.enable_lanes
            print(f"\n  Lanes: {'ON ✓' if system.enable_lanes else 'OFF ✗'}")
        elif key == ord('3'):
            system.enable_signs = not system.enable_signs
            print(f"\n  Signs: {'ON ✓' if system.enable_signs else 'OFF ✗'}")
        elif key == ord('4'):
            system.enable_depth = not system.enable_depth
            print(f"\n  Depth: {'ON ✓' if system.enable_depth else 'OFF ✗'}")
        elif key == ord('5'):
            system.enable_drowsiness = not system.enable_drowsiness
            print(f"\n  Drowsiness: {'ON ✓' if system.enable_drowsiness else 'OFF ✗'}")
        elif key == ord('6'):
            system.enable_hazards = not system.enable_hazards
            print(f"\n  Hazards: {'ON ✓' if system.enable_hazards else 'OFF ✗'}")
        elif key == ord('7'):
            system.enable_bev = not system.enable_bev
            print(f"\n  BEV: {'ON ✓' if system.enable_bev else 'OFF ✗'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    if system.processing_times:
        avg_time = np.mean(system.processing_times)
        print(f"\n{'='*70}")
        print(f"SUMMARY: {system.frame_count} frames | "
              f"Avg: {avg_time*1000:.0f}ms ({1/avg_time:.1f} FPS)")
        print(f"{'='*70}")


def run_webcam(config=None):
    """Run intelligent ADAS on webcam"""
    print("\n" + "="*70)
    print("RUNNING INTELLIGENT ADAS ON WEBCAM (REAL-TIME)")
    print("="*70)
    
    system = IntelligentADAS(config)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n❌ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n{'='*70}")
    print("REAL-TIME INTELLIGENT ADAS")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n⚠ Failed to capture frame")
            break
        
        result_frame = system.process_frame(frame)
        cv2.imshow("Intelligent ADAS - Real-Time", result_frame)
        
        if system.frame_count % 60 == 0:
            avg_time = np.mean(system.processing_times[-30:]) if system.processing_times else 0
            print(f"Frame {system.frame_count} | Avg: {avg_time*1000:.0f}ms | "
                  f"Decision: {system.lane_avoidance.decision if system.lane_avoidance else 'N/A'}")
        
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("   EDGE DRIVE 3D - INTELLIGENT ADAS WITH LANE AVOIDANCE")
    print("="*70)
    
    parser = argparse.ArgumentParser(description='Intelligent ADAS with Dynamic Lane Avoidance')
    parser.add_argument('mode', choices=['video', 'webcam'], help='Processing mode')
    parser.add_argument('input', nargs='?', help='Video file path')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--lane-preset', default='default')
    parser.add_argument('--max-depth', type=float, default=50.0)
    parser.add_argument('--depth-skip', type=int, default=2)
    parser.add_argument('--sign-mode', default='opencv')
    parser.add_argument('--enable-signs', action='store_true', default=True)
    parser.add_argument('--enable-depth', action='store_true', default=True)
    
    args = parser.parse_args()
    
    config = {
        'yolo_model': args.model,
        'confidence': args.confidence,
        'lane_preset': args.lane_preset,
        'max_depth': args.max_depth,
        'depth_skip': args.depth_skip,
        'sign_mode': args.sign_mode,
        'enable_signs': args.enable_signs,
        'enable_depth': args.enable_depth,
    }
    
    print(f"\nConfiguration:")
    print(f"  YOLO Model: {config['yolo_model']}")
    print(f"  Confidence: {config['confidence']}")
    print(f"  Lane Preset: {config['lane_preset']}")
    
    if args.mode == 'video':
        if not args.input:
            print("\n❌ Error: Video mode requires input file")
            print("Usage: python main_intelligent.py video your_video.mp4")
            return
        run_video(args.input, config)
    elif args.mode == 'webcam':
        run_webcam(config)


if __name__ == "__main__":
    main()
