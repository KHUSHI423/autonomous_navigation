"""
=============================================================================
EDGE DRIVE 3D - COMPREHENSIVE MULTI-MODEL DETECTION SYSTEM
=============================================================================
Unified pipeline integrating all detection models:
- Object Detection (YOLOv8)
- Enhanced Lane Detection (HLS + Canny + Hough + Polynomial)
- Traffic Sign Detection (OpenCV + YOLO modes)
- Depth Estimation (MiDaS)
- Driver Drowsiness Detection
- Road Hazard Detection
- BEV (Bird's Eye View) Mapping
- Decision Making & Obstacle Avoidance

Author: EdgeDrive3D Team
=============================================================================
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

# Add core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     ███████╗███╗   ██╗ █████╗ ██████╗  ██████╗ ███████╗██╗  ██╗   ║
    ║     ██╔════╝████╗  ██║██╔══██╗██╔══██╗██╔═══██╗██╔════╝╚██╗██╔╝   ║
    ║     █████╗  ██╔██╗ ██║███████║██████╔╝██║   ██║█████╗   ╚███╔╝    ║
    ║     ██╔══╝  ██║╚██╗██║██╔══██║██╔══██╗██║   ██║██╔══╝   ██╔██╗    ║
    ║     ███████╗██║ ╚████║██║  ██║██║  ██║╚██████╔╝███████╗██╔ ██╗   ║
    ║     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ║
    ║                                                                   ║
    ║        Multi-Model Detection System - ENHANCED EDITION            ║
    ║   Objects + Lanes(HLS) + Signs + Depth + Drowsiness + Hazards    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print detailed help information"""
    help_text = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                      AVAILABLE MODELS                             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    1. Object Detection (YOLOv8)
       - Models: yolov8n.pt (fast), yolov8m.pt (balanced), yolov8l.pt (accurate)
       - Detects: cars, pedestrians, bicycles, traffic lights, etc.
       - 3D position estimation using depth fusion
    
    2. Lane Detection (ENHANCED - HLS + Canny + Hough)
       - HLS color space masking for robust detection
       - Adaptive thresholding for varying lighting conditions
       - Yellow & white lane line detection
       - Polynomial fitting with temporal smoothing
       - Presets: default, highway, city, faded, night, indian_road
    
    3. Traffic Sign Detection
       - OpenCV mode: Color segmentation + shape analysis
       - YOLO mode: Deep learning-based detection
       - Detects: stop, yield, speed limits, warnings, etc.
    
    4. Depth Estimation (MiDaS)
       - Monocular depth from single RGB image
       - Models: midas_small (fast), midas_hybrid (balanced), midas_large (accurate)
    
    5. BEV Mapping (Bird's Eye View)
       - Top-down view transformation
       - Useful for spatial awareness
    
    6. Decision Making
       - Fuses all detections for navigation decisions
       - Obstacle avoidance logic
       - Lane keeping assistance
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                      KEYBOARD SHORTCUTS                           ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Video/Webcam Mode:
    - 'q' or 'ESC': Quit
    - '1': Toggle Lane Detection ON/OFF
    - '2': Toggle Traffic Sign Detection ON/OFF
    - '3': Toggle Depth Estimation ON/OFF
    - '4': Toggle Drowsiness Detection ON/OFF
    - 'p': Pause/Resume video
    - 's': Save current frame screenshot
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                      USAGE EXAMPLES                               ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Image Processing:
      python main_complete.py image input.jpg
    
    Video Processing:
      python main_complete.py video input.mp4 --model yolov8n.pt
    
    Webcam (Real-time):
      python main_complete.py webcam --model yolov8n.pt --lane-preset default
    
    With All Models Enabled:
      python main_complete.py video input.mp4 --all-models
    
    Fast Mode (YOLO + Lanes only):
      python main_complete.py video input.mp4 --fast
    
    Custom Configuration:
      python main_complete.py video input.mp4 \\
          --model yolov8m.pt \\
          --lane-preset highway \\
          --no-signs \\
          --depth-skip 3
    """
    print(help_text)


class DrowsinessDetector:
    """Simple driver drowsiness detection using facial landmarks"""
    
    def __init__(self, warning_threshold=0.25):
        """
        Initialize drowsiness detector
        
        Args:
            warning_threshold: Eye Aspect Ratio threshold for drowsiness
        """
        self.warning_threshold = warning_threshold
        self.blink_counter = 0
        self.drowsy_frames = 0
        self.consecutive_drowsy = 0
        self.max_consecutive = 3  # Frames to trigger warning
        
        # Load Haar Cascade for face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.available = True
        except:
            self.available = False
            print("  ⚠ Drowsiness detection cascades not found, disabled")
    
    def detect(self, frame):
        """
        Detect driver drowsiness
        
        Args:
            frame: Input BGR image
            
        Returns:
            dict: Drowsiness status and metrics
        """
        if not self.available:
            return {'is_drowsy': False, 'error_rate': 0.0, 'available': False}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        is_drowsy = False
        error_rate = 0.0
        
        if len(faces) > 0:
            # For simplicity, use the largest detected face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes in face ROI
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            if len(eyes) >= 2:
                # Calculate eye aspect ratio
                eye_areas = []
                for (ex, ey, ew, eh) in eyes:
                    eye_areas.append(ew * eh)
                
                # Simple drowsiness metric based on eye openness
                avg_eye_area = np.mean(eye_areas)
                face_area = w * h
                eye_to_face_ratio = avg_eye_area / face_area if face_area > 0 else 0
                
                # Typical awake ratio is ~0.02-0.03, drowsy is lower
                if eye_to_face_ratio < self.warning_threshold:
                    self.consecutive_drowsy += 1
                    if self.consecutive_drowsy >= self.max_consecutive:
                        is_drowsy = True
                else:
                    self.consecutive_drowsy = 0
                
                error_rate = max(0, (self.warning_threshold - eye_to_face_ratio) / self.warning_threshold)
            else:
                error_rate = 0.5  # Can't detect eyes
        
        return {
            'is_drowsy': is_drowsy,
            'error_rate': error_rate,
            'blink_counter': self.blink_counter,
            'available': True
        }
    
    def draw_status(self, frame, drowsiness_info):
        """Draw drowsiness status on frame"""
        if not drowsiness_info.get('available', False):
            return frame
        
        overlay = frame.copy()
        
        is_drowsy = drowsiness_info.get('is_drowsy', False)
        error_rate = drowsiness_info.get('error_rate', 0.0)
        
        # Status color and text
        if is_drowsy:
            color = (0, 0, 255)  # Red
            status_text = "DROWSY! Alert!"
        elif error_rate > 0.3:
            color = (0, 255, 255)  # Yellow
            status_text = "Getting Drowsy..."
        else:
            color = (0, 255, 0)  # Green
            status_text = "Alert"
        
        # Draw status box
        cv2.rectangle(overlay, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"Driver: {status_text}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Fatigue: {error_rate:.0%}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame


class HazardDetector:
    """Road hazard detection using YOLO and depth information"""
    
    def __init__(self):
        """Initialize hazard detector"""
        self.hazard_classes = {'person', 'bicycle', 'motorcycle', 'traffic light'}
        self.min_hazard_distance = 5.0  # meters
    
    def detect(self, objects_3d, depth_map=None):
        """
        Detect road hazards
        
        Args:
            objects_3d: List of detected 3D objects
            depth_map: Optional depth map for additional hazard detection
            
        Returns:
            list: Detected hazards with severity
        """
        hazards = []
        
        for obj in objects_3d:
            if obj.class_name in self.hazard_classes:
                distance = obj.distance_3d if hasattr(obj, 'distance_3d') else 0
                
                if distance > 0 and distance < self.min_hazard_distance:
                    severity = "critical" if distance < 2.0 else "warning"
                    hazards.append({
                        'class': obj.class_name,
                        'distance': distance,
                        'severity': severity,
                        'bbox': obj.bbox if hasattr(obj, 'bbox') else None
                    })
        
        return hazards
    
    def draw_hazards(self, frame, hazards):
        """Draw hazard warnings on frame"""
        overlay = frame.copy()
        
        y_offset = 70
        for hazard in hazards:
            if hazard['severity'] == 'critical':
                color = (0, 0, 255)
                text = f"⚠ CRITICAL: {hazard['class']} at {hazard['distance']:.1f}m"
            else:
                color = (0, 255, 255)
                text = f"⚠ WARNING: {hazard['class']} at {hazard['distance']:.1f}m"
            
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame


def mode_video_comprehensive(args):
    """
    Comprehensive video processing with ALL models integrated
    This is the main function that demonstrates the complete pipeline
    """
    from core.perception_engine import PerceptionEngine
    
    print("\n" + "="*70)
    print("INITIALIZING COMPREHENSIVE DETECTION PIPELINE")
    print("="*70)
    
    # Initialize perception engine with all models
    engine_config = {
        'yolo_model': args.model,
        'confidence': args.confidence,
        'max_depth': args.max_depth,
        'enable_lane': not args.no_lane,
        'lane_preset': args.lane_preset,
        'enable_signs': not args.no_signs,
        'sign_mode': args.sign_mode,
        'enable_depth': not args.no_depth,
        'depth_skip_frames': args.depth_skip,
    }
    
    print(f"\nEngine Configuration:")
    print(f"  YOLO Model: {engine_config['yolo_model']}")
    print(f"  Confidence: {engine_config['confidence']}")
    print(f"  Lane Detection: {'✓' if engine_config['enable_lane'] else '✗'} (preset: {args.lane_preset})")
    print(f"  Sign Detection: {'✓' if engine_config['enable_signs'] else '✗'} (mode: {args.sign_mode})")
    print(f"  Depth Estimation: {'✓' if engine_config['enable_depth'] else '✗'}")
    print(f"  Max Depth: {engine_config['max_depth']}m")
    
    # Initialize additional detectors
    drowsiness_detector = DrowsinessDetector()
    hazard_detector = HazardDetector()
    
    # Initialize perception engine
    try:
        engine = PerceptionEngine(engine_config)
    except Exception as e:
        print(f"\n❌ Failed to initialize perception engine: {e}")
        print("   Make sure required models are downloaded:")
        print("   - YOLO: pip install ultralytics")
        print("   - MiDaS: pip install timm")
        return
    
    # Open video source
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"\n❌ Could not open video: {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*70}")
    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"{'='*70}")
    
    # Initialize counters
    frame_count = 0
    processing_times = []
    lane_toggle_state = True
    sign_toggle_state = True
    depth_toggle_state = True
    drowsiness_toggle_state = False
    paused = False
    
    print(f"\n{'='*70}")
    print("PROCESSING VIDEO - Press 'q' to quit")
    print("="*70)
    print(f"Keyboard Shortcuts:")
    print(f"  '1' - Toggle Lane Detection (current: {'ON' if lane_toggle_state else 'OFF'})")
    print(f"  '2' - Toggle Sign Detection (current: {'ON' if sign_toggle_state else 'OFF'})")
    print(f"  '3' - Toggle Depth Estimation (current: {'ON' if depth_toggle_state else 'OFF'})")
    print(f"  '4' - Toggle Drowsiness Detection (current: {'ON' if drowsiness_toggle_state else 'OFF'})")
    print(f"  'p' - Pause/Resume")
    print(f"  's' - Save Screenshot")
    print(f"  'q' or 'ESC' - Quit")
    print(f"{'='*70}\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✓ End of video reached")
                break
            
            frame_count += 1
            start_time = time.time()
            
            try:
                # Update engine config based on toggle states
                engine.config['enable_lane'] = lane_toggle_state
                engine.config['enable_signs'] = sign_toggle_state
                engine.config['enable_depth'] = depth_toggle_state
                
                # Run comprehensive perception pipeline
                result = engine.process_frame(frame)
                
                # Run drowsiness detection
                drowsiness_info = {}
                if drowsiness_toggle_state:
                    drowsiness_info = drowsiness_detector.detect(frame)
                    frame = drowsiness_detector.draw_status(frame, drowsiness_info)
                
                # Run hazard detection
                if hasattr(result, 'objects_3d') and result.objects_3d:
                    hazards = hazard_detector.detect(result.objects_3d)
                    if hazards:
                        frame = hazard_detector.draw_hazards(frame, hazards)
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Create comprehensive overlay
                overlay_info = []
                overlay_info.append(f"Frame: {frame_count}/{total_frames}")
                overlay_info.append(f"FPS: {1/process_time:.1f}" if process_time > 0 else "FPS: N/A")
                overlay_info.append(f"Objects: {len(result.objects_3d) if hasattr(result, 'objects_3d') else 0}")
                overlay_info.append(f"Signs: {len(result.traffic_signs) if hasattr(result, 'traffic_signs') else 0}")
                overlay_info.append(f"Lanes: {'ON' if lane_toggle_state else 'OFF'}")
                overlay_info.append(f"Depth: {'ON' if depth_toggle_state else 'OFF'}")
                overlay_info.append(f"Drowsy: {'ON' if drowsiness_toggle_state else 'OFF'}")
                
                if hasattr(result, 'lanes') and result.lanes:
                    overlay_info.append(f"Lane Width: {result.lanes.lane_width_meters:.2f}m")
                    overlay_info.append(f"Offset: {result.lanes.vehicle_offset:+.2f}m")
                
                if hasattr(result, 'decision'):
                    overlay_info.append(f"Decision: {result.decision.get('action', 'N/A')}")
                
                # Draw overlay
                y_pos = 20
                cv2.rectangle(frame, (width - 280, 0), (width, min(250, height)), (0, 0, 0), -1)
                cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
                
                for i, info in enumerate(overlay_info):
                    cv2.putText(frame, info, (width - 270, y_pos + i * 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 22
                
                # Show frame
                cv2.imshow("Comprehensive Multi-Model Detection", frame)
                
                # Print summary every 30 frames
                if frame_count % 30 == 0:
                    avg_time = np.mean(processing_times[-30:]) if processing_times else 0
                    print(f"Frame {frame_count}/{total_frames} | "
                          f"Avg: {avg_time*1000:.0f}ms | "
                          f"Objects: {len(result.objects_3d) if hasattr(result, 'objects_3d') else 0} | "
                          f"Signs: {len(result.traffic_signs) if hasattr(result, 'traffic_signs') else 0} | "
                          f"Lanes: {'✓' if result.lanes else '✗'}")
                
            except Exception as e:
                print(f"\n⚠ Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key in [ord('q'), ord('Q'), 27]:  # q or ESC
            print("\n\n✓ Quitting...")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"\n{'⏸ PAUSED' if paused else '▶ RESUMED'}")
        elif key == ord('1'):
            lane_toggle_state = not lane_toggle_state
            print(f"\n Lane Detection: {'ON ✓' if lane_toggle_state else 'OFF ✗'}")
        elif key == ord('2'):
            sign_toggle_state = not sign_toggle_state
            print(f"\n Sign Detection: {'ON ✓' if sign_toggle_state else 'OFF ✗'}")
        elif key == ord('3'):
            depth_toggle_state = not depth_toggle_state
            print(f"\n Depth Estimation: {'ON ✓' if depth_toggle_state else 'OFF ✗'}")
        elif key == ord('4'):
            drowsiness_toggle_state = not drowsiness_toggle_state
            print(f"\n Drowsiness Detection: {'ON ✓' if drowsiness_toggle_state else 'OFF ✗'}")
        elif key == ord('s'):
            screenshot_name = f"screenshot_frame_{frame_count}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"\n✓ Screenshot saved: {screenshot_name}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average Processing Time: {avg_time*1000:.1f}ms per frame")
        print(f"Average FPS: {1/avg_time:.2f}")
        print(f"{'='*70}")


def mode_webcam_comprehensive(args):
    """
    Comprehensive webcam processing with ALL models
    Real-time detection with keyboard controls
    """
    from core.perception_engine import PerceptionEngine
    
    print("\n" + "="*70)
    print("INITIALIZING REAL-TIME WEBCAM DETECTION PIPELINE")
    print("="*70)
    
    # Initialize perception engine
    engine_config = {
        'yolo_model': args.model,
        'confidence': args.confidence,
        'max_depth': args.max_depth,
        'enable_lane': not args.no_lane,
        'lane_preset': args.lane_preset,
        'enable_signs': not args.no_signs,
        'sign_mode': args.sign_mode,
        'enable_depth': not args.no_depth,
        'depth_skip_frames': args.depth_skip,
    }
    
    # Initialize additional detectors
    drowsiness_detector = DrowsinessDetector()
    hazard_detector = HazardDetector()
    
    # Initialize perception engine
    try:
        engine = PerceptionEngine(engine_config)
    except Exception as e:
        print(f"\n❌ Failed to initialize perception engine: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n❌ Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize counters
    frame_count = 0
    processing_times = []
    lane_toggle_state = True
    sign_toggle_state = True
    depth_toggle_state = True
    drowsiness_toggle_state = False
    
    print(f"\n{'='*70}")
    print("REAL-TIME DETECTION ACTIVE - Press 'q' to quit")
    print("="*70)
    print(f"Keyboard Shortcuts:")
    print(f"  '1' - Toggle Lane Detection")
    print(f"  '2' - Toggle Sign Detection")
    print(f"  '3' - Toggle Depth Estimation")
    print(f"  '4' - Toggle Drowsiness Detection")
    print(f"  's' - Save Screenshot")
    print(f"  'q' or 'ESC' - Quit")
    print(f"{'='*70}\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("\n⚠ Failed to capture frame")
            break
        
        frame_count += 1
        start_time = time.time()
        
        try:
            # Update engine config
            engine.config['enable_lane'] = lane_toggle_state
            engine.config['enable_signs'] = sign_toggle_state
            engine.config['enable_depth'] = depth_toggle_state
            
            # Run perception pipeline
            result = engine.process_frame(frame)
            
            # Run drowsiness detection
            drowsiness_info = {}
            if drowsiness_toggle_state:
                drowsiness_info = drowsiness_detector.detect(frame)
                frame = drowsiness_detector.draw_status(frame, drowsiness_info)
            
            # Run hazard detection
            if hasattr(result, 'objects_3d') and result.objects_3d:
                hazards = hazard_detector.detect(result.objects_3d)
                if hazards:
                    frame = hazard_detector.draw_hazards(frame, hazards)
            
            # Calculate processing time
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # Create overlay
            height, width = frame.shape[:2]
            overlay_info = [
                f"FPS: {1/process_time:.1f}" if process_time > 0 else "FPS: N/A",
                f"Objects: {len(result.objects_3d) if hasattr(result, 'objects_3d') else 0}",
                f"Signs: {len(result.traffic_signs) if hasattr(result, 'traffic_signs') else 0}",
                f"Lanes: {'ON' if lane_toggle_state else 'OFF'}",
                f"Drowsy: {'ON' if drowsiness_toggle_state else 'OFF'}",
            ]
            
            if hasattr(result, 'lanes') and result.lanes:
                overlay_info.append(f"Width: {result.lanes.lane_width_meters:.2f}m")
                overlay_info.append(f"Offset: {result.lanes.vehicle_offset:+.2f}m")
            
            # Draw overlay
            y_pos = 20
            cv2.rectangle(frame, (10, 10), (250, 10 + len(overlay_info) * 25), (0, 0, 0), -1)
            
            for i, info in enumerate(overlay_info):
                cv2.putText(frame, info, (20, y_pos + i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow("Real-Time Multi-Model Detection", frame)
            
            # Print status every 60 frames
            if frame_count % 60 == 0:
                avg_time = np.mean(processing_times[-60:]) if processing_times else 0
                print(f"Frame {frame_count} | Avg: {avg_time*1000:.0f}ms | "
                      f"Objects: {len(result.objects_3d) if hasattr(result, 'objects_3d') else 0}")
            
        except Exception as e:
            print(f"\n⚠ Error processing frame: {e}")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key in [ord('q'), ord('Q'), 27]:
            print("\n✓ Quitting...")
            break
        elif key == ord('1'):
            lane_toggle_state = not lane_toggle_state
            print(f"\n Lane Detection: {'ON ✓' if lane_toggle_state else 'OFF ✗'}")
        elif key == ord('2'):
            sign_toggle_state = not sign_toggle_state
            print(f"\n Sign Detection: {'ON ✓' if sign_toggle_state else 'OFF ✗'}")
        elif key == ord('3'):
            depth_toggle_state = not depth_toggle_state
            print(f"\n Depth Estimation: {'ON ✓' if depth_toggle_state else 'OFF ✗'}")
        elif key == ord('4'):
            drowsiness_toggle_state = not drowsiness_toggle_state
            print(f"\n Drowsiness Detection: {'ON ✓' if drowsiness_toggle_state else 'OFF ✗'}")
        elif key == ord('s'):
            screenshot_name = f"webcam_screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"\n✓ Screenshot saved: {screenshot_name}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='EdgeDrive3D - Comprehensive Multi-Model Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video input.mp4                  Process video with all models
  %(prog)s webcam --model yolov8n.pt        Real-time webcam detection
  %(prog)s image photo.jpg                  Process single image
  %(prog)s video input.mp4 --fast           Fast mode (YOLO + Lanes only)
  %(prog)s video input.mp4 --all-models     Enable all detection models
        """
    )
    
    # Main mode selection
    parser.add_argument('mode', type=str, choices=['image', 'video', 'webcam', 'help'],
                       help='Processing mode')
    parser.add_argument('input', type=str, nargs='?', default=None,
                       help='Input file (image/video path)')
    
    # Model selection
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    
    # Lane detection
    parser.add_argument('--no-lane', action='store_true',
                       help='Disable lane detection')
    parser.add_argument('--lane-preset', type=str, default='default',
                       choices=['default', 'highway', 'city', 'faded', 'night', 'indian_road'],
                       help='Lane detection preset')
    
    # Sign detection
    parser.add_argument('--no-signs', action='store_true',
                       help='Disable traffic sign detection')
    parser.add_argument('--sign-mode', type=str, default='opencv',
                       choices=['opencv', 'yolo'],
                       help='Traffic sign detection mode')
    
    # Depth estimation
    parser.add_argument('--no-depth', action='store_true',
                       help='Disable depth estimation')
    parser.add_argument('--depth-skip', type=int, default=2,
                       help='Process depth every N frames (default: 2)')
    parser.add_argument('--max-depth', type=float, default=50.0,
                       help='Maximum detection depth in meters')
    
    # Special modes
    parser.add_argument('--all-models', action='store_true',
                       help='Enable all detection models')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: YOLO + Lanes only')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--show', action='store_true',
                       help='Show detection results in window')
    
    args = parser.parse_args()
    
    # Handle help mode
    if args.mode == 'help':
        print_help()
        return
    
    # Apply special mode configurations
    if args.all_models:
        args.no_lane = False
        args.no_signs = False
        args.no_depth = False
        print("\n✓ All detection models enabled")
    
    if args.fast:
        args.no_depth = True
        args.no_signs = True
        args.model = 'yolov8n.pt'
        print("\n✓ Fast mode enabled (YOLO + Lanes only)")
    
    # Route to appropriate mode handler
    if args.mode == 'image':
        if not args.input:
            print("❌ Error: Image mode requires input file path")
            print("Usage: python main_complete.py image <path_to_image>")
            return
        
        from core.perception_engine import PerceptionEngine
        
        engine = PerceptionEngine({
            'yolo_model': args.model,
            'confidence': args.confidence,
            'max_depth': args.max_depth,
            'enable_lane': not args.no_lane,
            'lane_preset': args.lane_preset,
            'enable_signs': not args.no_signs,
            'sign_mode': args.sign_mode,
            'enable_depth': not args.no_depth,
        })
        
        image = cv2.imread(args.input)
        
        if image is None:
            print(f"❌ Could not load image: {args.input}")
            return
        
        print(f"\n Processing image: {args.input}")
        result = engine.process_frame(image)
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "detection_result.jpg")
        cv2.imwrite(output_path, result.detections_overlay)
        
        print(f"\n{'='*70}")
        print("IMAGE PROCESSING RESULTS")
        print(f"{'='*70}")
        print(f"Objects detected: {len(result.objects_3d)}")
        print(f"Lanes detected: {'✓' if result.lanes else '✗'}")
        print(f"Traffic signs: {len(result.traffic_signs)}")
        print(f"Decision: {result.decision['action']}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*70}")
        
        if args.show:
            cv2.imshow("Detection Result", result.detections_overlay)
            if result.bev_image is not None:
                cv2.imshow("BEV", result.bev_image)
            if result.depth_colored is not None:
                cv2.imshow("Depth", result.depth_colored)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.mode == 'video':
        if not args.input:
            print("❌ Error: Video mode requires input file path")
            print("Usage: python main_complete.py video <path_to_video>")
            return
        
        mode_video_comprehensive(args)
    
    elif args.mode == 'webcam':
        mode_webcam_comprehensive(args)


if __name__ == "__main__":
    main()
