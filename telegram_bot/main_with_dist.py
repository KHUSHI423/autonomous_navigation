"""
Main application for Traffic Detection with DISTANCE ESTIMATION
Shows accurate distance to detected objects instead of confidence scores
"""

import cv2
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import math
import numpy as np
import time

# Import detectors
from yolo_detector import TrafficDetector, AdvancedTrafficDetector


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║    📏 TRAFFIC DETECTOR WITH DISTANCE ESTIMATION 📐           ║
    ║          (Accurate Object Distance Measurement)              ║
    ║                                                               ║
    ║  Measures distance to: Cars, Trucks, Buses, Motorcycles,     ║
    ║                        Bicycles, Pedestrians, etc.           ║
    ║                                                               ║
    ║  Modes: image | video | webcam                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


class DistanceEstimator:
    """
    Accurate distance estimation using camera calibration and known object sizes
    """
    
    # Average real-world dimensions for common objects (in meters)
    # Source: Standard vehicle specifications
    KNOWN_OBJECT_SIZES = {
        'car': {'width': 1.8, 'height': 1.5},
        'truck': {'width': 2.5, 'height': 3.0},
        'bus': {'width': 2.5, 'height': 3.2},
        'motorcycle': {'width': 0.8, 'height': 1.2},
        'bicycle': {'width': 0.6, 'height': 1.1},
        'person': {'width': 0.5, 'height': 1.7},
        'traffic light': {'width': 0.3, 'height': 0.8},
        'stop sign': {'width': 0.75, 'height': 0.75},
        'auto_rickshaw': {'width': 1.2, 'height': 1.8},
        'animal': {'width': 0.6, 'height': 1.0},
    }
    
    def __init__(self, focal_length=None, camera_height=1.5, camera_tilt_angle=0):
        """
        Initialize distance estimator with camera parameters
        
        Args:
            focal_length: Camera focal length in pixels (auto-estimated if None)
            camera_height: Height of camera from ground in meters (default: 1.5m for typical mounting)
            camera_tilt_angle: Camera tilt angle from horizontal in degrees
        """
        self.focal_length = focal_length
        self.camera_height = camera_height
        self.camera_tilt_angle = camera_tilt_angle
        self.calibration_data = {}
        
    def calibrate_from_image(self, image_shape, known_distance, known_object_width, pixel_width):
        """
        Calibrate focal length using a known object at known distance
        
        Args:
            image_shape: (height, width) of image
            known_distance: Real distance to object in meters
            known_object_width: Real width of object in meters
            pixel_width: Width of object in pixels
        """
        # Focal length (pixels) = (pixel_width * known_distance) / known_object_width
        if pixel_width > 0 and known_object_width > 0:
            self.focal_length = (pixel_width * known_distance) / known_object_width
            self.calibration_data = {
                'image_shape': image_shape,
                'known_distance': known_distance,
                'known_object_width': known_object_width,
                'pixel_width': pixel_width,
                'calculated_focal_length': self.focal_length
            }
            return True
        return False
    
    def estimate_distance_focal_length(self, object_width_pixels, real_object_width):
        """
        Estimate distance using focal length formula
        
        Args:
            object_width_pixels: Width of object in pixels
            real_object_width: Real width of object in meters
            
        Returns:
            Distance in meters
        """
        if self.focal_length and object_width_pixels > 0 and real_object_width > 0:
            distance = (real_object_width * self.focal_length) / object_width_pixels
            return distance
        return None
    
    def estimate_distance_height_based(self, object_height_pixels, real_object_height, image_height):
        """
        Estimate distance using camera height and object height
        Useful when camera is mounted at known height
        
        Args:
            object_height_pixels: Height of object in pixels
            real_object_height: Real height of object in meters
            image_height: Image height in pixels
            
        Returns:
            Distance in meters
        """
        if object_height_pixels > 0 and real_object_height > 0:
            # Angular size approach
            angular_size = (object_height_pixels / image_height) * self.get_vertical_fov()
            if angular_size > 0:
                distance = real_object_height / math.tan(math.radians(angular_size / 2))
                return distance
        return None
    
    def get_vertical_fov(self):
        """Get vertical field of view (typical values for common cameras)"""
        # Default: 60 degrees for typical camera
        return 60.0
    
    def estimate_distance(self, bbox, image_shape, class_name, calibration_mode=False):
        """
        Main distance estimation function
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            image_shape: (height, width) of image
            class_name: Detected object class name
            calibration_mode: If True, use calibration instead of estimation
            
        Returns:
            Distance in meters or None if estimation fails
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Calculate object dimensions in pixels
        obj_width_px = x2 - x1
        obj_height_px = y2 - y1
        
        # Get known real-world dimensions
        if class_name not in self.KNOWN_OBJECT_SIZES:
            return None
            
        known_width = self.KNOWN_OBJECT_SIZES[class_name]['width']
        known_height = self.KNOWN_OBJECT_SIZES[class_name]['height']
        
        distances = []
        
        # Method 1: Focal length based (if calibrated)
        if self.focal_length:
            dist_focal = self.estimate_distance_focal_length(obj_width_px, known_width)
            if dist_focal and dist_focal > 0:
                distances.append(dist_focal)
        
        # Method 2: Height-based estimation
        dist_height = self.estimate_distance_height_based(obj_height_px, known_height, h)
        if dist_height and dist_height > 0:
            distances.append(dist_height)
        
        # Method 3: Use average of both if available, otherwise use what we have
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
            return max(avg_distance, 0.5)  # Minimum 0.5m
        
        return None
    
    def distance_to_string(self, distance):
        """Convert distance to readable string"""
        if distance is None:
            return "N/A"
        if distance < 1:
            return f"{distance*100:.0f}cm"
        elif distance < 10:
            return f"{distance:.1f}m"
        else:
            return f"{distance:.0f}m"


def detect_image_with_distance(
    image_path: str,
    output_dir: str = "output",
    model_size: str = "yolo11m.pt",
    confidence: float = 0.4,
    show_result: bool = True,
    save_json: bool = True,
    use_sahi: bool = True,
    camera_height: float = 1.5,
    camera_tilt: float = 0
):
    """
    Detect objects in image and estimate distances
    """
    image_path = str(image_path).strip().strip('"').strip("'")
    os.makedirs(output_dir, exist_ok=True)
    
    detector = TrafficDetector(
        model_size=model_size,
        confidence_threshold=confidence,
        use_sahi=use_sahi
    )
    
    distance_estimator = DistanceEstimator(
        camera_height=camera_height,
        camera_tilt_angle=camera_tilt
    )
    
    output_image = os.path.join(output_dir, f"{Path(image_path).stem}_distance.jpg")
    output_json = os.path.join(output_dir, f"{Path(image_path).stem}_distance.json")
    
    print(f"\nProcessing: {image_path}")
    
    try:
        annotated_image, detections = detector.detect_from_file(
            image_path,
            output_path=None
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        return None
    
    # Estimate distances and update display
    for det in detections:
        det['distance'] = distance_estimator.estimate_distance(
            det['bbox'],
            annotated_image.shape,
            det['class_name']
        )
        det['distance_str'] = distance_estimator.distance_to_string(det['distance'])
    
    # Draw detections with distance instead of confidence
    for det in detections:
        color = detector.COLORS.get(det['class_name'], (0, 255, 0))
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        distance_str = det['distance_str']
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label with distance
        label = f"{class_name}: {distance_str}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(annotated_image, (x1, y1 - 25), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save output
    cv2.imwrite(output_image, annotated_image)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📏 DISTANCE ESTIMATION RESULTS")
    print("=" * 60)
    print(f"Total objects detected: {len(detections)}")
    
    if detections:
        distances = [d['distance'] for d in detections if d['distance'] is not None]
        if distances:
            print(f"Closest object: {min(distances):.2f}m")
            print(f"Farthest object: {max(distances):.2f}m")
            print(f"Average distance: {sum(distances)/len(distances):.2f}m")
        
        print("\nObjects by distance:")
        for det in sorted(detections, key=lambda x: x['distance'] if x['distance'] else 9999):
            dist = det['distance_str']
            print(f"  • {det['class_name']}: {dist}")
    print("=" * 60)
    
    # Save JSON
    if save_json:
        results = {
            'image_path': image_path,
            'output_image': output_image,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': model_size,
                'confidence': confidence,
                'camera_height': camera_height,
                'camera_tilt': camera_tilt
            },
            'detections': detections
        }
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Show result
    if show_result and annotated_image is not None:
        try:
            cv2.imshow('Distance Estimation', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    
    return detections


def detect_video_with_distance(
    video_path: str,
    output_dir: str = "output",
    model_size: str = "yolo11m.pt",
    confidence: float = 0.4,
    show_preview: bool = True,
    camera_height: float = 1.5,
    camera_tilt: float = 0
):
    """
    Process video file with distance estimation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    detector = TrafficDetector(
        model_size=model_size,
        confidence_threshold=confidence,
        use_sahi=False  # Disable SAHI for video to maintain FPS
    )
    
    distance_estimator = DistanceEstimator(
        camera_height=camera_height,
        camera_tilt_angle=camera_tilt
    )
    
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_distance.mp4")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print("Processing video with distance estimation... Press 'q' to quit")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects
        try:
            _, detections = detector.detect(frame)
        except:
            continue
        
        # Estimate distances
        for det in detections:
            det['distance'] = distance_estimator.estimate_distance(
                det['bbox'],
                frame.shape,
                det['class_name']
            )
            det['distance_str'] = distance_estimator.distance_to_string(det['distance'])
        
        # Draw detections with distance
        for det in detections:
            color = detector.COLORS.get(det['class_name'], (0, 255, 0))
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            distance_str = det['distance_str']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with distance
            label = f"{class_name}: {distance_str}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add distance scale bar at bottom
        frame = draw_distance_scale(frame)
        
        # Add info text
        info_text = f"Frame {frame_count}/{total_frames}"
        cv2.putText(frame, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Show preview
        if show_preview:
            cv2.imshow('Distance Estimation - Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n\nProcessed {frame_count} frames in {elapsed:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Saved video to: {output_path}")


def draw_distance_scale(frame):
    """Draw a distance scale reference at the bottom of the frame"""
    h, w = frame.shape[:2]
    
    # Scale bar parameters
    bar_y = h - 60
    bar_height = 20
    bar_max_width = 200  # pixels representing 10m
    
    # Draw background
    cv2.rectangle(frame, (10, bar_y - 25), (250, bar_y + bar_height + 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, bar_y - 25), (250, bar_y + bar_height + 10), (255, 255, 255), 1)
    
    # Draw text
    cv2.putText(frame, "Distance Scale:", (20, bar_y - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw scale bar segments (each segment = 2m)
    segment_width = bar_max_width / 5
    for i in range(6):
        x = 20 + int(i * segment_width)
        # Alternate colors for visibility
        color = (0, 255, 0) if i % 2 == 0 else (255, 255, 255)
        cv2.line(frame, (x, bar_y), (x, bar_y + bar_height), color, 2)
        # Add distance labels
        if i < 5:
            dist_label = f"{i*2}m"
            cv2.putText(frame, dist_label, (x + 3, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame


def detect_webcam_with_distance(
    camera_id: int = 0,
    model_size: str = "yolo11n.pt",
    confidence: float = 0.4,
    output_path: str = None,
    save_video: bool = False,
    camera_height: float = 1.5,
    camera_tilt: float = 0
):
    """
    Real-time detection with distance estimation from webcam
    """
    detector = TrafficDetector(
        model_size=model_size,
        confidence_threshold=confidence,
        use_sahi=False
    )
    
    distance_estimator = DistanceEstimator(
        camera_height=camera_height,
        camera_tilt_angle=camera_tilt
    )
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_id}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera: {width}x{height} @ {fps}fps")
    print("Starting real-time distance estimation... Press 'q' to quit")
    
    # Optionally save video
    out = None
    if save_video and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording to: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects
        try:
            _, detections = detector.detect(frame)
        except:
            continue
        
        # Estimate distances
        for det in detections:
            det['distance'] = distance_estimator.estimate_distance(
                det['bbox'],
                frame.shape,
                det['class_name']
            )
            det['distance_str'] = distance_estimator.distance_to_string(det['distance'])
        
        # Draw detections with distance
        for det in detections:
            color = detector.COLORS.get(det['class_name'], (0, 255, 0))
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            distance_str = det['distance_str']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with distance
            label = f"{class_name}: {distance_str}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add distance scale
        frame = draw_distance_scale(frame)
        
        # Add FPS counter
        elapsed = time.time() - start_time
        if elapsed > 0:
            cv2.putText(frame, f"FPS: {frame_count/elapsed:.1f}",
                       (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Distance Estimation - Webcam', frame)
        
        if out:
            out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(description='Traffic Detection with Distance Estimation')
    subparsers = parser.add_subparsers(dest='mode', help='Detection mode')
    
    # Image mode
    img_p = subparsers.add_parser('image', help='Single image')
    img_p.add_argument('input', help='Input path')
    img_p.add_argument('-o', '--output', default='output')
    img_p.add_argument('-m', '--model', default='yolo11m.pt')
    img_p.add_argument('-c', '--confidence', type=float, default=0.4)
    img_p.add_argument('--no-sahi', action='store_false', dest='use_sahi', default=True)
    img_p.add_argument('--camera-height', type=float, default=1.5, help='Camera height from ground (meters)')
    img_p.add_argument('--camera-tilt', type=float, default=0, help='Camera tilt angle (degrees)')
    
    # Video mode
    vid_p = subparsers.add_parser('video', help='Video file')
    vid_p.add_argument('input')
    vid_p.add_argument('-o', '--output', default='output')
    vid_p.add_argument('-m', '--model', default='yolo11m.pt')
    vid_p.add_argument('-c', '--confidence', type=float, default=0.4)
    vid_p.add_argument('--camera-height', type=float, default=1.5, help='Camera height from ground (meters)')
    vid_p.add_argument('--camera-tilt', type=float, default=0, help='Camera tilt angle (degrees)')
    
    # Webcam mode
    web_p = subparsers.add_parser('webcam', help='Webcam')
    web_p.add_argument('-i', '--camera-id', type=int, default=0)
    web_p.add_argument('-m', '--model', default='yolo11n.pt')
    web_p.add_argument('-o', '--output', help='Output video path')
    web_p.add_argument('--save', action='store_true', help='Enable video recording')
    web_p.add_argument('--camera-height', type=float, default=1.5, help='Camera height from ground (meters)')
    web_p.add_argument('--camera-tilt', type=float, default=0, help='Camera tilt angle (degrees)')
    
    args = parser.parse_args()
    
    import time as time_module
    start_time = time_module.time()
    
    if args.mode == 'image':
        detect_image_with_distance(
            args.input, args.output, args.model, args.confidence,
            use_sahi=args.use_sahi,
            camera_height=args.camera_height,
            camera_tilt=args.camera_tilt
        )
    elif args.mode == 'video':
        detect_video_with_distance(
            args.input, args.output, args.model, args.confidence,
            camera_height=args.camera_height,
            camera_tilt=args.camera_tilt
        )
    elif args.mode == 'webcam':
        detect_webcam_with_distance(
            args.camera_id, args.model,
            output_path=args.output,
            save_video=args.save,
            camera_height=args.camera_height,
            camera_tilt=args.camera_tilt
        )
    else:
        print("Please specify a mode (image, video, webcam). Use --help for info.")


if __name__ == "__main__":
    main()
