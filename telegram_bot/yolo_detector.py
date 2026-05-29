"""
YOLOv8-based Road and Traffic Infrastructure Detection
Detects: cars, trucks, buses, motorcycles, bicycles, pedestrians,
         traffic lights, stop signs, and more
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import os
from typing import List, Tuple, Dict, Optional
import json
import torch

# SAHI imports for high-precision small object detection
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


class TrafficDetector:
    """
    Traffic and Road Infrastructure Object Detector upgraded for Indian Road Conditions.
    Uses YOLOv11 and SAHI for 95%+ accuracy targets.
    """
    
    # IDD (Indian Driving Dataset) inspired class mapping
    # This expands COCO to handle unstructured Indian traffic
    INDIAN_TRAFFIC_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        9: 'traffic light',
        11: 'stop sign',
        # Mapping additional IDD-specific or common Indian vehicles if using custom weights
        # For base YOLO, we map the most relevant COCO classes
        'auto_rickshaw': 2, # Often detected as car/truck in base YOLO
        'animal': 15,       # COCO 'cat', 'dog', 'horse', 'sheep', 'cow' (15-19)
        'rider': 3          # COCO motorcycle/bicycle rider
    }
    
    # Color palette for different object types (BGR format)
    COLORS = {
        'person': (0, 255, 255),      # Yellow
        'bicycle': (255, 165, 0),      # Orange
        'car': (0, 255, 0),            # Green
        'motorcycle': (255, 0, 255),   # Magenta
        'bus': (255, 0, 0),            # Blue
        'truck': (0, 165, 255),        # Orange-Red
        'traffic light': (0, 0, 255),  # Red
        'stop sign': (0, 0, 200),      # Dark Red
        'auto_rickshaw': (0, 200, 255),# Sky Blue
        'animal': (255, 255, 255),     # White
    }
    
    def __init__(
        self, 
        model_size: str = 'yolo11m.pt',
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = None,
        use_sahi: bool = True
    ):
        """
        Initialize the Upgraded Traffic Detector
        
        Args:
            model_size: YOLO model size ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt')
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ('cpu', 'cuda', or None for auto)
            use_sahi: Enable Slicing Aided Hyper Inference for small objects
        """
        print(f"Loading Model: {model_size}...")
        self.model_path = model_size
        self.model = YOLO(model_size)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_sahi = use_sahi and SAHI_AVAILABLE
        
        if use_sahi and not SAHI_AVAILABLE:
            print("  ⚠ SAHI not found. Falling back to standard inference.")
        
        # Auto-detect device
        if device is None or device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"  ✓ Running on: {self.device}")
        
        if self.use_sahi:
            print("  ✓ SAHI Enabled (Optimized for Small Objects)")
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',
                model_path=model_size,
                confidence_threshold=confidence_threshold,
                device=self.device
            )
        
        print("Model loaded successfully!")
        
    def detect(
        self, 
        image: np.ndarray,
        filter_traffic_only: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects in an image with optional SAHI slicing
        """
        if self.use_sahi:
            # Sliced inference for high accuracy on small objects (pedestrians, distant bikes)
            results = get_sliced_prediction(
                image,
                self.sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            detections = []
            for obj in results.object_prediction_list:
                class_name = obj.category.name
                confidence = obj.score.value
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Filter for traffic-related classes
                if filter_traffic_only and class_name not in self.COLORS:
                    # check if it maps to any of our target classes
                    is_traffic = False
                    for t_class in self.COLORS.keys():
                        if t_class in class_name:
                            is_traffic = True
                            break
                    if not is_traffic: continue

                bbox = obj.bbox.to_xyxy() # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)
                
                det = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                detections.append(det)
        else:
            # Standard YOLOv11 inference - OPTIMIZED FOR SPEED
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=320,  # Smaller input size = MUCH faster inference
                agnostic_nms=False  # Faster NMS
            )[0]
            
            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                detections.append(det)

        # Annotate
        annotated_image = image.copy()
        for det in detections:
            color = self.COLORS.get(det['class_name'], (0, 255, 0))
            annotated_image = self._draw_detection(annotated_image, det, color)
        
        return annotated_image, detections
    
    def _draw_detection(self, image: np.ndarray, detection: Dict, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw bounding box and label on image with sharp text"""
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get image dimensions for adaptive sizing
        img_height, img_width = image.shape[:2]
        
        # Adaptive thickness and font scale based on image size
        base_size = min(img_width, img_height)
        box_thickness = max(2, int(base_size / 300))
        font_scale = max(0.3, base_size / 1000)
        font_thickness = max(1, int(base_size / 500))
        
        # Draw bounding box with rounded corners effect
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
        
        # Prepare label
        label = f"{class_name}: {confidence:.0%}"
        
        # Use better font
        font = cv2.FONT_HERSHEY_DUPLEX  # Sharper than SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Calculate label position (above the box)
        label_x = x1
        label_y = y1 - 10
        
        # If label goes above image, put it inside the box
        if label_y - text_height < 0:
            label_y = y1 + text_height + 10
        
        # Draw label background with padding
        padding = 5
        bg_x1 = label_x
        bg_y1 = label_y - text_height - padding
        bg_x2 = label_x + text_width + (padding * 2)
        bg_y2 = label_y + padding
        
        # Draw filled rectangle for background
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1, cv2.LINE_AA)
        
        # Draw border around label background
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw text with anti-aliasing (white text with black outline for contrast)
        text_x = label_x + padding
        text_y = label_y
        
        # Draw black outline for better visibility
        cv2.putText(
            image, label, (text_x, text_y),
            font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA
        )
        
        # Draw white text on top
        cv2.putText(
            image, label, (text_x, text_y),
            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
        )
        
        return image
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load image with multiple fallback methods for Windows compatibility
        """
        # Clean up the path
        image_path = str(image_path).strip().strip('"').strip("'")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        print(f"  Loading image from: {image_path}")
        
        image = None
        
        # Method 1: Standard cv2.imread
        try:
            image = cv2.imread(image_path)
            if image is not None:
                print("  ✓ Loaded with cv2.imread")
                return image
        except Exception as e:
            print(f"  cv2.imread failed: {e}")
        
        # Method 2: Read with numpy for unicode paths
        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                print("  ✓ Loaded with cv2.imdecode")
                return image
        except Exception as e:
            print(f"  cv2.imdecode failed: {e}")
        
        # Method 3: PIL as fallback
        try:
            from PIL import Image
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if image is not None:
                print("  ✓ Loaded with PIL")
                return image
        except Exception as e:
            print(f"  PIL loading failed: {e}")
        
        raise ValueError(f"Could not load image: {image_path}")
    
    def detect_from_file(
        self, 
        image_path: str,
        output_path: Optional[str] = None,
        filter_traffic_only: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects from an image file
        """
        # Load image using robust method
        image = self.load_image(image_path)
        
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
        print("  Running detection...")
        
        # Run detection
        start_time = time.time()
        annotated_image, detections = self.detect(image, filter_traffic_only)
        inference_time = time.time() - start_time
        
        print(f"  ✓ Detection completed in {inference_time:.2f}s")
        print(f"  ✓ Found {len(detections)} traffic objects")
        
        # Save output if path provided
        if output_path:
            output_path = str(output_path).strip().strip('"').strip("'")
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            success = cv2.imwrite(output_path, annotated_image)
            if not success:
                try:
                    from PIL import Image
                    pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    pil_image.save(output_path)
                    success = True
                except:
                    pass
            
            if success:
                print(f"  ✓ Saved annotated image to: {output_path}")
        
        return annotated_image, detections
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = True,
        filter_traffic_only: bool = True
    ) -> None:
        """
        Detect objects in a video file
        """
        video_path = str(video_path).strip().strip('"').strip("'")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        writer = None
        if output_path:
            output_path = str(output_path).strip().strip('"').strip("'")
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing video... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, detections = self.detect(frame, filter_traffic_only)
            
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if writer:
                writer.write(annotated_frame)
            
            if show_preview:
                display_frame = annotated_frame
                if width > 1280:
                    scale = 1280 / width
                    display_frame = cv2.resize(annotated_frame, 
                                              (int(width * scale), int(height * scale)))
                try:
                    cv2.imshow('Traffic Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                except cv2.error as e:
                    if "The function is not implemented" in str(e):
                        print("\n⚠ Preview window not available (headless environment). Continuing processing without preview...")
                        show_preview = False
                    else:
                        print(f"\n⚠ Error showing preview: {e}")
                        show_preview = False
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames in {elapsed_time:.1f}s")
        print(f"Average FPS: {frame_count/elapsed_time:.1f}")
        if output_path:
            print(f"Saved video to: {output_path}")
    
    def detect_webcam(
        self,
        camera_id: int = 0,
        filter_traffic_only: bool = True,
        output_path: Optional[str] = None,
        save_video: bool = False
    ) -> None:
        """
        Run real-time detection on webcam feed
        
        Args:
            camera_id: Camera device ID
            filter_traffic_only: Filter for traffic objects only
            output_path: Path to save output video (if save_video=True)
            save_video: Enable video recording
        """
        print("Opening camera...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")

        print("Setting camera resolution...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get actual camera resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if not detected

        print(f"Camera resolution: {width}x{height} @ {fps}fps")

        # Setup video writer if saving
        writer = None
        if save_video and output_path:
            output_path = str(output_path).strip().strip('"').strip("'")
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Video will be saved to: {output_path}")

        # Warm-up inference
        print("Warming up model (first inference may take 30-60 seconds)...")
        warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        warmup_start = time.time()
        _, _ = self.detect(warmup_frame, filter_traffic_only)
        print(f"✓ Warm-up completed in {time.time() - warmup_start:.1f}s")

        print("\n✅ Starting webcam detection... Press 'q' to quit\n")
        print("   Press 's' to start/stop recording")

        frame_count = 0
        start_time = time.time()
        recording = save_video  # Start recording if save_video flag is set
        recording_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            annotated_frame, detections = self.detect(frame, filter_traffic_only)

            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Add status overlay
            status_text = f"FPS: {current_fps:.1f} | Objects: {len(detections)}"
            if recording:
                rec_time = time.time() - recording_start if recording_start else 0
                status_text += f" | ● REC {rec_time:.0f}s"
                cv2.circle(annotated_frame, (width - 20, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save frame if recording
            if writer and recording:
                writer.write(annotated_frame)

            try:
                cv2.imshow('Traffic Detection - Webcam (Press Q to quit)', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nStopped by user")
                    break
                elif key == ord('s'):
                    if recording:
                        recording = False
                        print("\n⏹ Recording stopped")
                    else:
                        recording = True
                        recording_start = time.time()
                        print("\n▶ Recording started")
            except cv2.error as e:
                if "The function is not implemented" in str(e):
                    print("\n❌ Error: Cannot run webcam detection in a headless environment (no GUI support).")
                    print("  Please install: pip install opencv-python")
                    break
                else:
                    print(f"\n⚠ Error displaying webcam feed: {e}")
                    break

        cap.release()
        if writer:
            writer.release()
            print(f"\n✓ Video saved to: {output_path}")
        cv2.destroyAllWindows()
        print(f"\nSession ended. Total frames: {frame_count}, Avg FPS: {current_fps:.1f}")
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """Get summary statistics of detections"""
        summary = {
            'total_objects': len(detections),
            'by_class': {},
            'average_confidence': 0.0
        }
        
        if not detections:
            return summary
        
        for det in detections:
            class_name = det['class_name']
            if class_name not in summary['by_class']:
                summary['by_class'][class_name] = 0
            summary['by_class'][class_name] += 1
        
        summary['average_confidence'] = float(np.mean(
            [det['confidence'] for det in detections]
        ))
        
        return summary


class AdvancedTrafficDetector(TrafficDetector):
    """Extended detector with road segmentation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Loading segmentation model...")
        self.seg_model = YOLO('yolov8m-seg.pt')
        print("Segmentation model loaded!")
    
    def detect_with_segmentation(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Detect objects and segment areas"""
        annotated_image, detections = self.detect(image)
        
        seg_results = self.seg_model(
            image, 
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        seg_mask = np.zeros_like(image)
        
        if seg_results.masks is not None:
            for i, mask in enumerate(seg_results.masks.data):
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                
                colored_mask = np.zeros_like(image)
                color = (np.random.randint(100, 255), 
                        np.random.randint(100, 255), 
                        np.random.randint(100, 255))
                colored_mask[mask_resized > 0.5] = color
                seg_mask = cv2.addWeighted(seg_mask, 1, colored_mask, 0.5, 0)
        
        combined = cv2.addWeighted(annotated_image, 0.7, seg_mask, 0.3, 0)
        
        return combined, detections, seg_mask
