"""
Video Processor Module
Handles video capture from various sources (webcam, file, RTSP stream)
Provides frame buffering and preprocessing for real-time processing
"""

import cv2
import numpy as np
from typing import Generator, Optional, Tuple, Dict, Any
from pathlib import Path
import yaml
from loguru import logger
import time
from collections import deque
from threading import Lock


class VideoProcessor:
    """
    Video capture and preprocessing module
    
    Supports:
    - Webcam (device index)
    - Video files (mp4, avi, etc.)
    - RTSP streams
    - HTTP streams
    """
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize video processor
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        self.frame_buffer: deque = deque(maxlen=3)  # Buffer for smooth playback
        self.frame_lock = Lock()
        self.fps_counter = deque(maxlen=30)
        self.start_time: Optional[float] = None
        
        # Camera settings
        self.source = self.config['camera']['source']
        self.width = self.config['camera']['width']
        self.height = self.config['camera']['height']
        self.target_fps = self.config['camera']['fps']
        self.flip_horizontal = self.config['camera']['flip_horizontal']
        
        logger.info(f"VideoProcessor initialized with source: {self.source}")
    
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
            'camera': {
                'source': 0,
                'width': 1280,
                'height': 720,
                'fps': 30,
                'flip_horizontal': False
            }
        }
    
    def initialize(self) -> bool:
        """
        Initialize video capture device
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine capture type
            if isinstance(self.source, int):
                # Webcam
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                # File or stream
                self.cap = cv2.VideoCapture(str(self.source))
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set camera properties (for webcams)
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual properties
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f} FPS")
            
            self.is_initialized = True
            self.start_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video processor: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get next frame from video source
        
        Returns:
            Frame as numpy array (BGR format), or None if failed
        """
        if not self.is_initialized or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            # Try to reopen for streams
            if not isinstance(self.source, int):
                logger.info("Attempting to reopen stream...")
                time.sleep(1)
                self.initialize()
            return None
        
        # Preprocessing
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        # Add to buffer
        with self.frame_lock:
            self.frame_buffer.append(frame)
        
        # Calculate FPS
        current_time = time.time()
        if len(self.fps_counter) > 0:
            last_time = self.fps_counter[-1]
            fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
            self.fps_counter.append(current_time)
        else:
            fps = 0
            self.fps_counter.append(current_time)
        
        return frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame from buffer
        
        Returns:
            Latest frame or None
        """
        with self.frame_lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1].copy()
        return None
    
    def get_fps(self) -> float:
        """
        Get current FPS
        
        Returns:
            Frames per second
        """
        if len(self.fps_counter) < 2:
            return 0.0
        
        times = list(self.fps_counter)
        return len(times) / (times[-1] - times[0]) if times[-1] > times[0] else 0.0
    
    def get_frame_count(self) -> int:
        """
        Get total frame count (for video files)
        
        Returns:
            Total frames or -1 for streams
        """
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_position(self) -> int:
        """
        Get current frame position (for video files)
        
        Returns:
            Current frame index
        """
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def frames_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously
        
        Yields:
            Video frames (BGR format)
        """
        if not self.is_initialized:
            if not self.initialize():
                return
        
        frame_count = 0
        
        while True:
            frame = self.get_frame()
            
            if frame is None:
                # End of video file
                if isinstance(self.source, int):
                    # For webcam, wait and retry
                    time.sleep(0.1)
                    continue
                else:
                    # For file, stop
                    break
            
            frame_count += 1
            yield frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to frame for detection
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame (RGB, normalized)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed (YOLO expects specific sizes)
        # YOLOv8 handles resizing internally, so we skip this
        
        return rgb
    
    def draw_fps(self, frame: np.ndarray, fps: Optional[float] = None) -> np.ndarray:
        """
        Draw FPS counter on frame
        
        Args:
            frame: Input frame
            fps: FPS value (uses calculated if None)
            
        Returns:
            Frame with FPS overlay
        """
        if fps is None:
            fps = self.get_fps()
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw FPS background
        cv2.rectangle(overlay, (10, 10), (150, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw FPS text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def draw_info(self, frame: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Draw information overlay on frame
        
        Args:
            frame: Input frame
            info: Dictionary with info to display
            
        Returns:
            Frame with info overlay
        """
        y_offset = 60
        line_height = 25
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += line_height
        
        return frame
    
    def release(self) -> None:
        """Release video capture device"""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            logger.info("Video processor released")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def test_video_processor(config_path: str = "config_realtime.yaml"):
    """
    Test video processor with display
    
    Args:
        config_path: Path to configuration file
    """
    import sys
    
    processor = VideoProcessor(config_path)
    
    if not processor.initialize():
        logger.error("Failed to initialize video processor")
        return
    
    logger.info("Press 'q' to exit")
    
    try:
        for frame in processor.frames_generator():
            # Draw FPS
            fps = processor.get_fps()
            frame = processor.draw_fps(frame, fps)
            
            # Draw info
            info = {
                "Resolution": f"{processor.actual_width}x{processor.actual_height}",
                "Frame": processor.get_frame_count()
            }
            frame = processor.draw_info(frame, info)
            
            # Display
            cv2.imshow("Video Processor Test", frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        processor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test the video processor
    config_file = "config_realtime.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    test_video_processor(config_file)
