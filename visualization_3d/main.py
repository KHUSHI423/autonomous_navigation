#!/usr/bin/env python3
"""
Real-Time Video Visualization System
Main entry point for the traffic visualization pipeline

Usage:
    python main.py                      # Run with default webcam
    python main.py --source 0           # Run with webcam 0
    python main.py --source video.mp4   # Run with video file
    python main.py --source rtsp://...  # Run with RTSP stream
    python main.py --server-only        # Run only WebSocket server
    python main.py --dashboard          # Run Streamlit dashboard
"""

import argparse
import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional
from loguru import logger
import time

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "logs/realtime_visualizer.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


class RealTimeVisualizer:
    """
    Main class for real-time video visualization
    
    Orchestrates all components:
    - Video capture
    - Object detection
    - Object tracking
    - 3D scene mapping
    - WebSocket streaming
    """
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize the visualizer
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Components (initialized on demand)
        self.processor = None
        self.detector = None
        self.tracker = None
        self.mapper = None
        self.model_selector = None
        self.ws_server = None
        
        # State
        self.is_running = False
        self.is_calibrated = False
        self.frame_count = 0
        self.start_time = None
        
        logger.info("RealTimeVisualizer initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def initialize_components(self, source: any = 0) -> bool:
        """
        Initialize all components
        
        Args:
            source: Video source (device index, file path, or URL)
            
        Returns:
            True if successful
        """
        try:
            # Import components
            from video_processor import VideoProcessor
            from object_detector import ObjectDetector
            from tracker import ByteTrack
            from scene_mapper import SceneMapper
            from model_selector import ModelSelector
            
            # Initialize video processor
            logger.info("Initializing video processor...")
            self.processor = VideoProcessor(self.config_path)
            self.processor.source = source
            
            if isinstance(source, int):
                logger.info(f"Using webcam: {source}")
            else:
                logger.info(f"Using video source: {source}")
            
            if not self.processor.initialize():
                logger.error("Failed to initialize video processor")
                return False
            
            # Initialize detector
            logger.info("Initializing object detector...")
            self.detector = ObjectDetector(self.config_path)
            
            if not self.detector.initialize():
                logger.error("Failed to initialize object detector")
                return False
            
            # Initialize tracker
            logger.info("Initializing tracker...")
            self.tracker = ByteTrack(self.config_path)
            
            # Initialize scene mapper
            logger.info("Initializing scene mapper...")
            self.mapper = SceneMapper(self.config_path)
            
            # Initialize model selector
            logger.info("Initializing model selector...")
            self.model_selector = ModelSelector(self.config_path)
            
            logger.info("All components initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install -r requirements_realtime.txt")
            return False
        
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def initialize_websocket_server(self) -> bool:
        """
        Initialize WebSocket server
        
        Returns:
            True if successful
        """
        try:
            from websocket_server import WebSocketServer
            
            logger.info("Initializing WebSocket server...")
            self.ws_server = WebSocketServer(self.config_path)
            
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install fastapi websockets uvicorn")
            return False
        
        except Exception as e:
            logger.error(f"Error initializing WebSocket server: {e}")
            return False
    
    def process_frame(self, frame) -> dict:
        """
        Process a single frame through the pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with processed data
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Track objects
        tracks = self.tracker.update(detections)
        
        # Calibrate scene mapper on first frame
        if not self.is_calibrated:
            self.mapper.auto_calibrate_homography(
                frame.shape[1],
                frame.shape[0]
            )
            self.is_calibrated = True
            logger.info("Scene mapper calibrated")
        
        # Get model mappings
        model_paths = self.model_selector.get_models_for_scene(tracks)
        
        # Map to 3D scene
        scene_objects = self.mapper.create_scene_from_tracks(tracks, model_paths)
        
        # Convert to dict for streaming
        objects_data = [obj.to_dict() for obj in scene_objects]
        
        # Get statistics
        track_stats = self.tracker.get_statistics()
        det_stats = self.detector.get_statistics(detections)
        
        statistics = {
            'tracks': track_stats,
            'detections': det_stats,
            'frame_shape': frame.shape,
            'frame_count': self.frame_count
        }
        
        # Get FPS
        fps = self.processor.get_fps()
        
        return {
            'objects': objects_data,
            'statistics': statistics,
            'fps': fps,
            'tracks': tracks,
            'detections': detections,
            'frame': frame
        }
    
    def draw_visualization(self, frame, result: dict) -> None:
        """
        Draw visualization on frame
        
        Args:
            frame: Input frame
            result: Processing result dictionary
        """
        import cv2
        
        # Draw detections
        frame = self.detector.draw_detections(frame, result['detections'])
        
        # Draw tracks
        frame = self.tracker.draw_tracks(frame, result['tracks'])
        
        # Draw ground plane
        frame = self.mapper.draw_ground_plane(frame)
        
        # Draw FPS
        frame = self.processor.draw_fps(frame, result['fps'])
        
        # Draw statistics
        stats = result['statistics']['tracks']
        info_text = f"Objects: {stats['total_tracks']} | FPS: {result['fps']:.1f}"
        cv2.putText(
            frame, info_text, (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    
    async def run_async(self, source: any = 0, with_websocket: bool = True):
        """
        Run the visualizer asynchronously
        
        Args:
            source: Video source
            with_websocket: Enable WebSocket streaming
        """
        import cv2
        
        # Initialize components
        if not self.initialize_components(source):
            return
        
        # Initialize WebSocket if requested
        if with_websocket:
            if not self.initialize_websocket_server():
                with_websocket = False
        
        # Start WebSocket server in background
        ws_task = None
        if with_websocket:
            loop = asyncio.get_event_loop()
            ws_task = loop.create_task(self.ws_server.run_server())
            logger.info("WebSocket server started in background")
            
            # Give server time to start
            await asyncio.sleep(0.5)
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("Starting video processing loop...")
        logger.info("Press 'q' to exit")
        
        try:
            for frame in self.processor.frames_generator():
                if not self.is_running:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Update WebSocket server
                if with_websocket and self.ws_server:
                    self.ws_server.update_scene(
                        result['objects'],
                        result['statistics'],
                        result['fps']
                    )
                
                # Draw visualization
                frame = self.draw_visualization(frame, result)
                
                # Display
                cv2.imshow("Real-Time Traffic Visualization", frame)
                
                self.frame_count += 1
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.is_running = False
            
            # Cleanup
            if self.processor:
                self.processor.release()
            
            if with_websocket and self.ws_server:
                self.ws_server.is_running = False
            
            cv2.destroyAllWindows()
            
            # Print summary
            if self.start_time:
                elapsed = time.time() - self.start_time
                logger.info(f"Processed {self.frame_count} frames in {elapsed:.1f}s")
                logger.info(f"Average FPS: {self.frame_count / elapsed:.1f}")
    
    def run(self, source: any = 0, with_websocket: bool = True):
        """
        Run the visualizer (synchronous wrapper)
        
        Args:
            source: Video source
            with_websocket: Enable WebSocket streaming
        """
        try:
            asyncio.run(self.run_async(source, with_websocket))
        except KeyboardInterrupt:
            logger.info("Stopped by user")


def run_server_only(config_path: str):
    """
    Run only the WebSocket server
    
    Args:
        config_path: Configuration file path
    """
    from websocket_server import WebSocketServer
    import asyncio
    
    ws_server = WebSocketServer(config_path)
    
    logger.info("Starting WebSocket server only...")
    logger.info("Open viewer_realtime.html in browser to view")
    logger.info("Press Ctrl+C to stop")
    
    try:
        asyncio.run(ws_server.run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped")


def run_dashboard(config_path: str):
    """
    Run the Streamlit dashboard
    
    Args:
        config_path: Configuration file path
    """
    import subprocess
    
    logger.info("Starting Streamlit dashboard...")
    logger.info("Dashboard will open in browser automatically")
    
    dashboard_path = Path(__file__).parent / "live_dashboard.py"
    
    if not dashboard_path.exists():
        logger.error("Dashboard file not found")
        return
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true"
    ])


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-Time Video Visualization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Run with default webcam
  python main.py --source 0           # Run with webcam 0
  python main.py --source video.mp4   # Run with video file
  python main.py --server-only        # Run only WebSocket server
  python main.py --dashboard          # Run Streamlit dashboard
  python main.py --no-websocket       # Run without WebSocket streaming
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="0",
        help="Video source: device index (0, 1, ...), file path, or URL"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_realtime.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Run only WebSocket server (no video processing)"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run Streamlit dashboard"
    )
    
    parser.add_argument(
        "--no-websocket",
        action="store_true",
        help="Disable WebSocket streaming"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with simulated data"
    )
    
    args = parser.parse_args()
    
    # Parse source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run based on mode
    if args.server_only:
        run_server_only(args.config)
    
    elif args.dashboard:
        run_dashboard(args.config)
    
    elif args.demo:
        logger.info("Running in demo mode...")
        logger.info("This will simulate data without actual video processing")
        
        # Import and run dashboard in simulation mode
        from live_dashboard import main as dashboard_main
        dashboard_main()
    
    else:
        # Run full visualizer
        visualizer = RealTimeVisualizer(args.config)
        
        with_websocket = not args.no_websocket
        
        if with_websocket:
            logger.info("WebSocket streaming enabled")
            logger.info("Open viewer_realtime.html in browser for 3D view")
        else:
            logger.info("WebSocket streaming disabled")
        
        visualizer.run(source=source, with_websocket=with_websocket)


if __name__ == "__main__":
    main()
