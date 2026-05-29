"""
WebSocket Server Module
Real-time data streaming to web-based 3D visualizer
FastAPI + WebSocket for low-latency communication
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import yaml
from loguru import logger
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
except ImportError:
    logger.error("FastAPI or uvicorn not installed. Run: pip install fastapi websockets uvicorn")
    FastAPI = None
    WebSocket = None


class SceneData:
    """Container for scene data to stream to clients"""
    
    def __init__(self):
        self.objects: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {}
        self.timestamp: float = 0
        self.fps: float = 0
    
    def add_object(self, obj: Dict[str, Any]) -> None:
        """Add object to scene"""
        self.objects.append(obj)
    
    def clear(self) -> None:
        """Clear all objects"""
        self.objects.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'objects': self.objects,
            'statistics': self.statistics,
            'timestamp': self.timestamp,
            'fps': self.fps
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class WebSocketServer:
    """
    WebSocket server for real-time 3D visualization
    
    Features:
    - Multiple client support
    - Real-time scene updates
    - Statistics streaming
    - Automatic reconnection handling
    """
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize WebSocket server
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        ws_config = self.config.get('websocket', {})
        
        self.host = ws_config.get('host', 'localhost')
        self.port = ws_config.get('port', 8765)
        self.update_rate = ws_config.get('update_rate', 30)
        self.max_clients = ws_config.get('max_clients', 10)
        
        # FastAPI app
        self.app = FastAPI(title="Real-Time 3D Visualization")
        
        # Connected clients
        self.clients: Set[WebSocket] = set()
        
        # Scene data
        self.scene_data = SceneData()
        
        # Server state
        self.is_running = False
        self.update_task: Optional[asyncio.Task] = None
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"WebSocketServer initialized on {self.host}:{self.port}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def get_root():
            """Serve the viewer HTML file"""
            return FileResponse("viewer_realtime.html")
        
        @self.app.get("/viewer")
        async def get_viewer():
            """Alternative endpoint for viewer"""
            return FileResponse("viewer_realtime.html")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "clients": len(self.clients),
                "fps": self.scene_data.fps
            }
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            """Get current statistics"""
            return self.scene_data.statistics
        
        @self.app.get("/api/objects")
        async def get_objects():
            """Get current scene objects"""
            return {"objects": self.scene_data.objects}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle WebSocket connections"""
            await self._handle_websocket(websocket)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """
        Handle WebSocket connection
        
        Args:
            websocket: WebSocket connection
        """
        # Accept connection
        await websocket.accept()
        
        # Check client limit
        if len(self.clients) >= self.max_clients:
            await websocket.send_json({"error": "Maximum clients reached"})
            await websocket.close()
            return
        
        # Add client
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial scene data
        await websocket.send_json(self.scene_data.to_dict())
        
        try:
            # Keep connection alive
            while True:
                # Receive messages from client (if any)
                data = await websocket.receive_text()
                
                # Handle client commands
                try:
                    message = json.loads(data)
                    await self._handle_client_message(websocket, message)
                except json.JSONDecodeError:
                    pass
                    
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Remove client
            self.clients.discard(websocket)
            logger.info(f"Client removed. Total clients: {len(self.clients)}")
    
    async def _handle_client_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Handle message from client
        
        Args:
            websocket: WebSocket connection
            message: Received message
        """
        msg_type = message.get('type', '')
        
        if msg_type == 'ping':
            await websocket.send_json({'type': 'pong', 'timestamp': datetime.now().timestamp()})
        
        elif msg_type == 'request_full_update':
            # Send full scene data
            await websocket.send_json(self.scene_data.to_dict())
        
        elif msg_type == 'set_view':
            # Handle view change request
            view = message.get('view', 'perspective')
            logger.info(f"Client requested view: {view}")
    
    def update_scene(self, objects: List[Dict[str, Any]], 
                     statistics: Dict[str, Any], fps: float) -> None:
        """
        Update scene data (called from main processing loop)
        
        Args:
            objects: List of scene objects
            statistics: Statistics dictionary
            fps: Current FPS
        """
        self.scene_data.objects = objects
        self.scene_data.statistics = statistics
        self.scene_data.fps = fps
        self.scene_data.timestamp = datetime.now().timestamp()
    
    async def broadcast_updates(self):
        """Broadcast scene updates to all clients"""
        if not self.clients:
            return
        
        message = self.scene_data.to_json()
        
        # Send to all clients
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    async def run_server(self):
        """Run the WebSocket server"""
        if FastAPI is None:
            logger.error("FastAPI not available")
            return
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        self.is_running = True
        
        await server.serve()
    
    def start_background_task(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start background update task"""
        async def update_loop():
            while self.is_running:
                await self.broadcast_updates()
                await asyncio.sleep(1.0 / self.update_rate)
        
        self.update_task = loop.create_task(update_loop())
        logger.info(f"Background update task started at {self.update_rate} Hz")


class RealTimePipeline:
    """
    Integrates all components into a real-time processing pipeline
    
    Usage:
        pipeline = RealTimePipeline()
        pipeline.run()
    """
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize real-time pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        # Import components
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from video_processor import VideoProcessor
        from object_detector import ObjectDetector
        from tracker import ByteTrack
        from scene_mapper import SceneMapper
        from model_selector import ModelSelector
        
        self.processor = VideoProcessor(config_path)
        self.detector = ObjectDetector(config_path)
        self.tracker = ByteTrack(config_path)
        self.mapper = SceneMapper(config_path)
        self.model_selector = ModelSelector(config_path)
        self.ws_server = WebSocketServer(config_path)
        
        # Calibrated flag
        self.is_calibrated = False
        
        logger.info("RealTimePipeline initialized")
    
    def process_frame(self, frame) -> Dict[str, Any]:
        """
        Process single frame through pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with processed data
        """
        # Detect
        detections = self.detector.detect(frame)
        
        # Track
        tracks = self.tracker.update(detections)
        
        # Calibrate on first frame
        if not self.is_calibrated:
            self.mapper.auto_calibrate_homography(frame.shape[1], frame.shape[0])
            self.is_calibrated = True
        
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
            'frame_shape': frame.shape
        }
        
        # Get FPS
        fps = self.processor.get_fps()
        
        return {
            'objects': objects_data,
            'statistics': statistics,
            'fps': fps,
            'tracks': tracks,
            'detections': detections
        }
    
    async def run_async(self):
        """Run pipeline asynchronously"""
        import cv2
        
        # Initialize components
        if not self.processor.initialize():
            return
        if not self.detector.initialize():
            return
        
        # Start WebSocket server in background
        loop = asyncio.get_event_loop()
        
        server_task = loop.create_task(self.ws_server.run_server())
        logger.info("WebSocket server started")
        
        frame_count = 0
        
        try:
            for frame in self.processor.frames_generator():
                # Process frame
                result = self.process_frame(frame)
                
                # Update WebSocket server
                self.ws_server.update_scene(
                    result['objects'],
                    result['statistics'],
                    result['fps']
                )
                
                # Draw visualization
                frame = self._draw_visualization(frame, result)
                
                # Display
                cv2.imshow("Real-Time Pipeline", frame)
                
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.processor.release()
            cv2.destroyAllWindows()
            self.ws_server.is_running = False
    
    def _draw_visualization(self, frame, result: Dict[str, Any]):
        """Draw visualization on frame"""
        import cv2
        
        # Draw detections
        frame = self.detector.draw_detections(frame, result['detections'])
        
        # Draw FPS
        frame = self.processor.draw_fps(frame, result['fps'])
        
        # Draw statistics
        stats = result['statistics']['tracks']
        info_text = f"Objects: {stats['total_tracks']} | FPS: {result['fps']:.1f}"
        cv2.putText(frame, info_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Run the pipeline (synchronous wrapper)"""
        import asyncio
        
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Pipeline stopped")


def run_server(config_path: str = "config_realtime.yaml"):
    """
    Run only the WebSocket server (standalone mode)
    
    Args:
        config_path: Path to configuration file
    """
    import asyncio
    
    ws_server = WebSocketServer(config_path)
    
    # Mount static files
    ws_server.app.mount("/models", StaticFiles(directory="models"), name="models")
    
    try:
        asyncio.run(ws_server.run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == "__main__":
    import sys
    
    config_file = "config_realtime.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Run full pipeline
    pipeline = RealTimePipeline(config_file)
    pipeline.run()
