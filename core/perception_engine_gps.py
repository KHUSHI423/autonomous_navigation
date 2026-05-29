"""
=============================================================================
EDGE DRIVE 3D - GPS-ENHANCED PERCEPTION ENGINE
=============================================================================
Extended perception engine with GPS integration

Features:
- All original perception capabilities
- GPS position fusion
- Object geolocation (camera coords -> GPS)
- Trajectory logging
- Real-time map overlay preparation

Author: EdgeDrive3D Team
=============================================================================
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base perception engine
from core.perception_engine import (
    PerceptionEngine, PerceptionResult, Object3D,
    CameraIntrinsics, DepthEstimator, ObjectDetector3D,
    BEVMapper, DecisionMaker
)

# Import GPS and coordinate transform
from hardware.gps_reader import GPSReceiver, GPSReading
from utils.coordinate_transform import (
    CoordinateTransformer, ObjectGPSProjector,
    haversine_distance, bearing
)


# ============================================================================
# EXTENDED DATA STRUCTURES
# ============================================================================

@dataclass
class GPSPerceptionResult(PerceptionResult):
    """Extended perception result with GPS data"""
    
    # GPS data
    gps_reading: Optional[GPSReading] = None
    
    # Objects with GPS coordinates
    objects_with_gps: List[Dict] = field(default_factory=list)
    
    # Vehicle trajectory
    trajectory_point: Optional[Tuple[float, float]] = None
    
    # Map bounds (for auto-zoom)
    map_bounds: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Extended serialization"""
        base = super().to_dict()
        
        if self.gps_reading:
            base['gps'] = self.gps_reading.to_dict()
        
        base['objects_with_gps'] = self.objects_with_gps
        base['trajectory'] = self.trajectory_point
        
        return base


# ============================================================================
# GPS PERCEPTION ENGINE
# ============================================================================

class GPSPerceptionEngine(PerceptionEngine):
    """
    Perception engine with GPS integration
    
    Extends the base PerceptionEngine with:
    - GPS position tracking
    - Object geolocation
    - Coordinate transformation
    - Trajectory logging
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Initialize base engine
        super().__init__(config)
        
        # GPS configuration
        self.gps_config = {
            'simulate': config.get('gps_simulate', True),
            'gps_port': config.get('gps_port', 'auto'),
            'baudrate': config.get('baudrate', 9600),
        }
        
        # Initialize GPS
        print("\nInitializing GPS...")
        self.gps = GPSReceiver(
            port=self.gps_config['gps_port'],
            baudrate=self.gps_config['baudrate'],
            simulate=self.gps_config['simulate']
        )
        
        # Coordinate transformer (initialized after GPS fix)
        self.transformer: Optional[CoordinateTransformer] = None
        self.projector: Optional[ObjectGPSProjector] = None
        
        # Trajectory storage
        self.trajectory: List[Tuple[float, float]] = []
        self.max_trajectory_points = 1000
        
        # Statistics
        self.gps_updates = 0
        self.objects_geolocated = 0
        
        print("  ✓ GPS Perception Engine Ready!\n")
    
    def start_gps(self):
        """Start GPS receiver"""
        self.gps.start()
        
        # Wait for initial fix
        print("Waiting for GPS fix...")
        gps_data = self.gps.wait_for_fix(timeout=10.0)
        
        if gps_data:
            print(f"  ✓ GPS Fix: {gps_data.latitude:.6f}, {gps_data.longitude:.6f}")
            self._initialize_transformer(gps_data)
        else:
            print("  ⚠ No GPS fix yet (will continue in simulated mode)")
    
    def stop_gps(self):
        """Stop GPS receiver"""
        self.gps.stop()
    
    def _initialize_transformer(self, gps_data: GPSReading):
        """Initialize coordinate transformer with GPS position"""
        self.transformer = CoordinateTransformer(
            base_lat=gps_data.latitude,
            base_lon=gps_data.longitude,
            base_alt=gps_data.altitude,
            base_heading=gps_data.heading
        )
        self.projector = ObjectGPSProjector(self.transformer)
    
    def process_frame_gps(self, frame: np.ndarray) -> GPSPerceptionResult:
        """
        Process frame with GPS integration
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            GPSPerceptionResult with all perception + GPS data
        """
        # Get GPS position
        gps_data = self.gps.get_position()
        
        # Initialize transformer on first valid GPS
        if gps_data and gps_data.is_valid and self.transformer is None:
            self._initialize_transformer(gps_data)
            self.gps_updates += 1
        
        # Update transformer with latest position (for moving vehicle)
        if gps_data and gps_data.is_valid and self.transformer:
            self.transformer.update_base_position(
                latitude=gps_data.latitude,
                longitude=gps_data.longitude,
                altitude=gps_data.altitude,
                heading=gps_data.heading
            )
            self.gps_updates += 1
        
        # Run base perception pipeline
        base_result = self.process_frame(frame)
        
        # Create extended result
        gps_result = GPSPerceptionResult(
            timestamp=base_result.timestamp,
            fps=base_result.fps,
            depth_map=base_result.depth_map,
            objects_3d=base_result.objects_3d,
            lanes=base_result.lanes,
            depth_colored=base_result.depth_colored,
            detections_overlay=base_result.detections_overlay,
            bev_image=base_result.bev_image,
            point_cloud=base_result.point_cloud,
            decision=base_result.decision,
            gps_reading=gps_data
        )
        
        # Project objects to GPS coordinates
        if gps_data and gps_data.is_valid and self.transformer and base_result.objects_3d:
            gps_result.objects_with_gps = self._project_objects_to_gps(
                base_result.objects_3d,
                frame.shape
            )
            self.objects_geolocated += len(gps_result.objects_with_gps)
        
        # Add trajectory point
        if gps_data and gps_data.is_valid:
            gps_result.trajectory_point = (gps_data.latitude, gps_data.longitude)
            self.trajectory.append(gps_result.trajectory_point)
            
            # Limit trajectory size
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory = self.trajectory[-self.max_trajectory_points:]
        
        # Calculate map bounds for auto-zoom
        gps_result.map_bounds = self._calculate_map_bounds()
        
        return gps_result
    
    def _project_objects_to_gps(
        self,
        objects: List[Object3D],
        image_shape: Tuple[int, int, int]
    ) -> List[Dict]:
        """
        Project detected objects to GPS coordinates
        
        Args:
            objects: 3D objects from perception
            image_shape: Image dimensions (H, W, C)
        
        Returns:
            List of objects with GPS coordinates
        """
        objects_with_gps = []
        
        for obj in objects:
            if obj.position_3d is None or obj.bbox_2d is None:
                continue
            
            # Project to GPS
            lat, lon, alt = self.transformer.camera_to_gps(
                obj.position_3d[0],  # X (right)
                obj.position_3d[1],  # Y (down)
                obj.position_3d[2]   # Z (forward)
            )
            
            obj_with_gps = {
                'class_id': obj.class_id,
                'class_name': obj.class_name,
                'confidence': obj.confidence,
                'distance': obj.distance,
                'position_3d': obj.position_3d.tolist() if obj.position_3d is not None else None,
                'bbox_2d': obj.bbox_2d,
                'gps_coordinates': {
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt
                },
                'dimensions': obj.dimensions
            }
            
            objects_with_gps.append(obj_with_gps)
        
        return objects_with_gps
    
    def _calculate_map_bounds(self) -> Dict:
        """Calculate map bounds from trajectory"""
        if not self.trajectory:
            return None
        
        lats = [p[0] for p in self.trajectory]
        lons = [p[1] for p in self.trajectory]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': sum(lats) / len(lats),
            'center_lon': sum(lons) / len(lons)
        }
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get current trajectory"""
        return self.trajectory.copy()
    
    def clear_trajectory(self):
        """Clear trajectory"""
        self.trajectory.clear()
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        base_stats = {
            'gps_updates': self.gps_updates,
            'objects_geolocated': self.objects_geolocated,
            'trajectory_points': len(self.trajectory),
            'gps_connected': self.gps.serial_conn is not None if not self.gps.simulate else True,
            'simulate_mode': self.gps.simulate
        }
        return base_stats
    
    def save_trajectory(self, filepath: str):
        """Save trajectory to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'trajectory': self.trajectory,
                'statistics': self.get_statistics(),
                'timestamp': time.time()
            }, f, indent=2)
        print(f"  ✓ Trajectory saved to: {filepath}")
    
    def load_trajectory(self, filepath: str) -> List[Tuple[float, float]]:
        """Load trajectory from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.trajectory = [tuple(p) for p in data.get('trajectory', [])]
        print(f"  ✓ Trajectory loaded from: {filepath}")
        return self.trajectory


# ============================================================================
# GPS PERCEPTION LOGGER
# ============================================================================

class GPSPerceptionLogger:
    """
    Log GPS perception results for later replay/analysis
    """
    
    def __init__(self, output_dir: str = "output/gps_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records_logged = 0
    
    def start_session(self):
        """Start logging session"""
        filepath = self.output_dir / f"{self.session_id}.jsonl"
        self.file = open(filepath, 'w')
        print(f"  ✓ GPS Logger: Recording to {filepath}")
        return filepath
    
    def stop_session(self):
        """Stop logging session"""
        if self.file:
            self.file.close()
            self.file = None
        print(f"  ✓ GPS Logger: Session complete ({self.records_logged} records)")
    
    def log(self, result: GPSPerceptionResult, frame: np.ndarray = None):
        """Log a perception result"""
        if not self.file:
            return
        
        # Create record
        record = {
            'timestamp': time.time(),
            'frame_index': self.records_logged,
            'gps': result.gps_reading.to_dict() if result.gps_reading else None,
            'objects': result.objects_with_gps,
            'decision': result.decision,
            'fps': result.fps,
            'trajectory': result.trajectory_point
        }
        
        # Write record
        self.file.write(json.dumps(record) + '\n')
        self.file.flush()
        self.records_logged += 1
    
    def replay_session(self, filepath: str = None):
        """
        Generator to replay logged session
        
        Yields:
            Dict with logged data
        """
        if filepath is None:
            filepath = self.output_dir / f"{self.session_id}.jsonl"
        
        with open(filepath, 'r') as f:
            for line in f:
                yield json.loads(line.strip())


# ============================================================================
# MAIN (Test)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GPS Perception Engine Test")
    print("=" * 60)
    
    # Create engine with simulated GPS
    engine = GPSPerceptionEngine({'gps_simulate': True})
    
    # Start GPS
    engine.start_gps()
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
    
    print("\nProcessing test frames...")
    
    for i in range(10):
        result = engine.process_frame_gps(test_frame)
        
        if result.gps_reading and result.gps_reading.is_valid:
            print(f"\rFrame {i+1}: GPS({result.gps_reading.latitude:.6f}, "
                  f"{result.gps_reading.longitude:.6f}) | "
                  f"Objects: {len(result.objects_with_gps)}", end="")
        
        time.sleep(0.1)
    
    print("\n\nStatistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save trajectory
    engine.save_trajectory("output/gps_logs/test_trajectory.json")
    
    # Cleanup
    engine.stop_gps()
    print("\n✓ Test complete")
