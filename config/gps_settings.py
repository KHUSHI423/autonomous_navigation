"""
=============================================================================
EDGE DRIVE 3D - GPS CONFIGURATION
=============================================================================
GPS-specific configuration settings

Author: EdgeDrive3D Team
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import json
import yaml
from pathlib import Path


# ============================================================================
# GPS CONFIGURATION
# ============================================================================

@dataclass
class GPSConfig:
    """GPS receiver settings"""
    # Connection
    port: str = "auto"  # 'auto', 'COM3', '/dev/ttyUSB0', etc.
    baudrate: int = 9600
    timeout: float = 2.0
    
    # Mode
    simulate: bool = True  # Use simulated GPS for testing
    
    # Filtering
    use_kalman_filter: bool = True
    process_noise: float = 0.1
    measurement_noise: float = 5.0
    
    # Update rates
    min_update_interval: float = 0.05  # 20Hz max
    stale_threshold: float = 5.0  # Mark stale after 5s
    
    # Accuracy thresholds
    min_satellites: int = 4
    min_fix_quality: int = 1
    max_accuracy_meters: float = 10.0


@dataclass
class MapConfig:
    """OpenStreetMap visualization settings"""
    # Default view
    default_lat: float = 12.9716  # Bangalore (example)
    default_lon: float = 77.5946
    default_zoom: int = 15
    
    # Map tiles
    tile_provider: str = "OpenStreetMap"  # OSM, Satellite, Terrain, Dark
    
    # Overlays
    show_trajectory: bool = True
    trajectory_color: str = "#00ff00"
    trajectory_width: int = 3
    max_trajectory_points: int = 1000
    
    show_fov: bool = True
    fov_color: str = "#00ff00"
    fov_opacity: float = 0.2
    fov_range_meters: float = 50.0
    fov_degrees: float = 70.0
    
    show_objects: bool = True
    object_marker_size: int = 12
    show_detection_rays: bool = True
    
    # Markers
    vehicle_marker_color: str = "#00ff00"
    vehicle_marker_size: int = 40
    
    # Legend
    show_legend: bool = True
    legend_position: str = "bottomleft"


@dataclass
class CoordinateTransformConfig:
    """Coordinate transformation settings"""
    # Reference frame
    reference_altitude: float = 0.0
    
    # Camera mounting
    camera_height_m: float = 1.5
    camera_pitch_deg: float = -10.0  # Downward tilt
    camera_yaw_deg: float = 0.0  # Straight ahead
    
    # Earth model
    earth_radius_m: float = 6378137.0
    
    # Precision
    gps_precision_degrees: int = 8  # Decimal places


@dataclass
class GPSLoggerConfig:
    """GPS logging settings"""
    # Output
    output_dir: str = "output/gps_logs"
    filename_format: str = "gps_%Y%m%d_%H%M%S.jsonl"
    
    # What to log
    log_raw_gps: bool = True
    log_objects: bool = True
    log_trajectory: bool = True
    log_decisions: bool = True
    
    # Recording
    auto_start: bool = False
    max_file_size_mb: int = 100
    rotate_files: bool = True


@dataclass
class GPSPerceptionConfig:
    """GPS-enhanced perception settings"""
    # Enable/disable
    enable_gps: bool = True
    enable_geolocation: bool = True
    enable_trajectory: bool = True
    
    # Object projection
    project_objects_to_gps: bool = True
    min_object_distance: float = 0.5  # Ignore very close objects
    max_object_distance: float = 100.0  # Ignore very far objects
    
    # Fusion
    fuse_gps_with_perception: bool = True
    gps_weight: float = 0.7  # Weight for GPS in fusion
    
    # Dashboard
    dashboard_port: int = 8502  # Different from main dashboard
    dashboard_refresh_rate: float = 30.0  # Hz


# ============================================================================
# MASTER GPS CONFIGURATION
# ============================================================================

@dataclass
class GPSSystemConfig:
    """Master GPS system configuration"""
    gps: GPSConfig = field(default_factory=GPSConfig)
    map: MapConfig = field(default_factory=MapConfig)
    transform: CoordinateTransformConfig = field(default_factory=CoordinateTransformConfig)
    logger: GPSLoggerConfig = field(default_factory=GPSLoggerConfig)
    perception: GPSPerceptionConfig = field(default_factory=GPSPerceptionConfig)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'gps': {
                'port': self.gps.port,
                'baudrate': self.gps.baudrate,
                'simulate': self.gps.simulate,
                'use_kalman_filter': self.gps.use_kalman_filter,
                'min_satellites': self.gps.min_satellites,
            },
            'map': {
                'default_zoom': self.map.default_zoom,
                'tile_provider': self.map.tile_provider,
                'show_trajectory': self.map.show_trajectory,
                'show_fov': self.map.show_fov,
                'show_objects': self.map.show_objects,
            },
            'transform': {
                'camera_height_m': self.transform.camera_height_m,
                'camera_pitch_deg': self.transform.camera_pitch_deg,
            },
            'logger': {
                'output_dir': self.logger.output_dir,
                'log_raw_gps': self.logger.log_raw_gps,
                'log_objects': self.logger.log_objects,
            },
            'perception': {
                'enable_gps': self.perception.enable_gps,
                'enable_geolocation': self.perception.enable_geolocation,
                'dashboard_port': self.perception.dashboard_port,
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save configuration to YAML file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        print(f"✓ GPS Config saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GPSSystemConfig':
        """Load configuration from YAML file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"⚠ GPS Config not found: {filepath}, using defaults")
            return cls()
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Apply loaded values
        if 'gps' in data:
            for k, v in data['gps'].items():
                if hasattr(config.gps, k):
                    setattr(config.gps, k, v)
        
        if 'map' in data:
            for k, v in data['map'].items():
                if hasattr(config.map, k):
                    setattr(config.map, k, v)
        
        if 'transform' in data:
            for k, v in data['transform'].items():
                if hasattr(config.transform, k):
                    setattr(config.transform, k, v)
        
        if 'logger' in data:
            for k, v in data['logger'].items():
                if hasattr(config.logger, k):
                    setattr(config.logger, k, v)
        
        if 'perception' in data:
            for k, v in data['perception'].items():
                if hasattr(config.perception, k):
                    setattr(config.perception, k, v)
        
        print(f"✓ GPS Config loaded from: {filepath}")
        return config


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class PresetGPSConfigs:
    """Pre-defined GPS configuration presets"""
    
    @staticmethod
    def simulated_config() -> GPSSystemConfig:
        """Configuration for simulated GPS (testing)"""
        config = GPSSystemConfig()
        config.gps.simulate = True
        config.gps.port = "auto"
        config.map.default_zoom = 16
        config.perception.dashboard_port = 8502
        return config
    
    @staticmethod
    def hardware_config() -> GPSSystemConfig:
        """Configuration for hardware GPS"""
        config = GPSSystemConfig()
        config.gps.simulate = False
        config.gps.port = "auto"
        config.gps.baudrate = 9600
        config.gps.use_kalman_filter = True
        config.map.show_trajectory = True
        config.map.show_fov = True
        config.logger.auto_start = True
        return config
    
    @staticmethod
    def usb_dongle_config() -> GPSSystemConfig:
        """Configuration for USB GPS dongle"""
        config = GPSSystemConfig()
        config.gps.simulate = False
        config.gps.port = "auto"  # Auto-detect
        config.gps.baudrate = 9600
        config.gps.min_satellites = 4
        config.map.tile_provider = "OpenStreetMap"
        config.map.default_zoom = 17
        return config
    
    @staticmethod
    def serial_config(port: str = "/dev/ttyUSB0", baudrate: int = 9600) -> GPSSystemConfig:
        """Configuration for serial GPS module"""
        config = GPSSystemConfig()
        config.gps.simulate = False
        config.gps.port = port
        config.gps.baudrate = baudrate
        config.gps.timeout = 3.0
        config.transform.camera_height_m = 1.5
        return config
    
    @staticmethod
    def high_accuracy_config() -> GPSSystemConfig:
        """Configuration for high-accuracy RTK GPS"""
        config = GPSSystemConfig()
        config.gps.simulate = False
        config.gps.min_satellites = 8
        config.gps.min_fix_quality = 2  # Require DGPS fix
        config.gps.max_accuracy_meters = 2.0
        config.gps.use_kalman_filter = True
        config.gps.process_noise = 0.05
        config.gps.measurement_noise = 1.0
        config.map.default_zoom = 19  # Higher zoom for accuracy
        return config
    
    @staticmethod
    def demo_config() -> GPSSystemConfig:
        """Configuration for demonstrations"""
        config = GPSSystemConfig()
        config.gps.simulate = True  # Use simulated for reliability
        config.map.show_trajectory = True
        config.map.show_fov = True
        config.map.show_objects = True
        config.map.show_legend = True
        config.perception.dashboard_port = 8502
        config.logger.log_objects = True
        config.logger.log_trajectory = True
        return config


# ============================================================================
# DEFAULT CONFIGURATION CREATION
# ============================================================================

def create_default_gps_config(filepath: str = "config/gps_settings.yaml"):
    """Create default GPS configuration file"""
    config = GPSSystemConfig()
    config.save(filepath)
    print(f"\nDefault GPS configuration created: {filepath}")
    print("\nTo customize, edit the YAML file and load with:")
    print("  config = GPSSystemConfig.load('config/gps_settings.yaml')")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GPS Configuration")
    print("=" * 60)
    
    # Create default config
    create_default_gps_config()
    
    # Load and display
    config = GPSSystemConfig.load("config/gps_settings.yaml")
    
    print("\nCurrent GPS Configuration:")
    print(config.to_json())
    
    # Show presets
    print("\n\nAvailable Presets:")
    print("  - simulated_config: For testing without hardware")
    print("  - hardware_config: For USB GPS dongle")
    print("  - usb_dongle_config: Optimized for USB GPS")
    print("  - serial_config: For UART GPS modules")
    print("  - high_accuracy_config: For RTK GPS")
    print("  - demo_config: For demonstrations")
