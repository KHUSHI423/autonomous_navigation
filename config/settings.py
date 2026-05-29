"""
=============================================================================
EDGE DRIVE 3D - SYSTEM CONFIGURATION
=============================================================================
Centralized configuration management for the entire system

Author: EdgeDrive3D Team
=============================================================================
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import json
import yaml
from pathlib import Path


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class PerceptionConfig:
    """Perception pipeline settings"""
    # Object detection
    yolo_model: str = "yolov8m.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    
    # Depth estimation
    depth_model: str = "midas_hybrid"  # midas_small, midas_hybrid, midas_large
    max_depth_meters: float = 50.0
    min_depth_meters: float = 0.5
    
    # Camera
    fov_degrees: float = 70.0
    camera_width: int = 640
    camera_height: int = 480
    
    # Performance
    target_fps: int = 30
    skip_frames: int = 1
    use_gpu: bool = True


@dataclass
class HardwareConfig:
    """Hardware interface settings"""
    # UDP streaming
    udp_port: int = 5000
    udp_buffer_size: int = 2**20  # 1MB
    
    # ESP32
    esp32_ip: str = "192.168.4.1"
    esp32_timeout: float = 2.0
    auto_control: bool = False
    
    # Motor control
    default_speed: int = 180
    max_speed: int = 255
    safety_stop_distance: float = 2.0
    warning_distance: float = 5.0


@dataclass
class VisualizationConfig:
    """Dashboard and visualization settings"""
    # Dashboard
    dashboard_port: int = 8501
    dashboard_address: str = "localhost"
    
    # 3D visualization
    point_size: float = 2.0
    point_cloud_downsample: int = 2
    show_coordinate_frame: bool = True
    
    # BEV
    bev_range_x: Tuple[float, float] = (-20.0, 20.0)
    bev_range_z: Tuple[float, float] = (0.0, 50.0)
    bev_size: Tuple[int, int] = (600, 800)
    
    # Recording
    save_snapshots: bool = True
    save_videos: bool = True
    save_pointclouds: bool = True


@dataclass
class OutputConfig:
    """Output directory settings"""
    base_dir: str = "output"
    recordings_dir: str = "output/recordings"
    snapshots_dir: str = "output/snapshots"
    pointclouds_dir: str = "output/pointclouds"
    maps_dir: str = "output/maps"
    logs_dir: str = "output/logs"


@dataclass
class SystemConfig:
    """Master system configuration"""
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # System mode
    debug_mode: bool = False
    demo_mode: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save configuration to YAML file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        print(f"✓ Config saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"⚠ Config not found: {filepath}, using defaults")
            return cls()
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create config with loaded values
        config = cls()
        
        if 'perception' in data:
            for k, v in data['perception'].items():
                if hasattr(config.perception, k):
                    setattr(config.perception, k, v)
        
        if 'hardware' in data:
            for k, v in data['hardware'].items():
                if hasattr(config.hardware, k):
                    setattr(config.hardware, k, v)
        
        if 'visualization' in data:
            for k, v in data['visualization'].items():
                if hasattr(config.visualization, k):
                    setattr(config.visualization, k, v)
        
        if 'output' in data:
            for k, v in data['output'].items():
                if hasattr(config.output, k):
                    setattr(config.output, k, v)
        
        print(f"✓ Config loaded from: {filepath}")
        return config


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class PresetConfigs:
    """Pre-defined configuration presets"""
    
    @staticmethod
    def fast_config() -> SystemConfig:
        """Fast inference configuration (lower accuracy, higher FPS)"""
        config = SystemConfig()
        config.perception.yolo_model = "yolov8n.pt"
        config.perception.depth_model = "midas_small"
        config.perception.confidence_threshold = 0.3
        config.perception.target_fps = 60
        config.debug_mode = False
        return config
    
    @staticmethod
    def balanced_config() -> SystemConfig:
        """Balanced configuration (default)"""
        return SystemConfig()
    
    @staticmethod
    def accurate_config() -> SystemConfig:
        """High accuracy configuration (lower FPS, better detection)"""
        config = SystemConfig()
        config.perception.yolo_model = "yolov8l.pt"
        config.perception.depth_model = "midas_large"
        config.perception.confidence_threshold = 0.5
        config.perception.target_fps = 15
        return config
    
    @staticmethod
    def hardware_config() -> SystemConfig:
        """Hardware integration configuration"""
        config = SystemConfig()
        config.hardware.auto_control = True
        config.hardware.safety_stop_distance = 2.0
        config.hardware.warning_distance = 5.0
        config.visualization.save_snapshots = True
        return config
    
    @staticmethod
    def demo_config() -> SystemConfig:
        """Demo mode configuration"""
        config = SystemConfig()
        config.demo_mode = True
        config.visualization.dashboard_port = 8501
        config.perception.yolo_model = "yolov8m.pt"
        return config


# ============================================================================
# DEFAULT CONFIGURATION CREATION
# ============================================================================

def create_default_config(filepath: str = "config/settings.yaml"):
    """Create default configuration file"""
    config = SystemConfig()
    config.save(filepath)
    print(f"\nDefault configuration created: {filepath}")
    print("\nTo customize, edit the YAML file and load with:")
    print("  config = SystemConfig.load('config/settings.yaml')")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Create default config
    create_default_config()
    
    # Example: Load and modify
    config = SystemConfig.load("config/settings.yaml")
    print("\nCurrent configuration:")
    print(config.to_json())
