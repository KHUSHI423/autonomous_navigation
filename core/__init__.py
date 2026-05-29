"""
=============================================================================
EDGE DRIVE 3D - CORE MODULES
=============================================================================
Enhanced perception system with:
- Depth Estimation (MiDaS)
- 3D Object Detection (YOLOv8)
- Lane Detection (OpenCV)
- Traffic Sign Detection
- BEV Mapping
- Decision Making
=============================================================================
"""

from .perception_engine import (
    PerceptionEngine,
    CameraIntrinsics,
    Object3D,
    PerceptionResult,
    DepthEstimator,
    ObjectDetector3D,
    BEVMapper,
    DecisionMaker,
)

from .lane_detector import LaneDetector, LaneResult
from .sign_detector import SignDetector, SignDetectionResult, TrafficSign

__version__ = '2.0.0'
__all__ = [
    # Main engine
    'PerceptionEngine',
    'PerceptionResult',
    
    # Data structures
    'CameraIntrinsics',
    'Object3D',
    
    # Detection modules
    'DepthEstimator',
    'ObjectDetector3D',
    'LaneDetector',
    'LaneResult',
    'SignDetector',
    'SignDetectionResult',
    'TrafficSign',
    
    # Mapping & decisions
    'BEVMapper',
    'DecisionMaker',
]
