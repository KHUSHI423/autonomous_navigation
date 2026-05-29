"""
EdgeDrive3D - Unified Autonomous Perception System
"""

__version__ = "1.0.0"
__author__ = "EdgeDrive3D Team"

from core.perception_engine import PerceptionEngine, Object3D, CameraIntrinsics, PerceptionResult
from config.settings import SystemConfig, PerceptionConfig, HardwareConfig

__all__ = [
    'PerceptionEngine',
    'Object3D',
    'CameraIntrinsics',
    'PerceptionResult',
    'SystemConfig',
    'PerceptionConfig',
    'HardwareConfig',
]
