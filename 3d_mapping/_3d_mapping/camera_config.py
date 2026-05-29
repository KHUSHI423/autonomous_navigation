"""
Camera Configuration and Calibration Parameters
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @classmethod
    def from_image_size(cls, width: int, height: int, fov_degrees: float = 70.0):
        """
        Create camera intrinsics from image size and field of view
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_degrees: Horizontal field of view in degrees
        """
        fov_rad = np.radians(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
    
    @classmethod
    def default_smartphone(cls, width: int, height: int):
        """Default parameters for smartphone camera"""
        return cls.from_image_size(width, height, fov_degrees=70)
    
    @classmethod
    def default_dashcam(cls, width: int, height: int):
        """Default parameters for dashcam/wide angle"""
        return cls.from_image_size(width, height, fov_degrees=120)
    
    @classmethod
    def default_webcam(cls, width: int, height: int):
        """Default parameters for webcam"""
        return cls.from_image_size(width, height, fov_degrees=60)


# Standard object dimensions in meters (length, width, height)
OBJECT_DIMENSIONS = {
    'car': {'length': 4.5, 'width': 1.8, 'height': 1.5},
    'truck': {'length': 7.0, 'width': 2.5, 'height': 3.0},
    'bus': {'length': 12.0, 'width': 2.5, 'height': 3.5},
    'motorcycle': {'length': 2.2, 'width': 0.8, 'height': 1.2},
    'bicycle': {'length': 1.8, 'width': 0.5, 'height': 1.1},
    'person': {'length': 0.5, 'width': 0.5, 'height': 1.7},
    'traffic light': {'length': 0.3, 'width': 0.3, 'height': 1.0},
    'stop sign': {'length': 0.6, 'width': 0.05, 'height': 0.6},
    'fire hydrant': {'length': 0.4, 'width': 0.4, 'height': 0.8},
    'bench': {'length': 1.5, 'width': 0.5, 'height': 0.8},
    'parking meter': {'length': 0.3, 'width': 0.3, 'height': 1.3},
}


def get_object_dimensions(class_name: str) -> dict:
    """Get standard dimensions for an object class"""
    return OBJECT_DIMENSIONS.get(class_name.lower(), {
        'length': 1.0, 'width': 1.0, 'height': 1.0
    })
