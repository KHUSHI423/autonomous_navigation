"""
Monocular Depth Estimation using MiDaS and Depth Anything models
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Literal
from pathlib import Path
import time


class DepthEstimator:
    """
    Depth estimation using state-of-the-art monocular depth models
    """
    
    MODELS = {
        'midas_small': 'Intel/dpt-small',
        'midas_base': 'Intel/dpt-base',
        'midas_large': 'Intel/dpt-large',
        'midas_hybrid': 'Intel/dpt-hybrid-midas',
        'depth_anything_small': 'LiheYoung/depth-anything-small-hf',
        'depth_anything_base': 'LiheYoung/depth-anything-base-hf',
        'depth_anything_large': 'LiheYoung/depth-anything-large-hf',
    }
    
    def __init__(
        self,
        model_type: str = 'midas_hybrid',
        device: str = None
    ):
        """
        Initialize depth estimator
        
        Args:
            model_type: Model to use (see MODELS dict)
            device: 'cuda', 'cpu', or None for auto-detect
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing Depth Estimator...")
        print(f"  Model: {model_type}")
        print(f"  Device: {self.device}")
        
        self.model_type = model_type
        self._load_model(model_type)
        
        print("  ✓ Depth estimator ready!")
    
    def _load_model(self, model_type: str):
        """Load the depth estimation model"""
        
        if 'depth_anything' in model_type:
            self._load_depth_anything(model_type)
        else:
            self._load_midas(model_type)
    
    def _load_midas(self, model_type: str):
        """Load MiDaS model"""
        try:
            # Try loading from torch hub
            if model_type == 'midas_small':
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            elif model_type == 'midas_large':
                self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
            else:  # hybrid (default)
                self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
            
            self.model.to(self.device)
            self.model.eval()
            self.use_transformers = False
            
        except Exception as e:
            print(f"  Warning: Could not load from torch hub: {e}")
            print("  Falling back to transformers...")
            self._load_from_transformers(self.MODELS.get(model_type, 'Intel/dpt-hybrid-midas'))
    
    def _load_depth_anything(self, model_type: str):
        """Load Depth Anything model"""
        self._load_from_transformers(self.MODELS[model_type])
    
    def _load_from_transformers(self, model_name: str):
        """Load model using transformers library"""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        
        print(f"  Loading from HuggingFace: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.use_transformers = True
    
    def estimate_depth(
        self, 
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Estimate depth from a single image
        
        Args:
            image: Input image (BGR or RGB format)
            normalize: If True, normalize depth to 0-1 range
            
        Returns:
            Depth map as numpy array (same size as input)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        original_size = (image.shape[1], image.shape[0])
        
        with torch.no_grad():
            if self.use_transformers:
                depth_map = self._estimate_transformers(image_rgb)
            else:
                depth_map = self._estimate_midas(image_rgb)
        
        # Resize to original size
        depth_map = cv2.resize(depth_map, original_size)
        
        # Normalize if requested
        if normalize:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_map.astype(np.float32)
    
    def _estimate_midas(self, image_rgb: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS"""
        # Transform image
        input_batch = self.transform(image_rgb).to(self.device)
        
        # Predict depth
        prediction = self.model(input_batch)
        
        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()
        
        return depth
    
    def _estimate_transformers(self, image_rgb: np.ndarray) -> np.ndarray:
        """Estimate depth using transformers model"""
        from PIL import Image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()
        
        return depth
    
    def estimate_metric_depth(
        self,
        image: np.ndarray,
        max_depth_meters: float = 100.0
    ) -> np.ndarray:
        """
        Estimate depth in approximate metric units (meters)
        
        Note: Monocular depth is relative, so this is an approximation
        based on typical scene depths.
        
        Args:
            image: Input image
            max_depth_meters: Maximum expected depth in the scene
            
        Returns:
            Depth map in approximate meters
        """
        # Get normalized depth (0-1, where 1 is closest)
        relative_depth = self.estimate_depth(image, normalize=True)
        
        # Invert so that higher values = farther
        # MiDaS outputs inverse depth, so close objects have high values
        inverted = 1.0 - relative_depth
        
        # Scale to metric range
        metric_depth = inverted * max_depth_meters
        
        # Avoid zero depth
        metric_depth = np.clip(metric_depth, 0.1, max_depth_meters)
        
        return metric_depth
    
    def create_depth_colormap(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Create a colored visualization of the depth map
        
        Args:
            depth_map: Depth map (normalized 0-1)
            colormap: OpenCV colormap to use
            
        Returns:
            Colored depth visualization (BGR)
        """
        # Normalize to 0-255
        depth_normalized = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        
        return depth_colored


class DepthAnythingV2Estimator(DepthEstimator):
    """
    Specialized estimator for Depth Anything V2 (if available)
    Falls back to V1 if V2 is not available
    """
    
    def __init__(self, model_size: str = 'base', device: str = None):
        """
        Args:
            model_size: 'small', 'base', or 'large'
        """
        model_type = f'depth_anything_{model_size}'
        super().__init__(model_type=model_type, device=device)
