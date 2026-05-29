"""
Model Selector Module
Maps detected object classes to appropriate 3D GLB models
Handles model variations and randomization for visual diversity
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import yaml
from loguru import logger


class ModelSelector:
    """
    Selects appropriate 3D GLB models based on detection class
    and object characteristics
    """
    
    # Default model mappings for COCO classes
    DEFAULT_MAPPINGS = {
        # Vehicles
        'car': [
            'models/vehicles/veh_car_sedan_blue.glb',
            'models/vehicles/veh_car_sedan_silver.glb',
            'models/vehicles/veh_car_suv_dark.glb',
            'models/vehicles/veh_car_suv_white.glb',
            'models/vehicles/veh_car_hatchback_yellow.glb'
        ],
        'bus': [
            'models/vehicles/veh_bus_city_red.glb',
            'models/vehicles/veh_bus_mini_blue.glb'
        ],
        'truck': [
            'models/vehicles/veh_truck_delivery.glb',
            'models/vehicles/veh_tempo_white.glb',
            'models/vehicles/veh_tanker_red.glb'
        ],
        'motorcycle': [
            'models/vehicles/veh_bike_motorcycle.glb',
            'models/vehicles/veh_scooter_yellow.glb'
        ],
        'bicycle': [
            'models/vehicles/veh_bicycle_blue.glb'
        ],
        
        # People
        'person': [
            'models/humans/hum_pedestrian_blue.glb',
            'models/humans/hum_pedestrian_red.glb',
            'models/humans/hum_standing.glb'
        ],
        
        # Default fallback
        'default': [
            'models/markers/mrk_pin_location.glb'
        ]
    }
    
    def __init__(self, config_path: str = "config_realtime.yaml", 
                 models_dir: str = "models"):
        """
        Initialize model selector
        
        Args:
            config_path: Path to YAML configuration file
            models_dir: Base directory for 3D models
        """
        self.config = self._load_config(config_path)
        self.models_dir = Path(models_dir)
        
        # Load model mappings from config or use defaults
        model_config = self.config.get('models', {})
        self.base_dir = Path(model_config.get('base_dir', 'models'))
        self.auto_select = model_config.get('auto_select', True)
        self.default_model = model_config.get('default', 'models/markers/mrk_pin_location.glb')
        
        # Build model mapping
        self.model_mappings = self._build_model_mappings(model_config.get('mappings', {}))
        
        # Model cache (for performance)
        self._model_cache: Dict[str, Any] = {}
        
        # Randomization seed for reproducibility
        self.seed = random.randint(0, 10000)
        
        logger.info(f"ModelSelector initialized with {len(self.model_mappings)} class mappings")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def _build_model_mappings(self, config_mappings: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Build model mappings from config and defaults
        
        Args:
            config_mappings: Mappings from config file
            
        Returns:
            Dictionary mapping class names to model lists
        """
        mappings = {}
        
        # Start with config mappings (single model per class)
        for class_name, model_path in config_mappings.items():
            full_path = str(self.base_dir / model_path.replace('models/', ''))
            if class_name not in mappings:
                mappings[class_name] = []
            mappings[class_name].append(full_path)
        
        # Merge with defaults (add variations)
        for class_name, models in self.DEFAULT_MAPPINGS.items():
            if class_name not in mappings:
                mappings[class_name] = models
            else:
                # Add default models as variations
                for model in models:
                    if model not in mappings[class_name]:
                        mappings[class_name].append(model)
        
        # Verify models exist
        for class_name, models in mappings.items():
            existing_models = []
            for model in models:
                model_path = Path(model)
                if model_path.exists():
                    existing_models.append(model)
                else:
                    logger.debug(f"Model not found: {model}")
            
            if existing_models:
                mappings[class_name] = existing_models
            else:
                mappings[class_name] = [self.default_model]
        
        return mappings
    
    def select_model(self, class_name: str, detection: Optional[Any] = None,
                     track: Optional[Any] = None) -> str:
        """
        Select best 3D model for a detection/track
        
        Args:
            class_name: Object class name
            detection: Detection object (optional, for smart selection)
            track: Track object (optional, for smart selection)
            
        Returns:
            Path to selected GLB model
        """
        # Get available models for this class
        available_models = self.model_mappings.get(class_name, [self.default_model])
        
        if len(available_models) == 0:
            return self.default_model
        
        if len(available_models) == 1:
            return available_models[0]
        
        # Smart selection based on object characteristics
        if detection is not None or track is not None:
            selected = self._smart_select(class_name, available_models, detection, track)
            if selected:
                return selected
        
        # Random selection for variety
        return random.choice(available_models)
    
    def _smart_select(self, class_name: str, models: List[str],
                      detection: Optional[Any], track: Optional[Any]) -> Optional[str]:
        """
        Intelligently select model based on object characteristics
        
        Args:
            class_name: Object class
            models: Available models
            detection: Detection object
            track: Track object
            
        Returns:
            Selected model path or None
        """
        # Get bbox dimensions if available
        bbox = None
        if detection is not None:
            bbox = detection.bbox
        elif track is not None:
            bbox = track.bbox
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            
            # Car selection based on aspect ratio
            if class_name == 'car':
                if aspect_ratio > 1.5:  # Wider = SUV
                    suv_models = [m for m in models if 'suv' in m.lower()]
                    if suv_models:
                        return random.choice(suv_models)
                else:  # Standard = sedan
                    sedan_models = [m for m in models if 'sedan' in m.lower()]
                    if sedan_models:
                        return random.choice(sedan_models)
            
            # Person selection based on speed (cyclist vs pedestrian)
            if class_name == 'person' and track is not None:
                speed = track.get_average_speed() if hasattr(track, 'get_average_speed') else 0
                if speed > 5:  # Fast moving = cyclist
                    cyclist_models = [m for m in models if 'cyclist' in m.lower()]
                    if cyclist_models:
                        return random.choice(cyclist_models)
        
        return None
    
    def get_model_for_track(self, track: Any) -> str:
        """
        Get model for a tracked object
        
        Args:
            track: TrackedObject
            
        Returns:
            Model path
        """
        class_name = track.class_name
        return self.select_model(class_name, track=track)
    
    def get_models_for_scene(self, tracks: List[Any]) -> Dict[int, str]:
        """
        Get models for all tracks in scene
        
        Args:
            tracks: List of TrackedObject
            
        Returns:
            Dictionary mapping track IDs to model paths
        """
        model_map = {}
        
        for track in tracks:
            model_path = self.get_model_for_track(track)
            model_map[track.track_id] = model_path
        
        return model_map
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model info
        """
        path = Path(model_path)
        
        if not path.exists():
            return {
                'path': model_path,
                'exists': False,
                'size': 0
            }
        
        # Parse model name for info
        parts = path.stem.split('_')
        
        return {
            'path': str(path),
            'exists': True,
            'size': path.stat().st_size,
            'category': parts[0] if len(parts) > 0 else 'unknown',
            'type': parts[1] if len(parts) > 1 else 'unknown',
            'variant': parts[2] if len(parts) > 2 else 'unknown',
            'color': parts[3] if len(parts) > 3 else 'unknown'
        }
    
    def list_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available models organized by class
        
        Returns:
            Dictionary mapping class names to model info lists
        """
        result = {}
        
        for class_name, models in self.model_mappings.items():
            result[class_name] = []
            for model in models:
                info = self.get_model_info(model)
                result[class_name].append(info)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get model selector statistics
        
        Returns:
            Dictionary with statistics
        """
        total_models = sum(len(models) for models in self.model_mappings.values())
        
        return {
            'total_classes': len(self.model_mappings),
            'total_models': total_models,
            'average_models_per_class': total_models / max(len(self.model_mappings), 1),
            'classes': list(self.model_mappings.keys())
        }
    
    def validate_models(self) -> Tuple[List[str], List[str]]:
        """
        Validate all model files exist
        
        Returns:
            Tuple of (existing models, missing models)
        """
        existing = []
        missing = []
        
        for models in self.model_mappings.values():
            for model in models:
                if Path(model).exists():
                    existing.append(model)
                else:
                    missing.append(model)
        
        return existing, missing


def test_selector(config_path: str = "config_realtime.yaml"):
    """Test model selector"""
    selector = ModelSelector(config_path)
    
    # Print statistics
    stats = selector.get_statistics()
    print(f"\nModel Selector Statistics:")
    print(f"  Total classes: {stats['total_classes']}")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Classes: {', '.join(stats['classes'])}")
    
    # Validate models
    existing, missing = selector.validate_models()
    print(f"\nModel Validation:")
    print(f"  Existing: {len(existing)}")
    print(f"  Missing: {len(missing)}")
    
    if missing:
        print(f"\nMissing models:")
        for model in missing[:10]:
            print(f"    - {model}")
    
    # Test model selection
    print(f"\nModel Selection Test:")
    test_classes = ['car', 'bus', 'person', 'motorcycle', 'truck']
    
    for class_name in test_classes:
        model = selector.select_model(class_name)
        print(f"  {class_name}: {model}")


if __name__ == "__main__":
    import sys
    
    config_file = "config_realtime.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    test_selector(config_file)
