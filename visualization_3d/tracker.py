"""
Object Tracker Module
Multi-object tracking with unique IDs using ByteTrack algorithm
Maintains consistent object identities across video frames
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from pathlib import Path
import yaml
from loguru import logger

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    logger.warning("filterpy not installed. Run: pip install filterpy")
    KalmanFilter = None

try:
    import lap
except ImportError:
    logger.warning("lap/lapx not installed. Run: pip install lapx")
    lap = None


class TrackedObject:
    """Represents a tracked object with unique ID"""
    
    def __init__(self, track_id: int, detection: Any, frame_count: int):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        
        # Position and velocity
        self.center = detection.center
        self.bottom_center = detection.bottom_center
        self.velocity = (0, 0)
        
        # History
        self.bbox_history = deque(maxlen=30)
        self.center_history = deque(maxlen=30)
        self.bbox_history.append(self.bbox)
        self.center_history.append(self.center)
        
        # Tracking state
        self.age = 0
        self.total_age = 0
        self.consecutive_misses = 0
        self.first_seen = frame_count
        self.last_seen = frame_count
        self.is_confirmed = False
        
        # Speed estimation (pixels per frame)
        self.speed_history = deque(maxlen=10)
    
    def update(self, detection: Any, frame_count: int) -> None:
        """
        Update track with new detection
        
        Args:
            detection: New detection
            frame_count: Current frame number
        """
        old_center = self.center
        self.bbox = detection.bbox
        self.center = detection.center
        self.bottom_center = detection.bottom_center
        self.confidence = detection.confidence
        
        # Calculate velocity
        dx = self.center[0] - old_center[0]
        dy = self.center[1] - old_center[1]
        self.velocity = (dx, dy)
        
        # Update history
        self.bbox_history.append(self.bbox)
        self.center_history.append(self.center)
        
        # Update speed
        speed = np.sqrt(dx**2 + dy**2)
        self.speed_history.append(speed)
        
        # Update state
        self.age = 0
        self.total_age += 1
        self.consecutive_misses = 0
        self.last_seen = frame_count
        
        if self.total_age >= 3:
            self.is_confirmed = True
    
    def predict(self) -> None:
        """Predict next position (when detection is missing)"""
        # Simple linear prediction
        predicted_x = self.center[0] + self.velocity[0]
        predicted_y = self.center[1] + self.velocity[1]
        self.center = (int(predicted_x), int(predicted_y))
        self.consecutive_misses += 1
        self.age += 1
    
    def get_average_speed(self) -> float:
        """Get average speed over history"""
        if len(self.speed_history) == 0:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get trajectory path"""
        return list(self.center_history)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'track_id': self.track_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'center': self.center,
            'bottom_center': self.bottom_center,
            'velocity': self.velocity,
            'speed': self.get_average_speed(),
            'confidence': self.confidence,
            'age': self.total_age,
            'is_confirmed': self.is_confirmed,
            'trajectory': list(self.center_history)
        }
    
    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, class={self.class_name}, pos={self.center})"


class ByteTrack:
    """
    ByteTrack-inspired multi-object tracker
    
    Features:
    - High confidence detection association
    - Low confidence detection recovery
    - Kalman filter prediction
    - Track management (birth/death)
    """
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        """
        Initialize tracker
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        track_config = self.config['tracking']
        
        self.track_threshold = track_config['track_threshold']
        self.min_track_length = track_config['min_track_length']
        self.max_age = track_config['max_age']
        self.match_threshold = track_config['match_threshold']
        
        self.frame_count = 0
        self.next_track_id = 0
        self.tracks: List[TrackedObject] = []
        self.lost_tracks: List[TrackedObject] = []
        
        logger.info("ByteTrack initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'tracking': {
                'enabled': True,
                'tracker': 'bytetrack',
                'track_threshold': 0.3,
                'min_track_length': 3,
                'max_age': 30,
                'match_threshold': 0.8
            }
        }
    
    def _iou_distance(self, tracks: List[TrackedObject], 
                      detections: List[Any]) -> np.ndarray:
        """
        Calculate IoU distance matrix between tracks and detections
        
        Args:
            tracks: List of tracked objects
            detections: List of detections
            
        Returns:
            Distance matrix (num_tracks x num_detections)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))
        
        tracks_bbox = np.array([t.bbox for t in tracks])
        dets_bbox = np.array([d.bbox for d in detections])
        
        # Expand dimensions for broadcasting
        tracks_bbox = tracks_bbox[:, np.newaxis, :]  # (N, 1, 4)
        dets_bbox = dets_bbox[np.newaxis, :, :]      # (1, M, 4)
        
        # Calculate intersection
        xx1 = np.maximum(tracks_bbox[..., 0], dets_bbox[..., 0])
        yy1 = np.maximum(tracks_bbox[..., 1], dets_bbox[..., 1])
        xx2 = np.minimum(tracks_bbox[..., 2], dets_bbox[..., 2])
        yy2 = np.minimum(tracks_bbox[..., 3], dets_bbox[..., 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate union
        track_area = (tracks_bbox[..., 2] - tracks_bbox[..., 0]) * \
                     (tracks_bbox[..., 3] - tracks_bbox[..., 1])
        det_area = (dets_bbox[..., 2] - dets_bbox[..., 0]) * \
                   (dets_bbox[..., 3] - dets_bbox[..., 1])
        union = track_area + det_area - intersection
        
        # IoU and distance
        iou = intersection / (union + 1e-8)
        distance = 1 - iou
        
        return distance
    
    def _match(self, cost_matrix: np.ndarray, threshold: float) -> Tuple[List, List, List]:
        """
        Match tracks to detections using linear assignment
        
        Args:
            cost_matrix: Distance matrix
            threshold: Matching threshold
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if lap is None:
            # Fallback: greedy matching
            return self._greedy_match(cost_matrix, threshold)
        
        # Hungarian algorithm
        matched_rows, matched_cols = [], []
        row_ind, col_ind = lap.lapjv(cost_matrix, threshold=threshold, extend_cost=True)
        
        for i, j in zip(row_ind, col_ind):
            if i >= 0 and j >= 0:
                matched_rows.append(i)
                matched_cols.append(j)
        
        matched_pairs = list(zip(matched_rows, matched_cols))
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def _greedy_match(self, cost_matrix: np.ndarray, 
                      threshold: float) -> Tuple[List, List, List]:
        """Greedy matching fallback"""
        matched_pairs = []
        used_rows = set()
        used_cols = set()
        
        # Sort by cost
        indices = np.argwhere(cost_matrix < threshold)
        costs = [cost_matrix[i, j] for i, j in indices]
        sorted_indices = np.argsort(costs)
        
        for idx in sorted_indices:
            i, j = indices[idx]
            if i not in used_rows and j not in used_cols:
                matched_pairs.append((i, j))
                used_rows.add(i)
                used_cols.add(j)
        
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_rows]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in used_cols]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List[Any]) -> List[TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of TrackedObject with updated positions
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            # No detections, predict all tracks
            for track in self.tracks:
                track.predict()
            # Remove old tracks
            self.tracks = [t for t in self.tracks if t.age < self.max_age]
            return self.tracks
        
        # Split detections by confidence
        high_conf_dets = [d for d in detections if d.confidence >= self.match_threshold]
        low_conf_dets = [d for d in detections if d.confidence < self.match_threshold]
        
        # First association: high confidence detections
        if len(self.tracks) > 0 and len(high_conf_dets) > 0:
            cost_matrix = self._iou_distance(self.tracks, high_conf_dets)
            matched, unmatched_tracks, unmatched_dets = self._match(
                cost_matrix, self.match_threshold
            )
            
            # Update matched tracks
            for track_idx, det_idx in matched:
                self.tracks[track_idx].update(high_conf_dets[det_idx], self.frame_count)
            
            # Handle unmatched tracks
            unmatched_track_indices = unmatched_tracks
        else:
            unmatched_track_indices = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(high_conf_dets)))
        
        # Second association: low confidence detections with unmatched tracks
        if len(unmatched_track_indices) > 0 and len(low_conf_dets) > 0:
            unmatched_tracks_list = [self.tracks[i] for i in unmatched_track_indices]
            cost_matrix = self._iou_distance(unmatched_tracks_list, low_conf_dets)
            matched, _, unmatched_low_dets = self._match(cost_matrix, 0.5)
            
            for track_idx, det_idx in matched:
                orig_track_idx = unmatched_track_indices[track_idx]
                self.tracks[orig_track_idx].update(low_conf_dets[det_idx], self.frame_count)
                unmatched_track_indices.remove(orig_track_idx)
        
        # Create new tracks for unmatched detections
        new_detections = [high_conf_dets[i] for i in unmatched_dets]
        new_detections.extend([low_conf_dets[i] for i in unmatched_low_dets if i < len(low_conf_dets)])
        
        for det in new_detections:
            if det.confidence >= self.track_threshold:
                new_track = TrackedObject(self.next_track_id, det, self.frame_count)
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Predict unmatched tracks
        for idx in unmatched_track_indices:
            self.tracks[idx].predict()
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.age < self.max_age]
        
        return self.tracks
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active (confirmed) tracks"""
        return [t for t in self.tracks if t.is_confirmed]
    
    def get_tracks_by_class(self, class_name: str) -> List[TrackedObject]:
        """Get tracks filtered by class name"""
        return [t for t in self.tracks if t.class_name == class_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        stats = {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len([t for t in self.tracks if t.is_confirmed]),
            'new_tracks': len([t for t in self.tracks if t.total_age == 1]),
            'lost_tracks': len(self.lost_tracks),
            'by_class': {}
        }
        
        for track in self.tracks:
            if track.class_name not in stats['by_class']:
                stats['by_class'][track.class_name] = 0
            stats['by_class'][track.class_name] += 1
        
        return stats
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[TrackedObject]) -> np.ndarray:
        """
        Draw tracks on frame
        
        Args:
            frame: Input frame (BGR)
            tracks: List of tracked objects
            
        Returns:
            Frame with drawn tracks
        """
        for track in tracks:
            # Color based on track state
            if track.is_confirmed:
                color = (0, 255, 0)  # Green for confirmed
            else:
                color = (0, 128, 255)  # Orange for unconfirmed
            
            # Draw bounding box
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw trajectory
            trajectory = track.get_trajectory()
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i - 1]
                    pt2 = trajectory[i]
                    alpha = i / len(trajectory)
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
        
        return frame


def test_tracker(config_path: str = "config_realtime.yaml", source: int = 0):
    """
    Test tracker with video feed
    
    Args:
        config_path: Path to configuration file
        source: Video source
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from video_processor import VideoProcessor
    from object_detector import ObjectDetector
    import cv2
    
    detector = ObjectDetector(config_path)
    processor = VideoProcessor(config_path)
    tracker = ByteTrack(config_path)
    
    if not detector.initialize():
        logger.error("Failed to initialize detector")
        return
    
    if not processor.initialize():
        logger.error("Failed to initialize video processor")
        return
    
    logger.info("Press 'q' to exit")
    
    try:
        for frame in processor.frames_generator():
            # Detect
            detections = detector.detect(frame)
            
            # Track
            tracks = tracker.update(detections)
            
            # Draw
            frame = tracker.draw_tracks(frame, tracks)
            
            # Draw stats
            stats = tracker.get_statistics()
            info = f"Tracks: {stats['confirmed_tracks']} | Classes: {stats['by_class']}"
            cv2.putText(frame, info, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # FPS
            fps = processor.get_fps()
            frame = processor.draw_fps(frame, fps)
            
            # Display
            cv2.imshow("Tracker Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        processor.release()
        cv2.destroyAllWindows()


# Import cv2 for drawing
import cv2

if __name__ == "__main__":
    import sys
    
    config_file = "config_realtime.yaml"
    source = 0
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    if len(sys.argv) > 2:
        source = int(sys.argv[2])
    
    test_tracker(config_file, source)
