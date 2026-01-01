from typing import List, Dict, Tuple


class PathPlanner:
    """
    Rule-based path planner.
    Takes object detections and decides navigation action.
    """

    def __init__(
        self,
        stop_size_threshold: float = 0.15,
        center_threshold: float = 0.15,
        min_confidence: float = 0.5,
    ):
        """
        Args:
            stop_size_threshold: bbox area above which object is considered too close
            center_threshold: how close to center object must be to block forward path
            min_confidence: ignore detections below this confidence
        """
        self.stop_size_threshold = stop_size_threshold
        self.center_threshold = center_threshold
        self.min_confidence = min_confidence

    def _bbox_area(self, bbox: List[float]) -> float:
        """Compute normalized bounding-box area."""
        xmin, ymin, xmax, ymax = bbox
        return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

    def _bbox_center(self, bbox: List[float]) -> float:
        """Return x-center of bounding box (0–1)."""
        xmin, _, xmax, _ = bbox
        return (xmin + xmax) / 2.0

    def plan(self, detections: List[Dict]) -> Dict:
        """
        Decide navigation action based on detections.

        Returns:
            dict with keys:
            - action_name
            - reasoning
        """

        if not detections:
            return {
                "action_name": "FORWARD",
                "reasoning": "No obstacles detected",
            }

        left_blocked = False
        right_blocked = False

        for det in detections:
            if det["score"] < self.min_confidence:
                continue

            bbox = det["bbox"]
            area = self._bbox_area(bbox)
            center_x = self._bbox_center(bbox)

            # Object is close and in front → STOP
            if area > self.stop_size_threshold and abs(center_x - 0.5) < self.center_threshold:
                return {
                    "action_name": "STOP",
                    "reasoning": f"Obstacle ({det['class_name']}) directly ahead",
                }

            # Mark left/right blocks
            if center_x < 0.4:
                left_blocked = True
            elif center_x > 0.6:
                right_blocked = True

        if left_blocked and not right_blocked:
            return {
                "action_name": "TURN_RIGHT",
                "reasoning": "Obstacle on left side",
            }

        if right_blocked and not left_blocked:
            return {
                "action_name": "TURN_LEFT",
                "reasoning": "Obstacle on right side",
            }

        if left_blocked and right_blocked:
            return {
                "action_name": "STOP",
                "reasoning": "Obstacles on both sides",
            }

        return {
            "action_name": "FORWARD",
            "reasoning": "Path clear",
        }
