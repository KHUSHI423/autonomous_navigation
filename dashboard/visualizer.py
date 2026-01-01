import cv2
import numpy as np
from typing import List, Dict


class Visualizer:
    """
    Visualization dashboard for autonomous navigation.
    """

    def __init__(self, window_name="Autonomous Navigation"):
        self.window_name = window_name

    def draw_center_guides(self, frame):
        """Draw center vertical and horizontal guides."""
        h, w = frame.shape[:2]

        # Vertical center line
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # Horizontal mid line
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

        return frame

    def draw_detections(self, frame, detections: List[Dict]):
        """Draw bounding boxes and labels."""
        h, w = frame.shape[:2]

        for det in detections:
            xmin, ymin, xmax, ymax = det["bbox"]

            # Clamp values (important!)
            xmin = int(max(0, xmin) * w)
            ymin = int(max(0, ymin) * h)
            xmax = int(min(1, xmax) * w)
            ymax = int(min(1, ymax) * h)

            label = det["class_name"]
            score = det["score"]

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (xmin, max(0, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return frame

    def draw_decision(self, frame, decision: Dict):
        """Draw navigation decision text."""
        action = decision.get("action_name", "UNKNOWN")
        reason = decision.get("reasoning", "")

        color_map = {
            "FORWARD": (0, 255, 0),
            "TURN_LEFT": (255, 255, 0),
            "TURN_RIGHT": (255, 0, 255),
            "STOP": (0, 0, 255),
        }

        color = color_map.get(action, (255, 255, 255))

        cv2.rectangle(frame, (10, 10), (420, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"ACTION: {action}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            frame,
            reason,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return frame

    def draw_fps(self, frame, fps: float):
        """Draw FPS counter."""
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return frame

    def visualize(self, frame, detections, decision, fps):
        """Apply all visualization layers."""
        frame = self.draw_center_guides(frame)
        frame = self.draw_detections(frame, detections)
        frame = self.draw_decision(frame, decision)
        frame = self.draw_fps(frame, fps)
        return frame

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def cleanup(self):
        cv2.destroyAllWindows()
