import time
from typing import Dict


class Controller:
    """
    Controller layer.
    Converts planner decisions into actions.
    Currently runs in simulation mode (prints actions).
    """

    def __init__(self):
        self.current_action = "STOP"
        self.last_action_time = time.time()
        self.running = False

    def initialize(self):
        """Initialize controller (placeholder for GPIO setup)."""
        self.running = True
        print("[CONTROLLER] Initialized (simulation mode)")

    def execute_decision(self, decision: Dict):
        """
        Execute decision from PathPlanner.

        Args:
            decision: dict with key 'action_name'
        """
        if not self.running:
            return

        action = decision.get("action_name", "STOP")

        if action != self.current_action:
            self.current_action = action
            self.last_action_time = time.time()
            self._apply_action(action)

    def _apply_action(self, action: str):
        """
        Apply action (simulation).
        Replace this logic with GPIO motor control later.
        """
        if action == "FORWARD":
            print("[ACTION] Moving forward")
        elif action == "TURN_LEFT":
            print("[ACTION] Turning left")
        elif action == "TURN_RIGHT":
            print("[ACTION] Turning right")
        elif action == "STOP":
            print("[ACTION] Stopping")
        else:
            print(f"[ACTION] Unknown action: {action}")

    def get_status(self) -> Dict:
        """Return current controller state."""
        return {
            "action": self.current_action,
            "running": self.running,
            "last_action_time": self.last_action_time,
        }

    def shutdown(self):
        """Shutdown controller safely."""
        if self.running:
            self._apply_action("STOP")
            self.running = False
            print("[CONTROLLER] Shutdown complete")
