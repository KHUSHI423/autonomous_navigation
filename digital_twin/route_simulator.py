"""
=============================================================================
ROUTE SIMULATOR & INCIDENT RECORDER - Digital Twin Companion
=============================================================================
Advanced route planning, simulation, and incident analysis for the robot car

Features:
- Plan routes on map
- Simulate route execution before physical run
- Record all incidents with full state snapshots
- Replay incidents with timeline scrubbing
- Export incident reports

Run: python route_simulator.py
=============================================================================
"""

import json
import time
import math
import threading
import socket
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

# ============ CONFIGURATION ============
UDP_COMMAND_PORT = 9000  # Send to ESP32 motor controller
UDP_STATUS_PORT = 9001   # Receive from robot
SIMULATION_STEP_MS = 100

print("="*70)
print("  ROUTE SIMULATOR & INCIDENT RECORDER")
print("="*70)
print("Plan routes, simulate, and analyze incidents")
print("\nCommands:")
print("  plan     - Create new route plan")
print("  simulate - Run route simulation")
print("  incidents - View recorded incidents")
print("  replay   - Replay specific incident")
print("  help     - Show this help")
print("="*70)

# ============ DATA CLASSES ============
@dataclass
class Waypoint:
    x: float
    z: float
    heading: float = 0.0
    speed: float = 100  # cm/s
    action: str = "navigate"  # navigate, stop, turn, wait

@dataclass
class RoutePlan:
    name: str
    waypoints: List[Waypoint]
    created_at: str = ""
    total_distance: float = 0.0
    estimated_time: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate total distance and estimated time"""
        self.total_distance = 0
        self.estimated_time = 0
        
        for i in range(1, len(self.waypoints)):
            prev = self.waypoints[i-1]
            curr = self.waypoints[i]
            dist = math.sqrt((curr.x - prev.x)**2 + **(curr.z - prev.z)2)
            self.total_distance += dist
            
            if curr.speed > 0:
                self.estimated_time += dist / curr.speed

@dataclass
class Incident:
    id: int
    type: str
    timestamp: str
    position: Dict[str, float]
    state: Dict
    severity: str  # low, medium, high, critical
    description: str
    resolution: str = ""

# ============ ROUTE PLANNER ============
class RoutePlanner:
    def __init__(self):
        self.routes: Dict[str, RoutePlan] = {}
        self.current_route: Optional[RoutePlan] = None
        self.current_waypoint_index = 0
        
    def create_route(self, name: str, waypoints: List[Tuple[float, float, float]]) -> RoutePlan:
        """Create a new route from waypoints (x, z, heading)"""
        wp_objects = [Waypoint(x=w[0], z=w[1], heading=w[2] if len(w) > 2 else 0) 
                      for w in waypoints]
        route = RoutePlan(name=name, waypoints=wp_objects)
        self.routes[name] = route
        print(f"✅ Route '{name}' created with {len(waypoints)} waypoints")
        print(f"   Distance: {route.total_distance:.1f}cm, ETA: {route.estimated_time:.1f}s")
        return route
    
    def load_sample_routes(self):
        """Load sample routes for demonstration"""
        # Route 1: Square pattern
        self.create_route("square_loop", [
            (0, 0, 0),
            (50, 0, 90),
            (50, 50, 180),
            (0, 50, 270),
            (0, 0, 0)
        ])
        
        # Route 2: Figure 8
        self.create_route("figure_eight", [
            (0, 0, 45),
            (30, 30, 90),
            (0, 60, 135),
            (-30, 30, 180),
            (0, 0, 225),
            (30, -30, 270),
            (0, -60, 315),
            (-30, -30, 0),
            (0, 0, 45)
        ])
        
        # Route 3: Obstacle avoidance course
        self.create_route("obstacle_course", [
            (0, 0, 0),
            (20, 0, 0),
            (20, 10, 90),
            (10, 10, 90),
            (10, 20, 0),
            (30, 20, 0),
            (30, 30, 90),
            (0, 30, 180),
            (0, 0, 270)
        ])
        
        print(f"📍 Loaded {len(self.routes)} sample routes")
    
    def select_route(self, name: str) -> bool:
        """Select a route for simulation/execution"""
        if name in self.routes:
            self.current_route = self.routes[name]
            self.current_waypoint_index = 0
            print(f"🎯 Selected route: {name}")
            return True
        print(f"❌ Route '{name}' not found")
        return False
    
    def get_next_waypoint(self) -> Optional[Waypoint]:
        """Get next waypoint in current route"""
        if not self.current_route:
            return None
        if self.current_waypoint_index < len(self.current_route.waypoints):
            return self.current_route.waypoints[self.current_waypoint_index]
        return None
    
    def advance_waypoint(self):
        """Move to next waypoint"""
        if self.current_route:
            self.current_waypoint_index += 1

# ============ INCIDENT RECORDER ============
class IncidentRecorder:
    def __init__(self):
        self.incidents: List[Incident] = []
        self.incident_id_counter = 0
        self.state_history: deque = deque(maxlen=1000)  # Last 1000 states
        self.recording = True
        
    def record_state(self, state: Dict):
        """Record robot state for replay"""
        if self.recording:
            state_with_time = state.copy()
            state_with_time["recorded_at"] = time.time()
            self.state_history.append(state_with_time)
    
    def add_incident(self, incident_type: str, state: Dict, severity: str = "medium") -> Incident:
        """Record an incident"""
        self.incident_id_counter += 1
        
        # Classify incident type
        descriptions = {
            "ULTRASONIC_CLOSE": "Object detected within critical distance",
            "EMERGENCY_STOP": "Emergency stop triggered",
            "OBSTACLE_AVOIDANCE": "Obstacle avoidance maneuver executed",
            "COLLISION_WARNING": "Potential collision detected",
            "SENSOR_FAILURE": "Sensor data unavailable or invalid",
            "ROUTE_DEVIATION": "Robot deviated from planned route",
            "LOW_BATTERY": "Battery voltage below threshold",
            "COMMUNICATION_LOSS": "Lost connection with robot"
        }
        
        incident = Incident(
            id=self.incident_id_counter,
            type=incident_type,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            position=state.get("position", {"x": 0, "y": 0, "z": 0}),
            state=state,
            severity=severity,
            description=descriptions.get(incident_type, "Unknown incident type"),
            resolution=""
        )
        
        self.incidents.append(incident)
        
        # Print notification
        severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "⚠️"}
        print(f"{severity_emoji.get(severity, '📝')} INCIDENT #{incident.id}: {incident_type}")
        print(f"   {incident.description}")
        print(f"   Position: ({state.get('position', {}).get('x', 0):.1f}, {state.get('position', {}).get('z', 0):.1f})")
        
        return incident
    
    def get_incidents(self) -> List[Incident]:
        """Get all recorded incidents"""
        return self.incidents
    
    def get_incident(self, incident_id: int) -> Optional[Incident]:
        """Get specific incident by ID"""
        for inc in self.incidents:
            if inc.id == incident_id:
                return inc
        return None
    
    def export_incidents(self, filename: str = "incidents.json"):
        """Export incidents to JSON file"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_incidents": len(self.incidents),
            "incidents": [asdict(inc) for inc in self.incidents]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Exported {len(self.incidents)} incidents to {filename}")
    
    def get_state_at_time(self, timestamp: float) -> Optional[Dict]:
        """Get closest recorded state to given timestamp"""
        if not self.state_history:
            return None
        
        closest = min(self.state_history, 
                     key=lambda s: abs(s.get("recorded_at", 0) - timestamp))
        return closest

# ============ ROUTE SIMULATOR ============
class RouteSimulator:
    def __init__(self, planner: RoutePlanner, recorder: IncidentRecorder):
        self.planner = planner
        self.recorder = recorder
        self.is_simulating = False
        self.simulation_state = {
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": 0,
            "throttle": 0,
            "steering": 0,
            "speed": 0,
            "ultrasonic": 999
        }
        
    def simulate_route(self, route_name: str, add_obstacles: bool = True):
        """Simulate a route with optional obstacles"""
        if not self.planner.select_route(route_name):
            return
        
        print(f"\n🚀 Starting simulation of '{route_name}'...")
        self.is_simulating = True
        
        route = self.planner.current_route
        current_pos = np.array([0.0, 0.0])
        current_heading = 0.0
        
        for i, waypoint in enumerate(route.waypoints):
            if not self.is_simulating:
                break
                
            target = np.array([waypoint.x, waypoint.z])
            print(f"\n📍 Waypoint {i+1}/{len(route.waypoints)}: ({waypoint.x}, {waypoint.z})")
            
            # Simulate movement to waypoint
            while True:
                direction = target - current_pos
                distance = np.linalg.norm(direction)
                
                if distance < 2:  # Close enough to waypoint
                    break
                
                # Normalize and move
                direction = direction / distance
                step = min(5, distance)  # Move 5cm per step
                current_pos = current_pos + direction * step
                
                # Update heading
                current_heading = math.atan2(direction[1], direction[0])
                
                # Update simulation state
                self.simulation_state["position"] = {
                    "x": float(current_pos[0]),
                    "y": 0,
                    "z": float(current_pos[1])
                }
                self.simulation_state["rotation"] = current_heading
                self.simulation_state["throttle"] = waypoint.speed
                self.simulation_state["speed"] = waypoint.speed * 0.5
                self.simulation_state["steering"] = 0
                
                # Simulate obstacles
                if add_obstacles:
                    obstacle_dist = self._check_obstacle(current_pos)
                    self.simulation_state["ultrasonic"] = obstacle_dist
                    
                    if obstacle_dist < 30:
                        # Record incident
                        severity = "critical" if obstacle_dist < 15 else "high"
                        self.recorder.add_incident(
                            "SIMULATED_OBSTACLE",
                            self.simulation_state.copy(),
                            severity
                        )
                        print(f"  ⚠️ Obstacle at {obstacle_dist:.1f}cm!")
                        
                        # Simulate avoidance
                        self.simulation_state["throttle"] = 0
                        self.simulation_state["steering"] = -0.5
                
                # Record state
                self.recorder.record_state(self.simulation_state.copy())
                
                print(f"  Pos: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | Dist: {distance:.1f}cm | Ultra: {self.simulation_state['ultrasonic']:.0f}cm")
                
                time.sleep(0.1)  # Simulation step
        
        self.is_simulating = False
        print(f"\n✅ Simulation complete!")
        print(f"   Total incidents recorded: {len(self.recorder.get_incidents())}")
    
    def _check_obstacle(self, position: np.ndarray) -> float:
        """Simulate ultrasonic sensor with random obstacles"""
        # Create some fixed obstacle positions
        obstacles = [
            np.array([25, 5]),
            np.array([25, 45]),
            np.array([5, 25]),
            np.array([45, 25])
        ]
        
        min_dist = 999.0
        for obs in obstacles:
            dist = np.linalg.norm(position - obs)
            if dist < min_dist:
                min_dist = dist
        
        # Return distance minus some noise
        return max(0, min_dist - 20 + np.random.random() * 5)
    
    def stop_simulation(self):
        """Stop current simulation"""
        self.is_simulating = False
        print("🛑 Simulation stopped")

# ============ INTERACTIVE CLI ============
def main():
    planner = RoutePlanner()
    recorder = IncidentRecorder()
    simulator = RouteSimulator(planner, recorder)
    
    # Load sample routes
    planner.load_sample_routes()
    
    # Command loop
    while True:
        try:
            cmd = input("\n🔮 digital_twin> ").strip().lower()
            
            if cmd == "help":
                print("""
Commands:
  plan                     - List all routes
  plan <name>              - Select a route
  plan new <name>          - Create new route (interactive)
  simulate <route>         - Simulate a route
  simulate <route> --safe  - Simulate without obstacles
  stop                     - Stop current simulation
  incidents                - List all incidents
  incidents <id>           - Show details of incident
  incidents export         - Export incidents to JSON
  replay <id>              - Replay incident (opens viewer)
  status                   - Show current robot state
  clear                    - Clear all incidents
  quit                     - Exit program
""")
            
            elif cmd == "plan":
                print("\n📍 Available Routes:")
                for name, route in planner.routes.items():
                    current = " [SELECTED]" if planner.current_route == route else ""
                    print(f"  • {name}: {len(route.waypoints)} waypoints, {route.total_distance:.1f}cm{current}")
            
            elif cmd.startswith("plan "):
                parts = cmd.split()
                if len(parts) >= 2:
                    if parts[1] == "new":
                        # Interactive route creation
                        name = input("Route name: ")
                        waypoints = []
                        print("Enter waypoints (x z heading), empty line to finish:")
                        while True:
                            wp_input = input("  > ")
                            if not wp_input:
                                break
                            try:
                                coords = list(map(float, wp_input.split()))
                                waypoints.append(tuple(coords))
                            except:
                                print("  Invalid format. Use: x z heading")
                        if waypoints:
                            planner.create_route(name, waypoints)
                    else:
                        planner.select_route(parts[1])
            
            elif cmd.startswith("simulate"):
                parts = cmd.split()
                route_name = parts[1] if len(parts) > 1 else "square_loop"
                add_obstacles = "--safe" not in cmd
                simulator.simulate_route(route_name, add_obstacles)
            
            elif cmd == "stop":
                simulator.stop_simulation()
            
            elif cmd == "incidents":
                incidents = recorder.get_incidents()
                if not incidents:
                    print("  No incidents recorded")
                else:
                    print(f"\n📋 Recorded Incidents ({len(incidents)}):")
                    for inc in incidents:
                        severity_icon = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "⚠️"}
                        print(f"  #{inc.id} {severity_icon.get(inc.severity, '')} {inc.type}")
                        print(f"     {inc.timestamp} | {inc.description[:50]}...")
            
            elif cmd.startswith("incidents "):
                parts = cmd.split()
                if len(parts) >= 2:
                    if parts[1] == "export":
                        recorder.export_incidents()
                    elif parts[1].isdigit():
                        inc = recorder.get_incident(int(parts[1]))
                        if inc:
                            print(f"\n📋 Incident #{inc.id}")
                            print(f"  Type: {inc.type}")
                            print(f"  Time: {inc.timestamp}")
                            print(f"  Severity: {inc.severity}")
                            print(f"  Description: {inc.description}")
                            print(f"  Position: ({inc.position['x']:.1f}, {inc.position['z']:.1f})")
                            print(f"  State:")
                            for k, v in inc.state.items():
                                if k != "position":
                                    print(f"    {k}: {v}")
            
            elif cmd.startswith("replay"):
                print("  Replay functionality available in web interface")
                print("  Open http://localhost:8080 and use the Timeline feature")
            
            elif cmd == "status":
                print("\n📊 Current State:")
                for k, v in simulator.simulation_state.items():
                    print(f"  {k}: {v}")
            
            elif cmd == "clear":
                recorder.incidents.clear()
                recorder.state_history.clear()
                print("  ✅ Cleared all incidents")
            
            elif cmd == "quit" or cmd == "exit":
                print("  Goodbye!")
                break
            
            else:
                print(f"  Unknown command: {cmd}")
                print("  Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n  Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
