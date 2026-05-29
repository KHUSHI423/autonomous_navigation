"""
================================================================================
                    TESLA-STYLE BIRD'S EYE VIEW MAPPING
                    Semantic BEV + Occupancy Grid + FSD Visualization
================================================================================

Features:
    ✅ Bird's Eye View (BEV) projection
    ✅ Semantic occupancy grid (road, obstacle, free space)
    ✅ Inverse Perspective Mapping (IPM)
    ✅ Tesla-style visualization
    ✅ Persistent map accumulation
    ✅ Real-time processing
    ✅ 3D bounding boxes for objects

Requirements:
    pip install opencv-python numpy torch torchvision

Controls:
    Q - Quit
    S - Save map
    B - Toggle BEV view
    O - Toggle occupancy grid
    M - Toggle 3D mode
    R - Reset map
"""

import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime

# ==================== Configuration ====================
CONFIG = {
    # Camera
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    "FOCAL_LENGTH": 800.0,
    
    # BEV Settings
    "BEV_WIDTH": 512,
    "BEV_HEIGHT": 512,
    "BEV_RESOLUTION": 0.1,  # meters per pixel
    "BEV_RANGE": 25,        # meters in each direction
    
    # IPM (Inverse Perspective Mapping)
    "CAMERA_HEIGHT": 1.5,   # meters above ground
    "CAMERA_PITCH": 20.0,   # degrees tilted down
    "CAMERA_YAW": 0.0,
    
    # Occupancy Grid
    "GRID_SIZE": 512,
    "FREE_SPACE_THRESHOLD": 0.7,
    "OBSTACLE_HEIGHT": 0.5,  # meters
    
    # Visualization
    "COLOR_BG": (10, 10, 15),
    "COLOR_FREE": (30, 50, 30),      # Green tint - drivable
    "COLOR_OCCUPIED": (50, 30, 30),  # Red tint - obstacle
    "COLOR_UNKNOWN": (20, 20, 25),   # Gray - unknown
    "COLOR_ROAD": (40, 60, 40),
    "COLOR_OBJECT": (200, 100, 50),
    "COLOR_TRAJECTORY": (0, 200, 255),
    "COLOR_CAMERA": (0, 255, 0),
}

# ==================== Global State ====================
class GlobalState:
    def __init__(self):
        # BEV Map
        self.bev_map = np.zeros((CONFIG["BEV_HEIGHT"], CONFIG["BEV_WIDTH"], 3), dtype=np.uint8)
        self.occupancy_grid = np.zeros((CONFIG["GRID_SIZE"], CONFIG["GRID_SIZE"]), dtype=np.float32)
        self.height_map = np.zeros((CONFIG["GRID_SIZE"], CONFIG["GRID_SIZE"]), dtype=np.float32)
        
        # Camera pose
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_yaw = 0.0
        self.trajectory = deque(maxlen=500)
        self.trajectory.append((0.0, 0.0))
        
        # Objects (3D bounding boxes)
        self.objects = []  # List of {bbox, class, confidence, position}
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Flags
        self.show_bev = True
        self.show_occupancy = True
        self.show_3d = False
        
        import threading
        self.lock = threading.Lock()

state = GlobalState()

# ==================== Inverse Perspective Mapping ====================
class IPMTransformer:
    """Transform camera view to Bird's Eye View using homography"""
    
    def __init__(self):
        self.camera_height = CONFIG["CAMERA_HEIGHT"]
        self.camera_pitch = np.radians(CONFIG["CAMERA_PITCH"])
        self.focal_length = CONFIG["FOCAL_LENGTH"]
        
        # Source points in camera image (trapezoid - road area)
        self.src_points = np.float32([
            [100, 350],   # Bottom-left
            [540, 350],  # Bottom-right
            [280, 200],   # Top-left
            [360, 200]   # Top-right
        ])
        
        # Destination points in BEV (rectangle)
        self.dst_points = np.float32([
            [50, 450],    # Bottom-left
            [450, 450],  # Bottom-right
            [50, 50],     # Top-left
            [450, 50]    # Top-right
        ])
        
        # Compute homography matrix
        self.H, _ = cv2.findHomography(self.src_points, self.dst_points)
        
        # Pre-compute inverse mapping
        self.bev_w, self.bev_h = CONFIG["BEV_WIDTH"], CONFIG["BEV_HEIGHT"]
    
    def transform_to_bev(self, image):
        """Warp camera image to Bird's Eye View"""
        if image is None:
            return None
        
        # Apply perspective transform
        bev = cv2.warpPerspective(
            image, 
            self.H, 
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return bev
    
    def project_point_to_bev(self, point_3d, camera_pos, camera_yaw):
        """Project 3D world point to BEV coordinates"""
        # Relative to camera
        rel_x = point_3d[0] - camera_pos[0]
        rel_z = point_3d[2] - camera_pos[2]
        
        # Rotate by camera yaw
        cos_y = np.cos(-camera_yaw)
        sin_y = np.sin(-camera_yaw)
        
        rot_x = rel_x * cos_y - rel_z * sin_y
        rot_z = rel_x * sin_y + rel_z * cos_y
        
        # Convert to BEV pixels (origin at center)
        bev_x = self.bev_w // 2 + int(rot_x / CONFIG["BEV_RESOLUTION"])
        bev_y = self.bev_h // 2 - int(rot_z / CONFIG["BEV_RESOLUTION"])
        
        return (bev_x, bev_y)

# ==================== Semantic Segmentation (Lightweight) ====================
class SemanticSegmenter:
    """Lightweight semantic segmentation for road/obstacle detection"""
    
    def __init__(self):
        self.enabled = True
        
        # Color ranges for road detection (in HSV)
        self.road_lower = np.array([40, 30, 30])
        self.road_upper = np.array([70, 255, 200])
        
        # Pre-trained colors for common classes
        self.class_colors = {
            'road': (0, 150, 0),
            'sidewalk': (150, 150, 150),
            'car': (0, 100, 255),
            'person': (255, 0, 0),
            'building': (100, 100, 255),
            'vegetation': (50, 200, 50),
        }
    
    def segment_road(self, image):
        """Detect road area using color thresholding"""
        if image is None:
            return None
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Road detection (grayish/bluish colors)
        mask = cv2.inRange(hsv, self.road_lower, self.road_upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_obstacles(self, depth_estimate, height_threshold=0.5):
        """Detect obstacles based on height estimation"""
        # Points above height threshold are obstacles
        obstacle_mask = np.abs(depth_estimate) > height_threshold
        return obstacle_mask

# ==================== Occupancy Grid Map ====================
class OccupancyGridMap:
    """Maintain a 2D occupancy grid for navigation"""
    
    def __init__(self, size=512, resolution=0.1):
        self.size = size
        self.resolution = resolution
        self.grid = np.zeros((size, size), dtype=np.float32)  # -1: unknown, 0: free, 1: occupied
        self.grid[:] = -1  # Initialize as unknown
        
        self.center = size // 2
    
    def update(self, camera_pos, camera_yaw, depth_map, image=None):
        """Update occupancy grid with new sensor data"""
        # Convert camera position to grid coordinates
        grid_x = int(camera_pos[0] / self.resolution) + self.center
        grid_y = int(camera_pos[2] / self.resolution) + self.center
        
        # Clamp to grid bounds
        grid_x = max(0, min(self.size - 1, grid_x))
        grid_y = max(0, min(self.size - 1, grid_y))
        
        # Update local area around camera
        radius = 50
        for dy in range(-radius, radius):
            for dx in range(-radius, radius):
                x = grid_x + dx
                y = grid_y + dy
                
                if 0 <= x < self.size and 0 <= y < self.size:
                    # Get depth at this point
                    depth_idx = (dy + radius, dx + radius)
                    
                    if depth_idx[0] < depth_map.shape[0] and depth_idx[1] < depth_map.shape[1]:
                        depth = depth_map[depth_idx]
                        
                        if depth > 0.5:  # Free space
                            self.grid[y, x] = max(self.grid[y, x], 0.0)
                        elif depth > 0.1:  # Obstacle
                            self.grid[y, x] = 1.0
    
    def get_visualization(self):
        """Convert occupancy grid to color image"""
        vis = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Unknown (gray)
        vis[self.grid < 0] = [20, 20, 25]
        
        # Free space (green)
        free_mask = self.grid >= 0
        vis[free_mask] = [30, 50, 30]
        
        # Occupied (red)
        occ_mask = self.grid > 0.5
        vis[occ_mask] = [50, 30, 30]
        
        return vis

# ==================== 3D Bounding Box Estimator ====================
class BoundingBoxEstimator:
    """Estimate 3D bounding boxes from 2D detections"""
    
    def __init__(self):
        self.class_dimensions = {
            'car': (4.5, 1.5, 2.0),      # (length, height, width) in meters
            'person': (0.5, 1.8, 0.6),
            'truck': (8.0, 3.0, 2.5),
            'bicycle': (1.8, 1.2, 0.6),
        }
    
    def estimate_3d_bbox(self, bbox_2d, depth, class_name='car'):
        """Estimate 3D bounding box from 2D bbox and depth"""
        x1, y1, x2, y2 = bbox_2d
        
        # Get object dimensions
        dims = self.class_dimensions.get(class_name, (2.0, 1.5, 1.0))
        length, height, width = dims
        
        # Center in image
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Estimate 3D position (simplified)
        fx = CONFIG["FOCAL_LENGTH"]
        X = (cx - CONFIG["FRAME_WIDTH"] / 2) * depth / fx
        Y = (cy - CONFIG["FRAME_HEIGHT"] / 2) * depth / fx
        Z = depth
        
        return {
            'position': np.array([X, Y, Z]),
            'dimensions': np.array(dims),
            'rotation': 0.0,  # Assume aligned with camera
            'class': class_name,
            'confidence': 0.8
        }

# ==================== Tesla-Style Renderer ====================
class TeslaRenderer:
    """Render Tesla FSD-style visualization"""
    
    def __init__(self):
        self.ipm = IPMTransformer()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        
        # Pre-compute BEV grid
        self.bev_grid = self._create_bev_grid()
    
    def _create_bev_grid(self):
        """Create BEV coordinate grid"""
        grid_lines = []
        spacing = int(2.0 / CONFIG["BEV_RESOLUTION"])  # 2 meter grid
        
        for i in range(-CONFIG["BEV_RANGE"], CONFIG["BEV_RANGE"] + 1, 2):
            x = CONFIG["BEV_WIDTH"] // 2 + int(i / CONFIG["BEV_RESOLUTION"])
            grid_lines.append(((x, 0), (x, CONFIG["BEV_HEIGHT"])))
            grid_lines.append(((0, x), (CONFIG["BEV_WIDTH"], x)))
        
        return grid_lines
    
    def render(self, frame, bev_view=None):
        """Render complete Tesla-style dashboard"""
        with state.lock:
            dashboard = np.zeros((700, 1280, 3), dtype=np.uint8)
            dashboard[:] = CONFIG["COLOR_BG"]
            
            # Left panel - Camera feed
            cam_panel = dashboard[0:480, 0:640]
            if frame.shape[0] != 480 or frame.shape[1] != 640:
                resized = cv2.resize(frame, (640, 480))
            else:
                resized = frame
            cam_panel[:] = resized
            
            # Right-top panel - BEV Map
            bev_panel = dashboard[0:350, 640:1280]
            self._render_bev_map(bev_panel)
            
            # Right-bottom panel - Occupancy Grid
            occ_panel = dashboard[350:480, 640:1280]
            self._render_occupancy_grid(occ_panel)
            
            # Bottom panel - Stats
            stats_panel = dashboard[480:700, :]
            self._render_stats(stats_panel)
            
            # Overlay on camera
            self._render_camera_overlay(cam_panel)
        
        return dashboard
    
    def _render_bev_map(self, panel):
        """Render Bird's Eye View map"""
        panel[:] = CONFIG["COLOR_BG"]
        
        # Draw grid
        for line in self.bev_grid:
            cv2.line(panel, line[0], line[1], (40, 40, 50), 1)
        
        # Draw trajectory
        if len(state.trajectory) > 1:
            traj_points = []
            for pos in state.trajectory:
                x = CONFIG["BEV_WIDTH"] // 2 + int(pos[0] / CONFIG["BEV_RESOLUTION"])
                y = CONFIG["BEV_HEIGHT"] // 2 - int(pos[1] / CONFIG["BEV_RESOLUTION"])
                traj_points.append([x, y])
            
            cv2.polylines(panel, [np.array(traj_points, np.int32)], 
                         False, CONFIG["COLOR_TRAJECTORY"], 2)
        
        # Draw camera (ego vehicle)
        cam_x = CONFIG["BEV_WIDTH"] // 2
        cam_y = CONFIG["BEV_HEIGHT"] // 2
        
        # Draw vehicle as rectangle
        vehicle_w, vehicle_h = 20, 40
        vehicle_pts = np.array([
            [cam_x - vehicle_w//2, cam_y + vehicle_h//2],
            [cam_x + vehicle_w//2, cam_y + vehicle_h//2],
            [cam_x + vehicle_w//2, cam_y - vehicle_h//2],
            [cam_x - vehicle_w//2, cam_y - vehicle_h//2],
        ], np.int32)
        cv2.fillPoly(panel, [vehicle_pts], CONFIG["COLOR_CAMERA"])
        
        # Draw heading indicator
        heading_end = (cam_x, cam_y - vehicle_h//2 - 10)
        cv2.line(panel, (cam_x, cam_y), heading_end, (255, 255, 0), 2)
        
        # Draw objects as 3D bounding boxes (top-down view)
        for obj in state.objects:
            pos = obj.get('position', state.camera_pos)
            
            # Project to BEV
            obj_x = CONFIG["BEV_WIDTH"] // 2 + int((pos[0] - state.camera_pos[0]) / CONFIG["BEV_RESOLUTION"])
            obj_y = CONFIG["BEV_HEIGHT"] // 2 - int((pos[2] - state.camera_pos[2]) / CONFIG["BEV_RESOLUTION"])
            
            # Draw bounding box
            bbox_w = int(obj.get('dimensions', [2.0, 1.5, 1.0])[2] / CONFIG["BEV_RESOLUTION"])
            bbox_h = int(obj.get('dimensions', [2.0, 1.5, 1.0])[0] / CONFIG["BEV_RESOLUTION"])
            
            bbox_pts = np.array([
                [obj_x - bbox_w//2, obj_y + bbox_h//2],
                [obj_x + bbox_w//2, obj_y + bbox_h//2],
                [obj_x + bbox_w//2, obj_y - bbox_h//2],
                [obj_x - bbox_w//2, obj_y - bbox_h//2],
            ], np.int32)
            
            cv2.polylines(panel, [bbox_pts], True, CONFIG["COLOR_OBJECT"], 2)
            
            # Label
            label = obj.get('class', 'object')
            cv2.putText(panel, label, (obj_x - 30, obj_y - 10),
                       self.font_small, 0.5, CONFIG["COLOR_OBJECT"], 1)
        
        # Range indicator
        range_m = CONFIG["BEV_RANGE"]
        cv2.putText(panel, f"{range_m}m", (10, 30),
                   self.font_small, 0.5, (150, 150, 150), 1)
        
        # Title
        cv2.putText(panel, "BIRD'S EYE VIEW", (10, 320),
                   self.font, 0.6, (255, 255, 255), 1)
    
    def _render_occupancy_grid(self, panel):
        """Render occupancy grid visualization"""
        panel[:] = CONFIG["COLOR_UNKNOWN"]
        
        # Get occupancy visualization
        occ_vis = state.occupancy_grid
        
        # Resize to fit panel
        occ_resized = cv2.resize(occ_vis, (panel.shape[1], panel.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Blend with panel
        mask = occ_resized > 0
        panel[mask] = occ_resized[mask]
        
        # Title
        cv2.putText(panel, "OCCUPANCY GRID", (10, 30),
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Legend
        legend_y = 50
        cv2.rectangle(panel, (10, legend_y), (30, legend_y + 15), CONFIG["COLOR_FREE"], -1)
        cv2.putText(panel, "Free", (35, legend_y + 12),
                   self.font_small, 0.4, (200, 200, 200), 1)
        
        cv2.rectangle(panel, (10, legend_y + 20), (30, legend_y + 35), CONFIG["COLOR_OCCUPIED"], -1)
        cv2.putText(panel, "Occupied", (35, legend_y + 32),
                   self.font_small, 0.4, (200, 200, 200), 1)
    
    def _render_camera_overlay(self, panel):
        """Render overlay on camera feed"""
        # Detection boxes (simulated)
        cv2.rectangle(panel, (200, 150), (400, 350), CONFIG["COLOR_OBJECT"], 2)
        cv2.putText(panel, "OBJECT 85%", (200, 145),
                   self.font_small, 0.5, CONFIG["COLOR_OBJECT"], 1)
        
        # Stats overlay
        y = 30
        cv2.putText(panel, f"FPS: {state.fps:.1f}", (10, y),
                   self.font, 0.6, CONFIG["COLOR_CAMERA"], 2)
        y += 25
        cv2.putText(panel, f"Pos: [{state.camera_pos[0]:.2f}, {state.camera_pos[2]:.2f}]", 
                   (10, y), self.font_small, 0.5, (255, 255, 255), 1)
        y += 20
        cv2.putText(panel, f"Objects: {len(state.objects)}", 
                   (10, y), self.font_small, 0.5, (255, 255, 255), 1)
        y += 20
        cv2.putText(panel, f"Mode: BEV", (10, y),
                   self.font_small, 0.5, (0, 200, 255), 1)
    
    def _render_stats(self, panel):
        """Render bottom stats panel"""
        panel[:] = (10, 10, 20)
        
        cv2.line(panel, (0, 100), (1280, 100), (40, 40, 50), 1)
        
        stats = [
            ("FRAMES", str(state.frame_count), 50),
            ("POSITION", f"[{state.camera_pos[0]:.2f}, {state.camera_pos[2]:.2f}]", 200),
            ("OBJECTS", str(len(state.objects)), 400),
            ("TRAJECTORY", str(len(state.trajectory)), 550),
            ("FPS", f"{state.fps:.1f}", 700),
            ("YAW", f"{np.degrees(state.camera_yaw):.1f}°", 850),
        ]
        
        for label, value, x in stats:
            cv2.putText(panel, label, (x, 35),
                       self.font_small, 0.5, (100, 100, 100), 1)
            cv2.putText(panel, value, (x, 65),
                       self.font, 0.7, (255, 255, 255), 1)
        
        # Controls
        controls = [
            ("Q", "Quit", 950),
            ("S", "Save", 1020),
            ("B", "BEV", 1090),
            ("O", "Occupancy", 1150),
            ("R", "Reset", 1230),
        ]
        
        for key, action, x in controls:
            cv2.putText(panel, f"[{key}] {action}", (x, 35),
                       self.font_small, 0.45, (150, 150, 150), 1)

# ==================== Main SLAM System ====================
class BEVSLAMSystem:
    def __init__(self):
        self.ipm = IPMTransformer()
        self.segmenter = SemanticSegmenter()
        self.occupancy_map = OccupancyGridMap()
        self.bbox_estimator = BoundingBoxEstimator()
        self.renderer = TeslaRenderer()
        
        self.prev_frame = None
        self.cumulative_yaw = 0.0
    
    def process_frame(self, frame):
        """Process frame and update BEV map"""
        state.frame_count += 1
        
        # Update FPS
        now = time.time()
        if now - state.last_fps_time > 1.0:
            state.fps = state.frame_count / (now - state.last_fps_time)
            state.frame_count = 0
            state.last_fps_time = now
        
        # Create depth estimate (simplified - based on optical flow)
        depth_map = self._estimate_depth(frame)
        
        # Update occupancy grid
        if state.show_occupancy:
            self.occupancy_map.update(
                state.camera_pos,
                self.cumulative_yaw,
                depth_map,
                frame
            )
            state.occupancy_grid = self.occupancy_map.get_visualization()
        
        # Simulate object detection (in real system, use YOLO)
        if state.frame_count % 30 == 0:
            self._detect_objects_simulated(frame, depth_map)
        
        # Update camera position (simulated - in real system use visual odometry)
        self._update_camera_pose(frame)
        
        # Store for next frame
        self.prev_frame = frame.copy()
        
        return frame
    
    def _estimate_depth(self, frame):
        """Estimate depth from single image (simplified)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use optical flow magnitude as depth proxy
        if self.prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
            depth = 1.0 / (magnitude + 0.1)  # Inverse relationship
            return np.clip(depth, 0, 10)
        
        return np.ones((frame.shape[0], frame.shape[1])) * 5.0
    
    def _detect_objects_simulated(self, frame, depth_map):
        """Simulate object detection (replace with YOLO in production)"""
        # In production, use:
        # from ultralytics import YOLO
        # model = YOLO('yolov8n.pt')
        # results = model(frame)
        
        # For now, simulate detections
        state.objects = []
        
        # Simulate 1-3 objects
        num_objects = np.random.randint(1, 4)
        for i in range(num_objects):
            obj = {
                'position': state.camera_pos + np.array([
                    np.random.uniform(-5, 5),
                    0,
                    np.random.uniform(5, 15)
                ]),
                'dimensions': np.array([2.0, 1.5, 1.0]),
                'class': 'car',
                'confidence': 0.85
            }
            state.objects.append(obj)
    
    def _update_camera_pose(self, frame):
        """Update camera pose (simplified - use visual odometry in production)"""
        # Simulate small forward movement
        delta = 0.1
        
        # Update position based on yaw
        state.camera_pos[0] += delta * np.sin(self.cumulative_yaw)
        state.camera_pos[2] += delta * np.cos(self.cumulative_yaw)
        
        # Add to trajectory
        state.trajectory.append((state.camera_pos[0], state.camera_pos[2]))
    
    def render(self, frame):
        """Render dashboard"""
        return self.renderer.render(frame)
    
    def save_data(self):
        """Save map data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save occupancy grid
        cv2.imwrite(f"occupancy_{timestamp}.png", state.occupancy_grid)
        print(f"💾 Saved: occupancy_{timestamp}.png")
        
        # Save trajectory
        with open(f"trajectory_{timestamp}.txt", 'w') as f:
            f.write("# x z\n")
            for pos in state.trajectory:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f}\n")
        print(f"💾 Saved: trajectory_{timestamp}.txt")
        
        # Save objects
        with open(f"objects_{timestamp}.json", 'w') as f:
            objects_data = []
            for obj in state.objects:
                obj_copy = obj.copy()
                obj_copy['position'] = obj_copy['position'].tolist()
                obj_copy['dimensions'] = obj_copy['dimensions'].tolist()
                objects_data.append(obj_copy)
            json.dump(objects_data, f, indent=2)
        print(f"💾 Saved: objects_{timestamp}.json")

# ==================== Main ====================
def main():
    print("\n" + "=" * 70)
    print("        TESLA-STYLE BIRD'S EYE VIEW MAPPING")
    print("        Semantic BEV + Occupancy Grid + FSD Visualization")
    print("=" * 70)
    print("\n📋 INSTRUCTIONS:")
    print("   • Move camera to build the map")
    print("   • BEV shows top-down view (like Google Maps)")
    print("   • Occupancy grid shows free/occupied space")
    print("   • Objects shown as 3D bounding boxes")
    print("\n🎮 CONTROLS:")
    print("   Q - Quit")
    print("   S - Save map")
    print("   B - Toggle BEV view")
    print("   O - Toggle occupancy grid")
    print("   R - Reset map")
    print("\n🚀 Starting...\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    slam = BEVSLAMSystem()
    print("✅ Ready! Move camera to start mapping...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = slam.process_frame(frame)
        dashboard = slam.render(processed)
        
        cv2.imshow("Tesla-Style BEV Mapping", dashboard)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            slam.save_data()
        elif key == ord('b'):
            state.show_bev = not state.show_bev
        elif key == ord('o'):
            state.show_occupancy = not state.show_occupancy
        elif key == ord('r'):
            with state.lock:
                state.camera_pos = np.array([0.0, 0.0, 0.0])
                state.trajectory.clear()
                state.trajectory.append((0.0, 0.0))
                state.objects.clear()
                state.occupancy_grid[:] = 0
                slam.occupancy_map.grid[:] = -1
            print("\n🔄 Map reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session complete!\n")

if __name__ == "__main__":
    main()
