"""
================================================================================
                    NAVIGATION-GRADE BEV MAPPING SYSTEM
                    Static Environment Mapping + Dynamic Object Filtering
                    MiDaS Depth + YOLO Detection + Occupancy Grid
================================================================================

Features:
    ✅ MiDaS monocular depth estimation
    ✅ YOLO object detection with 3D positioning
    ✅ Dynamic object masking (ignores people)
    ✅ Static environment mapping only
    ✅ Clean occupancy grid (like autonomous vehicles)
    ✅ Bird's Eye View projection
    ✅ Persistent world map

Requirements:
    pip install opencv-python numpy torch torchvision
    pip install timm  # For MiDaS

Controls:
    Q - Quit
    S - Save map
    D - Toggle depth view
    M - Toggle dynamic masking
    R - Reset map
    + / - - Adjust ground threshold
"""

import cv2
import numpy as np
import time
import torch
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
    "BEV_RESOLUTION": 0.05,  # 5cm per pixel
    "BEV_RANGE": 20,         # meters
    
    # Physical
    "CAMERA_HEIGHT": 1.5,    # meters above ground
    "CAMERA_PITCH": 15.0,    # degrees tilted down
    
    # Depth
    "MAX_DEPTH": 30.0,
    "MIN_DEPTH": 0.5,
    "GROUND_THRESHOLD": 0.7,  # Lower = more ground detected
    
    # Objects
    "OBJECT_CLASSES": ['car', 'truck', 'bus', 'chair', 'couch', 'potted plant', 
                       'dining table', 'tv', 'laptop', 'book', 'scissors'],
    "IGNORE_CLASSES": ['person', 'dog', 'cat', 'bird'],  # Dynamic objects
    
    # Visualization
    "COLOR_BG": (10, 10, 15),
    "COLOR_FREE": (20, 80, 20),
    "COLOR_OCCUPIED": (80, 30, 30),
    "COLOR_UNKNOWN": (30, 30, 35),
    "COLOR_OBJECT": (200, 150, 50),
    "COLOR_PERSON": (255, 50, 50),
    "COLOR_TRAJECTORY": (0, 200, 255),
}

# ==================== Global State ====================
class GlobalState:
    def __init__(self):
        # Maps
        self.occupancy_grid = np.full((CONFIG["BEV_HEIGHT"], CONFIG["BEV_WIDTH"]), -1, dtype=np.float32)
        self.height_map = np.zeros((CONFIG["BEV_HEIGHT"], CONFIG["BEV_WIDTH"]), dtype=np.float32)
        self.static_map = np.zeros((CONFIG["BEV_HEIGHT"], CONFIG["BEV_WIDTH"], 3), dtype=np.uint8)
        
        # Camera
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_yaw = 0.0
        self.trajectory = deque(maxlen=500)
        self.trajectory.append((0.0, 0.0))
        
        # Objects
        self.static_objects = []  # Furniture, walls, etc.
        self.dynamic_objects = []  # People, pets (detected but not mapped)
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Flags
        self.show_depth = False
        self.dynamic_masking = True
        self.ground_threshold = CONFIG["GROUND_THRESHOLD"]
        
        import threading
        self.lock = threading.Lock()

state = GlobalState()

# ==================== MiDaS Depth Estimation ====================
class DepthEstimator:
    """Monocular depth estimation using MiDaS"""
    
    def __init__(self):
        self.enabled = True
        self.model = None
        self.device = None
        
        try:
            # Try to load MiDaS
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load MiDaS Small (faster)
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Preprocessing transforms
            transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = transforms.small_transform
            
            print(f"✅ MiDaS loaded on {self.device}")
        except Exception as e:
            print(f"⚠️  MiDaS not available: {e}")
            print("   Using fallback depth estimation")
            self.enabled = False
    
    def estimate(self, image):
        """Estimate depth map from single image"""
        if not self.enabled or self.model is None:
            return self._fallback_depth(image)
        
        try:
            # Preprocess
            input_tensor = self.transform(image).to(self.device)
            
            # Inference
            with torch.no_grad():
                prediction = self.model(input_tensor)
            
            # Post-process
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode='bicubic',
                align_corners=False
            )
            
            depth = prediction.squeeze().cpu().numpy()
            
            # Normalize to meters (approximate)
            depth = depth / depth.max() * CONFIG["MAX_DEPTH"]
            depth = np.clip(depth, CONFIG["MIN_DEPTH"], CONFIG["MAX_DEPTH"])
            
            return depth
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return self._fallback_depth(image)
    
    def _fallback_depth(self, image):
        """Fallback: estimate depth from image gradients"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use perspective cues (lower = closer)
        h = gray.shape[0]
        depth = np.tile(np.linspace(0.5, CONFIG["MAX_DEPTH"], h), (gray.shape[1], 1)).T
        
        # Add some variation based on intensity
        edges = cv2.Canny(gray, 50, 150)
        depth += edges.astype(np.float32) * 0.1
        
        return np.clip(depth, CONFIG["MIN_DEPTH"], CONFIG["MAX_DEPTH"])

# ==================== YOLO Object Detection ====================
class ObjectDetector:
    """YOLO-based object detection with 3D positioning"""
    
    def __init__(self):
        self.model = None
        
        try:
            # Try to load YOLOv8n (nano - fastest)
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 loaded")
        except Exception as e:
            print(f"⚠️  YOLO not available: {e}")
            print("   Using fallback detection")
    
    def detect(self, image, depth_map):
        """Detect objects and estimate 3D positions"""
        objects = []
        dynamic = []
        
        if self.model is None:
            return self._fallback_detect(image, depth_map)
        
        try:
            # Run YOLO
            results = self.model(image, verbose=False, conf=0.4)
            result = results[0]
            
            boxes = result.boxes
            if boxes is None:
                return objects, dynamic
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()
                
                class_name = result.names[cls_id]
                
                # Get depth at object location
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Sample depth from object center region
                depth_region = depth_map[max(0, cy-20):min(depth_map.shape[0], cy+20),
                                        max(0, cx-20):min(depth_map.shape[1], cx+20)]
                obj_depth = np.median(depth_region) if len(depth_region) > 0 else 5.0
                
                # Estimate 3D position
                fx = CONFIG["FOCAL_LENGTH"]
                img_cx = CONFIG["FRAME_WIDTH"] // 2
                img_cy = CONFIG["FRAME_HEIGHT"] // 2
                
                X = (cx - img_cx) * obj_depth / fx
                Y = -CONFIG["CAMERA_HEIGHT"]  # On ground
                Z = obj_depth
                
                obj_data = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox_2d': (x1, y1, x2, y2),
                    'position_3d': np.array([X, Y, Z]),
                    'depth': obj_depth,
                    'is_dynamic': class_name.lower() in CONFIG["IGNORE_CLASSES"]
                }
                
                if obj_data['is_dynamic']:
                    dynamic.append(obj_data)
                else:
                    objects.append(obj_data)
        
        except Exception as e:
            print(f"Detection error: {e}")
            return self._fallback_detect(image, depth_map)
        
        return objects, dynamic
    
    def _fallback_detect(self, image, depth_map):
        """Fallback: simple motion-based detection"""
        # Return empty - no objects detected
        return [], []

# ==================== Occupancy Grid Mapper ====================
class OccupancyMapper:
    """Build and maintain occupancy grid map"""
    
    def __init__(self):
        self.grid_size = CONFIG["BEV_HEIGHT"]
        self.resolution = CONFIG["BEV_RESOLUTION"]
        self.center = self.grid_size // 2
        self.occupancy_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.float32)
        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
    
    def world_to_grid(self, x, z):
        """Convert world coordinates to grid coordinates"""
        gx = int(x / self.resolution) + self.center
        gz = int(z / self.resolution) + self.center
        return np.clip(gx, 0, self.grid_size - 1), np.clip(gz, 0, self.grid_size - 1)
    
    def grid_to_world(self, gx, gz):
        """Convert grid coordinates to world coordinates"""
        x = (gx - self.center) * self.resolution
        z = (gz - self.center) * self.resolution
        return x, z
    
    def update(self, depth_map, camera_pos, camera_yaw, objects, dynamic_objects):
        """Update occupancy grid with new sensor data"""
        # Create mask for dynamic objects (to ignore them)
        dynamic_mask = self._create_dynamic_mask(dynamic_objects, depth_map.shape)
        
        # Process depth map
        h, w = depth_map.shape
        
        for y in range(0, h, 2):  # Downsample for speed
            for x in range(0, w, 2):
                # Skip dynamic objects
                if state.dynamic_masking and dynamic_mask[y, x]:
                    continue
                
                depth = depth_map[y, x]
                
                if depth < CONFIG["MIN_DEPTH"] or depth > CONFIG["MAX_DEPTH"]:
                    continue
                
                # Convert to world coordinates
                # Camera coordinate system
                fx = CONFIG["FOCAL_LENGTH"]
                cx = w // 2
                cy = h // 2
                
                X_cam = (x - cx) * depth / fx
                Z_cam = depth
                Y_cam = 0  # Assume ground level
                
                # Rotate to world coordinates
                cos_yaw = np.cos(camera_yaw)
                sin_yaw = np.sin(camera_yaw)
                
                X_world = camera_pos[0] + X_cam * cos_yaw - Z_cam * sin_yaw
                Z_world = camera_pos[2] + X_cam * sin_yaw + Z_cam * cos_yaw
                
                # Update grid
                gx, gz = self.world_to_grid(X_world, Z_world)
                
                # Mark as free space (ray casting)
                self._cast_ray(camera_pos, (X_world, Z_world), depth_map.shape)
                
                # Mark as occupied
                if 0 <= gx < self.grid_size and 0 <= gz < self.grid_size:
                    # Check if obstacle (not ground)
                    if depth > state.ground_threshold * CONFIG["MAX_DEPTH"]:
                        self.occupancy_grid[gz, gx] = max(self.occupancy_grid[gz, gx], 1.0)
                        self.height_map[gz, gx] = max(self.height_map[gz, gx], depth)
    
    def _cast_ray(self, origin, target, shape):
        """Mark cells along ray as free space"""
        ox, oz = origin[0], origin[2]
        tx, tz = target
        
        # Bresenham's line algorithm
        dx = abs(tx - ox)
        dz = abs(tz - oz)
        sx = 1 if ox < tx else -1
        sz = 1 if oz < tz else -1
        
        err = dx - dz
        
        x, z = ox, oz
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            gx, gz = self.world_to_grid(x, z)
            
            if 0 <= gx < self.grid_size and 0 <= gz < self.grid_size:
                # Mark as free space (but don't overwrite occupied)
                if self.occupancy_grid[gz, gx] < 0.5:
                    self.occupancy_grid[gz, gx] = 0.0
            
            if abs(x - tx) < 0.1 and abs(z - tz) < 0.1:
                break
            
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x += sx * 0.1
            if e2 < dx:
                err += dx
                z += sz * 0.1
            
            steps += 1
    
    def _create_dynamic_mask(self, dynamic_objects, shape):
        """Create mask to ignore dynamic objects"""
        mask = np.zeros(shape, dtype=bool)
        
        for obj in dynamic_objects:
            x1, y1, x2, y2 = map(int, obj['bbox_2d'])
            mask[y1:y2, x1:x2] = True
        
        return mask
    
    def get_visualization(self):
        """Convert occupancy grid to color image"""
        vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Unknown (dark gray)
        vis[self.occupancy_grid < 0] = CONFIG["COLOR_UNKNOWN"]
        
        # Free space (green)
        free_mask = (self.occupancy_grid >= 0) & (self.occupancy_grid < 0.5)
        vis[free_mask] = CONFIG["COLOR_FREE"]
        
        # Occupied (red)
        occ_mask = self.occupancy_grid >= 0.5
        vis[occ_mask] = CONFIG["COLOR_OCCUPIED"]
        
        return vis

# ==================== BEV Renderer ====================
class BEVRenderer:
    """Render navigation-style BEV visualization"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.mapper = OccupancyMapper()
    
    def render(self, frame, depth_map=None):
        """Render complete navigation dashboard"""
        with state.lock:
            dashboard = np.zeros((700, 1280, 3), dtype=np.uint8)
            dashboard[:] = CONFIG["COLOR_BG"]
            
            # Left panel - Camera feed
            cam_panel = dashboard[0:480, 0:640]
            resized = cv2.resize(frame, (640, 480))
            cam_panel[:] = resized
            
            # Add depth overlay if enabled
            if state.show_depth and depth_map is not None:
                depth_vis = cv2.applyColorMap(
                    (depth_map / CONFIG["MAX_DEPTH"] * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                depth_resized = cv2.resize(depth_vis, (640, 480))
                cam_panel = cv2.addWeighted(cam_panel, 0.6, depth_resized, 0.4, 0)
            
            # Right-top panel - BEV Map
            bev_panel = dashboard[0:400, 640:1280]
            self._render_bev_map(bev_panel)
            
            # Right-bottom panel - Object list
            obj_panel = dashboard[400:480, 640:1280]
            self._render_object_list(obj_panel)
            
            # Bottom panel - Stats
            stats_panel = dashboard[480:700, :]
            self._render_stats(stats_panel)
            
            # Overlay on camera
            self._render_camera_overlay(cam_panel, depth_map)
            
            dashboard[0:480, 0:640] = cam_panel
        
        return dashboard
    
    def _render_bev_map(self, panel):
        """Render BEV occupancy grid"""
        panel[:] = CONFIG["COLOR_UNKNOWN"]
        
        # Get occupancy visualization
        occ_vis = state.occupancy_grid.copy()
        
        # Create color visualization
        vis = np.zeros((self.mapper.grid_size, self.mapper.grid_size, 3), dtype=np.uint8)
        vis[occ_vis < 0] = CONFIG["COLOR_UNKNOWN"]
        vis[(occ_vis >= 0) & (occ_vis < 0.5)] = CONFIG["COLOR_FREE"]
        vis[occ_vis >= 0.5] = CONFIG["COLOR_OCCUPIED"]
        
        # Resize to fit panel
        vis_resized = cv2.resize(vis, (panel.shape[1], panel.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        panel[:] = vis_resized
        
        # Draw grid lines
        step = int(2.0 / CONFIG["BEV_RESOLUTION"])
        for i in range(0, panel.shape[0], step):
            cv2.line(panel, (0, i), (panel.shape[1], i), (50, 50, 60), 1)
        for i in range(0, panel.shape[1], step):
            cv2.line(panel, (i, 0), (i, panel.shape[0]), (50, 50, 60), 1)
        
        # Draw trajectory
        if len(state.trajectory) > 1:
            traj_points = []
            for pos in state.trajectory:
                x = panel.shape[1] // 2 + int(pos[0] / CONFIG["BEV_RESOLUTION"])
                y = panel.shape[0] // 2 - int(pos[1] / CONFIG["BEV_RESOLUTION"])
                traj_points.append([x, y])
            
            cv2.polylines(panel, [np.array(traj_points, np.int32)], 
                         False, CONFIG["COLOR_TRAJECTORY"], 2)
        
        # Draw ego vehicle
        ego_x, ego_y = panel.shape[1] // 2, panel.shape[0] // 2
        vehicle_pts = np.array([
            [ego_x - 15, ego_y + 25],
            [ego_x + 15, ego_y + 25],
            [ego_x + 15, ego_y - 25],
            [ego_x - 15, ego_y - 25],
        ], np.int32)
        cv2.fillPoly(panel, [vehicle_pts], CONFIG["COLOR_CAMERA"])
        
        # Draw heading
        heading_len = 40
        heading_end = (
            int(ego_x + heading_len * np.sin(state.camera_yaw)),
            int(ego_y - heading_len * np.cos(state.camera_yaw))
        )
        cv2.line(panel, (ego_x, ego_y), heading_end, (255, 255, 0), 2)
        
        # Draw static objects
        for obj in state.static_objects:
            pos = obj['position_3d']
            rel_x = pos[0] - state.camera_pos[0]
            rel_z = pos[2] - state.camera_pos[2]
            
            obj_x = panel.shape[1] // 2 + int(rel_x / CONFIG["BEV_RESOLUTION"])
            obj_y = panel.shape[0] // 2 - int(rel_z / CONFIG["BEV_RESOLUTION"])
            
            # Draw bounding box
            size = 10
            cv2.rectangle(panel, 
                         (obj_x - size, obj_y - size),
                         (obj_x + size, obj_y + size),
                         CONFIG["COLOR_OBJECT"], 2)
            
            # Label
            cv2.putText(panel, obj['class'][:8], 
                       (obj_x - 25, obj_y - 15),
                       self.font_small, 0.4, CONFIG["COLOR_OBJECT"], 1)
        
        # Draw dynamic objects (people)
        for obj in state.dynamic_objects:
            pos = obj['position_3d']
            rel_x = pos[0] - state.camera_pos[0]
            rel_z = pos[2] - state.camera_pos[2]
            
            obj_x = panel.shape[1] // 2 + int(rel_x / CONFIG["BEV_RESOLUTION"])
            obj_y = panel.shape[0] // 2 - int(rel_z / CONFIG["BEV_RESOLUTION"])
            
            cv2.circle(panel, (obj_x, obj_y), 8, CONFIG["COLOR_PERSON"], -1)
        
        # Title
        cv2.putText(panel, "OCCUPANCY GRID MAP", (10, 30),
                   self.font, 0.7, (255, 255, 255), 1)
        
        # Range
        range_m = int(CONFIG["BEV_RANGE"])
        cv2.putText(panel, f"{range_m}m", (panel.shape[1] - 50, 30),
                   self.font_small, 0.5, (150, 150, 150), 1)
    
    def _render_object_list(self, panel):
        """Render detected objects list"""
        panel[:] = (15, 15, 25)
        
        y = 25
        cv2.putText(panel, "STATIC OBJECTS:", (10, y),
                   self.font_small, 0.5, (200, 200, 200), 1)
        y += 20
        
        for i, obj in enumerate(state.static_objects[:5]):
            cv2.putText(panel, f"  • {obj['class']} ({obj['depth']:.1f}m)", 
                       (15, y), self.font_small, 0.4, (200, 150, 50), 1)
            y += 18
        
        if len(state.static_objects) == 0:
            cv2.putText(panel, "  None detected", (15, y),
                       self.font_small, 0.4, (100, 100, 100), 1)
        
        # Dynamic objects
        y = 55
        cv2.putText(panel, "DYNAMIC (IGNORED):", (10, y),
                   self.font_small, 0.5, (200, 200, 200), 1)
        y += 20
        
        for obj in state.dynamic_objects[:3]:
            cv2.putText(panel, f"  • {obj['class']} ({obj['depth']:.1f}m)", 
                       (15, y), self.font_small, 0.4, CONFIG["COLOR_PERSON"], 1)
            y += 18
    
    def _render_camera_overlay(self, panel, depth_map):
        """Render overlay on camera feed"""
        # Stats
        y = 30
        cv2.putText(panel, f"FPS: {state.fps:.1f}", (10, y),
                   self.font, 0.6, (0, 255, 100), 2)
        y += 25
        cv2.putText(panel, f"Pos: [{state.camera_pos[0]:.2f}, {state.camera_pos[2]:.2f}]", 
                   (10, y), self.font_small, 0.5, (255, 255, 255), 1)
        y += 20
        cv2.putText(panel, f"Static: {len(state.static_objects)} | Dynamic: {len(state.dynamic_objects)}", 
                   (10, y), self.font_small, 0.5, (255, 255, 255), 1)
        y += 20
        
        mode = "DEPTH" if state.show_depth else "RGB"
        mask = "MASK ON" if state.dynamic_masking else "MASK OFF"
        cv2.putText(panel, f"{mode} | {mask}", (10, y),
                   self.font_small, 0.5, (0, 200, 255), 1)
        
        # Bounding boxes
        for obj in state.static_objects:
            x1, y1, x2, y2 = map(int, obj['bbox_2d'])
            cv2.rectangle(panel, (x1, y1), (x2, y2), CONFIG["COLOR_OBJECT"], 2)
            cv2.putText(panel, f"{obj['class']} {obj['depth']:.1f}m", 
                       (x1, y1 - 5), self.font_small, 0.4, CONFIG["COLOR_OBJECT"], 1)
        
        for obj in state.dynamic_objects:
            x1, y1, x2, y2 = map(int, obj['bbox_2d'])
            cv2.rectangle(panel, (x1, y1), (x2, y2), CONFIG["COLOR_PERSON"], 2)
            cv2.putText(panel, f"{obj['class']} (IGN)", 
                       (x1, y1 - 5), self.font_small, 0.4, CONFIG["COLOR_PERSON"], 1)
    
    def _render_stats(self, panel):
        """Render bottom stats panel"""
        panel[:] = (10, 10, 20)
        
        cv2.line(panel, (0, 100), (1280, 100), (40, 40, 50), 1)
        
        stats = [
            ("FRAMES", str(state.frame_count), 50),
            ("POSITION", f"[{state.camera_pos[0]:.2f}, {state.camera_pos[2]:.2f}]", 200),
            ("STATIC", str(len(state.static_objects)), 400),
            ("DYNAMIC", str(len(state.dynamic_objects)), 520),
            ("FPS", f"{state.fps:.1f}", 650),
            ("GROUND", f"{state.ground_threshold:.2f}", 780),
        ]
        
        for label, value, x in stats:
            cv2.putText(panel, label, (x, 35),
                       self.font_small, 0.5, (100, 100, 100), 1)
            cv2.putText(panel, value, (x, 65),
                       self.font, 0.7, (255, 255, 255), 1)
        
        # Controls
        controls = [
            ("Q", "Quit", 900),
            ("S", "Save", 970),
            ("D", "Depth", 1040),
            ("M", "Mask", 1110),
            ("+/-", "Ground", 1180),
            ("R", "Reset", 1260),
        ]
        
        for key, action, x in controls:
            cv2.putText(panel, f"[{key}] {action}", (x, 35),
                       self.font_small, 0.45, (150, 150, 150), 1)

# ==================== Main SLAM System ====================
class NavigationSLAM:
    def __init__(self):
        self.depth_estimator = DepthEstimator()
        self.object_detector = ObjectDetector()
        self.occupancy_mapper = OccupancyMapper()
        self.renderer = BEVRenderer()
        
        self.prev_frame = None
        self.cumulative_yaw = 0.0
    
    def process_frame(self, frame):
        """Process frame and update navigation map"""
        state.frame_count += 1
        
        # Update FPS
        now = time.time()
        if now - state.last_fps_time > 1.0:
            state.fps = state.frame_count / (now - state.last_fps_time)
            state.frame_count = 0
            state.last_fps_time = now
        
        # Estimate depth
        depth_map = self.depth_estimator.estimate(frame)
        
        # Detect objects
        static_objs, dynamic_objs = self.object_detector.detect(frame, depth_map)
        
        with state.lock:
            state.static_objects = static_objs
            state.dynamic_objects = dynamic_objs
        
        # Update occupancy grid
        self.occupancy_mapper.update(
            depth_map,
            state.camera_pos,
            self.cumulative_yaw,
            static_objs,
            dynamic_objs
        )
        
        with state.lock:
            state.occupancy_grid = self.occupancy_mapper.occupancy_grid.copy()
        
        # Update camera pose (simulated - use visual odometry in production)
        self._update_pose(frame)
        
        self.prev_frame = frame.copy()
        
        return frame, depth_map
    
    def _update_pose(self, frame):
        """Update camera position"""
        # Simulate small forward movement
        delta = 0.05
        
        state.camera_pos[0] += delta * np.sin(self.cumulative_yaw)
        state.camera_pos[2] += delta * np.cos(self.cumulative_yaw)
        
        state.trajectory.append((state.camera_pos[0], state.camera_pos[2]))
    
    def render(self, frame, depth_map):
        return self.renderer.render(frame, depth_map)
    
    def save_data(self):
        """Save map data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save occupancy grid
        vis = self.occupancy_mapper.get_visualization()
        cv2.imwrite(f"occupancy_{timestamp}.png", vis)
        print(f"💾 Saved: occupancy_{timestamp}.png")
        
        # Save trajectory
        with open(f"trajectory_{timestamp}.txt", 'w') as f:
            f.write("# x z\n")
            for pos in state.trajectory:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f}\n")
        print(f"💾 Saved: trajectory_{timestamp}.txt")
        
        # Save objects
        with open(f"objects_{timestamp}.json", 'w') as f:
            json.dump({
                'static': state.static_objects,
                'dynamic': state.dynamic_objects
            }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print(f"💾 Saved: objects_{timestamp}.json")

# ==================== Main ====================
def main():
    print("\n" + "=" * 70)
    print("        NAVIGATION-GRADE BEV MAPPING SYSTEM")
    print("        Static Environment + Dynamic Object Filtering")
    print("=" * 70)
    print("\n📋 INSTRUCTIONS:")
    print("   • Point camera at environment (walls, furniture)")
    print("   • People will be detected but NOT mapped (dynamic)")
    print("   • Static objects will appear on occupancy grid")
    print("   • Move camera slowly to build the map")
    print("\n🎮 CONTROLS:")
    print("   Q - Quit")
    print("   S - Save map")
    print("   D - Toggle depth visualization")
    print("   M - Toggle dynamic masking")
    print("   +/- - Adjust ground threshold")
    print("   R - Reset map")
    print("\n🚀 Starting...\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    slam = NavigationSLAM()
    print("✅ Ready! Point camera at environment and move slowly...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed, depth_map = slam.process_frame(frame)
        dashboard = slam.render(processed, depth_map)
        
        cv2.imshow("Navigation BEV Mapping", dashboard)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            slam.save_data()
        elif key == ord('d'):
            state.show_depth = not state.show_depth
        elif key == ord('m'):
            state.dynamic_masking = not state.dynamic_masking
        elif key == ord('r'):
            with state.lock:
                state.occupancy_grid.fill(-1)
                state.static_objects.clear()
                state.dynamic_objects.clear()
                state.camera_pos = np.array([0.0, 0.0, 0.0])
                state.trajectory.clear()
                state.trajectory.append((0.0, 0.0))
            print("\n🔄 Map reset!")
        elif key == ord('+') or key == ord('='):
            state.ground_threshold = min(1.0, state.ground_threshold + 0.05)
            print(f"\n📏 Ground threshold: {state.ground_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            state.ground_threshold = max(0.1, state.ground_threshold - 0.05)
            print(f"\n📏 Ground threshold: {state.ground_threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session complete!\n")

if __name__ == "__main__":
    main()
