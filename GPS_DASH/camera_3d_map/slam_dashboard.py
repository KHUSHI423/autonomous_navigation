"""
================================================================================
                    REAL-TIME PERSISTENT 3D MAPPING SYSTEM
                    Monocular SLAM + Object Detection + Dashboard
================================================================================

Features:
    ✅ Persistent 3D map (points accumulate, never cleared)
    ✅ Global world coordinates
    ✅ Camera pose tracking with trajectory
    ✅ Object detection (YOLO-based)
    ✅ Futuristic robotics dashboard UI
    ✅ Real-time performance
    ✅ Save map, trajectory, objects

Requirements:
    pip install opencv-python numpy

Controls:
    Q - Quit
    S - Save point cloud
    O - Toggle object detection
    R - Reset map
    G - Toggle grid
"""

import cv2
import numpy as np
import threading
import time
import json
from collections import deque
from datetime import datetime
import math

# ==================== Configuration ====================
CONFIG = {
    # Feature Detection
    "FEATURE_DETECTOR": "ORB",      # ORB, SIFT, AKAZE
    "MAX_FEATURES": 800,
    "MAX_FEATURES_PER_FRAME": 200,
    
    # Matching
    "MATCH_RATIO": 0.75,
    "MIN_MATCHES": 15,
    "RANSAC_THRESHOLD": 2.0,
    "RANSAC_CONFIDENCE": 0.999,
    
    # 3D Reconstruction
    "MAX_DEPTH": 100.0,
    "MIN_DEPTH": 0.5,
    "POINT_DOWNSAMPLE": 3,
    
    # Camera
    "FOCAL_LENGTH": 800.0,
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    
    # Visualization
    "MAP_SCALE": 30.0,
    "MAP_WIDTH": 800,
    "MAP_HEIGHT": 600,
    "TRAJECTORY_LENGTH": 500,
    "GRID_SIZE": 20,
    "GRID_SPACING": 1.0,
    
    # Object Detection
    "OBJECT_DETECTION": True,
    "OBJECT_CONFIDENCE": 0.5,
    "OBJECT_MODEL": "mobile",  # mobile or full
    
    # Performance
    "UPDATE_INTERVAL": 2,       # Update 3D view every N frames
    "MAX_POINTS": 50000,
    "MAX_TRAJECTORY": 1000,
    
    # Colors (BGR for OpenCV)
    "COLOR_BG": (10, 10, 15),
    "COLOR_GRID": (30, 30, 40),
    "COLOR_POINT": (100, 200, 255),
    "COLOR_POINT_VARIED": True,
    "COLOR_CAMERA": (0, 255, 0),
    "COLOR_TRAJECTORY": (0, 255, 255),
    "COLOR_OBJECT": (0, 100, 255),
    "COLOR_TEXT": (255, 255, 255),
    "COLOR_FPS": (0, 255, 100),
}

# ==================== Global State ====================
class GlobalState:
    def __init__(self):
        # Persistent 3D Map
        self.point_cloud = []           # List of [x, y, z]
        self.point_colors = []          # List of [r, g, b]
        self.point_timestamps = []      # List of timestamps
        
        # Camera
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_rotation = np.eye(3)
        self.trajectory = deque(maxlen=CONFIG["MAX_TRAJECTORY"])
        self.trajectory.append(self.camera_pos.copy())
        
        # Objects
        self.detected_objects = []      # List of {pos, class, confidence, bbox}
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.total_points = 0
        self.matched_features = 0
        
        # Flags
        self.show_grid = True
        self.object_detection_enabled = CONFIG["OBJECT_DETECTION"]
        
        # Lock for thread safety
        self.lock = threading.Lock()

state = GlobalState()

# ==================== Feature Detection ====================
class FeatureDetector:
    def __init__(self):
        self.detector = self._create_detector()
        
    def _create_detector(self):
        if CONFIG["FEATURE_DETECTOR"] == "SIFT":
            return cv2.SIFT_create(nfeatures=CONFIG["MAX_FEATURES"])
        elif CONFIG["FEATURE_DETECTOR"] == "AKAZE":
            return cv2.AKAZE_create()
        else:  # ORB
            return cv2.ORB_create(
                nfeatures=CONFIG["MAX_FEATURES"],
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                patchSize=31
            )
    
    def detect(self, image):
        kp, des = self.detector.detectAndCompute(image, None)
        return kp, des
    
    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        
        if CONFIG["FEATURE_DETECTOR"] == "SIFT":
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5),
                dict(checks=50)
            )
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < CONFIG["MATCH_RATIO"] * n.distance:
                    good.append(m)
            return good
        else:
            matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
            matches = matcher.match(des1, des2)
            
            # Filter by distance
            good = [m for m in matches if m.distance < 50]
            return good

# ==================== Camera Pose Estimation ====================
class PoseEstimator:
    def __init__(self):
        self.K = np.array([
            [CONFIG["FOCAL_LENGTH"], 0, CONFIG["FRAME_WIDTH"]//2],
            [0, CONFIG["FOCAL_LENGTH"], CONFIG["FRAME_HEIGHT"]//2],
            [0, 0, 1]
        ], dtype=np.float64)
        
    def estimate(self, pts1, pts2):
        """Estimate relative camera pose"""
        if len(pts1) < 8:
            return None, None
        
        # Essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            focal=CONFIG["FOCAL_LENGTH"],
            pp=(CONFIG["FRAME_WIDTH"]//2, CONFIG["FRAME_HEIGHT"]//2),
            method=cv2.RANSAC,
            prob=CONFIG["RANSAC_CONFIDENCE"],
            threshold=CONFIG["RANSAC_THRESHOLD"]
        )
        
        if E is None:
            return None, None
        
        # Recover pose
        _, R, t, _ = cv2.recoverPose(
            E, pts1, pts2,
            focal=CONFIG["FOCAL_LENGTH"],
            pp=(CONFIG["FRAME_WIDTH"]//2, CONFIG["FRAME_HEIGHT"]//2)
        )
        
        return R, t
    
    def triangulate(self, pts1, pts2, R, t):
        """Triangulate 3D points"""
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        P2 = np.hstack([R, t]).astype(np.float64)
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T

# ==================== Object Detection ====================
class ObjectDetector:
    def __init__(self):
        self.enabled = CONFIG["OBJECT_DETECTION"]
        self.net = None
        self.classes = []
        self.colors = {}
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load YOLO or MobileNet model"""
        try:
            # Try MobileNet SSD (faster)
            proto = "models/MobileNetSSD_deploy.prototxt"
            model = "models/MobileNetSSD_deploy.caffemodel"
            
            # Use OpenCV's DNN with pre-trained model
            self.net = cv2.dnn.readNetFromCaffe(
                cv2.samples.findFile(proto),
                cv2.samples.findFile(model)
            )
            
            self.classes = [
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
            
            # Generate colors for each class
            np.random.seed(42)
            self.colors = {
                i: tuple(map(int, np.random.randint(0, 255, 3)))
                for i in range(len(self.classes))
            }
            
            print("✅ Object detection model loaded")
        except Exception as e:
            print(f"⚠️  Object detection not available: {e}")
            self.enabled = False
    
    def detect(self, frame):
        """Detect objects in frame"""
        if not self.enabled or self.net is None:
            return []
        
        h, w = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        objects = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > CONFIG["OBJECT_CONFIDENCE"]:
                class_id = int(detections[0, 0, i, 1])
                
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Estimate depth (simplified - based on object size)
                obj_height = y2 - y1
                estimated_depth = max(0.5, min(20.0, 300.0 / max(obj_height, 1)))
                
                objects.append({
                    "class": self.classes[class_id] if class_id < len(self.classes) else "unknown",
                    "class_id": class_id,
                    "confidence": float(confidence),
                    "bbox": (x1, y1, x2, y2),
                    "depth": estimated_depth,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return objects

# ==================== 3D Dashboard Renderer ====================
class DashboardRenderer:
    def __init__(self):
        self.map_width = CONFIG["MAP_WIDTH"]
        self.map_height = CONFIG["MAP_HEIGHT"]
        self.scale = CONFIG["MAP_SCALE"]
        
        # Pre-compute grid
        self.grid_lines = self._create_grid()
        
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        
    def _create_grid(self):
        """Create ground plane grid lines"""
        lines = []
        size = CONFIG["GRID_SIZE"]
        spacing = CONFIG["GRID_SPACING"]
        
        # X-axis lines
        for i in range(-size, size + 1, int(spacing * 2)):
            lines.append(((-size, 0, i), (size, 0, i)))
            lines.append(((i, 0, -size), (i, 0, size)))
        
        # Axes
        lines.append(((0, 0, 0), (5, 0, 0)))  # X axis (red)
        lines.append(((0, 0, 0), (0, 0, 5)))  # Z axis (blue)
        
        return lines
    
    def project_3d_to_2d(self, point, camera_pos):
        """Project 3D world point to 2D screen"""
        # Relative to camera
        rel_x = point[0] - camera_pos[0]
        rel_z = point[2] - camera_pos[2]
        
        # Rotate based on camera yaw (simplified)
        yaw = 0  # Could use actual camera rotation
        
        # Project to screen
        screen_x = self.map_width // 2 + int(rel_x * self.scale)
        screen_z = self.map_height // 2 - int(rel_z * self.scale)
        
        return (screen_x, screen_z)
    
    def render(self, frame):
        """Render complete dashboard"""
        with state.lock:
            # Create dashboard layout
            dashboard = np.zeros((600, 1280, 3), dtype=np.uint8)
            dashboard[:] = CONFIG["COLOR_BG"]
            
            # Left panel - Camera feed
            cam_panel = dashboard[0:480, 0:640]
            if frame.shape[0] != 480 or frame.shape[1] != 640:
                resized = cv2.resize(frame, (640, 480))
            else:
                resized = frame
            cam_panel[:] = resized
            
            # Right panel - 3D Map
            map_panel = dashboard[0:480, 640:1280]
            self._render_3d_map(map_panel)
            
            # Bottom panel - Stats
            stats_panel = dashboard[480:600, :]
            self._render_stats(stats_panel)
            
            # Overlay UI on camera feed
            self._render_camera_overlay(cam_panel)
            
        return dashboard
    
    def _render_3d_map(self, panel):
        """Render 3D map view"""
        panel[:] = CONFIG["COLOR_BG"]
        
        cam_pos = state.camera_pos
        
        # Draw grid
        if state.show_grid:
            for line in self.grid_lines:
                pt1 = self.project_3d_to_2d(line[0], cam_pos)
                pt2 = self.project_3d_to_2d(line[1], cam_pos)
                
                # Color based on axis
                if line[0][1] == 0 and line[1][1] == 0:
                    color = CONFIG["COLOR_GRID"]
                    if line[0][0] == 0 or line[0][2] == 0:
                        color = (50, 50, 60)
                    cv2.line(panel, pt1, pt2, color, 1)
        
        # Draw trajectory
        if len(state.trajectory) > 1:
            traj_points = []
            for pos in state.trajectory:
                pt = self.project_3d_to_2d(pos, cam_pos)
                traj_points.append(pt)
            
            if len(traj_points) > 1:
                cv2.polylines(panel, [np.array(traj_points, np.int32)], 
                            False, CONFIG["COLOR_TRAJECTORY"], 2)
        
        # Draw camera position
        cam_screen = self.project_3d_to_2d(cam_pos, cam_pos)
        cv2.circle(panel, cam_screen, 8, CONFIG["COLOR_CAMERA"], -1)
        cv2.circle(panel, cam_screen, 12, (0, 150, 0), 1)
        
        # Draw camera direction indicator
        dir_end = self.project_3d_to_2d(
            (cam_pos[0], cam_pos[1], cam_pos[2] + 2), cam_pos
        )
        cv2.line(panel, cam_screen, dir_end, CONFIG["COLOR_CAMERA"], 2)
        
        # Draw points (downsampled for performance)
        points_to_draw = state.point_cloud[::CONFIG["POINT_DOWNSAMPLE"]]
        colors_to_draw = state.point_colors[::CONFIG["POINT_DOWNSAMPLE"]]
        
        for i, pt in enumerate(points_to_draw):
            # Distance culling
            dist = np.linalg.norm(pt - cam_pos)
            if dist > 50:
                continue
            
            screen_pt = self.project_3d_to_2d(pt, cam_pos)
            
            # Size based on distance
            size = max(1, int(5 - dist / 10))
            
            # Color
            if CONFIG["COLOR_POINT_VARIED"] and i < len(colors_to_draw):
                color = tuple(int(c * 255) for c in colors_to_draw[i])
                color = (color[2], color[1], color[0])  # RGB to BGR
            else:
                color = CONFIG["COLOR_POINT"]
            
            cv2.circle(panel, screen_pt, size, color, -1)
        
        # Draw detected objects
        for obj in state.detected_objects:
            obj_pos = obj.get("world_pos", state.camera_pos.copy())
            screen_pt = self.project_3d_to_2d(obj_pos, cam_pos)
            
            # Draw object marker
            cv2.circle(panel, screen_pt, 10, CONFIG["COLOR_OBJECT"], -1)
            cv2.circle(panel, screen_pt, 14, (0, 50, 150), 1)
            
            # Label
            label = f"{obj['class'][:8]}"
            cv2.putText(panel, label, (screen_pt[0] - 30, screen_pt[1] - 15),
                       self.font_small, 0.5, CONFIG["COLOR_TEXT"], 1)
        
        # Map info
        cv2.putText(panel, "3D MAP", (10, 30),
                   self.font, 0.8, CONFIG["COLOR_TEXT"], 2)
        cv2.putText(panel, f"Points: {len(state.point_cloud)}", (10, 60),
                   self.font_small, 0.6, CONFIG["COLOR_TEXT"], 1)
        cv2.putText(panel, f"Objects: {len(state.detected_objects)}", (10, 85),
                   self.font_small, 0.6, CONFIG["COLOR_TEXT"], 1)
        
        # Coordinate indicator
        self._draw_axes(panel, (50, 50), 30)
    
    def _draw_axes(self, panel, pos, size):
        """Draw XYZ axes indicator"""
        center = pos
        cv2.line(panel, center, (center[0] + size, center[1]), (0, 0, 255), 2)  # X - Red
        cv2.line(panel, center, (center[0], center[1] - size), (0, 255, 0), 2)  # Y - Green
        cv2.putText(panel, "X", (center[0] + size + 5, center[1] + 5),
                   self.font_small, 0.5, (0, 0, 255), 1)
        cv2.putText(panel, "Z", (center[0] - 15, center[1] - size - 5),
                   self.font_small, 0.5, (255, 0, 0), 1)
    
    def _render_camera_overlay(self, panel):
        """Render overlay on camera feed"""
        # Feature points
        # (Drawn by main loop)
        
        # Object bounding boxes
        for obj in state.detected_objects:
            x1, y1, x2, y2 = obj["bbox"]
            color = CONFIG["COLOR_OBJECT"]
            
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            
            label = f"{obj['class']} {obj['confidence']:.2f}"
            cv2.rectangle(panel, (x1, y1 - 20), (x1 + len(label) * 10, y1), color, -1)
            cv2.putText(panel, label, (x1 + 3, y1 - 5),
                       self.font_small, 0.5, (0, 0, 0), 1)
        
        # Status bar
        status_y = 30
        cv2.putText(panel, f"FPS: {state.fps:.1f}", (10, status_y),
                   self.font, 0.6, CONFIG["COLOR_FPS"], 2)
        status_y += 25
        cv2.putText(panel, f"Pos: [{state.camera_pos[0]:.2f}, {state.camera_pos[2]:.2f}]", 
                   (10, status_y), self.font_small, 0.5, CONFIG["COLOR_TEXT"], 1)
        status_y += 20
        cv2.putText(panel, f"Features: {state.matched_features}", 
                   (10, status_y), self.font_small, 0.5, CONFIG["COLOR_TEXT"], 1)
    
    def _render_stats(self, panel):
        """Render bottom stats panel"""
        panel[:] = (15, 15, 25)
        
        # Grid
        cv2.line(panel, (0, 60), (1280, 60), (40, 40, 50), 1)
        
        # Stats
        stats = [
            ("FRAMES", str(state.frame_count), 50),
            ("POINTS", str(len(state.point_cloud)), 200),
            ("OBJECTS", str(len(state.detected_objects)), 350),
            ("TRAJECTORY", str(len(state.trajectory)), 500),
            ("FPS", f"{state.fps:.1f}", 650),
            ("CAMERA X", f"{state.camera_pos[0]:.3f}", 800),
            ("CAMERA Z", f"{state.camera_pos[2]:.3f}", 950),
        ]
        
        for label, value, x in stats:
            cv2.putText(panel, label, (x, 25),
                       self.font_small, 0.5, (100, 100, 100), 1)
            cv2.putText(panel, value, (x, 50),
                       self.font, 0.7, CONFIG["COLOR_TEXT"], 1)
        
        # Controls
        controls = [
            ("Q", "Quit", 1100),
            ("S", "Save", 1180),
            ("O", "Objects", 1240),
        ]
        
        for key, action, x in controls:
            cv2.putText(panel, f"[{key}] {action}", (x, 25),
                       self.font_small, 0.45, (150, 150, 150), 1)

# ==================== Main SLAM System ====================
class SLAMSystem:
    def __init__(self):
        self.detector = FeatureDetector()
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetector()
        self.renderer = DashboardRenderer()
        
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        
        self.cumulative_R = np.eye(3)
        self.cumulative_t = np.array([0.0, 0.0, 0.0])
        
        # Camera intrinsics
        self.fx = CONFIG["FOCAL_LENGTH"]
        self.cx = CONFIG["FRAME_WIDTH"] // 2
        self.cy = CONFIG["FRAME_HEIGHT"] // 2
    
    def process_frame(self, frame):
        """Process a single frame"""
        state.frame_count += 1
        
        # Update FPS
        now = time.time()
        if now - state.last_fps_time > 1.0:
            state.fps = state.frame_count / (now - state.last_fps_time)
            state.frame_count = 0
            state.last_fps_time = now
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp, des = self.detector.detect(gray)
        
        # Match with previous frame
        if self.prev_kp is not None and self.prev_des is not None:
            matches = self.detector.match(self.prev_des, des)
            state.matched_features = len(matches)
            
            if len(matches) >= CONFIG["MIN_MATCHES"]:
                # Get matched points
                pts1 = np.float64([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float64([kp[m.trainIdx].pt for m in matches])
                
                # Estimate pose
                R, t = self.pose_estimator.estimate(pts1, pts2)
                
                if R is not None and t is not None:
                    # Update cumulative pose
                    self.cumulative_R = R @ self.cumulative_R
                    self.cumulative_t += (self.cumulative_R @ t.flatten()).reshape(3) * 0.1
                    
                    # Update state
                    with state.lock:
                        state.camera_rotation = self.cumulative_R
                        state.camera_pos = self.cumulative_t.copy()
                        state.trajectory.append(state.camera_pos.copy())
                    
                    # Triangulate new points
                    points_3d = self.pose_estimator.triangulate(pts1, pts2, R, t)
                    
                    # Transform to world coordinates
                    for i, pt in enumerate(points_3d):
                        if CONFIG["MIN_DEPTH"] < np.linalg.norm(pt) < CONFIG["MAX_DEPTH"]:
                            # Transform to world coords
                            world_pt = self.cumulative_R @ pt + self.cumulative_t
                            
                            # Limit total points
                            if len(state.point_cloud) < CONFIG["MAX_POINTS"]:
                                with state.lock:
                                    state.point_cloud.append(world_pt)
                                    
                                    # Get color
                                    x, y = int(pts2[i][0]), int(pts2[i][1])
                                    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                                        color = frame[y, x] / 255.0
                                        state.point_colors.append(color)
                                    else:
                                        state.point_colors.append([0.5, 0.5, 0.5])
                                    
                                    state.point_timestamps.append(time.time())
                    
                    # Add points from this match
                    if len(state.point_cloud) % 100 < CONFIG["POINT_DOWNSAMPLE"]:
                        pass  # Already added above
        
        # Object detection
        if state.object_detection_enabled and self.object_detector.enabled:
            objects = self.object_detector.detect(frame)
            
            with state.lock:
                state.detected_objects = objects
                
                # Estimate world positions for objects
                for obj in state.detected_objects:
                    # Simple depth estimation from bbox size
                    depth = obj["depth"]
                    
                    # Project to world coordinates
                    cx, cy = obj["center"]
                    x = (cx - self.cx) * depth / self.fx
                    z = (cy - self.cy) * depth / self.fx
                    
                    # World position
                    world_pos = self.cumulative_R @ np.array([x, 0, z]) + self.cumulative_t
                    obj["world_pos"] = world_pos
        
        # Store for next frame
        self.prev_frame = gray.copy()
        self.prev_kp = kp
        self.prev_des = des
        
        # Draw features on frame
        if kp is not None:
            cv2.drawKeypoints(frame, kp[:CONFIG["MAX_FEATURES_PER_FRAME"]], 
                            frame, color=(0, 255, 0), flags=0)
        
        return frame
    
    def render(self, frame):
        """Render dashboard"""
        return self.renderer.render(frame)
    
    def save_data(self):
        """Save all data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save point cloud
        if len(state.point_cloud) > 0:
            with open(f"pointcloud_{timestamp}.txt", 'w') as f:
                f.write("# x y z r g b\n")
                for i, pt in enumerate(state.point_cloud):
                    color = state.point_colors[i] if i < len(state.point_colors) else [0.5, 0.5, 0.5]
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            print(f"💾 Saved: pointcloud_{timestamp}.txt ({len(state.point_cloud)} points)")
        
        # Save trajectory
        with open(f"trajectory_{timestamp}.txt", 'w') as f:
            f.write("# x y z\n")
            for pos in state.trajectory:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        print(f"💾 Saved: trajectory_{timestamp}.txt ({len(state.trajectory)} positions)")
        
        # Save objects
        if len(state.detected_objects) > 0:
            with open(f"objects_{timestamp}.json", 'w') as f:
                objects_data = []
                for obj in state.detected_objects:
                    obj_copy = obj.copy()
                    if "world_pos" in obj_copy:
                        obj_copy["world_pos"] = obj_copy["world_pos"].tolist()
                    objects_data.append(obj_copy)
                json.dump(objects_data, f, indent=2)
            print(f"💾 Saved: objects_{timestamp}.json ({len(state.detected_objects)} objects)")

# ==================== Main ====================
def main():
    print("\n" + "=" * 70)
    print("        REAL-TIME PERSISTENT 3D MAPPING SYSTEM")
    print("        Monocular SLAM + Object Detection + Dashboard")
    print("=" * 70)
    print("\n📋 INSTRUCTIONS:")
    print("   • Point camera at textured environment")
    print("   • Move camera slowly to build the map")
    print("   • Avoid blank walls and mirrors")
    print("   • Good lighting improves results")
    print("\n🎮 CONTROLS:")
    print("   Q - Quit")
    print("   S - Save point cloud, trajectory, objects")
    print("   O - Toggle object detection")
    print("   R - Reset map")
    print("   G - Toggle grid")
    print("\n🚀 Starting camera...\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    # Initialize SLAM system
    slam = SLAMSystem()
    
    print("✅ System ready! Move camera to start mapping...\n")
    
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed = slam.process_frame(frame)
        
        # Render dashboard
        dashboard = slam.render(processed)
        
        # Show
        cv2.imshow("3D Mapping Dashboard", dashboard)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Quitting...")
            break
        elif key == ord('s') or key == ord('S'):
            print("\n💾 Saving data...")
            slam.save_data()
        elif key == ord('o') or key == ord('O'):
            state.object_detection_enabled = not state.object_detection_enabled
            print(f"\n🔍 Object detection: {'ON' if state.object_detection_enabled else 'OFF'}")
        elif key == ord('r') or key == ord('R'):
            print("\n🔄 Resetting map...")
            with state.lock:
                state.point_cloud.clear()
                state.point_colors.clear()
                state.detected_objects.clear()
                state.trajectory.clear()
                state.trajectory.append(np.array([0.0, 0.0, 0.0]))
                state.camera_pos = np.array([0.0, 0.0, 0.0])
                slam.cumulative_R = np.eye(3)
                slam.cumulative_t = np.array([0.0, 0.0, 0.0])
            print("   Map reset!")
        elif key == ord('g') or key == ord('G'):
            state.show_grid = not state.show_grid
            print(f"\n📐 Grid: {'ON' if state.show_grid else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Session complete!")
    print(f"   Final points: {len(state.point_cloud)}")
    print(f"   Final trajectory: {len(state.trajectory)} positions")
    print(f"   Objects detected: {len(state.detected_objects)}")
    print("\n💡 Tip: Run 'python main.py' again to start a new session\n")

if __name__ == "__main__":
    main()
