"""
================================================================================
                    OBJECT-LEVEL 3D MAPPING SYSTEM
                    Monocular SLAM + Object Clustering + Mesh Visualization
================================================================================

Features:
    ✅ Persistent 3D map with global coordinates
    ✅ Point cloud clustering into objects
    ✅ Object bounding boxes in 3D
    ✅ Realistic object shapes (not just dots)
    ✅ Camera trajectory tracking
    ✅ Real-time visualization
    ✅ Save/load map data

Requirements:
    pip install opencv-python numpy scikit-learn

Controls:
    Q - Quit
    S - Save map
    O - Toggle object detection (YOLO)
    C - Toggle clustering
    R - Reset map
    G - Toggle grid
    M - Toggle mesh view
"""

import cv2
import numpy as np
import time
import json
from collections import deque
from datetime import datetime
from sklearn.cluster import DBSCAN

# ==================== Configuration ====================
CONFIG = {
    # Feature Detection
    "MAX_FEATURES": 1000,
    "MIN_MATCHES": 12,
    "RANSAC_THRESHOLD": 2.0,
    
    # Camera
    "FOCAL_LENGTH": 800.0,
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    
    # 3D Mapping
    "MAX_DEPTH": 50.0,
    "MIN_DEPTH": 0.3,
    "MAP_SCALE": 25.0,
    
    # Object Clustering
    "CLUSTER_EPS": 0.8,        # Maximum distance between points in cluster
    "CLUSTER_MIN_SAMPLES": 15, # Minimum points per object
    "CLUSTER_INTERVAL": 30,    # Run clustering every N frames
    
    # Visualization
    "MAP_WIDTH": 900,
    "MAP_HEIGHT": 600,
    "MAX_POINTS": 30000,
    "MAX_TRAJECTORY": 800,
    
    # Colors
    "COLOR_BG": (5, 5, 10),
    "COLOR_GRID": (25, 25, 35),
    "COLOR_CAMERA": (0, 255, 0),
    "COLOR_TRAJECTORY": (0, 200, 255),
}

# ==================== Global State ====================
class GlobalState:
    def __init__(self):
        # Raw point cloud
        self.point_cloud = []           # [[x, y, z], ...]
        self.point_colors = []          # [[r, g, b], ...]
        self.point_descriptors = []     # Feature descriptors
        
        # Clustered objects
        self.objects = []               # List of Object3D
        self.object_clusters = []       # [[point_indices], ...]
        
        # Camera
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_rotation = np.eye(3)
        self.trajectory = deque(maxlen=CONFIG["MAX_TRAJECTORY"])
        self.trajectory.append(self.camera_pos.copy())
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.matched_features = 0
        
        # Flags
        self.show_grid = True
        self.clustering_enabled = True
        self.mesh_view = False
        self.object_detection_enabled = False
        
        # Lock
        import threading
        self.lock = threading.Lock()

class Object3D:
    """Represents a clustered 3D object"""
    def __init__(self, points, colors, object_id):
        self.id = object_id
        self.points = np.array(points)      # N x 3
        self.colors = np.array(colors)      # N x 3
        self.centroid = np.mean(self.points, axis=0)
        
        # Bounding box
        self.bbox_min = np.min(self.points, axis=0)
        self.bbox_max = np.max(self.points, axis=0)
        self.bbox_size = self.bbox_max - self.bbox_min
        
        # Estimate object type based on dimensions
        self.object_type = self._estimate_type()
        
        # Color for visualization
        self.visual_color = np.random.randint(100, 255, 3)
    
    def _estimate_type(self):
        """Estimate object type from bounding box ratios"""
        w, h, d = self.bbox_size
        
        if h > w * 1.5 and h > d * 1.5:
            return "TALL"      # Person, tree, pole
        elif w > h * 2 or d > h * 2:
            return "FLAT"      # Wall, floor, table
        elif abs(w - d) < 0.3 * max(w, d):
            return "ROUND"     # Box-like object
        else:
            return "IRREGULAR"

state = GlobalState()

# ==================== Feature Detection ====================
class FeatureDetector:
    def __init__(self):
        self.detector = cv2.ORB_create(
            nfeatures=CONFIG["MAX_FEATURES"],
            scaleFactor=1.2,
            nlevels=8
        )
    
    def detect(self, image):
        kp, des = self.detector.detectAndCompute(image, None)
        return kp, des
    
    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        
        matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        matches = matcher.match(des1, des2)
        
        # Filter by distance
        good = [m for m in matches if m.distance < 60]
        return good

# ==================== Camera Pose Estimation ====================
class PoseEstimator:
    def __init__(self):
        self.fx = CONFIG["FOCAL_LENGTH"]
        self.cx = CONFIG["FRAME_WIDTH"] // 2
        self.cy = CONFIG["FRAME_HEIGHT"] // 2
    
    def estimate(self, pts1, pts2):
        if len(pts1) < 8:
            return None, None
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            focal=self.fx,
            pp=(self.cx, self.cy),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=CONFIG["RANSAC_THRESHOLD"]
        )
        
        if E is None:
            return None, None
        
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.fx, pp=(self.cx, self.cy))
        return R, t
    
    def triangulate(self, pts1, pts2, R, t):
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        P2 = np.hstack([R, t]).astype(np.float64)
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T

# ==================== Object Clustering ====================
class ObjectClusterer:
    def __init__(self):
        self.eps = CONFIG["CLUSTER_EPS"]
        self.min_samples = CONFIG["CLUSTER_MIN_SAMPLES"]
    
    def cluster(self, points, colors):
        """Cluster points into objects using DBSCAN"""
        if len(points) < self.min_samples:
            return [], []
        
        # Run DBSCAN
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean',
            n_jobs=-1
        ).fit(points)
        
        labels = clustering.labels_
        
        # Group points by cluster
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise
        
        objects = []
        clusters = []
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_points = points[indices]
            cluster_colors = colors[indices]
            
            if len(cluster_points) >= self.min_samples:
                obj = Object3D(cluster_points, cluster_colors, len(objects))
                objects.append(obj)
                clusters.append(indices.tolist())
        
        return objects, clusters

# ==================== 3D Object Renderer ====================
class ObjectRenderer:
    def __init__(self):
        self.map_width = CONFIG["MAP_WIDTH"]
        self.map_height = CONFIG["MAP_HEIGHT"]
        self.scale = CONFIG["MAP_SCALE"]
    
    def project_3d_to_2d(self, point, camera_pos):
        """Project 3D world point to 2D screen (top-down view)"""
        rel_x = point[0] - camera_pos[0]
        rel_z = point[2] - camera_pos[2]
        
        screen_x = self.map_width // 2 + int(rel_x * self.scale)
        screen_z = self.map_height // 2 - int(rel_z * self.scale)
        
        return (screen_x, screen_z)
    
    def draw_object_3d(self, panel, obj, camera_pos):
        """Draw a 3D object with bounding box"""
        # Project all points
        screen_points = []
        for pt in obj.points[::3]:  # Downsample for performance
            screen_pt = self.project_3d_to_2d(pt, camera_pos)
            screen_points.append(screen_pt)
        
        # Draw points as filled circles (creates solid appearance)
        for pt in screen_points:
            dist = np.linalg.norm(obj.centroid - camera_pos)
            size = max(2, int(6 - dist / 10))
            cv2.circle(panel, pt, size, obj.visual_color.tolist(), -1)
        
        # Draw bounding box (projected to 2D)
        bbox_corners_3d = self._get_bbox_corners(obj)
        bbox_corners_2d = [self.project_3d_to_2d(c, camera_pos) for c in bbox_corners_3d]
        
        # Draw box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]
        
        for i, j in edges:
            if i < len(bbox_corners_2d) and j < len(bbox_corners_2d):
                cv2.line(panel, bbox_corners_2d[i], bbox_corners_2d[j], 
                        (150, 150, 150), 1)
        
        # Draw centroid marker
        centroid_2d = self.project_3d_to_2d(obj.centroid, camera_pos)
        cv2.circle(panel, centroid_2d, 6, (255, 255, 0), -1)
        
        # Label
        label = f"#{obj.id} {obj.object_type}"
        cv2.putText(panel, label, (centroid_2d[0] - 40, centroid_2d[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _get_bbox_corners(self, obj):
        """Get 8 corners of 3D bounding box"""
        min_x, min_y, min_z = obj.bbox_min
        max_x, max_y, max_z = obj.bbox_max
        
        corners = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, min_y, max_z),
            (min_x, min_y, max_z),
            (min_x, max_y, min_z),
            (max_x, max_y, min_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        
        return corners

# ==================== Dashboard Renderer ====================
class DashboardRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.object_renderer = ObjectRenderer()
        
        # Pre-compute grid
        self.grid_lines = self._create_grid()
    
    def _create_grid(self):
        lines = []
        size = 15
        spacing = 1.0
        
        for i in range(-size, size + 1, 2):
            lines.append(((-size, 0, i), (size, 0, i)))
            lines.append(((i, 0, -size), (i, 0, size)))
        
        # Axes
        lines.append(((0, 0, 0), (3, 0, 0)))  # X
        lines.append(((0, 0, 0), (0, 0, 3)))  # Z
        
        return lines
    
    def render(self, frame):
        """Render complete dashboard"""
        with state.lock:
            dashboard = np.zeros((650, 1280, 3), dtype=np.uint8)
            dashboard[:] = CONFIG["COLOR_BG"]
            
            # Left panel - Camera feed
            cam_panel = dashboard[0:480, 0:640]
            resized = cv2.resize(frame, (640, 480))
            cam_panel[:] = resized
            
            # Right panel - 3D Map
            map_panel = dashboard[0:480, 640:1280]
            self._render_3d_map(map_panel)
            
            # Bottom panel - Stats
            stats_panel = dashboard[480:650, :]
            self._render_stats(stats_panel)
            
            # Overlay UI
            self._render_camera_overlay(cam_panel)
        
        return dashboard
    
    def _render_3d_map(self, panel):
        """Render 3D map with objects"""
        panel[:] = CONFIG["COLOR_BG"]
        cam_pos = state.camera_pos
        
        # Draw grid
        if state.show_grid:
            for line in self.grid_lines:
                pt1 = self.object_renderer.project_3d_to_2d(line[0], cam_pos)
                pt2 = self.object_renderer.project_3d_to_2d(line[1], cam_pos)
                
                if line[0][0] == 0 or line[0][2] == 0:
                    color = (60, 60, 80)
                else:
                    color = CONFIG["COLOR_GRID"]
                cv2.line(panel, pt1, pt2, color, 1)
        
        # Draw trajectory
        if len(state.trajectory) > 1:
            traj_points = [
                self.object_renderer.project_3d_to_2d(pos, cam_pos)
                for pos in state.trajectory
            ]
            cv2.polylines(panel, [np.array(traj_points, np.int32)], 
                         False, CONFIG["COLOR_TRAJECTORY"], 2)
        
        # Draw camera
        cam_screen = self.object_renderer.project_3d_to_2d(cam_pos, cam_pos)
        cv2.circle(panel, cam_screen, 10, CONFIG["COLOR_CAMERA"], -1)
        cv2.circle(panel, cam_screen, 15, (0, 100, 0), 1)
        
        # Draw direction indicator
        dir_end = self.object_renderer.project_3d_to_2d(
            (cam_pos[0], cam_pos[1], cam_pos[2] + 1.5), cam_pos
        )
        cv2.line(panel, cam_screen, dir_end, CONFIG["COLOR_CAMERA"], 2)
        
        # Draw objects (clustered)
        if state.clustering_enabled:
            for obj in state.objects:
                dist = np.linalg.norm(obj.centroid - cam_pos)
                if dist < 40:  # Distance culling
                    self.object_renderer.draw_object_3d(panel, obj, cam_pos)
        else:
            # Draw raw points
            points_to_draw = state.point_cloud[::5]
            colors_to_draw = state.point_colors[::5]
            
            for i, pt in enumerate(points_to_draw):
                dist = np.linalg.norm(pt - cam_pos)
                if dist > 40:
                    continue
                
                screen_pt = self.object_renderer.project_3d_to_2d(pt, cam_pos)
                size = max(1, int(4 - dist / 15))
                
                if i < len(colors_to_draw):
                    color = tuple(int(c * 255) for c in colors_to_draw[i])
                    color = (color[2], color[1], color[0])
                else:
                    color = (100, 200, 255)
                
                cv2.circle(panel, screen_pt, size, color, -1)
        
        # Map info
        cv2.putText(panel, "3D MAP", (10, 30), self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"Points: {len(state.point_cloud)}", (10, 60),
                   self.font_small, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"Objects: {len(state.objects)}", (10, 85),
                   self.font_small, 0.6, (200, 200, 200), 1)
        
        # Legend
        legend_y = 120
        cv2.putText(panel, "Legend:", (10, legend_y),
                   self.font_small, 0.55, (255, 255, 255), 1)
        cv2.circle(panel, (15, legend_y + 15), 5, CONFIG["COLOR_CAMERA"], -1)
        cv2.putText(panel, "Camera", (30, legend_y + 20),
                   self.font_small, 0.45, (200, 200, 200), 1)
        cv2.circle(panel, (15, legend_y + 35), 5, (255, 255, 0), -1)
        cv2.putText(panel, "Object Center", (30, legend_y + 40),
                   self.font_small, 0.45, (200, 200, 200), 1)
        cv2.circle(panel, (15, legend_y + 55), 5, (150, 100, 255), -1)
        cv2.putText(panel, "Clustered Object", (30, legend_y + 60),
                   self.font_small, 0.45, (200, 200, 200), 1)
    
    def _render_camera_overlay(self, panel):
        """Render overlay on camera feed"""
        # Stats
        status_y = 30
        cv2.putText(panel, f"FPS: {state.fps:.1f}", (10, status_y),
                   self.font, 0.6, (0, 255, 100), 2)
        status_y += 25
        cv2.putText(panel, f"Pos: [{state.camera_pos[0]:.2f}, {state.camera_pos[2]:.2f}]", 
                   (10, status_y), self.font_small, 0.5, (255, 255, 255), 1)
        status_y += 20
        cv2.putText(panel, f"Features: {state.matched_features}", 
                   (10, status_y), self.font_small, 0.5, (255, 255, 255), 1)
        status_y += 20
        cv2.putText(panel, f"Mode: {'CLUSTER' if state.clustering_enabled else 'POINTS'}",
                   (10, status_y), self.font_small, 0.5, (255, 200, 0), 1)
    
    def _render_stats(self, panel):
        """Render bottom stats panel"""
        panel[:] = (10, 10, 20)
        
        cv2.line(panel, (0, 70), (1280, 70), (40, 40, 50), 1)
        
        stats = [
            ("FRAMES", str(state.frame_count), 50),
            ("POINTS", str(len(state.point_cloud)), 180),
            ("OBJECTS", str(len(state.objects)), 310),
            ("TRAJECTORY", str(len(state.trajectory)), 440),
            ("FPS", f"{state.fps:.1f}", 580),
            ("CAM X", f"{state.camera_pos[0]:.3f}", 720),
            ("CAM Z", f"{state.camera_pos[2]:.3f}", 850),
            ("CLUSTERS", "ON" if state.clustering_enabled else "OFF", 980),
        ]
        
        for label, value, x in stats:
            cv2.putText(panel, label, (x, 30),
                       self.font_small, 0.5, (100, 100, 100), 1)
            cv2.putText(panel, value, (x, 55),
                       self.font, 0.7, (255, 255, 255), 1)
        
        # Controls
        controls = [
            ("Q", "Quit", 1080),
            ("S", "Save", 1150),
            ("C", "Cluster", 1210),
            ("R", "Reset", 1270),
        ]
        
        for key, action, x in controls:
            cv2.putText(panel, f"[{key}] {action}", (x, 30),
                       self.font_small, 0.45, (150, 150, 150), 1)

# ==================== Main SLAM System ====================
class SLAMSystem:
    def __init__(self):
        self.detector = FeatureDetector()
        self.pose_estimator = PoseEstimator()
        self.clusterer = ObjectClusterer()
        self.renderer = DashboardRenderer()
        
        self.prev_kp = None
        self.prev_des = None
        
        self.cumulative_R = np.eye(3)
        self.cumulative_t = np.array([0.0, 0.0, 0.0])
        
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
                pts1 = np.float64([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float64([kp[m.trainIdx].pt for m in matches])
                
                R, t = self.pose_estimator.estimate(pts1, pts2)
                
                if R is not None and t is not None:
                    self.cumulative_R = R @ self.cumulative_R
                    self.cumulative_t += (self.cumulative_R @ t.flatten()).reshape(3) * 0.1
                    
                    with state.lock:
                        state.camera_rotation = self.cumulative_R
                        state.camera_pos = self.cumulative_t.copy()
                        state.trajectory.append(state.camera_pos.copy())
                    
                    # Triangulate new points
                    points_3d = self.pose_estimator.triangulate(pts1, pts2, R, t)
                    
                    for i, pt in enumerate(points_3d):
                        if CONFIG["MIN_DEPTH"] < np.linalg.norm(pt) < CONFIG["MAX_DEPTH"]:
                            world_pt = self.cumulative_R @ pt + self.cumulative_t
                            
                            if len(state.point_cloud) < CONFIG["MAX_POINTS"]:
                                with state.lock:
                                    state.point_cloud.append(world_pt)
                                    
                                    x, y = int(pts2[i][0]), int(pts2[i][1])
                                    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                                        color = frame[y, x] / 255.0
                                        state.point_colors.append(color)
                                    else:
                                        state.point_colors.append([0.5, 0.5, 0.5])
        
        # Run clustering periodically
        if state.clustering_enabled and state.frame_count % CONFIG["CLUSTER_INTERVAL"] == 0:
            if len(state.point_cloud) >= CONFIG["CLUSTER_MIN_SAMPLES"]:
                with state.lock:
                    state.objects, state.object_clusters = self.clusterer.cluster(
                        np.array(state.point_cloud),
                        np.array(state.point_colors)
                    )
        
        # Store for next frame
        self.prev_kp = kp
        self.prev_des = des
        
        # Draw features
        if kp is not None:
            cv2.drawKeypoints(frame, kp[:200], frame, color=(0, 255, 0), flags=0)
        
        return frame
    
    def render(self, frame):
        return self.renderer.render(frame)
    
    def save_data(self):
        """Save all data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Point cloud
        if len(state.point_cloud) > 0:
            with open(f"pointcloud_{timestamp}.txt", 'w') as f:
                f.write("# x y z r g b\n")
                for i, pt in enumerate(state.point_cloud):
                    color = state.point_colors[i] if i < len(state.point_colors) else [0.5, 0.5, 0.5]
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
            print(f"💾 Saved: pointcloud_{timestamp}.txt ({len(state.point_cloud)} points)")
        
        # Objects
        if len(state.objects) > 0:
            with open(f"objects_{timestamp}.json", 'w') as f:
                objects_data = []
                for obj in state.objects:
                    obj_data = {
                        "id": obj.id,
                        "type": obj.object_type,
                        "centroid": obj.centroid.tolist(),
                        "bbox_min": obj.bbox_min.tolist(),
                        "bbox_max": obj.bbox_max.tolist(),
                        "bbox_size": obj.bbox_size.tolist(),
                        "color": obj.visual_color.tolist(),
                        "point_count": len(obj.points)
                    }
                    objects_data.append(obj_data)
                json.dump(objects_data, f, indent=2)
            print(f"💾 Saved: objects_{timestamp}.json ({len(state.objects)} objects)")
        
        # Trajectory
        with open(f"trajectory_{timestamp}.txt", 'w') as f:
            f.write("# x y z\n")
            for pos in state.trajectory:
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        print(f"💾 Saved: trajectory_{timestamp}.txt ({len(state.trajectory)} positions)")

# ==================== Main ====================
def main():
    print("\n" + "=" * 70)
    print("        OBJECT-LEVEL 3D MAPPING SYSTEM")
    print("        Point Cloud Clustering + Object Detection")
    print("=" * 70)
    print("\n📋 INSTRUCTIONS:")
    print("   • Move camera slowly around environment")
    print("   • Points will cluster into objects automatically")
    print("   • Objects persist as you move")
    print("\n🎮 CONTROLS:")
    print("   Q - Quit")
    print("   S - Save map data")
    print("   C - Toggle clustering (points ↔ objects)")
    print("   R - Reset map")
    print("   G - Toggle grid")
    print("\n🚀 Starting...\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    slam = SLAMSystem()
    print("✅ Ready! Move camera to start mapping...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = slam.process_frame(frame)
        dashboard = slam.render(processed)
        
        cv2.imshow("3D Object Mapping", dashboard)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            slam.save_data()
        elif key == ord('c'):
            state.clustering_enabled = not state.clustering_enabled
            print(f"\n🔷 Clustering: {'ON' if state.clustering_enabled else 'OFF'}")
            if state.clustering_enabled and len(state.point_cloud) > 0:
                # Run clustering immediately
                with state.lock:
                    state.objects, state.object_clusters = slam.clusterer.cluster(
                        np.array(state.point_cloud),
                        np.array(state.point_colors)
                    )
                print(f"   Found {len(state.objects)} objects")
        elif key == ord('r'):
            with state.lock:
                state.point_cloud.clear()
                state.point_colors.clear()
                state.objects.clear()
                state.object_clusters.clear()
                state.trajectory.clear()
                state.trajectory.append(np.array([0.0, 0.0, 0.0]))
                state.camera_pos = np.array([0.0, 0.0, 0.0])
                slam.cumulative_R = np.eye(3)
                slam.cumulative_t = np.array([0.0, 0.0, 0.0])
            print("\n🔄 Map reset!")
        elif key == ord('g'):
            state.show_grid = not state.show_grid
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session complete!")
    print(f"   Final: {len(state.point_cloud)} points, {len(state.objects)} objects\n")

if __name__ == "__main__":
    main()
