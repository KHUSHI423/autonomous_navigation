"""
3D Mapping from Video - Monocular SLAM (OpenCV Only)
Uses webcam to create 3D point cloud of environment

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
from collections import deque

# ==================== Configuration ====================
FEATURE_DETECTOR = "ORB"  # ORB, SIFT, or AKAZE
MAX_FEATURES = 500
RANSAC_THRESHOLD = 2.0

# ==================== Global Variables ====================
point_cloud = []
point_colors = []
camera_positions = []
frame_count = 0

# ==================== Feature Detection ====================
def detect_features(image):
    """Detect keypoints and descriptors"""
    if FEATURE_DETECTOR == "ORB":
        detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
        kp, des = detector.detectAndCompute(image, None)
    elif FEATURE_DETECTOR == "SIFT":
        detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        kp, des = detector.detectAndCompute(image, None)
    else:
        detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
        kp, des = detector.detectAndCompute(image, None)
    
    return kp, des

def match_features(des1, des2):
    """Match descriptors between two frames"""
    if des1 is None or des2 is None:
        return []
    
    if FEATURE_DETECTOR == "SIFT":
        matcher = cv2.DescriptorMatcher_create("FlannBased")
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        return good_matches
    else:
        matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        matches = matcher.match(des1, des2)
        return matches

# ==================== Camera Pose Estimation ====================
def estimate_camera_pose(pts1, pts2):
    """Estimate camera motion between two frames"""
    if len(pts1) < 8:
        return None, None
    
    E, mask = cv2.findEssentialMat(
        pts1, pts2, 
        focal=1.0, 
        pp=(0., 0.), 
        method=cv2.RANSAC,
        prob=0.999,
        threshold=RANSAC_THRESHOLD
    )
    
    if E is None:
        return None, None
    
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=1.0, pp=(0., 0.))
    
    return R, t

# ==================== 3D Triangulation ====================
def triangulate_points(pts1, pts2, R, t):
    """Triangulate 3D points from two views"""
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    P2 = np.hstack([R, t]).astype(np.float32)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T

# ==================== 3D Visualization (OpenCV) ====================
class Visualizer3D:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.scale = 50  # Scale factor for visualization
        self.center_x = width // 2
        self.center_y = height // 2
        
    def render(self, points, colors, camera_pos, frame):
        """Render 3D point cloud on 2D image"""
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw grid
        for i in range(-10, 11, 2):
            pt1 = (int(self.center_x + i * self.scale), 0)
            pt2 = (int(self.center_x + i * self.scale), self.height)
            cv2.line(vis, pt1, pt2, (30, 30, 30), 1)
        
        # Draw points
        for i, pt in enumerate(points):
            x = int(self.center_x + pt[0] * self.scale)
            y = int(self.center_y - pt[2] * self.scale)  # Y is up
            
            if 0 <= x < self.width and 0 <= y < self.height:
                color = (255, 100, 100) if colors is None else tuple(int(c * 255) for c in colors[i % len(colors)])
                cv2.circle(vis, (x, y), 3, color, -1)
        
        # Draw camera position
        if camera_pos is not None:
            cam_x = int(self.center_x + camera_pos[0] * self.scale)
            cam_y = int(self.center_y - camera_pos[2] * self.scale)
            cv2.circle(vis, (cam_x, cam_y), 8, (0, 255, 0), -1)
            cv2.putText(vis, "CAM", (cam_x - 15, cam_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Stats
        cv2.putText(vis, f"Points: {len(points)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Cam Pos: [{camera_pos[0]:.2f}, {camera_pos[2]:.2f}]", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Side-by-side display
        combined = np.hstack([frame, vis])
        cv2.putText(combined, "Video Feed", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "3D Map (Top-Down View)", (frame.shape[1] + 10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("3D Mapping", combined)

# ==================== Main Processing ====================
def process_video():
    """Main video processing loop"""
    global point_cloud, point_colors, camera_positions, frame_count
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    print("🎬 Camera opened. Press 'q' to quit, 's' to save point cloud")
    print("📡 Processing video...")
    print("\n💡 Tips:")
    print("   - Move camera slowly side to side")
    print("   - Point at textured surfaces (not blank walls)")
    print("   - Good lighting helps!\n")
    
    prev_frame = None
    prev_kp = None
    prev_des = None
    
    vis3d = Visualizer3D()
    
    current_pos = np.array([0.0, 0.0, 0.0])
    cumulative_R = np.eye(3)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp, des = detect_features(gray)
        
        if prev_kp is not None and prev_des is not None:
            matches = match_features(prev_des, des)
            
            if len(matches) > 20:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                
                R, t = estimate_camera_pose(pts1, pts2)
                
                if R is not None and t is not None:
                    cumulative_R = R @ cumulative_R
                    current_pos += (cumulative_R @ t).flatten() * 0.1
                    
                    points_3d = triangulate_points(pts1, pts2, R, t)
                    
                    for pt in points_3d:
                        if np.linalg.norm(pt) < 50:
                            point_cloud.append(pt)
                            
                            x, y = int(pts2[0][0]), int(pts2[0][1])
                            if 0 <= y < 480 and 0 <= x < 640:
                                color = gray[y, x] / 255.0
                                point_colors.append([color, color, color])
                    
                    camera_positions.append(current_pos.copy())
        
        # Display
        display_frame = frame.copy()
        if kp is not None:
            cv2.drawKeypoints(display_frame, kp[:50], display_frame, 
                            color=(0, 255, 0), flags=0)
        
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Points: {len(point_cloud)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' quit, 's' save", (10, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update 3D visualization
        vis3d.render(
            np.array(point_cloud[::5]),
            np.array(point_colors[::5]),
            current_pos,
            display_frame
        )
        
        prev_frame = gray.copy()
        prev_kp = kp
        prev_des = des
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_point_cloud()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Processing complete!")
    print(f"   Total frames: {frame_count}")
    print(f"   3D points: {len(point_cloud)}")

def save_point_cloud():
    """Save point cloud to file"""
    if len(point_cloud) == 0:
        print("⚠️  No points to save!")
        return
    
    filename = f"pointcloud_{frame_count}.txt"
    with open(filename, 'w') as f:
        f.write("# x y z r g b\n")
        for i, pt in enumerate(point_cloud):
            color = point_colors[i] if i < len(point_colors) else [0.5, 0.5, 0.5]
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
    
    print(f"💾 Saved: {filename}")

# ==================== Main ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎥 3D Mapping from Video - Monocular SLAM")
    print("=" * 60)
    print("\n📋 Instructions:")
    print("   1. Point camera at scene with textures")
    print("   2. Move camera slowly to capture different angles")
    print("   3. Avoid blank walls and mirrors")
    print("   4. Press 'q' to quit, 's' to save point cloud")
    print("\n🚀 Starting...\n")
    
    process_video()
