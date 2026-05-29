"""
Simple BEV Mapping - Saves output to file for viewing
No OpenCV window - runs headless
"""

import cv2
import numpy as np
import time
import torch
from datetime import datetime

CONFIG = {
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    "BEV_SIZE": 512,
    "BEV_RESOLUTION": 0.05,
    "CAMERA_HEIGHT": 1.5,
    "MAX_DEPTH": 20.0,
}

class SimpleBEVMapper:
    def __init__(self):
        self.grid = np.full((CONFIG["BEV_SIZE"], CONFIG["BEV_SIZE"]), -1, dtype=np.float32)
        self.center = CONFIG["BEV_SIZE"] // 2
        self.camera_pos = np.array([0.0, 0.0])
        self.trajectory = [(0.0, 0.0)]
        
        # Load MiDaS
        try:
            self.device = torch.device('cpu')
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = transforms.small_transform
            print("✅ MiDaS loaded")
        except Exception as e:
            print(f"⚠️  MiDaS not available: {e}")
            self.model = None
    
    def estimate_depth(self, image):
        if self.model is None:
            h, w = image.shape[:2]
            return np.tile(np.linspace(0.5, CONFIG["MAX_DEPTH"], h), (w, 1)).T
        
        input_tensor = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image.shape[:2], mode='bicubic', align_corners=False
        )
        depth = prediction.squeeze().cpu().numpy()
        return depth / depth.max() * CONFIG["MAX_DEPTH"]
    
    def update_grid(self, depth_map):
        h, w = depth_map.shape
        cx, cy = w // 2, h // 2
        
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                depth = depth_map[y, x]
                if depth < 0.5 or depth > CONFIG["MAX_DEPTH"]:
                    continue
                
                # Convert to world coords
                X = (x - cx) * depth / CONFIG["MAX_DEPTH"]
                Z = depth
                
                # Grid coords
                gx = int(X / CONFIG["BEV_RESOLUTION"]) + self.center
                gy = int(Z / CONFIG["BEV_RESOLUTION"]) + self.center
                
                if 0 <= gx < CONFIG["BEV_SIZE"] and 0 <= gy < CONFIG["BEV_SIZE"]:
                    self.grid[gy, gx] = max(self.grid[gy, gx], 0.0)
        
        # Mark camera position as free
        gx, gy = self.center, self.center
        self.grid[max(0, gy-5):min(CONFIG["BEV_SIZE"], gy+5), 
                  max(0, gx-5):min(CONFIG["BEV_SIZE"], gx+5)] = 0.0
    
    def get_visualization(self):
        vis = np.zeros((CONFIG["BEV_SIZE"], CONFIG["BEV_SIZE"], 3), dtype=np.uint8)
        vis[self.grid < 0] = [30, 30, 40]  # Unknown
        vis[(self.grid >= 0) & (self.grid < 0.5)] = [20, 80, 20]  # Free
        vis[self.grid >= 0.5] = [80, 30, 30]  # Occupied
        return vis
    
    def save_map(self, filename="bev_map.png"):
        vis = self.get_visualization()
        cv2.imwrite(filename, vis)
        print(f"💾 Saved: {filename}")
        return vis

def main():
    print("\n" + "=" * 60)
    print("Simple BEV Mapper - Headless Mode")
    print("=" * 60)
    print("\n📷 Point camera at environment...")
    print("📊 Will capture 100 frames and save map\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    mapper = SimpleBEVMapper()
    
    frame_count = 0
    max_frames = 100
    
    print(f"📡 Capturing {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        depth = mapper.estimate_depth(frame)
        mapper.update_grid(depth)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"   Frame {frame_count}/{max_frames}")
        
        cv2.waitKey(1)
    
    cap.release()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapper.save_map(f"bev_map_{timestamp}.png")
    
    # Also save depth visualization
    last_depth = mapper.estimate_depth(frame)
    depth_vis = cv2.applyColorMap((last_depth / CONFIG["MAX_DEPTH"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"depth_{timestamp}.png", depth_vis)
    print(f"💾 Saved: depth_{timestamp}.png")
    
    print("\n✅ Complete! Check the .png files in this folder\n")

if __name__ == "__main__":
    main()
