"""
=============================================================================
EDGE DRIVE 3D - MAP OUTPUTS GENERATOR
=============================================================================
Generate all map visualization outputs for a given input image

This script processes an input image and generates:
1. Bird's Eye View (BEV) map
2. Depth map visualization
3. 3D point cloud (PLY format)
4. Detection overlay with bounding boxes
5. OpenStreetMap HTML visualization
6. 3D Campus map HTML visualization
7. Combined poster with all visualizations

Location: Avinashi Road, near Coimbatore Institute of Technology (CIT)
=============================================================================
"""

import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.perception_engine import PerceptionEngine
from dashboard.components.osm_view import (
    create_osm_map, create_vehicle_marker, create_object_marker,
    create_object_icon_marker, create_trajectory_polyline,
    create_fov_sector, PerceptionOverlayManager, OBJECT_COLORS
)
from dashboard.components.map_3d import generate_3d_map_html
from utils.coordinate_transform import destination_point


# ============================================================================
# CONFIGURATION
# ============================================================================

# CIT Campus coordinates (Avinashi Road, Coimbatore)
CIT_LATITUDE = 11.0294
CIT_LONGITUDE = 76.9384
CIT_HEADING = 90.0  # Facing East

# Input/Output paths
INPUT_IMAGE = r"C:\SANJEEVI\PROJECTS\AUTONOMOUS_VEHICLE\_3d_mapping\output_3d\result_image_base_3d_boxes.jpg"
OUTPUT_DIR = Path(__file__).parent / "poster_results"


# ============================================================================
# MAP OUTPUTS GENERATOR
# ============================================================================

class MapOutputsGenerator:
    """
    Generate all map visualization outputs from an input image
    """

    def __init__(self, output_dir: str = "poster_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize perception engine
        print("Initializing Perception Engine...")
        self.engine = PerceptionEngine({
            'yolo_model': 'yolov8m.pt',
            'confidence': 0.4,
            'max_depth': 50.0,
        })
        
        # Storage for results
        self.current_result = None
        self.current_frame = None
        
        print(f"✓ Output directory: {self.output_dir.absolute()}")
        print("✓ Map Outputs Generator Ready!\n")

    def process_image(self, image_path: str) -> bool:
        """
        Process input image through perception pipeline
        
        Args:
            image_path: Path to input image
            
        Returns:
            True if successful
        """
        print(f"Loading image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        self.current_frame = frame
        print(f"✓ Image loaded: {frame.shape[1]}x{frame.shape[0]}")
        
        # Process through perception pipeline
        print("Running perception pipeline...")
        result = self.engine.process_frame(frame)
        self.current_result = result
        
        print(f"✓ Perception complete")
        print(f"  - Objects detected: {len(result.objects_3d)}")
        print(f"  - FPS: {result.fps:.1f}")
        print(f"  - Decision: {result.decision['action']}")
        
        return True

    def generate_bev_map(self) -> str:
        """
        Generate Bird's Eye View map
        
        Returns:
            Path to saved BEV image
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating Bird's Eye View map...")
        
        bev_image = self.current_result.bev_image
        
        if bev_image is None:
            # Generate BEV if not available
            bev_image = self.engine.bev_mapper.create_bev(
                self.current_result.objects_3d
            )
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_bev_map.jpg"
        cv2.imwrite(str(output_path), bev_image)
        
        print(f"✓ BEV map saved: {output_path.name}")
        return str(output_path)

    def generate_depth_map(self) -> str:
        """
        Generate colored depth map visualization
        
        Returns:
            Path to saved depth map image
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating depth map visualization...")
        
        depth_colored = self.current_result.depth_colored
        
        if depth_colored is None:
            # Generate depth map if not available
            depth_map = self.current_result.depth_map
            if depth_map is not None:
                depth_norm = ((depth_map - depth_map.min()) / 
                             (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            else:
                print("⚠ No depth map available")
                return ""
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_depth_map.jpg"
        cv2.imwrite(str(output_path), depth_colored)
        
        print(f"✓ Depth map saved: {output_path.name}")
        return str(output_path)

    def generate_detection_overlay(self) -> str:
        """
        Generate detection overlay with bounding boxes
        
        Returns:
            Path to saved overlay image
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating detection overlay...")
        
        overlay = self.current_result.detections_overlay
        
        if overlay is None:
            overlay = self.current_frame
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_detection_overlay.jpg"
        cv2.imwrite(str(output_path), overlay)
        
        print(f"✓ Detection overlay saved: {output_path.name}")
        return str(output_path)

    def generate_point_cloud(self) -> str:
        """
        Generate 3D point cloud in PLY format
        
        Returns:
            Path to saved PLY file
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating 3D point cloud...")
        
        point_cloud = self.current_result.point_cloud
        
        if point_cloud is None:
            # Generate point cloud if not available
            frame = self.current_frame
            depth_map = self.current_result.depth_map
            
            if depth_map is None:
                print("⚠ No depth data available")
                return ""
            
            point_cloud = self.engine._generate_pointcloud(frame, depth_map)
        
        # Save as PLY
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_pointcloud.ply"
        
        self._save_pointcloud(point_cloud, output_path)
        
        print(f"✓ Point cloud saved: {output_path.name}")
        return str(output_path)

    def _save_pointcloud(self, point_cloud: Tuple, filepath: Path):
        """Save point cloud to PLY file"""
        points, colors = point_cloud
        
        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write points
            for i, point in enumerate(points):
                color = colors[i] if i < len(colors) else [128, 128, 128]
                f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} ")
                f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

    def generate_osm_map(self) -> str:
        """
        Generate OpenStreetMap HTML visualization
        
        Returns:
            Path to saved HTML file
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating OpenStreetMap visualization...")
        
        # Create map centered at CIT
        m = create_osm_map(
            center_lat=CIT_LATITUDE,
            center_lon=CIT_LONGITUDE,
            zoom=17,
            height="600"
        )
        
        # Add vehicle marker at CIT location
        vehicle = create_vehicle_marker(
            latitude=CIT_LATITUDE,
            longitude=CIT_LONGITUDE,
            heading=CIT_HEADING,
            speed=5.0,
            popup_info={'location': 'Avinashi Road, Near CIT'}
        )
        vehicle.add_to(m)
        
        # Add detected objects
        objects_data = []
        for i, obj in enumerate(self.current_result.objects_3d):
            if obj.position_3d is not None:
                # Project to GPS coordinates
                obj_lat, obj_lon = destination_point(
                    CIT_LATITUDE, CIT_LONGITUDE,
                    CIT_HEADING + np.degrees(np.arctan2(obj.position_3d[0], obj.position_3d[2])),
                    obj.distance
                )
                
                marker = create_object_icon_marker(
                    latitude=obj_lat,
                    longitude=obj_lon,
                    object_class=obj.class_name,
                    distance=obj.distance,
                    confidence=obj.confidence
                )
                marker.add_to(m)
                
                objects_data.append({
                    'class_name': obj.class_name,
                    'distance': float(obj.distance),
                    'confidence': float(obj.confidence),
                    'position_3d': obj.position_3d.tolist() if obj.position_3d is not None else None
                })
        
        # Add FOV sector
        fov = create_fov_sector(
            CIT_LATITUDE, CIT_LONGITUDE,
            CIT_HEADING,
            fov_degrees=70.0,
            range_meters=50.0,
            color='#00ff00',
            opacity=0.2
        )
        fov.add_to(m)
        
        # Add trajectory (simulated)
        trajectory = [
            (CIT_LATITUDE - 0.0001, CIT_LONGITUDE - 0.0001),
            (CIT_LATITUDE, CIT_LONGITUDE)
        ]
        traj_line = create_trajectory_polyline(
            trajectory,
            color='#00ffff',
            weight=4,
            opacity=0.8
        )
        traj_line.add_to(m)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_osm_map.html"
        m.save(str(output_path))
        
        # Also save objects data
        objects_path = self.output_dir / f"{timestamp}_objects.json"
        with open(objects_path, 'w') as f:
            json.dump({
                'location': 'Avinashi Road, Near CIT, Coimbatore',
                'vehicle_position': {
                    'latitude': CIT_LATITUDE,
                    'longitude': CIT_LONGITUDE,
                    'heading': CIT_HEADING
                },
                'objects': objects_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✓ OpenStreetMap saved: {output_path.name}")
        print(f"✓ Objects data saved: {objects_path.name}")
        return str(output_path)

    def generate_3d_campus_map(self) -> str:
        """
        Generate 3D Campus map HTML visualization
        
        Returns:
            Path to saved HTML file
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating 3D Campus visualization...")
        
        # Prepare objects data
        objects_data = []
        for obj in self.current_result.objects_3d:
            if obj.position_3d is not None:
                # Calculate approximate GPS for objects
                obj_lat, obj_lon = destination_point(
                    CIT_LATITUDE, CIT_LONGITUDE,
                    CIT_HEADING + np.degrees(np.arctan2(obj.position_3d[0], obj.position_3d[2])),
                    obj.distance
                )
                
                objects_data.append({
                    'class_name': obj.class_name,
                    'distance': float(obj.distance),
                    'confidence': float(obj.confidence),
                    'gps_coordinates': {
                        'latitude': obj_lat,
                        'longitude': obj_lon,
                        'altitude': 300  # Approximate altitude
                    },
                    'position_3d': obj.position_3d.tolist()
                })
        
        # Generate HTML
        html_content = generate_3d_map_html(
            center_lat=CIT_LATITUDE,
            center_lon=CIT_LONGITUDE,
            zoom=17,
            vehicle_position={
                'lat': CIT_LATITUDE,
                'lon': CIT_LONGITUDE,
                'heading': CIT_HEADING,
                'speed': 5.0
            },
            objects=objects_data,
            trajectory=[
                (CIT_LATITUDE - 0.0001, CIT_LONGITUDE - 0.0001),
                (CIT_LATITUDE, CIT_LONGITUDE)
            ],
            map_style='dark',
            height=600
        )
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_3d_campus_map.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ 3D Campus map saved: {output_path.name}")
        return str(output_path)

    def generate_results_json(self) -> str:
        """
        Generate JSON file with all perception results
        
        Returns:
            Path to saved JSON file
        """
        if self.current_result is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating results JSON...")
        
        # Serialize results
        results_data = {
            'metadata': {
                'input_image': INPUT_IMAGE,
                'location': 'Avinashi Road, Near CIT, Coimbatore, Tamil Nadu',
                'coordinates': {
                    'latitude': CIT_LATITUDE,
                    'longitude': CIT_LONGITUDE
                },
                'timestamp': datetime.now().isoformat(),
                'generator': 'EdgeDrive3D Map Outputs Generator'
            },
            'perception': {
                'fps': self.current_result.fps,
                'objects_count': len(self.current_result.objects_3d),
                'decision': self.current_result.decision,
                'objects': []
            }
        }
        
        # Add object details
        for obj in self.current_result.objects_3d:
            obj_data = {
                'class_name': obj.class_name,
                'confidence': float(obj.confidence),
                'distance_m': float(obj.distance),
                'bbox_2d': list(obj.bbox_2d) if obj.bbox_2d is not None else None,
                'position_3d': obj.position_3d.tolist() if obj.position_3d is not None else None,
                'dimensions': obj.dimensions
            }
            results_data['perception']['objects'].append(obj_data)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_results.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Results JSON saved: {output_path.name}")
        return str(output_path)

    def generate_combined_poster(self) -> str:
        """
        Generate combined poster with all visualizations
        
        Returns:
            Path to saved poster image
        """
        if self.current_result is None or self.current_frame is None:
            print("❌ No perception result available")
            return ""
        
        print("\nGenerating combined poster...")
        
        # Get all visualization images
        bev = self.current_result.bev_image
        depth = self.current_result.depth_colored
        detections = self.current_result.detections_overlay
        
        if bev is None:
            bev = self.engine.bev_mapper.create_bev(self.current_result.objects_3d)
        
        if depth is None:
            depth_map = self.current_result.depth_map
            if depth_map is not None:
                depth_norm = ((depth_map - depth_map.min()) / 
                             (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
                depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            else:
                depth = np.zeros_like(self.current_frame)
        
        if detections is None:
            detections = self.current_frame
        
        # Create poster layout with consistent sizing
        # Use a standard size for all panels
        panel_w = 640
        panel_h = 360
        header_h = 100
        footer_h = 50
        
        # Top: Input image (full width)
        # Bottom: BEV | Depth | Detections (3 columns)
        poster_w = panel_w * 3
        poster_h = header_h + panel_h + 20 + panel_h + footer_h
        
        poster = np.ones((poster_h, poster_w, 3), dtype=np.uint8) * 240
        
        # Add title
        title_text = "AUTONOMOUS VEHICLE PERCEPTION SYSTEM - AVINASHI ROAD, COIMBATORE"
        cv2.putText(poster, title_text, (poster_w//2 - 400, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        subtitle_text = "3D Mapping & Object Detection near Coimbatore Institute of Technology"
        cv2.putText(poster, subtitle_text, (poster_w//2 - 320, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        
        # Top: Original image (full width)
        top_y = header_h
        frame_resized = cv2.resize(self.current_frame, (poster_w, panel_h))
        poster[top_y:top_y+panel_h, :] = frame_resized
        
        # Bottom row: BEV, Depth, Detections
        bottom_y = top_y + panel_h + 20
        
        # Resize panels to consistent size
        bev_resized = cv2.resize(bev, (panel_w, panel_h))
        depth_resized = cv2.resize(depth, (panel_w, panel_h))
        detections_resized = cv2.resize(detections, (panel_w, panel_h))
        
        poster[bottom_y:bottom_y+panel_h, 0:panel_w] = bev_resized
        poster[bottom_y:bottom_y+panel_h, panel_w:panel_w*2] = depth_resized
        poster[bottom_y:bottom_y+panel_h, panel_w*2:panel_w*3] = detections_resized
        
        # Add labels
        label_y = bottom_y + panel_h - 15
        cv2.putText(poster, "BIRD'S EYE VIEW", (panel_w//2 - 90, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(poster, "DEPTH MAP", (panel_w + panel_w//2 - 70, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(poster, "DETECTIONS", (panel_w*2 + panel_w//2 - 80, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add info bar at bottom
        info_y = bottom_y + panel_h + 35
        cv2.putText(poster, f"Objects Detected: {len(self.current_result.objects_3d)}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        cv2.putText(poster, f"Processing FPS: {self.current_result.fps:.1f}", (250, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        cv2.putText(poster, f"Decision: {self.current_result.decision['action'].upper()}", (480, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        cv2.putText(poster, f"Location: CIT, Coimbatore", (750, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_combined_poster.jpg"
        cv2.imwrite(str(output_path), poster)
        
        print(f"✓ Combined poster saved: {output_path.name}")
        return str(output_path)

    def generate_all_outputs(self, image_path: str = None) -> Dict[str, str]:
        """
        Generate all map outputs
        
        Args:
            image_path: Optional path to input image
            
        Returns:
            Dictionary of output paths
        """
        if image_path is None:
            image_path = INPUT_IMAGE
        
        print("=" * 70)
        print("  EDGE DRIVE 3D - MAP OUTPUTS GENERATOR")
        print("=" * 70)
        print(f"\nInput: {image_path}")
        print(f"Location: Avinashi Road, Near CIT, Coimbatore")
        print(f"Output: {self.output_dir.absolute()}\n")
        
        outputs = {}
        
        # Process image
        if not self.process_image(image_path):
            return outputs
        
        # Generate all outputs
        outputs['bev_map'] = self.generate_bev_map()
        outputs['depth_map'] = self.generate_depth_map()
        outputs['detection_overlay'] = self.generate_detection_overlay()
        outputs['point_cloud'] = self.generate_point_cloud()
        outputs['osm_map'] = self.generate_osm_map()
        outputs['3d_campus_map'] = self.generate_3d_campus_map()
        outputs['results_json'] = self.generate_results_json()
        outputs['combined_poster'] = self.generate_combined_poster()
        
        # Print summary
        print("\n" + "=" * 70)
        print("  GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated {len(outputs)} outputs:")
        for name, path in outputs.items():
            if path:
                print(f"  ✓ {name}: {Path(path).name}")
        
        print(f"\n📁 All outputs saved to: {self.output_dir.absolute()}")
        print("\n📌 Open the HTML files in a web browser for interactive 3D maps")
        print("📌 Use the combined poster for presentations")
        
        return outputs


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate map visualization outputs from input image'
    )
    parser.add_argument(
        '-i', '--input',
        default=INPUT_IMAGE,
        help=f'Input image path (default: {INPUT_IMAGE})'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = MapOutputsGenerator(output_dir=args.output)
    generator.generate_all_outputs(image_path=args.input)
