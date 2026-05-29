"""
Main Application for 3D Environment Mapping
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import time

from depth_estimator import DepthEstimator
from scene_3d_mapper import Scene3DMapper
from object_3d_detector import Object3DDetector, create_scene_summary, Object3D
from visualizer_3d import Visualizer3D, visualize_open3d
from camera_config import CameraIntrinsics


def print_banner():
    banner = """
    ╔════════════════════════════════════════════════════════════════════╗
    ║            🗺️  3D ENVIRONMENT MAPPING SYSTEM  🗺️                    ║
    ║                                                                    ║
    ║  Features:                                                         ║
    ║    • Monocular Depth Estimation                                    ║
    ║    • 3D Point Cloud Generation                                     ║
    ║    • Object Detection with 3D Positions                            ║
    ║    • Dimension Estimation                                          ║
    ║    • Interactive 3D Visualization                                  ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_image(image_path: str) -> np.ndarray:
    """Load image with multiple fallback methods"""
    image_path = str(image_path).strip().strip('"').strip("'")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Try cv2.imread
    image = cv2.imread(image_path)
    
    if image is None:
        # Fallback to PIL
        from PIL import Image
        pil_image = Image.open(image_path).convert('RGB')
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return image


def process_image_3d(
    image_path: str,
    output_dir: str = "output_3d",
    depth_model: str = "midas_hybrid",
    yolo_model: str = "yolov8m.pt",
    confidence: float = 0.5,
    max_depth: float = 50.0,
    fov: float = 70.0,
    save_pointcloud: bool = True,
    show_visualization: bool = True
) -> dict:
    """
    Process a single image for 3D mapping
    
    Args:
        image_path: Path to input image
        output_dir: Output directory
        depth_model: Depth estimation model
        yolo_model: Object detection model
        confidence: Detection confidence threshold
        max_depth: Maximum depth in meters
        fov: Camera field of view in degrees
        save_pointcloud: Whether to save point cloud
        show_visualization: Whether to show interactive visualization
        
    Returns:
        Dictionary with results
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    print("\n[1/5] Loading image...")
    image = load_image(image_path)
    height, width = image.shape[:2]
    print(f"  Image size: {width}x{height}")
    
    # Initialize camera
    camera = CameraIntrinsics.from_image_size(width, height, fov)
    print(f"  Camera FOV: {fov}°")
    print(f"  Focal length: {camera.fx:.1f}px")
    
    # Initialize components
    print("\n[2/5] Initializing models...")
    depth_estimator = DepthEstimator(model_type=depth_model)
    object_detector = Object3DDetector(model_size=yolo_model, confidence_threshold=confidence)
    scene_mapper = Scene3DMapper(camera)
    visualizer = Visualizer3D()
    
    # Estimate depth
    print("\n[3/5] Estimating depth...")
    start_time = time.time()
    depth_map = depth_estimator.estimate_metric_depth(image, max_depth_meters=max_depth)
    depth_time = time.time() - start_time
    print(f"  Depth estimation: {depth_time:.2f}s")
    print(f"  Depth range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")
    
    # Detect objects in 3D
    print("\n[4/5] Detecting objects...")
    start_time = time.time()
    objects_3d = object_detector.detect_3d(image, depth_map, camera)
    detect_time = time.time() - start_time
    print(f"  Detection time: {detect_time:.2f}s")
    print(f"  Objects detected: {len(objects_3d)}")
    
    # Print detected objects
    if objects_3d:
        print("\n  Detected Objects:")
        print("  " + "-" * 56)
        print(f"  {'Class':<15} {'Distance':>10} {'Width':>10} {'Height':>10}")
        print("  " + "-" * 56)
        for obj in objects_3d:
            dims = obj.estimated_dimensions or {}
            w = dims.get('width', 0)
            h = dims.get('height', 0)
            print(f"  {obj.class_name:<15} {obj.distance:>9.1f}m {w:>9.2f}m {h:>9.2f}m")
        print("  " + "-" * 56)
    
    # Generate point cloud
    print("\n[5/5] Generating 3D point cloud...")
    start_time = time.time()
    points, colors = scene_mapper.depth_to_pointcloud(
        image, depth_map, 
        downsample_factor=2,
        max_depth=max_depth
    )
    pcloud_time = time.time() - start_time
    print(f"  Point cloud generation: {pcloud_time:.2f}s")
    
    # ===== Save Results =====
    print("\n" + "=" * 40)
    print("Saving results...")
    print("=" * 40)
    
    # 1. Save depth visualization
    depth_colored = depth_estimator.create_depth_colormap(
        (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    )
    cv2.imwrite(str(output_path / f"{image_name}_depth.jpg"), depth_colored)
    print(f"  ✓ Depth map: {image_name}_depth.jpg")
    
    # 2. Save 2D detection result
    image_2d = object_detector.draw_detections_2d(image, objects_3d, show_3d_info=True)
    cv2.imwrite(str(output_path / f"{image_name}_detections.jpg"), image_2d)
    print(f"  ✓ Detections: {image_name}_detections.jpg")
    
    # 3. Save 3D boxes overlay
    image_3d = object_detector.draw_3d_boxes(image_2d, objects_3d, camera)
    cv2.imwrite(str(output_path / f"{image_name}_3d_boxes.jpg"), image_3d)
    print(f"  ✓ 3D boxes: {image_name}_3d_boxes.jpg")
    
    # 4. Save combined visualization
    combined = visualizer.create_depth_visualization(image_2d, depth_map, objects_3d)
    cv2.imwrite(str(output_path / f"{image_name}_combined.jpg"), combined)
    print(f"  ✓ Combined view: {image_name}_combined.jpg")
    
    # 5. Save top-down view
    topdown = visualizer.create_topdown_view(objects_3d)
    cv2.imwrite(str(output_path / f"{image_name}_topdown.jpg"), topdown)
    print(f"  ✓ Top-down view: {image_name}_topdown.jpg")
    
    # 6. Save point cloud
    if save_pointcloud:
        scene_mapper.save_pointcloud(str(output_path / f"{image_name}_pointcloud.ply"))
        print(f"  ✓ Point cloud: {image_name}_pointcloud.ply")
    
    # 7. Create and save interactive 3D visualization
    try:
        fig = visualizer.create_interactive_scene(
            points, colors, objects_3d,
            title=f"3D Scene: {image_name}",
            downsample=5
        )
        visualizer.save_interactive_html(fig, str(output_path / f"{image_name}_3d_scene.html"))
        print(f"  ✓ Interactive 3D: {image_name}_3d_scene.html")
    except Exception as e:
        print(f"  ⚠ Could not create interactive visualization: {e}")
    
    # 8. Save JSON summary
    summary = create_scene_summary(objects_3d)
    summary['processing'] = {
        'image_path': image_path,
        'image_size': {'width': width, 'height': height},
        'depth_model': depth_model,
        'yolo_model': yolo_model,
        'max_depth': max_depth,
        'fov': fov,
        'point_cloud_size': len(points),
        'processing_time': {
            'depth_estimation': round(depth_time, 2),
            'object_detection': round(detect_time, 2),
            'point_cloud': round(pcloud_time, 2),
            'total': round(depth_time + detect_time + pcloud_time, 2)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / f"{image_name}_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ Results JSON: {image_name}_results.json")
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 SCENE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total objects:     {summary['total_objects']}")
    
    if summary['by_class']:
        print(f"\n  Objects by class:")
        for cls, data in summary['by_class'].items():
            print(f"    • {cls}: {data['count']} (avg: {data['avg_distance']:.1f}m)")
    
    if summary['closest_object']:
        print(f"\n  Closest: {summary['closest_object']['class']} at {summary['closest_object']['distance']:.1f}m")
    if summary['farthest_object']:
        print(f"  Farthest: {summary['farthest_object']['class']} at {summary['farthest_object']['distance']:.1f}m")
    
    print(f"\n  Point cloud: {len(points):,} points")
    print("=" * 60)
    
    # Show visualization windows
    if show_visualization:
        print("\n👁️ Showing visualizations (press any key to continue)...")
        
        # Resize for display
        display_width = 1200
        scale = display_width / combined.shape[1]
        display_combined = cv2.resize(combined, None, fx=scale, fy=scale)
        
        cv2.imshow("3D Mapping Result", display_combined)
        cv2.imshow("Top-Down View", topdown)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Try to show interactive 3D
        try:
            fig.show()
        except:
            print("  Open the HTML file in a browser for interactive 3D view")
    
    return summary


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="3D Environment Mapping")
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Image mode
    img_parser = subparsers.add_parser('image', help='Process single image')
    img_parser.add_argument('input', help='Input image path')
    img_parser.add_argument('-o', '--output', default='output_3d', help='Output directory')
    img_parser.add_argument('--depth-model', default='midas_hybrid',
                           choices=['midas_small', 'midas_hybrid', 'midas_large',
                                   'depth_anything_small', 'depth_anything_base'],
                           help='Depth estimation model')
    img_parser.add_argument('--yolo-model', default='yolov8m.pt', help='YOLO model')
    img_parser.add_argument('-c', '--confidence', type=float, default=0.5)
    img_parser.add_argument('--max-depth', type=float, default=50.0, help='Max depth in meters')
    img_parser.add_argument('--fov', type=float, default=70.0, help='Camera FOV in degrees')
    img_parser.add_argument('--no-display', action='store_true', help='Do not show visualization')
    img_parser.add_argument('--no-pointcloud', action='store_true', help='Do not save point cloud')
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Process directory of images')
    batch_parser.add_argument('input', help='Input directory')
    batch_parser.add_argument('-o', '--output', default='output_3d')
    batch_parser.add_argument('--depth-model', default='midas_hybrid')
    batch_parser.add_argument('--yolo-model', default='yolov8m.pt')
    batch_parser.add_argument('-c', '--confidence', type=float, default=0.5)
    batch_parser.add_argument('--max-depth', type=float, default=50.0)
    batch_parser.add_argument('--fov', type=float, default=70.0)
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        process_image_3d(
            args.input,
            output_dir=args.output,
            depth_model=args.depth_model,
            yolo_model=args.yolo_model,
            confidence=args.confidence,
            max_depth=args.max_depth,
            fov=args.fov,
            save_pointcloud=not args.no_pointcloud,
            show_visualization=not args.no_display
        )
    
    elif args.mode == 'batch':
        input_path = Path(args.input)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        print(f"Found {len(images)} images")
        
        for i, img_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Processing {img_path.name}")
            try:
                process_image_3d(
                    str(img_path),
                    output_dir=args.output,
                    depth_model=args.depth_model,
                    yolo_model=args.yolo_model,
                    confidence=args.confidence,
                    max_depth=args.max_depth,
                    fov=args.fov,
                    show_visualization=False
                )
            except Exception as e:
                print(f"  Error: {e}")
    
    else:
        # Interactive mode
        print("\n📷 Interactive Mode\n")
        print("Enter image path (or 'quit' to exit):")
        
        while True:
            image_path = input("\nImage path: ").strip().strip('"').strip("'")
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not image_path:
                continue
            
            try:
                process_image_3d(image_path, show_visualization=True)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
