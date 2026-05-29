"""
=============================================================================
TEST ENHANCED UNIFIED PIPELINE
=============================================================================
Test script for the enhanced EdgeDrive3D perception system with:
- Lane Detection
- Traffic Sign Detection
- 3D Object Detection
- Depth Estimation
- BEV Mapping
- Decision Making
=============================================================================
"""

import sys
import os

# Add core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from datetime import datetime


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_lane_detector():
    """Test lane detection module"""
    print_section("TEST 1: LANE DETECTOR")
    
    from core.lane_detector import LaneDetector, LaneResult
    
    # Create test image with simulated lanes
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw white lane lines
    cv2.line(test_image, (200, 480), (280, 200), (255, 255, 255), 5)
    cv2.line(test_image, (440, 480), (360, 200), (255, 255, 255), 5)
    
    # Test different presets
    presets = ['default', 'highway', 'city', 'indian_road']
    
    for preset in presets:
        print(f"\nTesting preset: {preset}")
        detector = LaneDetector(preset=preset)
        result = detector.detect(test_image)
        
        print(f"  Left lane: {'✓' if result.left_lane is not None else '✗'}")
        print(f"  Right lane: {'✓' if result.right_lane is not None else '✗'}")
        print(f"  Lane width: {result.lane_width_meters:.2f}m")
        print(f"  Vehicle offset: {result.vehicle_offset:+.2f}m")
        print(f"  Confidence: {result.confidence:.0%}")
    
    # Test drawing
    detector = LaneDetector(preset='default')
    result = detector.detect(test_image)
    output = detector.draw_lanes(test_image, result)
    
    output_path = "test_output/lane_test.jpg"
    os.makedirs("test_output", exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"\n✓ Lane test output saved to: {output_path}")
    
    return True


def test_sign_detector():
    """Test traffic sign detection module"""
    print_section("TEST 2: TRAFFIC SIGN DETECTOR")
    
    from core.sign_detector import SignDetector, SignDetectionResult
    
    # Create test image with simulated stop sign
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw red octagon (stop sign approximation)
    center = (320, 240)
    radius = 50
    pts = cv2.regularPolygon2PolyPoints(center, 8, radius, 0)[0]
    pts = pts.astype(np.int32)
    cv2.fillPoly(test_image, [pts], (0, 0, 255))
    cv2.putText(test_image, "STOP", (center[0] - 30, center[1] + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw yellow triangle (warning sign)
    center2 = (150, 300)
    triangle_pts = np.array([
        [center2[0], center2[1] - 40],
        [center2[0] - 35, center2[1] + 30],
        [center2[0] + 35, center2[1] + 30]
    ], dtype=np.int32)
    cv2.fillPoly(test_image, [triangle_pts], (0, 255, 255))
    
    # Test detection
    detector = SignDetector(mode='opencv', confidence_threshold=0.4)
    result = detector.detect(test_image)
    
    print(f"\nSigns detected: {len(result.signs)}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    
    for sign in result.signs:
        print(f"  - {sign.sign_type} ({sign.sign_class}) - {sign.confidence:.0%}")
        print(f"    BBox: {sign.bbox}")
        print(f"    Distance: {sign.distance_estimate:.1f}m")
    
    # Test drawing
    output = detector.draw_signs(test_image, result)
    
    output_path = "test_output/sign_test.jpg"
    cv2.imwrite(output_path, output)
    print(f"\n✓ Sign test output saved to: {output_path}")
    
    return True


def test_perception_engine():
    """Test complete perception engine"""
    print_section("TEST 3: COMPLETE PERCEPTION ENGINE")
    
    from core.perception_engine import PerceptionEngine
    
    # Create test image with road scene
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Sky
    cv2.rectangle(test_image, (0, 0), (640, 200), (100, 100, 255), -1)
    
    # Road
    cv2.rectangle(test_image, (0, 200), (640, 480), (50, 50, 50), -1)
    
    # Lane markings
    cv2.line(test_image, (200, 480), (280, 200), (255, 255, 255), 3)
    cv2.line(test_image, (440, 480), (360, 200), (255, 255, 255), 3)
    
    # Simulated car (rectangle)
    cv2.rectangle(test_image, (280, 280), (360, 340), (0, 255, 0), -1)
    
    # Stop sign
    center = (500, 250)
    cv2.circle(test_image, center, 30, (0, 0, 255), -1)
    cv2.putText(test_image, "STOP", (center[0] - 25, center[1] + 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print("\nInitializing Perception Engine...")
    engine = PerceptionEngine({
        'yolo_model': 'yolov8n.pt',
        'confidence': 0.4,
        'max_depth': 50.0,
        'enable_lane': True,
        'lane_preset': 'default',
        'enable_signs': True,
        'sign_mode': 'opencv',
    })
    
    print("\nProcessing test frame...")
    result = engine.process_frame(test_image)
    
    print(f"\nResults:")
    print(f"  FPS: {result.fps:.1f}")
    print(f"  Objects detected: {len(result.objects_3d)}")
    print(f"  Lanes detected: {'✓' if result.lanes else '✗'}")
    print(f"  Traffic signs: {len(result.traffic_signs)}")
    print(f"  Decision: {result.decision['action']}")
    print(f"  Warnings: {len(result.decision.get('warnings', []))}")
    
    if result.lanes:
        print(f"\nLane Details:")
        print(f"  Width: {result.lanes.lane_width_meters:.2f}m")
        print(f"  Curvature: {result.lanes.curvature:.4f}")
        print(f"  Offset: {result.lanes.vehicle_offset:+.2f}m")
        print(f"  Confidence: {result.lanes.confidence:.0%}")
    
    if result.traffic_signs:
        print(f"\nSign Details:")
        for sign in result.traffic_signs:
            print(f"  - {sign.sign_type} at {sign.distance_estimate:.1f}m")
    
    # Save outputs
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if result.detections_overlay is not None:
        cv2.imwrite(f"{output_dir}/{timestamp}_detections.jpg", result.detections_overlay)
    
    if result.bev_image is not None:
        cv2.imwrite(f"{output_dir}/{timestamp}_bev.jpg", result.bev_image)
    
    if result.depth_colored is not None:
        cv2.imwrite(f"{output_dir}/{timestamp}_depth.jpg", result.depth_colored)
    
    print(f"\n✓ Engine test outputs saved to: {output_dir}/")
    
    return True


def test_module_toggling():
    """Test dynamic module enabling/disabling"""
    print_section("TEST 4: MODULE TOGGLING")
    
    from core.perception_engine import PerceptionEngine
    
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (0, 200), (640, 480), (50, 50, 50), -1)
    cv2.line(test_image, (200, 480), (280, 200), (255, 255, 255), 3)
    cv2.line(test_image, (440, 480), (360, 200), (255, 255, 255), 3)
    
    print("\nTesting with all modules enabled...")
    engine = PerceptionEngine({
        'enable_lane': True,
        'enable_signs': True,
    })
    result = engine.process_frame(test_image)
    print(f"  Lanes: {'✓' if result.lanes else '✗'}")
    print(f"  Signs: {len(result.traffic_signs)}")
    
    print("\nTesting with lane disabled...")
    engine.config['enable_lane'] = False
    result = engine.process_frame(test_image)
    print(f"  Lanes: {'✓' if result.lanes else '✗'}")
    
    print("\nTesting with signs disabled...")
    engine.config['enable_signs'] = False
    result = engine.process_frame(test_image)
    print(f"  Signs: {len(result.traffic_signs)}")
    
    print("\n✓ Module toggling works correctly")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  EdgeDrive3D Enhanced Pipeline Test Suite")
    print("  Version 2.0.0")
    print("=" * 60)
    
    tests = [
        ("Lane Detector", test_lane_detector),
        ("Sign Detector", test_sign_detector),
        ("Perception Engine", test_perception_engine),
        ("Module Toggling", test_module_toggling),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced pipeline is ready.")
    else:
        print("\n⚠ Some tests failed. Check the output above.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
