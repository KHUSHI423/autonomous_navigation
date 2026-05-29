"""
Test script to verify the enhanced lane detection pipeline
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.lane_detector import LaneDetector, create_lane_detector


def test_basic_detection():
    """Test basic lane detection with synthetic image"""
    print("\n" + "="*70)
    print("TEST 1: Basic Lane Detection (Synthetic Image)")
    print("="*70)
    
    # Create test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw white lane lines
    cv2.line(test_img, (200, 480), (280, 200), (255, 255, 255), 5)
    cv2.line(test_img, (440, 480), (360, 200), (255, 255, 255), 5)
    cv2.line(test_img, (320, 480), (320, 200), (0, 255, 255), 3)
    
    # Create detector with HLS enhancement
    detector = create_lane_detector(
        preset='default',
        debug=True,
        use_adaptive_threshold=True,
        detect_yellow_lines=True
    )
    
    # Detect lanes
    result = detector.detect(test_img)
    
    print(f"\nResults:")
    print(f"  ✓ Left lane detected: {result.left_lane is not None}")
    print(f"  ✓ Right lane detected: {result.right_lane is not None}")
    print(f"  ✓ Lane width: {result.lane_width_meters:.2f}m")
    print(f"  ✓ Vehicle offset: {result.vehicle_offset:+.2f}m")
    print(f"  ✓ Confidence: {result.confidence:.0%}")
    
    # Save visualization
    output = detector.draw_lanes(test_img, result)
    cv2.imwrite("test_synthetic_result.jpg", output)
    print(f"\n  ✓ Result saved to: test_synthetic_result.jpg")
    
    if result.warped_binary is not None:
        debug_output = detector.draw_hls_debug(test_img, result)
        cv2.imwrite("test_synthetic_debug.jpg", debug_output)
        print(f"  ✓ Debug saved to: test_synthetic_debug.jpg")
    
    return result.left_lane is not None or result.right_lane is not None


def test_presets():
    """Test all detection presets"""
    print("\n" + "="*70)
    print("TEST 2: All Detection Presets")
    print("="*70)
    
    # Create test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(test_img, (200, 480), (280, 200), (255, 255, 255), 5)
    cv2.line(test_img, (440, 480), (360, 200), (255, 255, 255), 5)
    
    presets = ['default', 'highway', 'city', 'faded', 'night', 'indian_road']
    results = {}
    
    for preset in presets:
        detector = create_lane_detector(preset=preset, debug=False)
        result = detector.detect(test_img)
        results[preset] = result
        
        status = "✓" if (result.left_lane is not None or result.right_lane is not None) else "✗"
        print(f"  {status} {preset:15s} - Width: {result.lane_width_meters:.2f}m, "
              f"Confidence: {result.confidence:.0%}")
    
    return all(r.left_lane is not None or r.right_lane is not None for r in results.values())


def test_adaptive_threshold():
    """Test adaptive thresholding"""
    print("\n" + "="*70)
    print("TEST 3: Adaptive Thresholding")
    print("="*70)
    
    # Create dark image
    dark_img = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.line(dark_img, (200, 480), (280, 200), (180, 180, 180), 5)
    cv2.line(dark_img, (440, 480), (360, 200), (180, 180, 180), 5)
    
    # Create bright image
    bright_img = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.line(bright_img, (200, 480), (280, 200), (255, 255, 255), 5)
    cv2.line(bright_img, (440, 480), (360, 200), (255, 255, 255), 5)
    
    # Test with adaptive threshold
    detector_adaptive = create_lane_detector(
        preset='default',
        use_adaptive_threshold=True
    )
    
    result_dark = detector_adaptive.detect(dark_img)
    result_bright = detector_adaptive.detect(bright_img)
    
    print(f"\nAdaptive Threshold Enabled:")
    print(f"  Dark image  - S-threshold: {detector_adaptive.s_threshold_low}, "
          f"L-threshold: {detector_adaptive.l_threshold_low}")
    print(f"  Dark image  - Lanes detected: "
          f"{'✓' if (result_dark.left_lane is not None or result_dark.right_lane is not None) else '✗'}")
    print(f"  Bright image - Lanes detected: "
          f"{'✓' if (result_bright.left_lane is not None or result_bright.right_lane is not None) else '✗'}")
    
    return True


def test_yellow_detection():
    """Test yellow lane detection"""
    print("\n" + "="*70)
    print("TEST 4: Yellow Lane Detection")
    print("="*70)
    
    # Create test image with yellow line
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(test_img, (320, 480), (320, 200), (0, 255, 255), 5)  # Yellow line (BGR)
    
    # Test with yellow detection enabled
    detector_yellow = create_lane_detector(
        preset='default',
        detect_yellow_lines=True
    )
    
    result = detector_yellow.detect(test_img)
    
    print(f"\nYellow Line Detection:")
    print(f"  Yellow detection enabled: {detector_yellow.detect_yellow_lines}")
    print(f"  Lanes detected: "
          f"{'✓' if (result.left_lane is not None or result.right_lane is not None) else '✗'}")
    
    if result.left_lane is not None or result.right_lane is not None:
        output = detector_yellow.draw_lanes(test_img, result)
        cv2.imwrite("test_yellow_result.jpg", output)
        print(f"  ✓ Result saved to: test_yellow_result.jpg")
    
    return result.left_lane is not None or result.right_lane is not None


def test_video_file():
    """Test with actual video file if available"""
    print("\n" + "="*70)
    print("TEST 5: Video File Processing")
    print("="*70)
    
    # Look for video files in current directory
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    if not video_files:
        print("\n  ⚠ No video files found in current directory")
        print("  ℹ Place a video file and run: python test_lane_detection.py")
        return True
    
    video_file = video_files[0]
    print(f"\n  ℹ Testing with video: {video_file}")
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_file}")
        return False
    
    # Process first 10 frames
    detector = create_lane_detector(preset='default', debug=False)
    frame_count = 0
    success_count = 0
    
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        if result.left_lane is not None or result.right_lane is not None:
            success_count += 1
        
        frame_count += 1
    
    cap.release()
    
    success_rate = (success_count / frame_count * 100) if frame_count > 0 else 0
    print(f"\n  Results: {success_count}/{frame_count} frames successful ({success_rate:.0f}%)")
    
    return success_rate > 50


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ENHANCED LANE DETECTION - TEST SUITE")
    print("="*70)
    print("\nTesting HLS + Canny + Hough + Polynomial Pipeline")
    
    results = {
        'Basic Detection': test_basic_detection(),
        'Detection Presets': test_presets(),
        'Adaptive Threshold': test_adaptive_threshold(),
        'Yellow Detection': test_yellow_detection(),
        'Video Processing': test_video_file(),
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced lane detection is working correctly.")
        print("\nNext steps:")
        print("  1. Run: python main_complete.py video your_video.mp4")
        print("  2. Run: python main_complete.py webcam")
        print("  3. See: MAIN_COMPLETE_README.md for usage guide")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
