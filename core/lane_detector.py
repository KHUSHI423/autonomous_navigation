"""
=============================================================================
LANE DETECTION MODULE
=============================================================================
Robust lane detection for autonomous navigation using OpenCV.
Supports multiple presets for different road conditions.

Features:
- Canny edge detection
- Hough Line Transform
- Polynomial fitting
- Lane curvature estimation
- Vehicle offset calculation
- Multiple presets for Indian roads

Author: EdgeDrive3D Team
=============================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LaneResult:
    """Lane detection result"""
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    left_points: Optional[np.ndarray] = None
    right_points: Optional[np.ndarray] = None
    lane_width_pixels: float = 0.0
    lane_width_meters: float = 0.0
    curvature: float = 0.0  # radians
    vehicle_offset: float = 0.0  # meters
    confidence: float = 0.0
    preset_name: str = 'default'
    
    # Debug outputs
    warped_binary: Optional[np.ndarray] = None
    hough_lines: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            'lane_width_meters': round(self.lane_width_meters, 2),
            'curvature': round(self.curvature, 4),
            'vehicle_offset_meters': round(self.vehicle_offset, 2),
            'confidence': round(self.confidence, 2),
            'preset': self.preset_name,
        }


# ============================================================================
# LANE DETECTOR
# ============================================================================

class LaneDetector:
    """
    Robust lane detection using OpenCV
    
    Usage:
        detector = LaneDetector(preset='highway')
        result = detector.detect(frame)
    """
    
    # Presets for different road conditions
    PRESETS = {
        'default': {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 20,
            'min_line_length': 40,
            'max_line_gap': 20,
            'roi_trap': True,
        },
        'highway': {
            'canny_low': 60,
            'canny_high': 180,
            'hough_threshold': 30,
            'min_line_length': 50,
            'max_line_gap': 15,
            'roi_trap': True,
        },
        'city': {
            'canny_low': 40,
            'canny_high': 120,
            'hough_threshold': 15,
            'min_line_length': 30,
            'max_line_gap': 25,
            'roi_trap': True,
        },
        'faded': {
            'canny_low': 30,
            'canny_high': 100,
            'hough_threshold': 10,
            'min_line_length': 25,
            'max_line_gap': 30,
            'roi_trap': False,
        },
        'night': {
            'canny_low': 30,
            'canny_high': 80,
            'hough_threshold': 15,
            'min_line_length': 30,
            'max_line_gap': 20,
            'roi_trap': True,
        },
        'indian_road': {
            'canny_low': 35,
            'canny_high': 110,
            'hough_threshold': 12,
            'min_line_length': 25,
            'max_line_gap': 35,
            'roi_trap': False,
        },
    }
    
    # Lane line colors (for detection enhancement)
    LANE_COLORS = {
        'white': ((200, 200, 200), (255, 255, 255)),
        'yellow': ((20, 20, 150), (35, 255, 255)),
    }

    def __init__(self, preset: str = 'default', debug: bool = False, 
                 use_adaptive_threshold: bool = True,
                 detect_yellow_lines: bool = True):
        """
        Initialize lane detector

        Args:
            preset: Detection preset name
            debug: Enable debug mode with extra outputs
            use_adaptive_threshold: Enable adaptive thresholding based on lighting
            detect_yellow_lines: Enable yellow lane line detection
        """
        self.preset_name = preset
        self.config = self.PRESETS.get(preset, self.PRESETS['default'])
        self.debug = debug
        self.use_adaptive_threshold = use_adaptive_threshold
        self.detect_yellow_lines = detect_yellow_lines

        # Camera parameters (will be updated per frame)
        self.image_height = 0
        self.image_width = 0

        # Lane history for smoothing - INCREASED to 10 frames for stability
        self.left_lane_history: List[np.ndarray] = []
        self.right_lane_history: List[np.ndarray] = []
        self.max_history = 10  # Increased from 5 to 10 for smoother transitions

        # Polynomial fits
        self.left_fit: Optional[np.ndarray] = None
        self.right_fit: Optional[np.ndarray] = None

        # HLS thresholds (will be adjusted per frame if adaptive)
        self.s_threshold_low = 170
        self.l_threshold_low = 150

        print(f"  ✓ Lane Detector initialized (preset: {preset}, adaptive: {use_adaptive_threshold})")
    
    def detect(self, image: np.ndarray) -> LaneResult:
        """
        Detect lanes - robust and reliable
        """
        self.image_height, self.image_width = image.shape[:2]
        height, width = self.image_height, self.image_width
        
        # ROI - bottom 40%
        roi_y_start = int(height * 0.60)
        roi_image = image[roi_y_start:, :]
        roi_height, roi_width = roi_image.shape[:2]
        
        # Convert to HLS
        hls = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HLS)
        h_channel, l_channel, s_channel = cv2.split(hls)
        
        # WHITE lane detection (lowered thresholds for reliability)
        white_mask = np.zeros_like(s_channel, dtype=np.uint8)
        white_mask[(s_channel >= 120) & (s_channel <= 255) & 
                   (l_channel >= 140) & (l_channel <= 255)] = 255
        
        # YELLOW lane detection
        yellow_mask = np.zeros_like(s_channel, dtype=np.uint8)
        if self.detect_yellow_lines:
            yellow_mask[(h_channel >= 10) & (h_channel <= 40) & 
                        (s_channel >= 80) & (s_channel <= 255)] = 255
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        
        # Canny edge detection (lowered thresholds)
        edges = cv2.Canny(blurred, 40, 120)
        
        # ROI mask (trapezoid)
        mask = self._create_roi_mask(edges.shape)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Find lane pixels using histogram
        histogram = np.sum(edges[roi_height//2:, :], axis=0)
        midpoint = roi_width // 2
        
        # Smooth histogram to find better peaks
        histogram_smooth = np.convolve(histogram, np.ones(20), mode='same')
        
        # Find left and right lane starting positions
        leftx_base = np.argmax(histogram_smooth[:midpoint])
        rightx_base = np.argmax(histogram_smooth[midpoint:]) + midpoint
        
        # Sliding window approach
        n_windows = 9
        window_height = roi_height // n_windows
        margin = 80  # Wider margin
        minpix = 20  # Lower threshold
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        nonzero = edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        for window in range(n_windows):
            win_y_low = roi_height - (window + 1) * window_height
            win_y_high = roi_height - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Extract pixel positions
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])
        
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        # Fit polynomials
        self.left_fit = None
        self.right_fit = None
        
        if len(leftx) > 5 and len(lefty) > 5:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        
        if len(rightx) > 5 and len(righty) > 5:
            self.right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate lane lines from polynomials
        left_lane = None
        right_lane = None
        
        if self.left_fit is not None:
            ploty = np.linspace(0, roi_height - 1, 100)
            plotx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
            left_lane = np.array([int(plotx[99]), int(ploty[99] + roi_y_start),
                                 int(plotx[0]), int(ploty[0] + roi_y_start)])
        
        if self.right_fit is not None:
            ploty = np.linspace(0, roi_height - 1, 100)
            plotx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]
            right_lane = np.array([int(plotx[99]), int(ploty[99] + roi_y_start),
                                  int(plotx[0]), int(ploty[0] + roi_y_start)])
        
        # TEMPORAL SMOOTHING - populate history buffer
        if left_lane is not None:
            self.left_lane_history.append(left_lane)
            if len(self.left_lane_history) > self.max_history:
                self.left_lane_history.pop(0)
        
        if right_lane is not None:
            self.right_lane_history.append(right_lane)
            if len(self.right_lane_history) > self.max_history:
                self.right_lane_history.pop(0)
        
        # WEIGHTED AVERAGING
        if len(self.left_lane_history) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.left_lane_history))
            weights = weights / weights.sum()
            recent_lanes = np.array(self.left_lane_history)
            left_lane = np.average(recent_lanes, axis=0, weights=weights).astype(np.int32)
        
        if len(self.right_lane_history) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.right_lane_history))
            weights = weights / weights.sum()
            recent_lanes = np.array(self.right_lane_history)
            right_lane = np.average(recent_lanes, axis=0, weights=weights).astype(np.int32)
        
        # FALLBACK
        if left_lane is None and self.left_lane_history:
            left_lane = self.left_lane_history[-1].copy()
        if right_lane is None and self.right_lane_history:
            right_lane = self.right_lane_history[-1].copy()
        
        # Calculate metrics
        lane_width_px, lane_width_m = self._calculate_lane_width()
        curvature = self._calculate_curvature()
        vehicle_offset = self._calculate_vehicle_offset()
        confidence = self._calculate_confidence(left_lane, right_lane)
        
        # Create result
        result = LaneResult(
            left_lane=left_lane,
            right_lane=right_lane,
            left_points=self._get_lane_points(self.left_fit) if self.left_fit is not None else None,
            right_points=self._get_lane_points(self.right_fit) if self.right_fit is not None else None,
            lane_width_pixels=lane_width_px,
            lane_width_meters=lane_width_m,
            curvature=curvature,
            vehicle_offset=vehicle_offset,
            confidence=confidence,
            preset_name=self.preset_name,
        )
        
        if self.debug:
            result.warped_binary = edges
            result.hough_lines = None
        
        return result
    
    def _find_lanes_sliding_window(self, binary_image: np.ndarray) -> Tuple:
        """
        Find lanes using sliding window approach for robust detection
        
        This creates a visible path on the road like the LinkedIn post shows.
        
        Returns:
            left_fit, right_fit, left_points, right_points
        """
        height, width = binary_image.shape
        
        # Create histogram to find lane starting positions
        histogram = np.sum(binary_image[height//2:, :], axis=0)
        
        # Find peaks in histogram
        midpoint = width // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Sliding window parameters
        n_windows = 9
        window_height = height // n_windows
        margin = 100  # Width of window ± margin
        minpix = 50   # Minimum pixels to recenter window
        
        # Find window positions
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        # Get nonzero pixels
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Slide through windows
        for window in range(n_windows):
            # Window boundaries
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Find pixels in window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter window if enough pixels found
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])
        
        # Extract pixel positions
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        # Fit polynomials
        left_fit = None
        right_fit = None
        left_points = None
        right_points = None
        
        if len(leftx) > 10 and len(lefty) > 10:
            left_fit = np.polyfit(lefty, leftx, 2)
            ploty = np.linspace(0, height - 1, height)
            plotx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            left_points = np.stack([plotx, ploty], axis=1).astype(np.int32)
        
        if len(rightx) > 10 and len(righty) > 10:
            right_fit = np.polyfit(righty, rightx, 2)
            ploty = np.linspace(0, height - 1, height)
            plotx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            right_points = np.stack([plotx, ploty], axis=1).astype(np.int32)
        
        return left_fit, right_fit, left_points, right_points

    def _create_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create region of interest mask (trapezoid)"""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define trapezoid vertices
        top_width = width * 0.55
        bottom_width = width * 0.9
        top_y = int(height * 0.35)
        bottom_y = int(height * 0.95)
        
        left_top = int((width - top_width) / 2)
        right_top = int((width + top_width) / 2)
        left_bottom = int((width - bottom_width) / 2)
        right_bottom = int((width + bottom_width) / 2)
        
        pts = np.array([[
            [left_bottom, bottom_y],
            [left_top, top_y],
            [right_top, top_y],
            [right_bottom, bottom_y]
        ]], dtype=np.int32)
        
        cv2.fillPoly(mask, pts, 255)
        return mask
    
    def _separate_lines(self, lines: Optional[np.ndarray]) -> Tuple[List, List]:
        """
        Separate detected lines into left and right lanes using slope analysis
        
        Left lane: positive slope (in image coordinates, goes up-right)
        Right lane: negative slope (in image coordinates, goes up-left)
        Reject near-horizontal or noisy lines
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        center_x = self.image_width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # SLOPE FILTERING - reject horizontal/vertical noise
            if abs(slope) < 0.5:  # Reject near-horizontal
                continue
            if abs(slope) > 15:  # Reject near-vertical
                continue

            # Determine line center position
            line_center_x = (x1 + x2) / 2
            line_center_y = (y1 + y2) / 2
            
            # Only consider lines in lower portion of ROI (road region)
            if line_center_y > self.image_height * 0.15:  # Within ROI
                # Left lane: positive slope (goes up as x increases)
                # Right lane: negative slope (goes down as x increases)
                if slope > 0.5:  # Positive slope → LEFT lane
                    if line_center_x < center_x * 1.1:
                        left_lines.append((x1, y1, x2, y2))
                elif slope < -0.5:  # Negative slope → RIGHT lane
                    if line_center_x > center_x * 0.9:
                        right_lines.append((x1, y1, x2, y2))

        return left_lines, right_lines
    
    def _average_lines(self, lines: List, is_left: bool) -> Optional[np.ndarray]:
        """Average multiple line segments into single lane line"""
        if not lines:
            return None
        
        # Weight lines by length
        weighted_x1, weighted_y1, weighted_x2, weighted_y2 = 0, 0, 0, 0
        total_weight = 0

        for x1, y1, x2, y2 in lines:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            weight = length

            weighted_x1 += x1 * weight
            weighted_y1 += y1 * weight
            weighted_x2 += x2 * weight
            weighted_y2 += y2 * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        avg_x1 = int(weighted_x1 / total_weight)
        avg_y1 = int(weighted_y1 / total_weight)
        avg_x2 = int(weighted_x2 / total_weight)
        avg_y2 = int(weighted_y2 / total_weight)
        
        # Extend line to cover full height
        if avg_x2 == avg_x1:
            return None
        
        slope = (avg_y2 - avg_y1) / (avg_x2 - avg_x1)
        
        # Extend to bottom of image
        bottom_y = int(self.image_height * 0.95)
        bottom_x = int(avg_x1 + (bottom_y - avg_y1) / slope)
        
        # Extend to top of ROI
        top_y = int(self.image_height * 0.35)
        top_x = int(avg_x1 + (top_y - avg_y1) / slope)
        
        return np.array([top_x, top_y, bottom_x, bottom_y])
    
    def _fit_polynomials(self):
        """Fit second-order polynomials to lane lines"""
        # Use history for smoother results
        if self.left_lane_history:
            left_pts = []
            for lane in self.left_lane_history:
                if lane is not None:
                    left_pts.append((lane[0], lane[1]))
                    left_pts.append((lane[2], lane[3]))
            
            if len(left_pts) >= 2:
                left_pts = np.array(left_pts)
                self.left_fit = np.polyfit(left_pts[:, 1], left_pts[:, 0], 2)
        
        if self.right_lane_history:
            right_pts = []
            for lane in self.right_lane_history:
                if lane is not None:
                    right_pts.append((lane[0], lane[1]))
                    right_pts.append((lane[2], lane[3]))
            
            if len(right_pts) >= 2:
                right_pts = np.array(right_pts)
                self.right_fit = np.polyfit(right_pts[:, 1], right_pts[:, 0], 2)
    
    def _get_lane_points(self, fit: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Get smooth points along fitted polynomial for visualization"""
        if fit is None:
            return None

        # Generate smooth curve points in ROI coordinates
        roi_height = int(self.image_height * 0.40)  # Bottom 40%
        roi_y_start = int(self.image_height * 0.60)
        
        ploty_roi = np.linspace(0, roi_height - 1, 100)
        plotx = fit[0] * ploty_roi**2 + fit[1] * ploty_roi + fit[2]

        # Convert to full frame coordinates
        ploty_full = ploty_roi + roi_y_start

        # Ensure points are within image bounds
        plotx = np.clip(plotx, 0, self.image_width - 1)
        ploty_full = np.clip(ploty_full, 0, self.image_height - 1)

        return np.stack([plotx, ploty_full], axis=1).astype(np.int32)
    
    def _calculate_lane_width(self) -> Tuple[float, float]:
        """Calculate lane width in pixels and meters"""
        if self.left_fit is None or self.right_fit is None:
            return 0.0, 0.0

        # Calculate width at bottom of image
        bottom_y = self.image_height - 1

        left_x = self.left_fit[0] * bottom_y**2 + self.left_fit[1] * bottom_y + self.left_fit[2]
        right_x = self.right_fit[0] * bottom_y**2 + self.right_fit[1] * bottom_y + self.right_fit[2]

        width_px = abs(right_x - left_x)

        # Convert to meters (typical lane width ~3.5m)
        # Estimate based on detected width: assume ~3.5m if reasonable
        if width_px > 50:
            width_m = 3.5
        else:
            width_m = 0.0

        return width_px, width_m

    def _calculate_curvature(self) -> float:
        """Calculate lane curvature (simplified)"""
        if self.left_fit is None and self.right_fit is None:
            return 0.0

        # Use average of both lanes
        fits = [f for f in [self.left_fit, self.right_fit] if f is not None]
        if not fits:
            return 0.0

        avg_fit = np.mean(fits, axis=0)

        # Simple curvature calculation
        y_eval = self.image_height // 2
        curvature = abs(2 * avg_fit[0]) / (1 + (2 * avg_fit[0] * y_eval + avg_fit[1])**2)**1.5

        return float(curvature)

    def _calculate_vehicle_offset(self) -> float:
        """Calculate vehicle offset from lane center"""
        if self.left_fit is None or self.right_fit is None:
            return 0.0

        bottom_y = self.image_height - 1

        left_x = self.left_fit[0] * bottom_y**2 + self.left_fit[1] * bottom_y + self.left_fit[2]
        right_x = self.right_fit[0] * bottom_y**2 + self.right_fit[1] * bottom_y + self.right_fit[2]

        lane_center = (left_x + right_x) / 2
        image_center = self.image_width / 2

        offset_px = image_center - lane_center
        # Normalize to -1 to 1 range
        max_offset = self.image_width / 2
        return float(offset_px / max_offset)
    
    def _calculate_confidence(self, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
        """Calculate detection confidence score"""
        confidence = 0.0
        
        if left is not None:
            confidence += 0.4
        if right is not None:
            confidence += 0.4
        
        # Bonus for having history (stable detection)
        history_bonus = min(len(self.left_lane_history) + len(self.right_lane_history), 10) * 0.02
        confidence += history_bonus
        
        return min(confidence, 1.0)
    
    def draw_lanes(self, image: np.ndarray, result: LaneResult, 
                   ui_style: str = 'minimal') -> np.ndarray:
        """
        Production-quality lane drawing:
        - NO filled polygon (road stays visible)
        - Thin, semi-transparent green lines ON lane markers only
        - Precise line detection using Hough transforms
        """
        output = image.copy()
        height, width = image.shape[:2]
        
        # ROI - bottom 40% only
        roi_y_start = int(height * 0.60)
        
        # ===== PRECISE LANE LINE DRAWING =====
        
        # Use polynomial fit points for smooth curves
        if result.left_points is not None and result.right_points is not None:
            left_pts = result.left_points
            right_pts = result.right_points
            
            # Filter to ROI only
            left_filtered = left_pts[left_pts[:, 1] >= roi_y_start]
            right_filtered = right_pts[right_pts[:, 1] >= roi_y_start]
            
            if len(left_filtered) > 2 and len(right_filtered) > 2:
                # Draw LEFT lane line - thin, semi-transparent green
                for i in range(len(left_filtered) - 1):
                    pt1 = tuple(left_filtered[i])
                    pt2 = tuple(left_filtered[i + 1])
                    # Smooth curve with anti-aliasing
                    cv2.line(output, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw RIGHT lane line - thin, semi-transparent green
                for i in range(len(right_filtered) - 1):
                    pt1 = tuple(right_filtered[i])
                    pt2 = tuple(right_filtered[i + 1])
                    cv2.line(output, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Fallback: draw from lane arrays if polynomial points unavailable
        elif result.left_lane is not None and result.right_lane is not None:
            x1l, y1l, x2l, y2l = map(int, result.left_lane)
            x1r, y1r, x2r, y2r = map(int, result.right_lane)
            
            # Ensure within ROI
            y1l = max(y1l, roi_y_start)
            y2l = max(y2l, roi_y_start)
            y1r = max(y1r, roi_y_start)
            y2r = max(y2r, roi_y_start)
            
            # Draw thin lane lines
            cv2.line(output, (x1l, y1l), (x2l, y2l), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(output, (x1r, y1r), (x2r, y2r), (0, 255, 0), 2, cv2.LINE_AA)
        
        # ===== MINIMAL HUD =====
        if ui_style == 'minimal':
            output = self._draw_minimal_hud(output, result, height, width)
        else:
            output = self._draw_detailed_hud(output, result, height, width)
        
        return output
    
    def _draw_minimal_hud(self, image: np.ndarray, result: LaneResult, 
                          height: int, width: int) -> np.ndarray:
        """
        Draw minimal Tesla-style HUD - NO BACKGROUND BOXES
        Text drawn directly on video with proper spacing
        """
        
        # Skip HUD if confidence too low
        if result.confidence < 0.2:
            return image
        
        # Position: bottom-left corner with spacing
        x_pos = 15
        y_base = height - 100
        line_spacing = 25
        
        # Lane Width
        lane_color = (0, 255, 0) if result.confidence > 0.7 else (0, 255, 255)
        cv2.putText(image, f"LANE: {result.lane_width_meters:.1f}m",
                   (x_pos, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, lane_color, 2, cv2.LINE_AA)
        
        # Vehicle Offset
        offset_val = abs(result.vehicle_offset)
        offset_color = (0, 255, 0) if offset_val < 0.3 else (0, 165, 255)
        cv2.putText(image, f"OFFSET: {result.vehicle_offset:+.2f}m",
                   (x_pos, y_base + line_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 1, cv2.LINE_AA)
        
        # Confidence
        conf_color = (0, 255, 0) if result.confidence > 0.8 else (0, 255, 255)
        cv2.putText(image, f"CONF: {result.confidence:.0%}",
                   (x_pos, y_base + line_spacing * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 1, cv2.LINE_AA)
        
        return image
    
    def _draw_detailed_hud(self, image: np.ndarray, result: LaneResult,
                           height: int, width: int) -> np.ndarray:
        """Draw detailed HUD - NO BACKGROUND BOXES"""
        
        x_pos = 15
        y_base = height - 130
        line_spacing = 22
        
        cv2.putText(image, f"LANE WIDTH: {result.lane_width_meters:.2f}m",
                   (x_pos, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        offset_color = (0, 255, 0) if abs(result.vehicle_offset) < 0.3 else (0, 255, 255)
        cv2.putText(image, f"OFFSET: {result.vehicle_offset:+.2f}m",
                   (x_pos, y_base + line_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 1)
        
        cv2.putText(image, f"CONFIDENCE: {result.confidence:.0%}",
                   (x_pos, y_base + line_spacing * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(image, f"PRESET: {result.preset_name}",
                   (x_pos, y_base + line_spacing * 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image
    
    def draw_hls_debug(self, image: np.ndarray, result: LaneResult) -> np.ndarray:
        """Draw debug visualization showing HLS processing stages"""
        if result.warped_binary is None:
            return image
        
        # Convert binary to BGR for visualization
        debug_img = cv2.cvtColor(result.warped_binary, cv2.COLOR_GRAY2BGR)
        
        # Add processing info
        info_lines = [
            f"HLS Lane Detection Pipeline",
            f"S-Threshold: {self.s_threshold_low}, L-Threshold: {self.l_threshold_low}",
            f"Adaptive: {self.use_adaptive_threshold}, Yellow: {self.detect_yellow_lines}",
            f"Lines detected: {len(result.hough_lines) if result.hough_lines is not None else 0}"
        ]
        
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(debug_img, line, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return debug_img
    
    def set_preset(self, preset: str):
        """Change detection preset"""
        if preset in self.PRESETS:
            self.preset_name = preset
            self.config = self.PRESETS[preset]
            # Clear history when changing presets
            self.left_lane_history.clear()
            self.right_lane_history.clear()
            print(f"  ✓ Lane preset changed to: {preset}")
        else:
            print(f"  ⚠ Unknown preset: {preset}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_lane_detector(preset: str = 'default', debug: bool = False, 
                        use_adaptive_threshold: bool = True,
                        detect_yellow_lines: bool = True) -> LaneDetector:
    """Factory function to create lane detector
    
    Args:
        preset: Detection preset name
        debug: Enable debug mode
        use_adaptive_threshold: Enable adaptive thresholding
        detect_yellow_lines: Enable yellow lane detection
    
    Returns:
        Configured LaneDetector instance
    """
    return LaneDetector(
        preset=preset, 
        debug=debug,
        use_adaptive_threshold=use_adaptive_threshold,
        detect_yellow_lines=detect_yellow_lines
    )


if __name__ == "__main__":
    # Test enhanced lane detector with HLS color space
    detector = LaneDetector(preset='default', debug=True, 
                           use_adaptive_threshold=True, 
                           detect_yellow_lines=True)

    # Create test image with simulated lanes (both white and yellow)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw white lane lines
    cv2.line(test_image, (200, 480), (280, 200), (255, 255, 255), 5)
    cv2.line(test_image, (440, 480), (360, 200), (255, 255, 255), 5)
    
    # Draw yellow lane line (center line)
    cv2.line(test_image, (320, 480), (320, 200), (0, 255, 255), 3)

    result = detector.detect(test_image)

    print(f"\n{'='*60}")
    print(f"Enhanced Lane Detection Test (HLS + Canny + Hough)")
    print(f"{'='*60}")
    print(f"  Left lane: {'✓' if result.left_lane is not None else '✗'}")
    print(f"  Right lane: {'✓' if result.right_lane is not None else '✗'}")
    print(f"  Lane width: {result.lane_width_meters:.2f}m")
    print(f"  Vehicle offset: {result.vehicle_offset:+.2f}m")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Adaptive threshold: {detector.use_adaptive_threshold}")
    print(f"  Yellow detection: {detector.detect_yellow_lines}")
    print(f"{'='*60}")

    # Draw results
    output = detector.draw_lanes(test_image, result)
    cv2.imwrite("test_lane_output_enhanced.jpg", output)
    print(f"\n✓ Test output saved to: test_lane_output_enhanced.jpg")
    
    # Save debug visualization
    if result.warped_binary is not None:
        debug_output = detector.draw_hls_debug(test_image, result)
        cv2.imwrite("test_lane_hls_debug.jpg", debug_output)
        print(f"✓ Debug output saved to: test_lane_hls_debug.jpg")
