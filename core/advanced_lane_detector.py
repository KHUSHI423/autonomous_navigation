"""
=============================================================================
ADVANCED LANE DETECTION MODULE - PRODUCTION QUALITY
=============================================================================
Robust lane detection for BOTH straight and curved roads using:
- HLS color space thresholding (white + yellow)
- Gaussian blur + Canny edge detection
- Dynamic trapezoidal ROI
- Perspective transform (Bird's Eye View)
- Histogram-based sliding window approach
- 2nd degree polynomial fitting (smooth curves)
- Temporal smoothing (no jitter)
- Warp back to original perspective

Author: Senior Computer Vision Engineer
=============================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LaneResult:
    """Lane detection result"""
    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    left_points: Optional[np.ndarray] = None
    right_points: Optional[np.ndarray] = None
    lane_width_pixels: float = 0.0
    lane_width_meters: float = 3.5  # Standard lane width
    curvature: float = 0.0
    vehicle_offset: float = 0.0
    confidence: float = 0.0
    preset_name: str = 'default'
    warped_binary: Optional[np.ndarray] = None
    debug_image: Optional[np.ndarray] = None


# ============================================================================
# ADVANCED LANE DETECTOR
# ============================================================================

class AdvancedLaneDetector:
    """
    Production-quality lane detection for straight AND curved roads
    
    Pipeline:
    1. HLS color space + thresholding
    2. Gaussian blur + Canny edges
    3. Dynamic trapezoidal ROI
    4. Perspective transform (BEV)
    5. Sliding window histogram
    6. Polynomial curve fitting
    7. Temporal smoothing
    8. Warp back to original view
    """
    
    def __init__(self, preset: str = 'default', debug: bool = False):
        """
        Initialize lane detector
        
        Args:
            preset: Detection preset: default, highway, city, night, faded, indian_road.
                    Unknown values print a warning and fall back to 'default'.
            debug: Enable debug mode
        """
        self.preset = preset
        self.debug = debug
        
        # Camera parameters
        self.image_height = 0
        self.image_width = 0
        
        # Perspective transform matrices
        self.M = None  # Forward transform
        self.Minv = None  # Inverse transform
        
        # Polynomial history for temporal smoothing (10 frames)
        self.left_fit_history = deque(maxlen=10)
        self.right_fit_history = deque(maxlen=10)
        
        # Previous valid fits (fallback)
        self.prev_left_fit = None
        self.prev_right_fit = None
        
        # Previous lane points for exponential moving average (stability)
        self.prev_left_pts = None
        self.prev_right_pts = None
        self.smoothing_alpha = 0.25  # Alpha for exponential moving average
        
        # Preset configurations. `s_min` / `l_min` / `yellow_s_min` control the
        # HLS color thresholds in _preprocess_and_transform — lower values catch
        # faded markings on Indian roads. Missing keys fall back to Western defaults.
        self.presets = {
            'default':     {'canny_low': 50, 'canny_high': 150, 'margin': 100, 'minpix': 50},
            'highway':     {'canny_low': 60, 'canny_high': 180, 'margin': 100, 'minpix': 50},
            'city':        {'canny_low': 40, 'canny_high': 120, 'margin':  80, 'minpix': 40},
            'night':       {'canny_low': 30, 'canny_high': 100, 'margin': 100, 'minpix': 40},
            'faded':       {'canny_low': 25, 'canny_high':  90, 'margin': 110, 'minpix': 30,
                            's_min':  70, 'l_min': 140, 'yellow_s_min':  60},
            'indian_road': {'canny_low': 30, 'canny_high': 100, 'margin': 120, 'minpix': 30,
                            's_min':  80, 'l_min': 150, 'yellow_s_min':  70},
        }

        if preset not in self.presets:
            print(f"  [WARN] Unknown lane preset '{preset}' -- using 'default'. "
                  f"Available: {sorted(self.presets.keys())}")
            preset = 'default'

        self.preset = preset
        self.config = self.presets[preset]

        print(f"  [OK] Advanced Lane Detector initialized (preset: {preset})")
    
    def detect(self, image: np.ndarray) -> LaneResult:
        """
        Full lane detection pipeline - STABILITY FIRST
        
        Args:
            image: Input BGR image
            
        Returns:
            LaneResult with detected lanes
        """
        self.image_height, self.image_width = image.shape[:2]
        
        # STEP 1-3: Preprocessing + Perspective Transform
        binary_warped = self._preprocess_and_transform(image)
        
        # STEP 4-5: Sliding window + polynomial fitting
        left_fit, right_fit = self._find_lane_pixels_curved(binary_warped)
        
        # STEP 6: HEAVY SMOOTHING - blend with history
        left_fit_smooth, right_fit_smooth = self._heavy_smooth_polynomials(left_fit, right_fit)
        
        # STEP 7-8: Generate curved lane points and warp back
        result = self._generate_curved_output(image, left_fit_smooth, right_fit_smooth)
        
        # STEP 9: Validate - reject if unreasonable
        result = self._validate_result(result)
        
        if self.debug:
            result.warped_binary = binary_warped
        
        return result
    
    def _heavy_smooth_polynomials(self, left_fit, right_fit) -> Tuple:
        """
        HEAVY temporal smoothing with HARD VALIDATION
        
        If new detection is too different from previous, REJECT it entirely
        """
        # Start with previous fits or defaults. Polynomial is x = a*y^2 + b*y + c,
        # so a straight vertical line at x = k must be [0, 0, k] — not [0, k, 0].
        if self.prev_left_fit is not None:
            left_smooth = self.prev_left_fit.copy()
        else:
            left_smooth = np.array([0.0, 0.0, 0.3 * self.image_width])

        if self.prev_right_fit is not None:
            right_smooth = self.prev_right_fit.copy()
        else:
            right_smooth = np.array([0.0, 0.0, 0.7 * self.image_width])
        
        # Validate new fits against previous
        if left_fit is not None and self.prev_left_fit is not None:
            # Check how different the new fit is at bottom of image
            y_eval = self.image_height - 1
            prev_x = self.prev_left_fit[0] * y_eval**2 + self.prev_left_fit[1] * y_eval + self.prev_left_fit[2]
            new_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            
            diff = abs(new_x - prev_x)
            
            # If too different (>50 pixels), REJECT new fit entirely
            if diff < 50:
                # Accept and blend slowly (90% previous, 10% current)
                self.left_fit_history.append(left_fit)
                if len(self.left_fit_history) >= 2:
                    recent = list(self.left_fit_history)[-3:]
                    left_avg = np.mean(recent, axis=0)
                    left_smooth = 0.9 * left_smooth + 0.1 * left_avg
        
        if right_fit is not None and self.prev_right_fit is not None:
            y_eval = self.image_height - 1
            prev_x = self.prev_right_fit[0] * y_eval**2 + self.prev_right_fit[1] * y_eval + self.prev_right_fit[2]
            new_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            
            diff = abs(new_x - prev_x)
            
            # If too different (>50 pixels), REJECT new fit entirely
            if diff < 50:
                self.right_fit_history.append(right_fit)
                if len(self.right_fit_history) >= 2:
                    recent = list(self.right_fit_history)[-3:]
                    right_avg = np.mean(recent, axis=0)
                    right_smooth = 0.9 * right_smooth + 0.1 * right_avg
        elif left_fit is not None:
            # First valid detection
            self.left_fit_history.append(left_fit)
            self.right_fit_history.append(right_fit)
            left_smooth = left_fit
            right_smooth = right_fit
        
        # Store as previous
        self.prev_left_fit = left_smooth.copy()
        self.prev_right_fit = right_smooth.copy()
        
        return left_smooth, right_smooth
    
    def _validate_result(self, result: LaneResult) -> LaneResult:
        """
        Validate lane detection - reject if unreasonable
        """
        if result.left_fit is not None and result.right_fit is not None:
            # Check lane width at bottom of image
            y_eval = self.image_height - 1
            leftx = result.left_fit[0] * y_eval**2 + result.left_fit[1] * y_eval + result.left_fit[2]
            rightx = result.right_fit[0] * y_eval**2 + result.right_fit[1] * y_eval + result.right_fit[2]
            
            lane_width = abs(rightx - leftx)
            
            # Reject if lane width is unreasonable
            if lane_width < 50 or lane_width > 900:
                # Use previous valid fits
                if self.prev_left_fit is not None:
                    result.left_fit = self.prev_left_fit.copy()
                if self.prev_right_fit is not None:
                    result.right_fit = self.prev_right_fit.copy()
        
        return result
    
    def _preprocess_and_transform(self, image: np.ndarray) -> np.ndarray:
        """
        STEP 1-4: Preprocessing + Perspective Transform
        
        1. HLS color space
        2. White + yellow thresholding
        3. Gaussian blur + Canny edges
        4. Dynamic trapezoidal ROI + Perspective transform
        """
        # Convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        h_channel, l_channel, s_channel = cv2.split(hls)
        
        # Color thresholds (preset-tunable). Indian-road / faded presets override
        # these to be more permissive so low-contrast markings survive the mask.
        s_min = self.config.get('s_min', 150)
        l_min = self.config.get('l_min', 180)
        yellow_s_min = self.config.get('yellow_s_min', 100)

        # White lane detection (high saturation + lightness)
        white_mask = np.zeros_like(s_channel, dtype=np.uint8)
        white_mask[(s_channel >= s_min) & (s_channel <= 255) &
                   (l_channel >= l_min) & (l_channel <= 255)] = 1

        # Yellow lane detection (specific hue range)
        yellow_mask = np.zeros_like(s_channel, dtype=np.uint8)
        yellow_mask[(h_channel >= 10) & (h_channel <= 35) &
                    (s_channel >= yellow_s_min) & (s_channel <= 255)] = 1
        
        # Combine masks
        color_binary = np.zeros_like(s_channel)
        color_binary[(white_mask == 1) | (yellow_mask == 1)] = 1
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(color_binary.astype(np.float32), (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny((blurred * 255).astype(np.uint8), 
                         self.config['canny_low'], 
                         self.config['canny_high'])
        
        # Combine color + edge detection
        combined = np.zeros_like(edges)
        combined[(color_binary == 1) | (edges > 0)] = 255
        
        # Apply dynamic trapezoidal ROI
        mask = self._create_dynamic_roi(combined.shape)
        roi_masked = cv2.bitwise_and(combined, combined, mask=mask)
        
        # STEP 4: Perspective Transform (Bird's Eye View)
        if self.M is None:
            self.M, self.Minv = self._get_perspective_transform(image.shape)
        
        warped = cv2.warpPerspective(roi_masked, self.M, 
                                     (self.image_width, self.image_height),
                                     flags=cv2.INTER_LINEAR)
        
        return warped
    
    def _create_dynamic_roi(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create dynamic trapezoidal ROI for road region"""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Dynamic ROI based on image dimensions
        # Top: 60% width at 35% height
        # Bottom: 95% width at 95% height
        top_width = width * 0.60
        bottom_width = width * 0.95
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
    
    def _get_perspective_transform(self, image_shape):
        """
        Calculate perspective transform matrices for bird's eye view
        Maps road region to top-down view for better curve detection
        """
        height, width = image_shape[:2]
        
        # Source points (trapezoid in original image)
        # These define the road region
        src = np.float32([
            [width * 0.05, height * 0.95],      # Bottom-left
            [width * 0.45, height * 0.35],      # Top-left
            [width * 0.55, height * 0.35],      # Top-right
            [width * 0.95, height * 0.95]       # Bottom-right
        ])
        
        # Destination points (rectangle in bird's eye view)
        offset_x = 300  # Padding
        offset_y = 0
        dst = np.float32([
            [offset_x, height],                          # Bottom-left
            [offset_x, offset_y],                        # Top-left
            [width - offset_x, offset_y],                # Top-right
            [width - offset_x, height]                   # Bottom-right
        ])
        
        # Calculate transform matrices
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        return M, Minv
    
    def _validate_fits(self, left_fit, right_fit) -> Tuple:
        """
        STEP 6: Validate polynomial fits - reject bad detections
        
        Checks:
        - Parallelism: lanes should be roughly parallel
        - Curvature: should not be too sharp
        - Distance: lanes should be reasonable distance apart
        """
        # If either fit is missing, use previous valid fit
        if left_fit is None:
            left_fit = self.prev_left_fit.copy() if self.prev_left_fit is not None else None
        
        if right_fit is None:
            right_fit = self.prev_right_fit.copy() if self.prev_right_fit is not None else None
        
        # If both fits available, validate them
        if left_fit is not None and right_fit is not None:
            # Check if lanes are too close or too far apart at bottom of image
            y_eval = self.image_height - 1
            leftx = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
            rightx = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
            
            lane_width = abs(rightx - leftx)
            
            # Reject if lane width is unreasonable (too narrow or too wide)
            if lane_width < 100 or lane_width > 800:  # Pixels
                # Use previous fits if current ones are bad
                left_fit = self.prev_left_fit.copy() if self.prev_left_fit is not None else left_fit
                right_fit = self.prev_right_fit.copy() if self.prev_right_fit is not None else right_fit
        
        return left_fit, right_fit
    
    def _find_lane_pixels_curved(self, binary_warped: np.ndarray) -> Tuple:
        """
        STEP 4-5: Optimized Sliding Window for Curved Roads
        
        Uses histogram peaks + adaptive sliding window to detect curved lanes
        Returns: (left_fit, right_fit) - 2nd degree polynomial coefficients
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = len(histogram) // 4
        
        # Find lane starting positions (left and right bottom)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Sliding window parameters - optimized for curves
        nwindows = 9
        window_height = np.int32(binary_warped.shape[0] / nwindows)
        margin = 120  # Wider margin for curves
        minpix = 40   # Lower threshold for curves
        
        # Get nonzero pixel indices
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current window positions
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Lists to store lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # STEP 4: Adaptive Sliding Window for Curves
        for window in range(nwindows):
            # Window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            # For curved roads, use previous polynomial to predict window position
            if window > 0 and len(left_lane_inds) > 0 and len(right_lane_inds) > 0:
                # Use recent polynomial to predict next window position
                if len(self.left_fit_history) > 0:
                    recent_left = self.left_fit_history[-1]
                    win_y_center = (win_y_low + win_y_high) // 2
                    leftx_current = int(recent_left[0] * win_y_center**2 + 
                                       recent_left[1] * win_y_center + recent_left[2])
                
                if len(self.right_fit_history) > 0:
                    recent_right = self.right_fit_history[-1]
                    win_y_center = (win_y_low + win_y_high) // 2
                    rightx_current = int(recent_right[0] * win_y_center**2 + 
                                        recent_right[1] * win_y_center + recent_right[2])
            
            # Window boundaries with margin
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Find pixels in windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter windows - for curves, use weighted average
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate indices
        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([])
        
        # Extract lane pixel positions
        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        
        # STEP 5: Polynomial Curve Fitting (2nd degree) - FOR CURVES
        left_fit = None
        right_fit = None
        
        if len(leftx) > 50 and len(lefty) > 50:
            left_fit = np.polyfit(lefty, leftx, 2)  # x = ay² + by + c
        
        if len(rightx) > 50 and len(righty) > 50:
            right_fit = np.polyfit(righty, rightx, 2)  # x = ay² + by + c
        
        return left_fit, right_fit
    
    def _smooth_polynomials(self, left_fit, right_fit) -> Tuple:
        """
        STEP 8: Temporal Smoothing
        
        Uses weighted average of recent polynomial fits to eliminate jitter
        """
        # Add current fits to history
        if left_fit is not None:
            self.left_fit_history.append(left_fit)
            self.prev_left_fit = left_fit.copy()
        
        if right_fit is not None:
            self.right_fit_history.append(right_fit)
            self.prev_right_fit = right_fit.copy()
        
        # Weighted average (recent frames matter more)
        if len(self.left_fit_history) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.left_fit_history))
            weights = weights / weights.sum()
            recent_fits = np.array(self.left_fit_history)
            left_fit_smooth = np.average(recent_fits, axis=0, weights=weights)
        elif self.prev_left_fit is not None:
            left_fit_smooth = self.prev_left_fit.copy()
        else:
            left_fit_smooth = None
        
        if len(self.right_fit_history) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.right_fit_history))
            weights = weights / weights.sum()
            recent_fits = np.array(self.right_fit_history)
            right_fit_smooth = np.average(recent_fits, axis=0, weights=weights)
        elif self.prev_right_fit is not None:
            right_fit_smooth = self.prev_right_fit.copy()
        else:
            right_fit_smooth = None
        
        return left_fit_smooth, right_fit_smooth
    
    def _generate_curved_output(self, image: np.ndarray, left_fit, right_fit) -> LaneResult:
        """
        STEP 6-7: Generate SMOOTH CURVED lane points and warp back to original view
        
        Creates 200 smooth points along polynomial curves for perfect curve visualization
        """
        height, width = image.shape[:2]
        
        # Generate HIGH-RESOLUTION points for smooth curves (200 points instead of 100)
        ploty = np.linspace(0, height - 1, num=200)
        left_points = None
        right_points = None
        left_lane = None
        right_lane = None
        
        # Calculate LEFT lane curve points
        if left_fit is not None:
            # Polynomial: x = ay² + by + c
            leftx_plot = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            
            # Clip to image bounds
            leftx_plot = np.clip(leftx_plot, 0, width - 1)
            
            # Warp back to original perspective if transform exists
            if self.Minv is not None:
                left_points_warped = np.column_stack((leftx_plot, ploty))
                left_points = cv2.perspectiveTransform(
                    np.array([left_points_warped], dtype=np.float32),
                    self.Minv
                )[0].astype(np.int32)
            else:
                left_points = np.column_stack((leftx_plot, ploty)).astype(np.int32)
            
            # Create lane line array (bottom to top)
            left_lane = np.array([
                int(leftx_plot[-1]), int(ploty[-1]),  # Bottom
                int(leftx_plot[0]), int(ploty[0])     # Top
            ])
        
        # Calculate RIGHT lane curve points
        if right_fit is not None:
            rightx_plot = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            
            # Clip to image bounds
            rightx_plot = np.clip(rightx_plot, 0, width - 1)
            
            if self.Minv is not None:
                right_points_warped = np.column_stack((rightx_plot, ploty))
                right_points = cv2.perspectiveTransform(
                    np.array([right_points_warped], dtype=np.float32),
                    self.Minv
                )[0].astype(np.int32)
            else:
                right_points = np.column_stack((rightx_plot, ploty)).astype(np.int32)
            
            # Create lane line array (bottom to top)
            right_lane = np.array([
                int(rightx_plot[-1]), int(ploty[-1]),  # Bottom
                int(rightx_plot[0]), int(ploty[0])     # Top
            ])
        
        # Calculate metrics
        lane_width, lane_width_m = self._calculate_lane_width(left_fit, right_fit, ploty)
        curvature = self._calculate_curvature(left_fit, right_fit)
        offset = self._calculate_vehicle_offset(left_fit, right_fit, width)
        confidence = self._calculate_confidence(left_fit, right_fit)
        
        return LaneResult(
            left_fit=left_fit,
            right_fit=right_fit,
            left_lane=left_lane,
            right_lane=right_lane,
            left_points=left_points,
            right_points=right_points,
            lane_width_pixels=lane_width,
            lane_width_meters=lane_width_m,
            curvature=curvature,
            vehicle_offset=offset,
            confidence=confidence,
            preset_name=self.preset,
        )
    
    def _handle_edge_cases(self, result, left_fit, right_fit) -> LaneResult:
        """
        STEP 9: Handle edge cases
        - Missing lanes → use previous frame
        - Curved roads → ensure smooth polynomial
        - Shadows → fallback to color thresholding
        """
        # If one lane missing, estimate from previous
        if result.left_points is None and self.prev_left_fit is not None:
            result.left_fit = self.prev_left_fit.copy()
            # Regenerate points
            ploty = np.linspace(0, self.image_height - 1, 100)
            leftx = self.prev_left_fit[0] * ploty**2 + self.prev_left_fit[1] * ploty + self.prev_left_fit[2]
            if self.Minv is not None:
                left_pts = cv2.perspectiveTransform(
                    np.column_stack((leftx, ploty)).astype(np.float32).reshape(1, -1, 2),
                    self.Minv
                )[0].astype(np.int32)
                result.left_points = left_pts
        
        if result.right_points is None and self.prev_right_fit is not None:
            result.right_fit = self.prev_right_fit.copy()
            ploty = np.linspace(0, self.image_height - 1, 100)
            rightx = self.prev_right_fit[0] * ploty**2 + self.prev_right_fit[1] * ploty + self.prev_right_fit[2]
            if self.Minv is not None:
                right_pts = cv2.perspectiveTransform(
                    np.column_stack((rightx, ploty)).astype(np.float32).reshape(1, -1, 2),
                    self.Minv
                )[0].astype(np.int32)
                result.right_points = right_pts
        
        return result
    
    def _calculate_lane_width(self, left_fit, right_fit, ploty) -> Tuple[float, float]:
        """Calculate lane width at bottom of image"""
        if left_fit is None or right_fit is None:
            return 0.0, 3.5
        
        y_eval = ploty[-1]
        leftx = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        rightx = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
        
        width_px = abs(rightx - leftx)
        # Approximate: 700px ≈ 3.5m (calibrated)
        width_m = (width_px / 700.0) * 3.5
        
        return width_px, width_m
    
    def _calculate_curvature(self, left_fit, right_fit) -> float:
        """Calculate road curvature in meters"""
        if left_fit is None and right_fit is None:
            return 0.0
        
        # Use average curvature
        fits = [f for f in [left_fit, right_fit] if f is not None]
        if not fits:
            return 0.0
        
        avg_fit = np.mean(fits, axis=0)
        y_eval = self.image_height - 1
        
        # Scale to meters
        ym_per_pix = 30 / self.image_height  # 30m visible
        xm_per_pix = 3.5 / 700  # 3.5m lane width
        
        fit_cr = np.polyfit(np.arange(self.image_height) * ym_per_pix,
                           (avg_fit[0] * np.arange(self.image_height)**2 + 
                            avg_fit[1] * np.arange(self.image_height) + 
                            avg_fit[2]) * xm_per_pix, 2)
        
        curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)**1.5) / abs(2 * fit_cr[0])
        return float(curvature)
    
    def _calculate_vehicle_offset(self, left_fit, right_fit, width) -> float:
        """Calculate vehicle offset from lane center"""
        if left_fit is None or right_fit is None:
            return 0.0
        
        y_eval = self.image_height - 1
        leftx = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
        rightx = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
        
        lane_center = (leftx + rightx) / 2
        vehicle_center = width / 2
        
        offset_px = vehicle_center - lane_center
        offset_m = (offset_px / 700.0) * 3.5
        
        return float(offset_m)
    
    def _calculate_confidence(self, left_fit, right_fit) -> float:
        """Calculate detection confidence (0-1)"""
        confidence = 0.0
        if left_fit is not None:
            confidence += 0.4
        if right_fit is not None:
            confidence += 0.4
        if len(self.left_fit_history) > 2 and len(self.right_fit_history) > 2:
            confidence += 0.2
        return min(confidence, 1.0)
    
    def draw_lanes(self, image: np.ndarray, result: LaneResult) -> np.ndarray:
        """
        Draw detected lanes - STABLE, FULL-WIDTH, bottom 20% only
        
        Visualization:
        - Solid filled driving region (not thin lines)
        - Restricted to bottom 20% of frame (cutoff_y = 80% height)
        - Exponential moving average for zero jitter
        - Handles missing lanes gracefully
        """
        output = image.copy()
        height, width = image.shape[:2]
        
        # STEP 2: Define bottom 20% region constraint
        cutoff_y = int(height * 0.80)  # Only draw below this line
        y_start = cutoff_y
        y_end = height
        
        # Generate Y points for the fixed bottom region
        ploty = np.linspace(y_start, y_end - 1, 50)
        
        # Get polynomial fits (current or fallback)
        left_fit = result.left_fit if result.left_fit is not None else self.prev_left_fit
        right_fit = result.right_fit if result.right_fit is not None else self.prev_right_fit
        
        # If still no fits, use default straight lanes spanning the lower-middle
        # region. Polynomial is x = a*y^2 + b*y + c, so a=0, b=0, c=x fixes the
        # line to a constant x position (was previously [0, k, 0] which produced
        # a diagonal that clipped to the frame edge and made the overlay vanish).
        if left_fit is None:
            left_fit = np.array([0.0, 0.0, 0.30 * width])
        if right_fit is None:
            right_fit = np.array([0.0, 0.0, 0.70 * width])
        
        # Calculate X positions
        leftx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # Clip to frame bounds
        leftx = np.clip(leftx, 0, width - 1)
        rightx = np.clip(rightx, 0, width - 1)
        
        # Create current lane points
        current_left_pts = np.column_stack((leftx, ploty)).astype(np.int32)
        current_right_pts = np.column_stack((rightx, ploty)).astype(np.int32)
        
        # STEP 1: Apply exponential moving average for stability
        if self.prev_left_pts is not None and len(self.prev_left_pts) == len(current_left_pts):
            # Smooth: new = alpha * current + (1 - alpha) * previous
            left_pts = cv2.addWeighted(current_left_pts, self.smoothing_alpha, 
                                      self.prev_left_pts, 1 - self.smoothing_alpha, 0).astype(np.int32)
            right_pts = cv2.addWeighted(current_right_pts, self.smoothing_alpha, 
                                       self.prev_right_pts, 1 - self.smoothing_alpha, 0).astype(np.int32)
        else:
            # First frame or size mismatch - use current
            left_pts = current_left_pts
            right_pts = current_right_pts
        
        # Store for next frame
        self.prev_left_pts = left_pts.copy()
        self.prev_right_pts = right_pts.copy()
        
        # STEP 3-4: Create filled polygon region (not thin lines)
        # Combine left and right boundaries to form driving region
        pts = np.vstack([
            left_pts,              # Left boundary (top to bottom)
            right_pts[::-1]        # Right boundary (bottom to top)
        ]).astype(np.int32)
        
        # Draw solid filled driving region
        lane_overlay = output.copy()
        cv2.fillPoly(lane_overlay, [pts], (0, 255, 0))
        
        # Blend with original (25% opacity - road still visible)
        cv2.addWeighted(lane_overlay, 0.25, output, 0.75, 0, output)
        
        # Draw thick boundary lines for clarity
        cv2.polylines(output, [left_pts], False, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.polylines(output, [right_pts], False, (0, 255, 0), 4, cv2.LINE_AA)
        
        return output


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_lane_detector(preset: str = 'default', debug: bool = False) -> AdvancedLaneDetector:
    """Factory function to create lane detector"""
    return AdvancedLaneDetector(preset=preset, debug=debug)
