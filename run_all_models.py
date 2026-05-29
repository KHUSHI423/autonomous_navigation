"""
Edge Drive 3D - real-time road safety pipeline.

YOLOv8 object detection + classical lane detection (HLS + Canny + sliding
window) + pinhole distance + proximity warnings. Tuned for a clean, stable
output: green driving lane, color-coded object boxes with distance, a
lane-departure badge, and a bottom banner for close obstacles.

Usage:
    python run_all_models.py video clip.mp4 --lane-preset indian_road
    python run_all_models.py webcam --model yolov8n.pt
    python run_all_models.py video clip.mp4 --save
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.advanced_lane_detector import AdvancedLaneDetector


# Road-relevant COCO classes and their BGR colors.
ROAD_CLASSES = {
    0:  ('person',        (0, 200, 255)),
    1:  ('bicycle',       (255, 165,   0)),
    2:  ('car',           (0,   255,   0)),
    3:  ('motorcycle',    (255,   0, 255)),
    5:  ('bus',           (255,   0,   0)),
    7:  ('truck',         (0,   165, 255)),
    9:  ('traffic light', (0,     0, 255)),
    11: ('stop sign',     (0,     0, 200)),
}

# Real-world heights (metres) for the pinhole distance fallback.
CLASS_HEIGHTS = {
    'person': 1.70, 'bicycle': 1.10, 'car': 1.50, 'motorcycle': 1.30,
    'bus': 3.20, 'truck': 3.00, 'traffic light': 0.80, 'stop sign': 0.60,
}

# Proximity thresholds (metres).
WARN_DISTANCE     = 15.0
BRAKE_DISTANCE    = 7.0
CRITICAL_DISTANCE = 3.0

# Lane-departure threshold (metres from lane centre).
LANE_DEPARTURE_OFFSET = 0.35

# Camera model.
FOV_DEG       = 70.0
CAMERA_HEIGHT = 1.5  # metres above road


class RoadSafetySystem:
    """
    Self-contained pipeline: YOLO detector + classical lane detector +
    pinhole distance + overlay renderer. No MiDaS / BEV / point cloud.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        print("\n" + "=" * 70)
        print("  EDGE DRIVE 3D - ROAD SAFETY PIPELINE")
        print("=" * 70)
        print(f"  [1/2] Loading YOLO ({cfg['yolo_model']})...")
        self.yolo = YOLO(cfg['yolo_model'])
        print(f"  [2/2] Loading Lane Detector (preset: {cfg['lane_preset']})...")
        self.lane = AdvancedLaneDetector(preset=cfg['lane_preset'])
        print("=" * 70 + "\n")

        # Camera intrinsics (set lazily on the first frame).
        self.fx = None
        self.cy = None

        # Runtime state.
        self.frame_count = 0
        self.frame_times = deque(maxlen=30)

        # EMA smoothing state for the lane polygon.
        self._prev_center = None
        self._lane_alpha = 0.30
        self._lane_drawn = False
        self._lane_is_detected = False

        # Remember the last confidently-detected centre so brief detector
        # failures keep the overlay on the road instead of snapping back
        # to the frame midline.
        self._last_good_center = None
        self._frames_since_detection = 10_000
        self._stale_limit = 60  # ~2 s at 30 FPS

    # ---------------------------------------------------------------
    # frame processing
    # ---------------------------------------------------------------

    def process_frame(self, frame):
        t0 = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]
        if self.fx is None:
            self._calibrate(w, h)

        lane_result = self.lane.detect(frame)

        output = frame.copy()
        output = self._draw_lane_overlay(output, lane_result)

        detections = self._detect_objects(frame)

        nearest = None
        for d in detections:
            if d['distance'] > 0 and (nearest is None or d['distance'] < nearest['distance']):
                nearest = d

        for d in detections:
            is_focus = (d is nearest) and d['distance'] < WARN_DISTANCE
            self._draw_object(output, d, is_focus)

        action = 'FORWARD'
        if nearest and nearest['distance'] < CRITICAL_DISTANCE:
            action = 'STOP'
        elif nearest and nearest['distance'] < BRAKE_DISTANCE:
            action = 'SLOW DOWN'

        dt = time.time() - t0
        self.frame_times.append(dt)

        self._draw_hud(output, lane_result, nearest, action)
        self._draw_lane_departure_badge(output, lane_result)
        self._draw_proximity_banner(output, nearest)

        return output

    # ---------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------

    def _calibrate(self, w, h):
        fov_rad = np.radians(FOV_DEG)
        self.fx = w / (2 * np.tan(fov_rad / 2))
        self.cy = h / 2

    def _detect_objects(self, frame):
        r = self.yolo(frame, conf=self.cfg['confidence'], verbose=False)[0]
        raw = []
        for b in r.boxes:
            cid = int(b.cls[0])
            if cid not in ROAD_CLASSES:
                continue
            name, color = ROAD_CLASSES[cid]
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            raw.append({
                'name': name,
                'color': color,
                'conf': float(b.conf[0]),
                'bbox': (x1, y1, x2, y2),
                'distance': self._distance(name, x1, y1, x2, y2),
            })
        return self._suppress_duplicates(raw, iou_thresh=0.4)

    @staticmethod
    def _suppress_duplicates(dets, iou_thresh=0.5):
        """Simple class-agnostic NMS to drop near-duplicate boxes that slip
        through YOLO's built-in suppression when two classes overlap."""
        dets = sorted(dets, key=lambda d: d['conf'], reverse=True)
        kept = []
        for d in dets:
            ax1, ay1, ax2, ay2 = d['bbox']
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            duplicate = False
            for k in kept:
                bx1, by1, bx2, by2 = k['bbox']
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter == 0:
                    continue
                b_area = max(1, (bx2 - bx1) * (by2 - by1))
                iou = inter / (a_area + b_area - inter)
                if iou > iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(d)
        return kept

    def _distance(self, name, x1, y1, x2, y2):
        # Ground-plane pinhole first (uses bbox bottom + fixed camera height).
        if y2 > self.cy + 5:
            d = CAMERA_HEIGHT * self.fx / (y2 - self.cy)
            if 0.5 < d < 100:
                return d
        # Fallback: class-height pinhole (uses bbox height).
        real_h = CLASS_HEIGHTS.get(name)
        bbox_h = y2 - y1
        if real_h and bbox_h > 3:
            return real_h * self.fx / bbox_h
        return 0.0

    def _proximity_color(self, d):
        if d < CRITICAL_DISTANCE:
            return (0, 0, 255)      # red
        if d < BRAKE_DISTANCE:
            return (0, 140, 255)    # orange
        return (0, 220, 255)        # amber

    # ---------------------------------------------------------------
    # drawing
    # ---------------------------------------------------------------

    def _draw_lane_overlay(self, frame, lane_result):
        """
        Filled green driving-lane polygon with yellow boundary lines.

        Detection priority:
          1. Classical lane detector with a sane centre curve (painted
             lines). Marks the lane as truly detected.
          2. Last good detection within ~2 s — for brief dropouts.
          3. Default straight-ahead trapezoid — keeps the overlay on
             screen even when the detector can't find painted lines.
        """
        h, w = frame.shape[:2]

        center = self._extract_lane_center(lane_result, h, w)
        self._lane_is_detected = center is not None

        if center is not None:
            self._last_good_center = center.copy()
            self._frames_since_detection = 0
        else:
            self._frames_since_detection += 1
            if (self._last_good_center is not None
                    and self._frames_since_detection <= self._stale_limit):
                center = self._last_good_center
            else:
                center = self._default_lane_center(h, w)

        # Temporal smoothing of the centre path.
        if self._prev_center is not None and len(self._prev_center) == len(center):
            center = (self._lane_alpha * center
                      + (1 - self._lane_alpha) * self._prev_center)
        self._prev_center = center.copy()

        # Taper: narrower near the vanishing point, wider at the bottom.
        taper = np.linspace(0.08, 0.22, len(center)) * w
        xs = center[:, 0]
        ys = center[:, 1]
        left_pts  = np.stack([xs - taper, ys], axis=1).astype(np.int32)
        right_pts = np.stack([xs + taper, ys], axis=1).astype(np.int32)

        poly = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], (0, 255, 120))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        cv2.polylines(frame, [left_pts],  False, (0, 240, 255), 4, cv2.LINE_AA)
        cv2.polylines(frame, [right_pts], False, (0, 240, 255), 4, cv2.LINE_AA)
        self._lane_drawn = True
        return frame

    @staticmethod
    def _extract_lane_center(lane_result, h, w):
        """
        Derive a smoothed centre curve from the detector output. Returns
        None when the detector's output is missing or obviously broken.
        """
        left = lane_result.left_points
        right = lane_result.right_points
        if left is None or right is None or len(left) < 5 or len(right) < 5:
            return None

        n = min(len(left), len(right))
        center = (left[:n].astype(np.float32) + right[:n].astype(np.float32)) / 2.0

        y_min = int(h * 0.58)
        center = center[center[:, 1] >= y_min]
        if len(center) < 5:
            return None

        center = center[np.argsort(center[:, 1])]

        cx = np.mean(center[:, 0])
        if cx < w * 0.28 or cx > w * 0.72:
            return None

        return center

    @staticmethod
    def _default_lane_center(h, w):
        """
        Straight-ahead default ego-lane, used when the detector has no
        confident fix. Runs from ~60% of the frame height down to the
        bottom along the horizontal midline, so the overlay stays visible
        even on roads without painted markings.
        """
        y_top = int(h * 0.60)
        y_bot = h - 1
        ys = np.linspace(y_top, y_bot, 60)
        xs = np.full_like(ys, w / 2.0)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def _draw_object(self, frame, det, is_focus):
        x1, y1, x2, y2 = det['bbox']
        color = det['color']

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Class name only — distance is surfaced via the HUD and banner.
        label = det['name'].capitalize()
        self._draw_label(frame, label, det['bbox'], color)

        if is_focus:
            self._draw_bracket(frame, det['bbox'], self._proximity_color(det['distance']))

    def _draw_label(self, frame, text, bbox, color):
        x1, y1, x2, y2 = bbox
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pad = 5
        # Place label above the box, flipping below if there isn't room.
        if y1 - th - 2 * pad > 0:
            box_y1, box_y2 = y1 - th - 2 * pad, y1
            text_y = y1 - pad
        else:
            box_y1, box_y2 = y2, y2 + th + 2 * pad
            text_y = y2 + th + pad
        cv2.rectangle(frame, (x1, box_y1), (x1 + tw + 2 * pad, box_y2), color, -1)
        cv2.putText(frame, text, (x1 + pad, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_bracket(self, frame, bbox, color):
        x1, y1, x2, y2 = bbox
        L = max(18, int(0.15 * max(1, x2 - x1)))
        t = 4
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)

    def _draw_hud(self, frame, lane, nearest, action):
        x_pos, y_base, lh = 20, 38, 26
        rows = [("EDGE DRIVE 3D", (0, 255, 255), 0.65, 2)]

        if self._lane_is_detected and lane and lane.confidence > 0.2:
            offset = lane.vehicle_offset
            if abs(offset) < 0.20:
                rows.append(("Lane: CENTERED", (0, 255, 0), 0.52, 1))
            else:
                side = "LEFT" if offset < 0 else "RIGHT"
                rows.append((f"Lane: {side} {abs(offset):.2f}m",
                             (0, 165, 255), 0.52, 1))
        elif self._lane_drawn:
            rows.append(("Lane: TRACKING", (0, 200, 0), 0.52, 1))
        else:
            rows.append(("Lane: searching", (160, 160, 160), 0.52, 1))

        if nearest and nearest['distance'] > 0:
            rows.append((f"Nearest: {nearest['name']} @ {nearest['distance']:.1f}m",
                         self._proximity_color(nearest['distance']), 0.52, 1))
        else:
            rows.append(("Path: CLEAR", (0, 255, 0), 0.52, 1))

        action_color = {
            'FORWARD':   (0, 255, 0),
            'SLOW DOWN': (0, 140, 255),
            'STOP':      (0, 0, 255),
        }[action]
        rows.append((f"Action: {action}", action_color, 0.58, 2))

        panel_h = 16 + lh * len(rows)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 16), (345, 16 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        y = y_base
        for text, color, scale, thick in rows:
            cv2.putText(frame, text, (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)
            y += lh

    def _draw_lane_departure_badge(self, frame, lane):
        # Only fire when an actual lane was detected (not the fallback).
        if not self._lane_is_detected:
            return
        if not lane or lane.confidence < 0.3:
            return
        offset = lane.vehicle_offset
        if abs(offset) <= LANE_DEPARTURE_OFFSET:
            return

        side = "LEFT" if offset < 0 else "RIGHT"
        text = f"LANE DEPARTURE - {side}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        h, w = frame.shape[:2]
        cx = w // 2
        pad_x, pad_y = 24, 12
        bx1, by1 = cx - tw // 2 - pad_x, 20
        bx2, by2 = cx + tw // 2 + pad_x, 20 + th + 2 * pad_y

        pulse = 0.55 + 0.35 * abs(np.sin(self.frame_count * 0.35))
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 140, 255), -1)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (bx1 + pad_x, by2 - pad_y),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def _draw_proximity_banner(self, frame, nearest):
        if not nearest or nearest['distance'] >= WARN_DISTANCE:
            return
        d = nearest['distance']
        if d < CRITICAL_DISTANCE:
            headline, bg = "COLLISION RISK", (0, 0, 255)
        elif d < BRAKE_DISTANCE:
            headline, bg = "BRAKE NOW",      (0, 0, 255)
        else:
            headline, bg = "OBJECT NEAR",    (0, 140, 255)

        text = f"{headline}  -  {nearest['name'].upper()} at {d:.1f} m"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        h, w = frame.shape[:2]
        px, py = 28, 18
        bx1 = max(0, (w - tw) // 2 - px)
        bx2 = min(w, (w + tw) // 2 + px)
        by2 = h - 30
        by1 = by2 - (th + 2 * py)

        pulse = 0.85
        if d < CRITICAL_DISTANCE:
            pulse = 0.55 + 0.35 * abs(np.sin(self.frame_count * 0.35))

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg, -1)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2, cv2.LINE_AA)

        tx, ty = bx1 + px, by2 - py
        cv2.putText(frame, text, (tx, ty), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # ---------------------------------------------------------------
    # keyboard
    # ---------------------------------------------------------------

    def handle_key(self, key, last_frame):
        if key in (ord('q'), ord('Q'), 27):
            return True
        if key == ord('s'):
            if last_frame is None:
                print("  [WARN] no frame to save yet")
            else:
                fn = f"screenshot_{self.frame_count}.jpg"
                cv2.imwrite(fn, last_frame)
                print(f"  [OK] screenshot saved: {fn}")
        return False


# -----------------------------------------------------------------------
# runners
# -----------------------------------------------------------------------

def _open_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps if fps > 0 else 25.0, size)


def run_video(video_path, cfg):
    print("\n" + "=" * 70)
    print("RUNNING ON VIDEO")
    print("=" * 70)

    system = RoadSafetySystem(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.1f} FPS, {total} frames")
    print(f"{'=' * 70}")
    print("Keys:  Q=quit   P=pause/resume   S=screenshot")
    print(f"{'=' * 70}\n")

    writer = None
    if cfg.get('save'):
        os.makedirs('output', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join('output', f'detection_{ts}.mp4')
        writer = _open_writer(out_path, fps, (w, h))
        print(f"Saving annotated video to: {out_path}\n")

    paused = False
    last_frame = None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("\n[END] video finished")
                break

            annotated = system.process_frame(frame)
            last_frame = annotated
            cv2.imshow("Edge Drive 3D - Detection", annotated)
            if writer is not None:
                writer.write(annotated)

            if system.frame_count % 30 == 0 and system.frame_times:
                cur_fps = len(system.frame_times) / sum(system.frame_times)
                print(f"Frame {system.frame_count}/{total} | FPS: {cur_fps:.1f}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused
            print(f"\n[{'PAUSED' if paused else 'RESUMED'}]")
            continue
        if system.handle_key(key, last_frame):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if system.frame_times:
        avg_fps = len(system.frame_times) / sum(system.frame_times)
        print(f"\n{'=' * 70}")
        print(f"SUMMARY: {system.frame_count} frames | avg FPS: {avg_fps:.1f}")
        print(f"{'=' * 70}")


def run_webcam(cfg):
    print("\n" + "=" * 70)
    print("RUNNING ON WEBCAM (real-time)")
    print("=" * 70)

    system = RoadSafetySystem(cfg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"{'=' * 70}")
    print("Keys:  Q=quit   S=screenshot")
    print(f"{'=' * 70}\n")

    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] failed to grab frame")
            break

        annotated = system.process_frame(frame)
        last_frame = annotated
        cv2.imshow("Edge Drive 3D - Webcam", annotated)

        if system.frame_count % 60 == 0 and system.frame_times:
            cur_fps = len(system.frame_times) / sum(system.frame_times)
            print(f"Frame {system.frame_count} | FPS: {cur_fps:.1f}")

        if system.handle_key(cv2.waitKey(1) & 0xFF, last_frame):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("\n" + "=" * 70)
    print("   EDGE DRIVE 3D - ROAD SAFETY PIPELINE")
    print("=" * 70)

    p = argparse.ArgumentParser(description='Edge Drive 3D road safety pipeline')
    p.add_argument('mode', choices=['video', 'webcam'])
    p.add_argument('input', nargs='?', help='Video file (video mode)')
    p.add_argument('--model', default='yolov8n.pt', help='YOLO weights')
    p.add_argument('--confidence', type=float, default=0.4, help='YOLO confidence')
    p.add_argument('--lane-preset', default='indian_road',
                   choices=['default', 'highway', 'city', 'faded', 'night', 'indian_road'])
    p.add_argument('--save', action='store_true', help='Save annotated video (video mode)')
    args = p.parse_args()

    cfg = {
        'yolo_model':  args.model,
        'confidence':  args.confidence,
        'lane_preset': args.lane_preset,
        'save':        args.save,
    }

    print(f"\nConfiguration:")
    print(f"  YOLO Model:  {cfg['yolo_model']}")
    print(f"  Confidence:  {cfg['confidence']}")
    print(f"  Lane Preset: {cfg['lane_preset']}")
    if cfg['save']:
        print(f"  Save output: yes")

    if args.mode == 'video':
        if not args.input:
            print("\n[ERR] video mode needs an input file.")
            print("Usage: python run_all_models.py video your_video.mp4")
            return
        run_video(args.input, cfg)
    else:
        run_webcam(cfg)


if __name__ == "__main__":
    main()
