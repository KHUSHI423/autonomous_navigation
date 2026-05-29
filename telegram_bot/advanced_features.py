"""
=============================================================================
🚀 ADVANCED FEATURES MODULE - EDGEDRIVE3D TELEGRAM BOT
=============================================================================
Unique Hackathon Features:
- 🎤 Speech Recognition (Voice Commands)
- 🖼️ Image Analysis & Object Detection Reports
- 📊 Advanced Telemetry with Charts
- 🎯 Smart Mission Execution
- 🔔 Intelligent Alert System
- 📈 Analytics Dashboard
- 🤖 AI Decision Visualizations
=============================================================================
"""

import os
import io
import asyncio
import logging
import time
import socket
import struct
import numpy as np
import cv2
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from collections import deque
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup, 
    InputMediaPhoto,
    Location
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
    JobQueue
)
from telegram.constants import ParseMode

# YOLO for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """Telemetry data point"""
    timestamp: datetime
    throttle: int
    steering: float
    speed: float
    heading: float
    latitude: float
    longitude: float
    satellites: int
    objects_detected: int
    decision: str
    battery_voltage: float = 12.0


class AdvancedTelemetry:
    """Advanced telemetry with chart generation"""
    
    def __init__(self, max_points: int = 100):
        self.data_points: deque = deque(maxlen=max_points)
        self.chart_dir = Path(__file__).parent / "charts"
        self.chart_dir.mkdir(exist_ok=True)
        
    def add_point(self, telemetry: TelemetryData):
        """Add telemetry data point"""
        self.data_points.append(telemetry)
    
    def generate_speed_chart(self) -> Optional[bytes]:
        """Generate speed over time chart"""
        if len(self.data_points) < 2:
            return None
        
        times = [dp.timestamp for dp in self.data_points]
        speeds = [dp.speed for dp in self.data_points]
        throttles = [dp.throttle for dp in self.data_points]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Speed chart
        ax1.plot(times, speeds, 'g-', linewidth=2, label='Speed (m/s)')
        ax1.fill_between(times, speeds, alpha=0.3, color='green')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('🚀 Speed & Throttle Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throttle chart
        ax2.plot(times, throttles, 'b-', linewidth=2, label='Throttle')
        ax2.fill_between(times, throttles, alpha=0.3, color='blue')
        ax2.set_ylabel('Throttle')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def generate_position_chart(self) -> Optional[bytes]:
        """Generate GPS position track"""
        positions = [(dp.latitude, dp.longitude) for dp in self.data_points if dp.latitude]
        
        if len(positions) < 2:
            return None
        
        lats = [p[0] for p in positions]
        lons = [p[1] for p in positions]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot track
        ax.plot(lons, lats, 'r-', linewidth=2, label='Track', marker='o', markersize=3)
        
        # Mark start and end
        ax.plot(lons[0], lats[0], 'go', markersize=15, label='Start', zorder=5)
        ax.plot(lons[-1], lats[-1], 'ro', markersize=15, label='End', zorder=5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('🗺️ GPS Track')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def generate_decision_pie_chart(self) -> Optional[bytes]:
        """Generate decision distribution pie chart"""
        if len(self.data_points) < 5:
            return None
        
        decisions = {}
        for dp in self.data_points:
            decisions[dp.decision] = decisions.get(dp.decision, 0) + 1
        
        labels = list(decisions.keys())
        sizes = list(decisions.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title('🧠 Decision Distribution')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if not self.data_points:
            return {}
        
        speeds = [dp.speed for dp in self.data_points]
        throttles = [dp.throttle for dp in self.data_points]
        
        return {
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'avg_throttle': np.mean(throttles),
            'total_points': len(self.data_points),
            'duration': (self.data_points[-1].timestamp - self.data_points[0].timestamp).total_seconds()
        }


class VisionAnalyzer:
    """AI Vision analysis for images"""
    
    def __init__(self, model_path: str = "../../yolov8n.pt"):
        self.model = None
        if YOLO_AVAILABLE and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                logger.info("✅ YOLO model loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
    
    def analyze_image(self, image_bytes: bytes) -> Dict:
        """Analyze image and return detections"""
        if not self.model:
            return {'error': 'Model not loaded', 'objects': []}
        
        try:
            # Convert bytes to numpy
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'error': 'Invalid image', 'objects': []}
            
            # Run detection
            results = self.model(frame, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1+x2)//2, (y1+y2)//2]
                })
            
            return {
                'objects': detections,
                'count': len(detections),
                'image_size': frame.shape
            }
        
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {'error': str(e), 'objects': []}
    
    def generate_analysis_report(self, detections: Dict) -> str:
        """Generate human-readable analysis report"""
        if 'error' in detections:
            return f"❌ Analysis Error: {detections['error']}"
        
        objects = detections.get('objects', [])
        
        if not objects:
            return "📷 Image Analysis: No objects detected"
        
        report = f"🔍 *Vision Analysis Report*\n\n"
        report += f"📊 Objects Detected: {len(objects)}\n\n"
        
        # Group by class
        by_class = {}
        for obj in objects:
            cls = obj['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(obj['confidence'])
        
        for cls, confs in sorted(by_class.items()):
            avg_conf = np.mean(confs)
            count = len(confs)
            emoji = self._get_emoji_for_class(cls)
            report += f"{emoji} {cls.title()}: {count} detected (avg: {avg_conf:.0%})\n"
        
        report += f"\n📐 Image Size: {detections.get('image_size', ['?'])[0]}x{detections.get('image_size', ['?', '?'])[1]}"
        
        return report
    
    def _get_emoji_for_class(self, cls_name: str) -> str:
        """Get emoji for object class"""
        emojis = {
            'person': '👤',
            'car': '🚗',
            'motorcycle': '🏍️',
            'bus': '🚌',
            'truck': '🚛',
            'bicycle': '🚲',
            'dog': '🐕',
            'cat': '🐈',
            'bird': '🐦',
            'traffic light': '🚦',
            'stop sign': '🛑',
        }
        return emojis.get(cls_name.lower(), '🎯')


class SmartAlerts:
    """Intelligent alert system"""
    
    def __init__(self):
        self.alert_history: deque = deque(maxlen=50)
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_handlers = {}
        
    def register_alert(self, alert_type: str, handler):
        """Register alert handler"""
        self.alert_handlers[alert_type] = handler
    
    def check_alerts(self, telemetry: TelemetryData) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        now = datetime.now()
        
        # Low battery alert
        if telemetry.battery_voltage < 11.5:
            if self._can_alert('low_battery'):
                alerts.append({
                    'type': 'low_battery',
                    'priority': 'high',
                    'message': f"🔋 LOW BATTERY: {telemetry.battery_voltage:.1f}V",
                    'emoji': '🔋'
                })
        
        # High speed alert
        if telemetry.speed > 5.0:
            if self._can_alert('high_speed'):
                alerts.append({
                    'type': 'high_speed',
                    'priority': 'medium',
                    'message': f"🚀 HIGH SPEED: {telemetry.speed:.1f} m/s",
                    'emoji': '🚀'
                })
        
        # GPS signal loss
        if telemetry.satellites < 4:
            if self._can_alert('gps_loss'):
                alerts.append({
                    'type': 'gps_loss',
                    'priority': 'high',
                    'message': f"📡 GPS SIGNAL WEAK: {telemetry.satellites} satellites",
                    'emoji': '📡'
                })
        
        # Emergency stop
        if telemetry.throttle == 0 and telemetry.speed > 0.1:
            if self._can_alert('emergency_stop'):
                alerts.append({
                    'type': 'emergency_stop',
                    'priority': 'critical',
                    'message': "🛑 EMERGENCY STOP DETECTED",
                    'emoji': '🛑'
                })
        
        # Record alerts
        for alert in alerts:
            self.alert_history.append({
                **alert,
                'timestamp': now
            })
        
        return alerts
    
    def _can_alert(self, alert_type: str) -> bool:
        """Check if alert is not on cooldown"""
        now = datetime.now()
        last_alert = self.alert_cooldowns.get(alert_type)
        
        if last_alert is None or (now - last_alert).total_seconds() > 60:
            self.alert_cooldowns[alert_type] = now
            return True
        return False
    
    def get_alert_summary(self) -> str:
        """Get alert history summary"""
        if not self.alert_history:
            return "✅ No alerts in history"
        
        summary = "🚨 *Alert History*\n\n"
        
        # Count by type
        by_type = {}
        for alert in self.alert_history:
            t = alert['type']
            by_type[t] = by_type.get(t, 0) + 1
        
        for alert_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            emoji = self.alert_history[-1].get('emoji', '⚠️')
            summary += f"{emoji} {alert_type.replace('_', ' ').title()}: {count}\n"
        
        summary += f"\n📊 Total Alerts: {len(self.alert_history)}"
        
        return summary


class MissionPlanner:
    """Advanced mission planning and execution"""
    
    def __init__(self):
        self.waypoints: List[Dict] = []
        self.current_waypoint = 0
        self.mission_active = False
        self.mission_log: deque = deque(maxlen=100)
    
    def add_waypoint(self, latitude: float, longitude: float, 
                     altitude: float = 0.0, action: str = "NAVIGATE") -> int:
        """Add waypoint to mission"""
        waypoint = {
            'id': len(self.waypoints) + 1,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'action': action,
            'reached': False,
            'timestamp': datetime.now()
        }
        self.waypoints.append(waypoint)
        return waypoint['id']
    
    def clear_mission(self):
        """Clear all waypoints"""
        self.waypoints = []
        self.current_waypoint = 0
        self.mission_active = False
    
    def start_mission(self) -> bool:
        """Start mission execution"""
        if not self.waypoints:
            return False
        self.mission_active = True
        self.current_waypoint = 0
        self._log("Mission started")
        return True
    
    def abort_mission(self):
        """Abort mission"""
        self.mission_active = False
        self._log("Mission aborted")
    
    def update_progress(self, current_lat: float, current_lon: float) -> Optional[Dict]:
        """Update mission progress"""
        if not self.mission_active or self.current_waypoint >= len(self.waypoints):
            return None
        
        target = self.waypoints[self.current_waypoint]
        distance = self._calculate_distance(
            current_lat, current_lon,
            target['latitude'], target['longitude']
        )
        
        # Check if waypoint reached (within 5 meters)
        if distance < 5.0:
            target['reached'] = True
            self._log(f"Waypoint {self.current_waypoint + 1} reached")
            self.current_waypoint += 1
            
            if self.current_waypoint >= len(self.waypoints):
                self.mission_active = False
                self._log("Mission completed!")
                return {'status': 'completed', 'waypoints_reached': self.current_waypoint}
            
            return {
                'status': 'waypoint_reached',
                'current': self.current_waypoint + 1,
                'total': len(self.waypoints)
            }
        
        return {
            'status': 'en_route',
            'current': self.current_waypoint + 1,
            'total': len(self.waypoints),
            'distance_to_target': distance
        }
    
    def get_mission_status(self) -> Dict:
        """Get current mission status"""
        return {
            'active': self.mission_active,
            'waypoints': len(self.waypoints),
            'current': self.current_waypoint + 1,
            'completed': sum(1 for wp in self.waypoints if wp['reached']),
            'log': list(self.mission_log)[-10:]
        }
    
    def _log(self, message: str):
        """Log mission event"""
        self.mission_log.append({
            'timestamp': datetime.now(),
            'message': message
        })
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Haversine distance"""
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c


class AnalyticsDashboard:
    """Analytics and statistics dashboard"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.total_commands = 0
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.detection_counts = {}
    
    def record_command(self, command: str):
        """Record command execution"""
        self.total_commands += 1
    
    def record_speed(self, speed: float):
        """Record speed for statistics"""
        if speed > self.max_speed:
            self.max_speed = speed
    
    def record_detection(self, object_class: str, count: int):
        """Record object detections"""
        self.detection_counts[object_class] = \
            self.detection_counts.get(object_class, 0) + count
    
    def get_session_report(self) -> str:
        """Generate session analytics report"""
        duration = datetime.now() - self.session_start
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        report = f"""
📊 *Session Analytics Report*
━━━━━━━━━━━━━━━━━━━━━━

⏱️ *Duration:* {hours:02d}:{minutes:02d}:{seconds:02d}
🎮 *Commands:* {self.total_commands}
🚀 *Max Speed:* {self.max_speed:.1f} m/s
📍 *Distance:* {self.total_distance:.1f} m

🔍 *Object Detections:*
        """
        
        if self.detection_counts:
            for cls, count in sorted(self.detection_counts.items(), 
                                    key=lambda x: -x[1]):
                report += f"\n• {cls.title()}: {count}"
        else:
            report += "\nNo objects detected"
        
        report += f"""

━━━━━━━━━━━━━━━━━━━━━━
🤖 *EdgeDrive3D Analytics*
Session ID: {id(self) % 10000}
        """
        
        return report


async def advanced_analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Advanced analytics command handler"""
    analytics = context.user_data.get('analytics')
    
    if not analytics:
        await update.message.reply_text("📊 No analytics data available")
        return
    
    report = analytics.get_session_report()
    await update.message.reply_text(report, parse_mode=ParseMode.HTML)


# Export classes for use in main bot
__all__ = [
    'AdvancedTelemetry',
    'VisionAnalyzer', 
    'SmartAlerts',
    'MissionPlanner',
    'AnalyticsDashboard',
    'TelemetryData'
]
