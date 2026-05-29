"""
=============================================================================
HARDWARE INTEGRATION - RASPBERRY PI SENDER & LAPTOP RECEIVER
=============================================================================
UDP-based video streaming from Raspberry Pi 3B+ to Laptop for ML processing
with intelligent decision feedback to ESP32 motor controller

Author: EdgeDrive3D Team
=============================================================================
"""

import cv2
import socket
import numpy as np
import struct
import time
import argparse
import threading
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import json


# ============================================================================
# RASPBERRY PI CAMERA SENDER
# ============================================================================

class PiCameraSender:
    """
    Captures webcam frames on Raspberry Pi and sends via UDP to laptop
    """
    
    def __init__(self, receiver_ip: str, port: int = 5000, camera_id: int = 0,
                 width: int = 640, height: int = 480, quality: int = 80):
        self.receiver_ip = receiver_ip
        self.port = port
        self.quality = quality
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        
        self.frame_count = 0
        self.running = False
        
        print(f"✓ Camera: {self.width}x{self.height}")
        print(f"✓ Sending to: {receiver_ip}:{port}")
    
    def start(self):
        """Start streaming"""
        print("\n" + "="*60)
        print("  🎥 Pi Camera Sender")
        print("="*60)
        print("Streaming... Press 'q' to stop\n")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Encode as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                _, jpeg_data = cv2.imencode('.jpg', frame, encode_param)
                
                # Pack: size (4 bytes) + jpeg data
                data = struct.pack("I", len(jpeg_data)) + jpeg_data.tobytes()
                
                # Send
                self.sock.sendto(data, (self.receiver_ip, self.port))
                
                self.frame_count += 1
                
                # Stats
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    print(f"📊 FPS: {fps:.1f} | Frames: {self.frame_count}")
                    start_time = time.time()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop streaming"""
        self.running = False
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.cap.release()
        self.sock.close()
        print(f"Total frames sent: {self.frame_count}")


# ============================================================================
# LAPTOP UDP RECEIVER
# ============================================================================

class LaptopReceiver:
    """
    Receives UDP stream from Pi and processes with ML
    """
    
    def __init__(self, udp_port: int = 5000, model_size: str = 'yolov8m.pt',
                 confidence: float = 0.4, save_output: bool = False,
                 output_dir: str = "output"):
        self.udp_port = udp_port
        self.save_output = save_output
        self.output_dir = Path(output_dir)
        
        if save_output:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        self.sock.bind(("0.0.0.0", udp_port))
        
        print(f"✓ Listening on UDP port {udp_port}")
        
        # Import perception engine
        from core.perception_engine import PerceptionEngine
        
        self.engine = PerceptionEngine({
            'yolo_model': model_size,
            'confidence': confidence,
        })
        
        self.frame_count = 0
        self.running = False
        self.callback = None
    
    def set_callback(self, callback: Callable):
        """Set callback for processed results"""
        self.callback = callback
    
    def start(self):
        """Start receiving and processing"""
        print("\n" + "="*60)
        print("  📡 Laptop Receiver + ML Processing")
        print("="*60)
        print("Waiting for stream... Press 'q' to stop\n")
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running:
                # Receive data
                data, addr = self.sock.recvfrom(65536)
                
                if len(data) < 4:
                    continue
                
                # Parse
                size = struct.unpack("I", data[:4])[0]
                jpeg_data = data[4:4+size]
                
                # Decode
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                self.frame_count += 1
                
                # Process with ML
                result = self.engine.process_frame(frame)
                
                # FPS
                fps = self.frame_count / (time.time() - start_time)
                
                # Add overlay
                cv2.putText(result.detections_overlay, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result.detections_overlay, 
                           f"Objects: {len(result.objects_3d)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Callback
                if self.callback:
                    self.callback(result)
                
                # Save
                if self.save_output and self.frame_count % 30 == 0:
                    self.engine.save_results(result, str(self.output_dir))
                
                # Display
                cv2.namedWindow('EdgeDrive3D - Laptop Processing', cv2.WINDOW_NORMAL)
                cv2.imshow('EdgeDrive3D - Laptop Processing', result.detections_overlay)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_snapshot(result.detections_overlay)
        
        except KeyboardInterrupt:
            print("\n⏹ Stopped")
        finally:
            self.cleanup()
    
    def _save_snapshot(self, image: np.ndarray):
        """Save snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(path), image)
        print(f"✓ Saved: {path}")
    
    def stop(self):
        """Stop receiver"""
        self.running = False
    
    def cleanup(self):
        """Cleanup"""
        print("\nCleaning up...")
        self.sock.close()
        cv2.destroyAllWindows()
        print(f"Total frames processed: {self.frame_count}")


# ============================================================================
# ESP32 CONTROLLER
# ============================================================================

class ESP32Controller:
    """
    Control ESP32 motor driver via HTTP requests
    """
    
    def __init__(self, esp32_ip: str = "192.168.4.1", timeout: float = 2.0):
        self.base_url = f"http://{esp32_ip}"
        self.timeout = timeout
        self.connected = False
        
        # Test connection
        try:
            resp = requests.get(self.base_url, timeout=timeout)
            self.connected = True
            print(f"✓ Connected to ESP32 at {esp32_ip}")
        except Exception as e:
            print(f"⚠ Could not connect to ESP32: {e}")
            print("  Will retry on command...")
    
    def _send_command(self, command: str) -> bool:
        """Send command to ESP32"""
        try:
            resp = requests.get(f"{self.base_url}/{command}", timeout=self.timeout)
            return resp.status_code == 200
        except Exception as e:
            print(f"⚠ Command failed: {e}")
            return False
    
    def move_forward(self, speed: int = 180) -> bool:
        """Move forward"""
        if speed != 180:
            self._send_command(f"speed?value={speed}")
        return self._send_command("forward")
    
    def move_backward(self, speed: int = 180) -> bool:
        """Move backward"""
        if speed != 180:
            self._send_command(f"speed?value={speed}")
        return self._send_command("backward")
    
    def turn_left(self) -> bool:
        """Turn left"""
        return self._send_command("left")
    
    def turn_right(self) -> bool:
        """Turn right"""
        return self._send_command("right")
    
    def stop(self) -> bool:
        """Stop motors"""
        return self._send_command("stop")
    
    def set_speed(self, speed: int) -> bool:
        """Set motor speed (0-255)"""
        return self._send_command(f"speed?value={speed}")
    
    def execute_decision(self, decision: Dict[str, Any]) -> str:
        """Execute perception-based decision"""
        action = decision.get('action', 'stop')
        speed = decision.get('speed', 180)
        
        if action == 'forward':
            self.move_forward(speed)
            return "forward"
        elif action == 'backward':
            self.move_backward(speed)
            return "backward"
        elif action == 'left':
            self.turn_left()
            return "left"
        elif action == 'right':
            self.turn_right()
            return "right"
        else:
            self.stop()
            return "stop"


# ============================================================================
# INTEGRATED SYSTEM
# ============================================================================

class IntegratedSystem:
    """
    Complete system: Pi → Laptop → ESP32
    """
    
    def __init__(self, esp32_ip: str = "192.168.4.1", udp_port: int = 5000,
                 model_size: str = 'yolov8m.pt', auto_control: bool = False):
        self.receiver = LaptopReceiver(udp_port, model_size)
        self.esp32 = ESP32Controller(esp32_ip)
        self.auto_control = auto_control
        
        self.receiver.set_callback(self._on_frame_processed)
    
    def _on_frame_processed(self, result):
        """Callback when frame is processed"""
        if self.auto_control and self.esp32.connected:
            action = self.esp32.execute_decision(result.decision)
            
            # Show decision on result
            overlay = result.detections_overlay
            y_pos = 110
            
            cv2.putText(overlay, f"Decision: {action.upper()}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if result.decision.get('warnings'):
                for i, warning in enumerate(result.decision['warnings']):
                    cv2.putText(overlay, warning, (10, y_pos + 40 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def start(self):
        """Start integrated system"""
        print("\n" + "="*60)
        print("  EdgeDrive3D - Integrated System")
        print("="*60)
        print(f"Auto Control: {'ENABLED' if self.auto_control else 'DISABLED'}")
        print("="*60 + "\n")
        
        self.receiver.start()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EdgeDrive3D Hardware Integration')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    
    # Pi sender
    pi_parser = subparsers.add_parser('pi', help='Raspberry Pi sender')
    pi_parser.add_argument('receiver_ip', type=str, help='Laptop IP')
    pi_parser.add_argument('-p', '--port', type=int, default=5000)
    pi_parser.add_argument('-W', '--width', type=int, default=640)
    pi_parser.add_argument('-H', '--height', type=int, default=480)
    pi_parser.add_argument('-q', '--quality', type=int, default=80)
    
    # Laptop receiver
    laptop_parser = subparsers.add_parser('laptop', help='Laptop receiver')
    laptop_parser.add_argument('-p', '--port', type=int, default=5000)
    laptop_parser.add_argument('-m', '--model', default='yolov8m.pt')
    laptop_parser.add_argument('-c', '--confidence', type=float, default=0.4)
    laptop_parser.add_argument('--save', action='store_true')
    laptop_parser.add_argument('--auto', action='store_true', help='Auto control ESP32')
    laptop_parser.add_argument('--esp32-ip', default='192.168.4.1')
    
    args = parser.parse_args()
    
    if args.mode == 'pi':
        sender = PiCameraSender(
            args.receiver_ip, args.port,
            width=args.width, height=args.height, quality=args.quality
        )
        sender.start()
    
    elif args.mode == 'laptop':
        if args.auto:
            system = IntegratedSystem(
                esp32_ip=args.esp32_ip,
                udp_port=args.port,
                model_size=args.model,
                auto_control=True
            )
            system.start()
        else:
            receiver = LaptopReceiver(
                args.port, args.model, args.confidence, args.save
            )
            receiver.start()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
