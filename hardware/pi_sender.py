"""
=============================================================================
EDGE DRIVE 3D - RASPBERRY PI CAMERA SENDER
=============================================================================
Sends webcam frames via UDP to laptop for ML processing
Optimized for Raspberry Pi 3B+ with camera module

Usage:
    python pi_sender.py <LAPTOP_IP> -p 5000 -W 640 -H 480

Author: EdgeDrive3D Team
=============================================================================
"""

import cv2
import socket
import numpy as np
import struct
import time
import argparse


class PiCameraSender:
    """Raspberry Pi UDP Camera Sender"""
    
    def __init__(self, receiver_ip: str, port: int = 5000, camera_id: int = 0,
                 width: int = 640, height: int = 480, quality: int = 80,
                 fps_target: int = 30):
        self.receiver_ip = receiver_ip
        self.port = port
        self.quality = quality
        self.fps_target = fps_target
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps_target)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or fps_target
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        
        self.frame_count = 0
        self.running = False
        
        print(f"\n{'='*60}")
        print(f"  🎥 Pi Camera Sender")
        print(f"{'='*60}")
        print(f"  Camera: {self.width}x{self.height} @ {self.fps}fps")
        print(f"  Sending to: {receiver_ip}:{port}")
        print(f"  JPEG quality: {quality}%")
        print(f"{'='*60}\n")
    
    def start(self):
        """Start streaming"""
        print("Streaming... Press 'q' to stop\n")
        
        self.running = True
        start_time = time.time()
        frame_start = time.time()
        
        try:
            while self.running:
                # Capture
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠ Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                _, jpeg_data = cv2.imencode('.jpg', frame, encode_param)
                
                # Pack
                data = struct.pack("I", len(jpeg_data)) + jpeg_data.tobytes()
                
                # Send
                self.sock.sendto(data, (self.receiver_ip, self.port))
                
                self.frame_count += 1
                
                # FPS control
                elapsed = time.time() - frame_start
                frame_time = 1.0 / self.fps_target
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                frame_start = time.time()
                
                # Stats
                if self.frame_count % 30 == 0:
                    total_elapsed = time.time() - start_time
                    fps_actual = 30 / total_elapsed if total_elapsed > 0 else 0
                    print(f"📊 FPS: {fps_actual:.1f} | Frames: {self.frame_count}")
                    start_time = time.time()
        
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop streaming"""
        self.running = False
    
    def cleanup(self):
        """Cleanup"""
        print("\nCleaning up...")
        self.cap.release()
        self.sock.close()
        
        total_elapsed = time.time() - start_time if 'start_time' in dir() else 1
        avg_fps = self.frame_count / total_elapsed
        
        print(f"Total frames sent: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Pi Camera Sender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send to laptop at 192.168.1.100
  python pi_sender.py 192.168.1.100

  # Custom port and resolution
  python pi_sender.py 192.168.1.100 -p 5001 -W 1280 -H 720

  # Lower quality for faster streaming
  python pi_sender.py 192.168.1.100 -q 60
        """
    )
    
    parser.add_argument('receiver_ip', type=str, help='Laptop IP address')
    parser.add_argument('-p', '--port', type=int, default=5000, help='UDP port')
    parser.add_argument('-i', '--camera-id', type=int, default=0, help='Camera ID')
    parser.add_argument('-W', '--width', type=int, default=640, help='Frame width')
    parser.add_argument('-H', '--height', type=int, default=480, help='Frame height')
    parser.add_argument('-q', '--quality', type=int, default=80, help='JPEG quality (1-100)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Target FPS')
    
    args = parser.parse_args()
    
    sender = PiCameraSender(
        args.receiver_ip, args.port, args.camera_id,
        args.width, args.height, args.quality, args.fps
    )
    
    sender.start()


if __name__ == "__main__":
    main()
