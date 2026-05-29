"""
=============================================================================
EDGE DRIVE 3D - MAIN ENTRY POINT
=============================================================================
Unified command-line interface for all system modes

Author: EdgeDrive3D Team
=============================================================================
"""

import argparse
import sys
import os

# Add core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     ███████╗███╗   ██╗ █████╗ ██████╗  ██████╗ ███████╗██╗  ██╗   ║
    ║     ██╔════╝████╗  ██║██╔══██╗██╔══██╗██╔═══██╗██╔════╝╚██╗██╔╝   ║
    ║     █████╗  ██╔██╗ ██║███████║██████╔╝██║   ██║█████╗   ╚███╔╝    ║
    ║     ██╔══╝  ██║╚██╗██║██╔══██║██╔══██╗██║   ██║██╔══╝   ██╔██╗    ║
    ║     ███████╗██║ ╚████║██║  ██║██║  ██║╚██████╔╝███████╗██╔ ██╗   ║
    ║     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ║
    ║                                                                   ║
    ║           Unified Autonomous Perception System v2.0               ║
    ║                    ENHANCED EDITION                               ║
    ║   Depth + 3D Objects + Lanes + Traffic Signs + BEV + Decisions   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def mode_image(args):
    """Process single image"""
    from core.perception_engine import PerceptionEngine

    engine = PerceptionEngine({
        'yolo_model': args.model,
        'confidence': args.confidence,
        'max_depth': args.max_depth,
        'enable_lane': not args.no_lane,
        'lane_preset': args.lane_preset,
        'enable_signs': not args.no_signs,
        'sign_mode': args.sign_mode,
    })

    import cv2
    image = cv2.imread(args.input)

    if image is None:
        print(f"❌ Could not load image: {args.input}")
        return

    result = engine.process_frame(image)
    engine.save_results(result, args.output)

    print(f"\n✓ Processed: {args.input}")
    print(f"  Objects detected: {len(result.objects_3d)}")
    print(f"  Lanes detected: {'✓' if result.lanes else '✗'}")
    print(f"  Traffic signs: {len(result.traffic_signs)}")
    print(f"  Decision: {result.decision['action']}")
    print(f"  Results saved to: {args.output}/")

    # Show if requested
    if args.show:
        cv2.imshow("Detections", result.detections_overlay)
        if result.bev_image is not None:
            cv2.imshow("BEV", result.bev_image)
        if result.depth_colored is not None:
            cv2.imshow("Depth", result.depth_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def mode_video(args):
    """Process video file"""
    from core.perception_engine import PerceptionEngine
    import cv2

    # For video, use fast settings by default
    engine = PerceptionEngine({
        'yolo_model': args.model,
        'confidence': args.confidence,
        'enable_lane': not args.no_lane,
        'enable_signs': not args.no_signs,
        'enable_depth': not args.no_depth,
        'depth_skip_frames': args.depth_skip if hasattr(args, 'depth_skip') else 2,
        'lane_preset': args.lane_preset if hasattr(args, 'lane_preset') else 'default',
        'sign_mode': 'opencv',
    })
    
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {args.input}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total} frames")
    
    # Writer
    writer = None
    if args.save:
        output_path = os.path.join(args.output, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Processing... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = engine.process_frame(frame)
        
        frame_count += 1
        
        if writer:
            writer.write(result.detections_overlay)
        
        if args.show:
            display = result.detections_overlay
            if width > 1280:
                scale = 1280 / width
                display = cv2.resize(display, (int(width*scale), int(height*scale)))
            
            cv2.imshow('EdgeDrive3D', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Processed {frame_count} frames")


def mode_webcam(args):
    """Webcam processing"""
    from core.perception_engine import PerceptionEngine
    import cv2
    import numpy as np

    engine = PerceptionEngine({
        'yolo_model': args.model,
        'confidence': args.confidence,
        'enable_lane': not args.no_lane,
        'lane_preset': args.lane_preset,
        'enable_signs': not args.no_signs,
        'sign_mode': args.sign_mode,
    })

    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"❌ Could not open camera: {args.camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {width}x{height}")
    print("Starting... Press 'q' to quit, 's' to save snapshot\n")
    print("  Controls:")
    print("    - 'q': Quit")
    print("    - 's': Save snapshot")
    print("    - '1': Toggle lane detection")
    print("    - '2': Toggle sign detection")
    print()

    # Warmup
    print("Warming up model (first frame may take 30-60s)...")
    _ = engine.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    print("✓ Warmup complete\n")

    frame_count = 0
    start_time = time.time()
    
    # Module toggle state
    lane_enabled = not args.no_lane
    signs_enabled = not args.no_signs

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update engine config on the fly
        engine.config['enable_lane'] = lane_enabled
        engine.config['enable_signs'] = signs_enabled

        result = engine.process_frame(frame)
        frame_count += 1

        # FPS
        fps = frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0

        # Overlay
        cv2.putText(result.detections_overlay, f"FPS: {fps:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Decision
        action = result.decision['action'].upper()
        cv2.putText(result.detections_overlay, f"ACTION: {action}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Module status
        status_y = 110
        cv2.putText(result.detections_overlay, f"LANE: {'ON' if lane_enabled else 'OFF'}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result.detections_overlay, f"SIGNS: {'ON' if signs_enabled else 'OFF'}",
                   (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('EdgeDrive3D - Webcam', result.detections_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            from datetime import datetime
            path = f"{args.output}/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(path, result.detections_overlay)
            print(f"✓ Saved: {path}")
        elif key == ord('1'):
            lane_enabled = not lane_enabled
            print(f"  {'✓' if lane_enabled else '⚠'} Lane detection {'enabled' if lane_enabled else 'disabled'}")
        elif key == ord('2'):
            signs_enabled = not signs_enabled
            print(f"  {'✓' if signs_enabled else '⚠'} Sign detection {'enabled' if signs_enabled else 'disabled'}")

    cap.release()
    cv2.destroyAllWindows()


def mode_pi_stream(args):
    """Receive from Raspberry Pi"""
    from hardware.hardware_integration import LaptopReceiver, IntegratedSystem
    
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
            args.port, args.model, args.confidence, args.save, args.output
        )
        receiver.start()


def mode_dashboard(args):
    """Launch Streamlit dashboard"""
    import subprocess

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")

    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard not found: {dashboard_path}")
        return

    print("Launching Streamlit Dashboard...")
    print(f"Open http://localhost:{args.port} in your browser\n")

    subprocess.run([
        "streamlit", "run", dashboard_path,
        "--server.port", str(args.port),
        "--server.address", args.address
    ])


def mode_gps_dashboard(args):
    """Launch GPS-enhanced Streamlit dashboard with OpenStreetMap"""
    import subprocess

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "gps_dashboard.py")

    if not os.path.exists(dashboard_path):
        print(f"❌ GPS Dashboard not found: {dashboard_path}")
        return

    print("Launching GPS Dashboard with OpenStreetMap...")
    print(f"Open http://localhost:{args.port} in your browser\n")
    gps_mode = "Simulated" if args.simulate else "Hardware"
    print(f"📍 GPS Mode: {gps_mode}")
    if not args.simulate:
        print(f"   Port: {args.gps_port} @ {args.baudrate}")
    print()

    subprocess.run([
        "streamlit", "run", dashboard_path,
        "--server.port", str(args.port),
        "--server.address", args.address
    ])


def mode_3d_dashboard(args):
    """Launch 3D GPS Dashboard with CIT Campus visualization"""
    import subprocess

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "gps_dashboard_3d.py")

    if not os.path.exists(dashboard_path):
        print(f"❌ 3D GPS Dashboard not found: {dashboard_path}")
        return

    print("Launching 3D GPS Dashboard with CIT Campus...")
    print(f"Open http://localhost:{args.port} in your browser\n")
    gps_mode = "Simulated" if args.simulate else "Hardware"
    print(f"📍 GPS Mode: {gps_mode}")
    if not args.simulate:
        print(f"   Port: {args.gps_port} @ {args.baudrate}")
    print()
    print("🏫 3D Campus Features:")
    print("   - 3D buildings and landmarks")
    print("   - Interactive camera controls")
    print("   - Real-time vehicle tracking")
    print("   - Object visualization in 3D")
    print()

    subprocess.run([
        "streamlit", "run", dashboard_path,
        "--server.port", str(args.port),
        "--server.address", args.address
    ])


def main():
    import time
    import numpy as np
    
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='EdgeDrive3D - Unified Perception System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process image
  python main.py image path/to/image.jpg -o output/

  # Process video
  python main.py video road_video.mp4 --save

  # Webcam
  python main.py webcam

  # Receive from Pi
  python main.py pi-stream --port 5000

  # With auto ESP32 control
  python main.py pi-stream --auto --esp32-ip 192.168.4.1

  # Dashboard
  python main.py dashboard

  # GPS Dashboard (OpenStreetMap)
  python main.py gps-dashboard

  # GPS Dashboard with hardware GPS
  python main.py gps-dashboard --simulate=False --gps-port /dev/ttyUSB0

  # 3D GPS Dashboard (CIT Campus)
  python main.py 3d-dashboard

  # 3D GPS Dashboard with hardware GPS
  python main.py 3d-dashboard --simulate=False --gps-port COM3
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Image mode
    img_p = subparsers.add_parser('image', help='Process single image')
    img_p.add_argument('input', help='Input image path')
    img_p.add_argument('-o', '--output', default='output')
    img_p.add_argument('-m', '--model', default='yolov8m.pt')
    img_p.add_argument('-c', '--confidence', type=float, default=0.4)
    img_p.add_argument('-d', '--max-depth', type=float, default=50.0)
    img_p.add_argument('-s', '--show', action='store_true')
    img_p.add_argument('--no-lane', action='store_true', help='Disable lane detection')
    img_p.add_argument('--no-signs', action='store_true', help='Disable traffic sign detection')
    img_p.add_argument('--lane-preset', default='default', 
                       choices=['default', 'highway', 'city', 'faded', 'night', 'indian_road'],
                       help='Lane detection preset')
    img_p.add_argument('--sign-mode', default='opencv', choices=['opencv', 'yolo'],
                       help='Sign detection mode')

    # Video mode
    vid_p = subparsers.add_parser('video', help='Process video')
    vid_p.add_argument('input', help='Input video path')
    vid_p.add_argument('-o', '--output', default='output')
    vid_p.add_argument('-m', '--model', default='yolov8n.pt', help='YOLO model (yolov8n.pt for speed, yolov8m.pt for accuracy)')
    vid_p.add_argument('-c', '--confidence', type=float, default=0.4)
    vid_p.add_argument('-s', '--show', action='store_true', default=True)
    vid_p.add_argument('--save', action='store_true')
    vid_p.add_argument('--no-lane', action='store_true', help='Disable lane detection')
    vid_p.add_argument('--no-signs', action='store_true', help='Disable traffic sign detection')
    vid_p.add_argument('--no-depth', action='store_true', help='Disable depth estimation (FASTEST)')
    vid_p.add_argument('--depth-skip', type=int, default=2, metavar='N',
                       help='Process depth every N+1 frames (default=2, 0=every frame)')
    vid_p.add_argument('--lane-preset', default='default',
                       choices=['default', 'highway', 'city', 'faded', 'night', 'indian_road'],
                       help='Lane detection preset')

    # Webcam mode
    web_p = subparsers.add_parser('webcam', help='Webcam processing')
    web_p.add_argument('-i', '--camera-id', type=int, default=0)
    web_p.add_argument('-m', '--model', default='yolov8n.pt')
    web_p.add_argument('-c', '--confidence', type=float, default=0.4)
    web_p.add_argument('-o', '--output', default='output')
    web_p.add_argument('--no-lane', action='store_true', help='Disable lane detection')
    web_p.add_argument('--no-signs', action='store_true', help='Disable traffic sign detection')
    web_p.add_argument('--lane-preset', default='default',
                       choices=['default', 'highway', 'city', 'faded', 'night', 'indian_road'],
                       help='Lane detection preset')
    web_p.add_argument('--sign-mode', default='opencv', choices=['opencv', 'yolo'],
                       help='Sign detection mode')
    
    # Pi stream mode
    pi_p = subparsers.add_parser('pi-stream', help='Receive from Pi')
    pi_p.add_argument('-p', '--port', type=int, default=5000)
    pi_p.add_argument('-m', '--model', default='yolov8n.pt')
    pi_p.add_argument('-c', '--confidence', type=float, default=0.4)
    pi_p.add_argument('-o', '--output', default='output')
    pi_p.add_argument('--save', action='store_true')
    pi_p.add_argument('--auto', action='store_true', help='Auto control ESP32')
    pi_p.add_argument('--esp32-ip', default='192.168.4.1')
    
    # Dashboard mode
    dash_p = subparsers.add_parser('dashboard', help='Launch dashboard')
    dash_p.add_argument('-p', '--port', type=int, default=8501)
    dash_p.add_argument('-a', '--address', default='localhost')

    # GPS Dashboard mode
    gps_p = subparsers.add_parser('gps-dashboard', help='Launch GPS dashboard with OpenStreetMap')
    gps_p.add_argument('-p', '--port', type=int, default=8502)
    gps_p.add_argument('-a', '--address', default='localhost')
    gps_p.add_argument('--simulate', action='store_true', default=True, help='Use simulated GPS')
    gps_p.add_argument('--gps-port', default='auto', help='GPS serial port')
    gps_p.add_argument('--baudrate', type=int, default=9600, help='GPS baud rate')

    # 3D GPS Dashboard mode
    _3d_p = subparsers.add_parser('3d-dashboard', help='Launch 3D GPS dashboard with CIT Campus')
    _3d_p.add_argument('-p', '--port', type=int, default=8503)
    _3d_p.add_argument('-a', '--address', default='localhost')
    _3d_p.add_argument('--simulate', action='store_true', default=True, help='Use simulated GPS')
    _3d_p.add_argument('--gps-port', default='auto', help='GPS serial port')
    _3d_p.add_argument('--baudrate', type=int, default=9600, help='GPS baud rate')

    args = parser.parse_args()

    if args.mode == 'image':
        mode_image(args)
    elif args.mode == 'video':
        mode_video(args)
    elif args.mode == 'webcam':
        mode_webcam(args)
    elif args.mode == 'pi-stream':
        mode_pi_stream(args)
    elif args.mode == 'dashboard':
        mode_dashboard(args)
    elif args.mode == 'gps-dashboard':
        mode_gps_dashboard(args)
    elif args.mode == '3d-dashboard':
        mode_3d_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
