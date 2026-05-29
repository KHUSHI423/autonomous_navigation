"""
EdgeDrive3D Real-Time Dashboard - FIXED VERSION
Run: python dashboard/realtime_fixed.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import threading
import time
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.perception_engine import PerceptionEngine, Object3D

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', max_http_buffer_size=1e8)

# Global state
state = {
    'running': False,
    'mode': 'demo',
    'engine': None,
    'current_frame': None,
    'current_objects': [],
    'current_decision': {},
    'current_bev': None,
    'current_depth': None,
    'video_cap': None,
    'video_total': 0,
    'video_frame': 0,
    'video_playing': False,
    'image_uploaded': False
}

# ============================================================================
# HTML TEMPLATE
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream)
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if state['engine'] is None:
        state['engine'] = PerceptionEngine({'yolo_model': 'yolov8m.pt', 'confidence': 0.4, 'max_depth': 50.0})
    
    result = state['engine'].process_frame(img_np)
    
    state['current_frame'] = result.detections_overlay
    state['current_objects'] = result.objects_3d
    state['current_decision'] = result.decision
    state['current_bev'] = result.bev_image
    state['current_depth'] = result.depth_colored
    state['image_uploaded'] = True
    
    # Send result immediately
    send_update()
    
    return jsonify({'success': True, 'objects': len(result.objects_3d)})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video'}), 400
    
    file = request.files['video']
    upload_dir = Path('output/uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = upload_dir / file.filename
    file.save(str(video_path))
    
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    state['video_cap'] = cap
    state['video_total'] = total
    state['video_frame'] = 0
    
    return jsonify({'success': True, 'frames': total})

@app.route('/play_video', methods=['POST'])
def play_video():
    state['video_playing'] = True
    return jsonify({'success': True})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    state['video_playing'] = False
    return jsonify({'success': True})

# ============================================================================
# SOCKET.IO EVENTS
# ============================================================================

@socketio.on('connect')
def connect():
    print('✅ Client connected')

@socketio.on('set_mode')
def set_mode(data):
    state['mode'] = data['mode']
    print(f'Mode: {state["mode"]}')

@socketio.on('start')
def start():
    state['running'] = True
    if state['engine'] is None:
        state['engine'] = PerceptionEngine({'yolo_model': 'yolov8m.pt', 'confidence': 0.4, 'max_depth': 50.0})
    
    thread = threading.Thread(target=process_loop, daemon=True)
    thread.start()

@socketio.on('stop')
def stop():
    state['running'] = False
    state['video_playing'] = False

# ============================================================================
# PROCESSING LOOP
# ============================================================================

def process_loop():
    demo_gen = DemoGenerator()
    fps_counter = 0
    start_time = time.time()
    
    while state['running']:
        try:
            # DEMO MODE
            if state['mode'] == 'demo':
                objects = demo_gen.generate_objects()
                decision = demo_gen.generate_decision(objects)
                bev = demo_gen.generate_bev(objects)
                depth = demo_gen.generate_depth()
                
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (18, 18, 26)
                cv2.putText(frame, "DEMO MODE", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                
                state['current_frame'] = frame
                state['current_objects'] = objects
                state['current_decision'] = decision
                state['current_bev'] = bev
                state['current_depth'] = cv2.applyColorMap((depth * 5).astype(np.uint8), cv2.COLORMAP_TURBO)
            
            # IMAGE MODE (shows last processed image)
            elif state['mode'] == 'image' and state['image_uploaded']:
                pass  # Already processed, just display
            
            # VIDEO MODE
            elif state['mode'] == 'video' and state['video_cap'] is not None and state['video_playing']:
                state['video_cap'].set(cv2.CAP_PROP_POS_FRAMES, state['video_frame'])
                ret, frame = state['video_cap'].read()
                
                if ret and state['engine']:
                    result = state['engine'].process_frame(frame)
                    state['current_frame'] = result.detections_overlay
                    state['current_objects'] = result.objects_3d
                    state['current_decision'] = result.decision
                    state['current_bev'] = result.bev_image
                    state['current_depth'] = result.depth_colored
                    
                    state['video_frame'] += 1
                    if state['video_frame'] >= state['video_total']:
                        state['video_frame'] = 0
                        state['video_playing'] = False
                    
                    socketio.emit('video_progress', {
                        'frame': state['video_frame'],
                        'total': state['video_total']
                    })
            
            # WEBCAM MODE
            elif state['mode'] == 'webcam':
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                
                if ret and state['engine']:
                    result = state['engine'].process_frame(frame)
                    state['current_frame'] = result.detections_overlay
                    state['current_objects'] = result.objects_3d
                    state['current_decision'] = result.decision
                    state['current_bev'] = result.bev_image
                    state['current_depth'] = result.depth_colored
            
            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - start_time
            fps = fps_counter / elapsed if elapsed > 0 else 0
            
            # Send updates
            send_update(fps)
            
            time.sleep(0.05)  # 20 FPS
            
        except Exception as e:
            print(f'Error: {e}')
            time.sleep(0.1)

def send_update(fps=0):
    try:
        # Encode frames
        det_data = None
        depth_data = None
        
        if state['current_frame'] is not None:
            _, buf = cv2.imencode('.jpg', state['current_frame'])
            det_data = base64.b64encode(buf).decode()
        
        if state['current_depth'] is not None:
            _, buf = cv2.imencode('.jpg', state['current_depth'])
            depth_data = base64.b64encode(buf).decode()
        
        # BEV SVG
        bev_svg = None
        if state['current_bev'] is not None and state['current_objects']:
            bev_svg = create_bev_svg(state['current_objects'])
        
        # 3D Plot
        plot_data = None
        if state['current_objects']:
            plot_data = create_3d_plot_data(state['current_objects'])
        
        # Detections list
        detections = [{
            'class_name': obj.class_name,
            'confidence': obj.confidence,
            'distance': obj.distance,
            'dimensions': obj.dimensions
        } for obj in state['current_objects']]
        
        # Send
        socketio.emit('update', {
            'detection_frame': det_data,
            'depth_frame': depth_data,
            'bev_svg': bev_svg,
            'plot_3d': plot_data,
            'detections': detections,
            'fps': fps,
            'objects': len(state['current_objects']),
            'decision': state['current_decision'].get('action', 'stop'),
            'warnings': state['current_decision'].get('warnings', [])
        })
        
    except Exception as e:
        print(f'Send error: {e}')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_bev_svg(objects):
    svg_objs = ""
    for obj in objects:
        x = 350 + obj.position_3d[0] * 18 if obj.position_3d is not None else 350
        y = 450 - obj.distance * 12
        svg_objs += f'<circle cx="{x}" cy="{y}" r="12" fill="#fff" stroke="#5D5FEF" stroke-width="2"/><text x="{x}" y="{y-15}" fill="#fff" font-size="10" text-anchor="middle">{obj.class_name[:4]} {obj.distance:.0f}m</text>'
    
    return f'''<svg viewBox="0 0 700 500" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e">
        <line x1="350" y1="50" x2="350" y2="450" stroke="#555" stroke-width="2" stroke-dasharray="15,15"/>
        <circle cx="350" cy="450" r="18" fill="#30d158"/><text x="350" y="475" fill="#30d158" font-size="12" text-anchor="middle">EGO</text>
        {svg_objs}
        <text x="20" y="25" fill="#fff" font-size="14">Bird's Eye View</text>
    </svg>'''

def create_3d_plot_data(objects):
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Scatter3d(
        x=[o.position_3d[0] for o in objects if o.position_3d is not None],
        y=[o.position_3d[1] for o in objects if o.position_3d is not None],
        z=[o.position_3d[2] for o in objects if o.position_3d is not None],
        mode='markers+text',
        marker=dict(size=8, color='#5D5FEF'),
        text=[o.class_name for o in objects if o.position_3d is not None]
    )])
    
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B8B9E'),
        height=300,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    
    return {'data': fig.data, 'layout': fig.layout}

class DemoGenerator:
    CLASSES = ['car', 'person', 'truck', 'motorcycle', 'bus']
    
    @staticmethod
    def generate_objects(num=None):
        if num is None:
            num = np.random.randint(2, 5)
        objects = []
        for i in range(num):
            cls = np.random.choice(DemoGenerator.CLASSES)
            distance = np.random.uniform(5, 25)
            obj = Object3D(
                class_id=i, class_name=cls, confidence=np.random.uniform(0.75, 0.95),
                bbox_2d=(100, 100, 200, 200),
                position_3d=np.array([np.random.uniform(-8, 8), 0, distance]),
                distance=distance,
                dimensions={'width': np.random.uniform(1.5, 2.5), 'height': np.random.uniform(1.2, 2.0), 'length': np.random.uniform(3, 5)},
                bbox_3d=np.array([[-1,-1,distance-2],[1,-1,distance-2],[1,1,distance-2],[-1,1,distance-2],[-1,-1,distance+2],[1,-1,distance+2],[1,1,distance+2],[-1,1,distance+2]])
            )
            objects.append(obj)
        return sorted(objects, key=lambda x: x.distance)
    
    @staticmethod
    def generate_decision(objects):
        if not objects:
            return {'action': 'forward', 'warnings': []}
        closest = objects[0]
        if closest.distance < 3.0:
            return {'action': 'stop', 'warnings': [f"⚠️ {closest.class_name} at {closest.distance:.1f}m"]}
        return {'action': 'forward', 'warnings': []}
    
    @staticmethod
    def generate_bev(objects):
        bev = np.zeros((500, 700, 3), dtype=np.uint8)
        for obj in objects:
            x = int(350 + obj.position_3d[0] * 18)
            y = int(450 - obj.distance * 12)
            cv2.circle(bev, (x, y), 14, (255, 255, 255), -1)
            cv2.putText(bev, f"{obj.class_name[:3]} {obj.distance:.0f}m", (x-25, y-17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(bev, (350, 450), 18, (48, 209, 88), -1)
        return bev
    
    @staticmethod
    def generate_depth():
        h, w = 480, 640
        y, x = np.ogrid[:h, :w]
        depth = np.sqrt((x - 320)**2 + **(y - 240)2)
        depth = depth / depth.max()
        return (depth * 50).astype(np.float32)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🚗 EdgeDrive3D Real-Time Dashboard")
    print("="*60)
    print("\n  🌐 Open: http://localhost:5000")
    print("  📹 Modes: Demo, Webcam, Image, Video")
    print("\n  Press Ctrl+C to stop\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
