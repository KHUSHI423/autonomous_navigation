"""
=============================================================================
EDGE DRIVE 3D - REAL-TIME DASHBOARD (Flask + SocketIO) - FIXED
=============================================================================
High-performance real-time dashboard with:
- No page refresh (WebSocket updates)
- Smooth video playback (30 FPS)
- Image upload support
- Video upload support
- Webcam support
- Demo mode

Run: python dashboard/realtime_app.py
Open: http://localhost:5000
=============================================================================
"""

from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import threading
import time
import socket
import struct
from pathlib import Path
import sys
from collections import deque
import json
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.perception_engine import PerceptionEngine, Object3D


# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'edgedrive3d_secret_key_2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', max_http_buffer_size=1e8)


# ============================================================================
# GLOBAL STATE
# ============================================================================

class SystemState:
    def __init__(self):
        self.running = False
        self.mode = 'demo'  # demo, webcam, video, image, pi_stream
        self.engine = None
        self.current_frame = None
        self.current_objects = []
        self.current_decision = {}
        self.current_bev = None
        self.current_depth = None
        self.current_detections = None
        self.current_pointcloud = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = None
        self.video_path = None
        self.video_cap = None
        self.video_total_frames = 0
        self.video_frame = 0
        self.video_playing = False
        self.image_path = None
        self.image_processed = False
        self.webcam_cap = None
        self.pi_socket = None
        self.pi_frame_count = 0
        self.metrics_history = deque(maxlen=100)
        self.process_thread = None


state = SystemState()
processing_lock = threading.Lock()


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeDrive3D - Real-Time Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg-main: #0B0B0F;
            --bg-card: #181824;
            --bg-card-hover: #1F1F2E;
            --text-primary: #FFFFFF;
            --text-secondary: #8B8B9E;
            --border-color: rgba(255, 255, 255, 0.06);
            --accent-primary: #5D5FEF;
            --accent-success: #10B981;
            --accent-danger: #EF4444;
            --accent-warning: #F59E0B;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-main);
            color: var(--text-primary);
            overflow-x: hidden;
        }

        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-title {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #5D5FEF, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .status-running {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .status-stopped {
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-danger);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .main-container {
            padding: 2rem;
            max-width: 1600px;
            margin: 0 auto;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s;
        }

        .metric-card:hover {
            background: var(--bg-card-hover);
            transform: translateY(-3px);
            border-color: rgba(93, 95, 239, 0.3);
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }

        .control-panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }

        .control-section {
            margin-bottom: 1.5rem;
        }

        .control-section h3 {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }

        .btn-group {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            border: none;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .btn-primary {
            background: var(--accent-primary);
            color: white;
            box-shadow: 0 4px 15px rgba(93, 95, 239, 0.4);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(93, 95, 239, 0.6);
        }

        .btn-secondary {
            background: var(--bg-main);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            border-color: var(--accent-primary);
            background: rgba(93, 95, 239, 0.1);
            color: var(--accent-primary);
        }

        .btn-danger {
            background: var(--accent-danger);
            color: white;
        }

        .btn-success {
            background: var(--accent-success);
            color: white;
        }

        .file-upload {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 0.5rem;
        }

        .file-upload-label {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            background: var(--bg-main);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .file-upload-label:hover {
            border-color: var(--accent-primary);
            color: var(--accent-primary);
        }

        input[type="file"] {
            display: none;
        }

        .video-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .video-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid var(--border-color);
        }

        .card-title {
            color: var(--text-primary);
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid var(--border-color);
        }

        .video-container {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
        }

        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .video-placeholder {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: var(--text-secondary);
            text-align: center;
        }

        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 300px;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .chart-container {
            height: 300px;
        }

        .detection-list {
            max-height: 280px;
            overflow-y: auto;
        }

        .detection-item {
            background: var(--bg-main);
            border-radius: 12px;
            padding: 12px;
            margin: 6px 0;
            display: flex;
            justify-content: space-between;
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }

        .detection-item:hover {
            border-color: var(--accent-primary);
            background: var(--bg-card-hover);
        }

        .detection-class {
            font-weight: 600;
            font-size: 0.9rem;
        }

        .detection-distance {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        .warnings-container {
            margin-bottom: 1rem;
        }

        .warning-box {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: var(--accent-danger);
            font-weight: 600;
        }

        .info-box {
            background: rgba(93, 95, 239, 0.1);
            border: 1px solid rgba(93, 95, 239, 0.3);
            border-radius: 12px;
            padding: 1rem;
            color: var(--accent-primary);
        }

        .progress-container {
            margin-top: 1rem;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--bg-main);
            border-radius: 10px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-success));
            transition: width 0.3s;
            width: 0%;
        }

        .progress-text {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            text-align: right;
        }

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-main);
        }

        ::-webkit-scrollbar-thumb {
            background: #333344;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-primary);
        }

        @media (max-width: 1200px) {
            .metrics-row {
                grid-template-columns: repeat(2, 1fr);
            }
            .video-grid {
                grid-template-columns: 1fr;
            }
            .bottom-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">🚗 EdgeDrive3D Real-Time</div>
        <div class="header-status">
            <span id="statusBadge" class="status-badge status-stopped">System Stopped</span>
            <span id="clock" style="color: var(--text-secondary); font-family: 'JetBrains Mono';"></span>
        </div>
    </div>

    <div class="main-container">
        <div class="control-panel">
            <div class="control-section">
                <h3>🎯 Input Mode</h3>
                <div class="btn-group">
                    <button class="btn btn-secondary" onclick="setMode('demo')">🎮 Demo</button>
                    <button class="btn btn-secondary" onclick="setMode('webcam')">📹 Webcam</button>
                    <button class="btn btn-secondary" onclick="setMode('pi_stream')">📡 Pi Stream</button>
                    <button class="btn btn-secondary" onclick="setMode('image')">🖼️ Image</button>
                    <button class="btn btn-secondary" onclick="setMode('video')">🎬 Video</button>
                </div>
            </div>

            <div class="control-section" id="imageControls" style="display: none;">
                <h3>📤 Upload Image</h3>
                <div class="file-upload">
                    <label for="imageUpload" class="file-upload-label">📁 Choose Image</label>
                    <input type="file" id="imageUpload" accept="image/jpeg,image/png,image/webp">
                    <span id="imageName" style="color: var(--text-secondary);"></span>
                </div>
                <div class="btn-group" style="margin-top: 1rem;">
                    <button class="btn btn-primary" onclick="processImage()">🔍 Process Image</button>
                </div>
            </div>

            <div class="control-section" id="videoControls" style="display: none;">
                <h3>📤 Upload Video</h3>
                <div class="file-upload">
                    <label for="videoUpload" class="file-upload-label">📁 Choose Video</label>
                    <input type="file" id="videoUpload" accept="video/mp4,video/avi,video/mov,video/mkv">
                    <span id="videoName" style="color: var(--text-secondary);"></span>
                </div>
                <div class="btn-group" style="margin-top: 1rem;">
                    <button class="btn btn-primary" onclick="initializeVideo()">▶ Initialize Video</button>
                    <button class="btn btn-success" onclick="playVideo()">▶ Play Video</button>
                </div>
                <div id="videoProgress" class="progress-container" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <div class="progress-text" id="progressText">0 / 0 frames</div>
                </div>
            </div>
        </div>

        <div class="warnings-container" id="warningsContainer">
            <div class="info-box">✅ System Ready - Select a mode and click START</div>
        </div>

        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">⚡ Framerate</div>
                <div class="metric-value" id="fpsMetric">0 FPS</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🎯 Objects</div>
                <div class="metric-value" id="objectsMetric">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⚙️ Decision</div>
                <div class="metric-value" id="decisionMetric" style="color: #10B981;">STOP</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⏱️ Uptime</div>
                <div class="metric-value" id="uptimeMetric">0s</div>
            </div>
        </div>

        <div class="video-grid">
            <div class="video-card">
                <div class="card-title">🎯 Object Detection</div>
                <div class="video-container">
                    <img id="detectionFrame" alt="Detection Feed">
                    <div class="video-placeholder" id="detectionPlaceholder">
                        <div style="font-size: 3rem;">📹</div>
                        <div style="margin-top: 1rem;">Waiting for video feed...</div>
                    </div>
                </div>
            </div>
            <div class="video-card">
                <div class="card-title">🌊 Depth Map (Combined)</div>
                <div class="video-container">
                    <img id="depthFrame" alt="Depth Feed">
                    <div class="video-placeholder" id="depthPlaceholder">
                        <div style="font-size: 3rem;">🌊</div>
                        <div style="margin-top: 1rem;">Depth data not available</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="bottom-grid">
            <div class="video-card">
                <div class="card-title">🗺️ BEV Map (SVG Icons)</div>
                <div class="video-container" id="bevContainer">
                    <div class="video-placeholder">
                        <div style="font-size: 3rem;">🗺️</div>
                        <div style="margin-top: 1rem;">BEV not available</div>
                    </div>
                </div>
            </div>
            <div class="video-card">
                <div class="card-title">📦 3D Processed Output</div>
                <div id="plot3D" class="chart-container"></div>
            </div>
            <div class="video-card">
                <div class="card-title">📏 Distance Data</div>
                <div class="detection-list" id="detectionList">
                    <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                        No objects detected
                    </div>
                </div>
            </div>
        </div>

        <div class="video-card">
            <div class="card-title">📈 System Telemetry</div>
            <div id="analyticsChart" class="chart-container"></div>
        </div>
    </div>

    <script>
        const socket = io();
        let currentMode = 'demo';
        let startTime = null;

        socket.on('connect', () => {
            console.log('✅ Connected to EdgeDrive3D server');
        });

        socket.on('frame_update', (data) => {
            updateFrame(data);
        });

        socket.on('metrics_update', (data) => {
            updateMetrics(data);
        });

        socket.on('decision_update', (data) => {
            updateDecision(data);
        });

        socket.on('processing_complete', (data) => {
            document.getElementById('progressContainer').style.display = 'none';
            console.log('Processing complete');
        });

        function setMode(mode) {
            currentMode = mode;
            socket.emit('set_mode', { mode: mode });

            document.getElementById('imageControls').style.display = mode === 'image' ? 'block' : 'none';
            document.getElementById('videoControls').style.display = mode === 'video' ? 'block' : 'none';

            // Show Pi stream info
            if (mode === 'pi_stream') {
                document.getElementById('warningsContainer').innerHTML = 
                    '<div class="info-box">📡 Pi Stream Mode - Make sure Pi sender is running on port 5000</div>';
            }

            document.getElementById('statusBadge').className = 'status-badge status-stopped';
            document.getElementById('statusBadge').textContent = 'System Stopped';
        }

        function toggleSystem() {
            socket.emit('start_system', { mode: currentMode });
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('statusBadge').className = 'status-badge status-running';
            document.getElementById('statusBadge').textContent = 'System Running';
            startTime = Date.now();
        }

        function stopSystem() {
            socket.emit('stop_system');
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('statusBadge').className = 'status-badge status-stopped';
            document.getElementById('statusBadge').textContent = 'System Stopped';
            startTime = null;
        }

        function processImage() {
            const fileInput = document.getElementById('imageUpload');
            if (!fileInput.files[0]) {
                alert('Please select an image first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            document.getElementById('warningsContainer').innerHTML = '<div class="info-box">⏳ Processing image...</div>';
            
            fetch('/process_image', {
                method: 'POST',
                body: formData
            }).then(res => res.json()).then(data => {
                if (data.success) {
                    document.getElementById('warningsContainer').innerHTML = '<div class="info-box">✅ Image processed successfully!</div>';
                } else {
                    document.getElementById('warningsContainer').innerHTML = '<div class="warning-box">❌ Error: ' + data.error + '</div>';
                }
            }).catch(err => {
                document.getElementById('warningsContainer').innerHTML = '<div class="warning-box">❌ Error: ' + err + '</div>';
            });
        }

        function initializeVideo() {
            const fileInput = document.getElementById('videoUpload');
            if (!fileInput.files[0]) {
                alert('Please select a video first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('video', fileInput.files[0]);
            
            fetch('/initialize_video', {
                method: 'POST',
                body: formData
            }).then(res => res.json()).then(data => {
                if (data.success) {
                    alert('Video initialized: ' + data.frames + ' frames');
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }

        function playVideo() {
            socket.emit('play_video');
            document.getElementById('videoProgress').style.display = 'block';
        }

        function updateFrame(data) {
            if (data.detection_frame) {
                document.getElementById('detectionFrame').src = 'data:image/jpeg;base64,' + data.detection_frame;
                document.getElementById('detectionPlaceholder').style.display = 'none';
            }
            
            if (data.depth_frame) {
                document.getElementById('depthFrame').src = 'data:image/jpeg;base64,' + data.depth_frame;
                document.getElementById('depthPlaceholder').style.display = 'none';
            }
            
            if (data.bev_svg) {
                document.getElementById('bevContainer').innerHTML = data.bev_svg;
            }
            
            if (data.detections) {
                updateDetectionList(data.detections);
            }
            
            if (data.plot_3d) {
                Plotly.react('plot3D', data.plot_3d.data, data.plot_3d.layout, {displayModeBar: false});
            }
        }

        function updateMetrics(data) {
            document.getElementById('fpsMetric').textContent = data.fps.toFixed(1) + ' FPS';
            document.getElementById('objectsMetric').textContent = data.objects;
            
            if (startTime) {
                const uptime = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById('uptimeMetric').textContent = uptime + 's';
            }
        }

        function updateDecision(data) {
            const decisionEl = document.getElementById('decisionMetric');
            decisionEl.textContent = data.action.toUpperCase();
            
            if (data.action === 'forward') {
                decisionEl.style.color = '#10B981';
            } else if (data.action === 'stop') {
                decisionEl.style.color = '#EF4444';
            } else {
                decisionEl.style.color = '#F59E0B';
            }
            
            const warningsContainer = document.getElementById('warningsContainer');
            if (data.warnings && data.warnings.length > 0) {
                warningsContainer.innerHTML = data.warnings.map(w => 
                    '<div class="warning-box">' + w + '</div>'
                ).join('');
            } else {
                warningsContainer.innerHTML = '<div class="info-box">✅ System Normal - All Clear</div>';
            }
        }

        function updateDetectionList(detections) {
            const listEl = document.getElementById('detectionList');
            if (!detections || detections.length === 0) {
                listEl.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 2rem;">No objects detected</div>';
                return;
            }
            
            const colors = {
                'car': '#0a84ff', 'truck': '#5ac8fa', 'bus': '#ff9f0a',
                'person': '#ffd60a', 'motorcycle': '#ff375f', 'bicycle': '#5ac8fa'
            };
            
            listEl.innerHTML = detections.slice(0, 10).map(obj => `
                <div class="detection-item" style="border-left-color: ${colors[obj.class_name] || '#ffffff'}">
                    <div>
                        <div class="detection-class" style="color: ${colors[obj.class_name] || '#ffffff'}">${obj.class_name}</div>
                        <div class="detection-distance">Conf: ${(obj.confidence * 100).toFixed(0)}%</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="detection-distance">${obj.distance.toFixed(1)}m</div>
                        <div class="detection-distance">${obj.dimensions.width.toFixed(1)}×${obj.dimensions.height.toFixed(1)}m</div>
                    </div>
                </div>
            `).join('');
        }

        document.getElementById('imageUpload').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('imageName').textContent = file.name;
            }
        });

        document.getElementById('videoUpload').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('videoName').textContent = file.name;
            }
        });

        setInterval(() => {
            const now = new Date();
            document.getElementById('clock').textContent = now.toLocaleTimeString();
        }, 1000);

        Plotly.newPlot('analyticsChart', [{
            y: [],
            name: 'FPS',
            line: { color: '#5D5FEF', width: 2 }
        }, {
            y: [],
            name: 'Objects',
            line: { color: '#10B981', width: 2 },
            yaxis: 'y2'
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#8B8B9E', size: 11 },
            height: 300,
            margin: { l: 40, r: 40, t: 20, b: 40 },
            yaxis2: { overlaying: 'y', side: 'right' }
        }, { displayModeBar: false });
    </script>
</body>
</html>
'''


# ============================================================================
# SOCKET.IO EVENT HANDLERS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
    emit('status', {'message': 'Connected to EdgeDrive3D server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')


@socketio.on('set_mode')
def handle_set_mode(data):
    state.mode = data['mode']
    print(f'📊 Mode set to: {state.mode}')


@socketio.on('start_system')
def handle_start(data):
    with processing_lock:
        state.running = True
        state.start_time = time.time()
        
        if state.engine is None:
            print('🧠 Initializing Perception Engine...')
            state.engine = PerceptionEngine({
                'yolo_model': 'yolov8m.pt',
                'confidence': 0.4,
                'max_depth': 50.0
            })
            print('✅ Perception Engine ready!')
    
    # Start processing thread
    if state.process_thread is None or not state.process_thread.is_alive():
        state.process_thread = threading.Thread(target=process_loop, daemon=True)
        state.process_thread.start()
        print('▶ Processing thread started')


@socketio.on('stop_system')
def handle_stop():
    state.running = False
    state.video_playing = False
    print('⏹ System stopped')


@socketio.on('play_video')
def handle_play_video():
    state.video_playing = True
    print('▶ Video playing')


# ============================================================================
# PROCESSING LOOP
# ============================================================================

def process_loop():
    """Main processing loop"""
    from demo_generator import DemoGenerator
    
    demo_gen = DemoGenerator()
    frame_count = 0
    start_time = time.time()
    last_emit_time = 0
    
    while state.running:
        try:
            # Process based on mode
            if state.mode == 'demo':
                objects = demo_gen.generate_objects()
                decision = demo_gen.generate_decision(objects)
                bev = demo_gen.generate_bev(objects)
                depth = demo_gen.generate_depth()
                
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (18, 18, 26)
                cv2.putText(frame, "DEMO MODE", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                
                state.current_frame = frame
                state.current_objects = objects
                state.current_decision = decision
                state.current_bev = bev
                state.current_depth = cv2.applyColorMap((depth * 5).astype(np.uint8), cv2.COLORMAP_TURBO)
                state.current_detections = frame
                state.current_pointcloud = (demo_gen.generate_point_cloud()[0], None)
            
            elif state.mode == 'webcam':
                if state.webcam_cap is None:
                    state.webcam_cap = cv2.VideoCapture(0)

                ret, frame = state.webcam_cap.read()
                if ret and state.engine:
                    result = state.engine.process_frame(frame)
                    state.current_frame = result.detections_overlay if result.detections_overlay else frame
                    state.current_objects = result.objects_3d
                    state.current_decision = result.decision
                    state.current_bev = result.bev_image
                    state.current_depth = result.depth_colored if result.depth_colored else frame
                    state.current_detections = result.detections_overlay
                    state.current_pointcloud = result.point_cloud if result.point_cloud else demo_gen.generate_point_cloud()

            elif state.mode == 'pi_stream':
                # Receive frame from Pi UDP stream
                if state.pi_socket is None:
                    state.pi_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    state.pi_socket.bind(("0.0.0.0", 5000))
                    state.pi_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
                    state.pi_socket.settimeout(0.1)
                    print("📡 Pi UDP socket initialized on port 5000")
                
                try:
                    data, addr = state.pi_socket.recvfrom(65536)
                    if len(data) >= 4:
                        size = struct.unpack("I", data[:4])[0]
                        jpeg_data = data[4:4+size]
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None and state.engine:
                            result = state.engine.process_frame(frame)
                            state.current_frame = result.detections_overlay if result.detections_overlay else frame
                            state.current_objects = result.objects_3d
                            state.current_decision = result.decision
                            state.current_bev = result.bev_image
                            state.current_depth = result.depth_colored if result.depth_colored else frame
                            state.current_detections = result.detections_overlay
                            state.current_pointcloud = result.point_cloud if result.point_cloud else demo_gen.generate_point_cloud()
                            state.pi_frame_count += 1
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"❌ Pi stream error: {e}")

            elif state.mode == 'video' and state.video_playing and state.video_cap is not None:
                state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, state.video_frame)
                ret, frame = state.video_cap.read()
                
                if ret and state.engine:
                    result = state.engine.process_frame(frame)
                    state.current_frame = result.detections_overlay if result.detections_overlay else frame
                    state.current_objects = result.objects_3d
                    state.current_decision = result.decision
                    state.current_bev = result.bev_image
                    state.current_depth = result.depth_colored if result.depth_colored else frame
                    state.current_detections = result.detections_overlay
                    state.current_pointcloud = result.point_cloud if result.point_cloud else demo_gen.generate_point_cloud()
                    
                    state.video_frame += 1
                    if state.video_frame >= state.video_total_frames:
                        state.video_frame = 0
                    
                    # Emit progress
                    progress = state.video_frame / state.video_total_frames
                    socketio.emit('video_progress', {
                        'frame': state.video_frame,
                        'total': state.video_total_frames,
                        'progress': progress
                    })
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            state.fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Emit updates (limit to 30 FPS)
            current_time = time.time()
            if current_time - last_emit_time >= 0.033:  # 30 FPS
                emit_updates()
                last_emit_time = current_time
            
            time.sleep(0.033)  # 30 FPS
            
        except Exception as e:
            print(f"❌ Error in process loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


def emit_updates():
    """Emit all updates to connected clients"""
    try:
        detection_data = None
        depth_data = None
        
        if state.current_frame is not None:
            _, buffer = cv2.imencode('.jpg', state.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            detection_data = base64.b64encode(buffer).decode('utf-8')
        
        if state.current_depth is not None:
            _, buffer = cv2.imencode('.jpg', state.current_depth, [cv2.IMWRITE_JPEG_QUALITY, 80])
            depth_data = base64.b64encode(buffer).decode('utf-8')
        
        bev_svg = None
        if state.current_bev is not None and state.current_objects:
            from dashboard.app import create_bev_svg
            bev_svg = create_bev_svg(state.current_objects)
        
        plot_3d = None
        if state.current_pointcloud:
            from dashboard.app import create_3d_plot
            points = state.current_pointcloud[0] if isinstance(state.current_pointcloud, tuple) else state.current_pointcloud
            fig = create_3d_plot(points, state.current_objects)
            plot_3d = {'data': fig.data, 'layout': fig.layout}
        
        socketio.emit('frame_update', {
            'detection_frame': detection_data,
            'depth_frame': depth_data,
            'bev_svg': bev_svg,
            'detections': [{
                'class_name': obj.class_name,
                'confidence': obj.confidence,
                'distance': obj.distance,
                'dimensions': obj.dimensions
            } for obj in state.current_objects],
            'plot_3d': plot_3d
        })
        
        socketio.emit('metrics_update', {
            'fps': state.fps,
            'objects': len(state.current_objects)
        })
        
        socketio.emit('decision_update', {
            'action': state.current_decision.get('action', 'stop'),
            'warnings': state.current_decision.get('warnings', [])
        })
        
    except Exception as e:
        print(f"❌ Error emitting updates: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save and process image
        img = Image.open(file.stream)
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if state.engine is None:
            state.engine = PerceptionEngine({
                'yolo_model': 'yolov8m.pt',
                'confidence': 0.4,
                'max_depth': 50.0
            })
        
        result = state.engine.process_frame(img_np)
        
        state.current_frame = result.detections_overlay if result.detections_overlay else img_np
        state.current_objects = result.objects_3d
        state.current_decision = result.decision
        state.current_bev = result.bev_image
        state.current_depth = result.depth_colored if result.depth_colored else img_np
        state.current_detections = result.detections_overlay
        state.current_pointcloud = result.point_cloud if result.point_cloud else DemoGenerator.generate_point_cloud()
        
        # Emit updates immediately
        emit_updates()
        
        return jsonify({'success': True, 'objects': len(result.objects_3d)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/initialize_video', methods=['POST'])
def initialize_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save video
        upload_dir = Path('output/uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = upload_dir / file.filename
        file.save(str(video_path))
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        state.video_path = str(video_path)
        state.video_cap = cap
        state.video_total_frames = total_frames
        state.video_frame = 0
        
        return jsonify({'success': True, 'frames': total_frames, 'path': str(video_path)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    if state.current_frame is None:
        return jsonify({'error': 'No frame to save'}), 400
    
    snapshot_dir = Path('output/snapshots')
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    snapshot_path = snapshot_dir / f'snapshot_{timestamp}.jpg'
    
    cv2.imwrite(str(snapshot_path), state.current_frame)
    
    return jsonify({'success': True, 'path': str(snapshot_path)})


# ============================================================================
# DEMO DATA GENERATOR
# ============================================================================

class DemoGenerator:
    CLASSES = ['car', 'person', 'truck', 'motorcycle', 'bus', 'bicycle']
    
    @staticmethod
    def generate_objects(num=None):
        if num is None:
            num = np.random.randint(2, 6)
        objects = []
        for i in range(num):
            cls = np.random.choice(DemoGenerator.CLASSES)
            distance = np.random.uniform(3, 30)
            obj = Object3D(
                class_id=i, class_name=cls, confidence=np.random.uniform(0.75, 0.95),
                bbox_2d=(100, 100, 200, 200),
                position_3d=np.array([np.random.uniform(-10, 10), 0, distance]),
                distance=distance,
                dimensions={'width': np.random.uniform(1.5, 2.5), 'height': np.random.uniform(1.2, 2.0), 'length': np.random.uniform(3, 5)},
                bbox_3d=np.array([[-1,-1,distance-2],[1,-1,distance-2],[1,1,distance-2],[-1,1,distance-2],[-1,-1,distance+2],[1,-1,distance+2],[1,1,distance+2],[-1,1,distance+2]])
            )
            objects.append(obj)
        return sorted(objects, key=lambda x: x.distance)
    
    @staticmethod
    def generate_decision(objects):
        if not objects:
            return {'action': 'forward', 'speed': 180, 'reason': 'clear', 'warnings': []}
        closest = objects[0]
        if closest.distance < 2.0:
            return {'action': 'stop', 'speed': 0, 'reason': 'obstacle', 'warnings': [f"⚠️ CRITICAL: {closest.class_name} at {closest.distance:.1f}m"]}
        elif closest.distance < 5.0:
            if closest.position_3d[0] < -1:
                return {'action': 'right', 'speed': 100, 'reason': 'avoid_left', 'warnings': [f"⚠️ {closest.class_name} on left at {closest.distance:.1f}m"]}
            elif closest.position_3d[0] > 1:
                return {'action': 'left', 'speed': 100, 'reason': 'avoid_right', 'warnings': [f"⚠️ {closest.class_name} on right at {closest.distance:.1f}m"]}
            return {'action': 'stop', 'speed': 0, 'reason': 'obstacle_ahead', 'warnings': [f"⚠️ {closest.class_name} ahead at {closest.distance:.1f}m"]}
        return {'action': 'forward', 'speed': 180, 'reason': 'clear', 'warnings': []}
    
    @staticmethod
    def generate_bev(objects):
        bev = np.zeros((500, 700, 3), dtype=np.uint8)
        for y in range(0, 500, 40):
            cv2.line(bev, (0, y), (700, y), (25, 25, 35), 1)
        for x in range(0, 700, 40):
            cv2.line(bev, (x, 0), (x, 500), (25, 25, 35), 1)
        cv2.line(bev, (350, 50), (350, 450), (60, 60, 70), 2)
        cv2.circle(bev, (350, 450), 18, (48, 209, 88), -1)
        cv2.putText(bev, "EGO", (335, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (48, 209, 88), 1)
        for obj in objects:
            x = int(350 + obj.position_3d[0] * 18)
            y = int(450 - obj.distance * 12)
            cv2.circle(bev, (x, y), 14, (255, 255, 255), -1)
            cv2.putText(bev, f"{obj.class_name[:3]} {obj.distance:.0f}m", (x-20, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return bev
    
    @staticmethod
    def generate_depth():
        h, w = 480, 640
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        depth = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        depth = depth / depth.max()
        depth = 1 - depth
        return (depth * 50).astype(np.float32)
    
    @staticmethod
    def generate_point_cloud(num=2000):
        x = np.random.uniform(-15, 15, num)
        y = np.random.uniform(-2, 2, num)
        z = np.random.uniform(1, 40, num)
        return np.column_stack([x, y, z]), np.random.rand(num, 3)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🚗 EdgeDrive3D Real-Time Dashboard")
    print("="*60)
    print("\n  ✅ Starting server...")
    print("  🌐 Open: http://localhost:5000")
    print("  🔌 Socket.IO: Enabled (real-time updates)")
    print("  📹 Modes: Demo, Webcam, Pi Stream, Image, Video")
    print("  📡 Pi Stream: Listening on UDP port 5000")
    print("\n  Press Ctrl+C to stop\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
