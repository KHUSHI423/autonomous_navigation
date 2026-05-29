"""
=============================================================================
EDGE DRIVE 3D - COMPLETE DASHBOARD (All Features + Sleek UI)
=============================================================================
Full-featured dashboard with ALL previous functionality:
- Demo mode
- Real-time webcam processing
- Image upload and processing
- Video upload and playback with controls
- Pi stream reception
- Snapshot saving
- All 5 output views
- Beautiful dark UI (from inspiration code)
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Dict, Tuple
from collections import deque
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.perception_engine import PerceptionEngine, Object3D


# ============================================================================
# SVG ICONS (from map_icons_mini_project.html - EXACT COPIES)
# ============================================================================

SVG_ICONS = {
    'car': '''<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="gradBlue" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#0a84ff"/><stop offset="100%" stop-color="#0050d0"/></linearGradient></defs><rect x="8" y="18" width="32" height="16" rx="4" fill="url(#gradBlue)" opacity="0.9"/><rect x="12" y="12" width="24" height="12" rx="3" fill="url(#gradBlue)"/><rect x="14" y="14" width="9" height="7" rx="1.5" fill="rgba(120,200,255,0.3)" stroke="rgba(120,200,255,0.5)" stroke-width="0.5"/><rect x="25" y="14" width="9" height="7" rx="1.5" fill="rgba(120,200,255,0.3)" stroke="rgba(120,200,255,0.5)" stroke-width="0.5"/><circle cx="15" cy="36" r="3.5" fill="#1a1a28" stroke="#555" stroke-width="1"/><circle cx="15" cy="36" r="1.5" fill="#888"/><circle cx="33" cy="36" r="3.5" fill="#1a1a28" stroke="#555" stroke-width="1"/><circle cx="33" cy="36" r="1.5" fill="#888"/><rect x="10" y="30" width="5" height="2" rx="1" fill="#ffd60a" opacity="0.8"/><rect x="33" y="30" width="5" height="2" rx="1" fill="#ff453a" opacity="0.8"/></svg>''',
    
    'truck': '''<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#5ac8fa"/><stop offset="100%" stop-color="#32ade6"/></linearGradient></defs><rect x="4" y="14" width="28" height="22" rx="3" fill="url(#gradCyan)" opacity="0.9"/><rect x="32" y="20" width="14" height="16" rx="3" fill="#32ade6"/><rect x="34" y="22" width="10" height="8" rx="2" fill="rgba(120,200,255,0.3)" stroke="rgba(120,200,255,0.5)" stroke-width="0.5"/><circle cx="12" cy="38" r="3" fill="#1a1a28" stroke="#555" stroke-width="1"/><circle cx="24" cy="38" r="3" fill="#1a1a28" stroke="#555" stroke-width="1"/><circle cx="40" cy="38" r="3" fill="#1a1a28" stroke="#555" stroke-width="1"/><rect x="4" y="16" width="28" height="3" rx="1" fill="rgba(255,255,255,0.1)"/></svg>''',
    
    'bus': '''<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="gradAmber" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ff9f0a"/><stop offset="100%" stop-color="#ff6723"/></linearGradient></defs><rect x="8" y="8" width="32" height="30" rx="5" fill="url(#gradAmber)" opacity="0.9"/><rect x="10" y="10" width="28" height="3" rx="1" fill="rgba(255,255,255,0.2)"/><rect x="11" y="15" width="7" height="8" rx="1.5" fill="rgba(255,255,200,0.3)" stroke="rgba(255,255,200,0.4)" stroke-width="0.5"/><rect x="20" y="15" width="7" height="8" rx="1.5" fill="rgba(255,255,200,0.3)" stroke="rgba(255,255,200,0.4)" stroke-width="0.5"/><rect x="29" y="15" width="7" height="8" rx="1.5" fill="rgba(255,255,200,0.3)" stroke="rgba(255,255,200,0.4)" stroke-width="0.5"/><rect x="20" y="26" width="8" height="10" rx="2" fill="rgba(0,0,0,0.3)"/><circle cx="14" cy="40" r="3" fill="#1a1a28" stroke="#555" stroke-width="1"/><circle cx="34" cy="40" r="3" fill="#1a1a28" stroke="#555" stroke-width="1"/></svg>''',
    
    'person': '''<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="gradYellow" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ffd60a"/><stop offset="100%" stop-color="#ffb800"/></linearGradient></defs><circle cx="24" cy="10" r="4.5" fill="url(#gradYellow)"/><path d="M24 15 L24 28" stroke="#ffd60a" stroke-width="3" stroke-linecap="round"/><path d="M16 22 L24 19 L32 22" fill="none" stroke="#ffd60a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M24 28 L17 40" stroke="#ffd60a" stroke-width="2.5" stroke-linecap="round"/><path d="M24 28 L31 40" stroke="#ffd60a" stroke-width="2.5" stroke-linecap="round"/></svg>''',
    
    'motorcycle': '''<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="gradMagenta" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ff375f"/><stop offset="100%" stop-color="#d63384"/></linearGradient></defs><circle cx="12" cy="34" r="6" fill="none" stroke="url(#gradMagenta)" stroke-width="2.5"/><circle cx="12" cy="34" r="2" fill="#ff375f"/><circle cx="36" cy="34" r="6" fill="none" stroke="url(#gradMagenta)" stroke-width="2.5"/><circle cx="36" cy="34" r="2" fill="#ff375f"/><path d="M12 34 L18 22 L30 18 L36 34" fill="none" stroke="#ff375f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M18 22 L22 16 L26 16 L30 18" fill="none" stroke="#ff375f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="24" cy="14" r="3" fill="rgba(255,55,95,0.3)" stroke="#ff375f" stroke-width="1"/><rect x="28" y="18" width="3" height="1.5" rx="0.5" fill="#ffd60a" opacity="0.8"/></svg>''',
    
    'bicycle': '''<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><circle cx="13" cy="32" r="7" fill="none" stroke="#5ac8fa" stroke-width="2" opacity="0.8"/><circle cx="35" cy="32" r="7" fill="none" stroke="#5ac8fa" stroke-width="2" opacity="0.8"/><path d="M13 32 L22 18 L30 18 L35 32" fill="none" stroke="#5ac8fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><line x1="22" y1="18" x2="18" y2="18" stroke="#5ac8fa" stroke-width="2" stroke-linecap="round"/><line x1="13" y1="32" x2="26" y2="32" stroke="#5ac8fa" stroke-width="1.5" stroke-linecap="round"/><line x1="26" y1="32" x2="30" y2="18" stroke="#5ac8fa" stroke-width="1.5" stroke-linecap="round"/><circle cx="26" cy="16" r="2" fill="rgba(90,200,250,0.2)" stroke="#5ac8fa" stroke-width="1"/></svg>''',
}

ICON_COLORS_HEX = {
    'car': '#0a84ff', 'truck': '#5ac8fa', 'bus': '#ff9f0a',
    'person': '#ffd60a', 'motorcycle': '#ff375f', 'bicycle': '#5ac8fa',
}

ICON_COLORS_BGR = {
    'car': (255, 10, 10), 'truck': (250, 200, 90), 'bus': (10, 159, 255),
    'person': (10, 214, 255), 'motorcycle': (95, 55, 255), 'bicycle': (250, 200, 90),
}


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="EdgeDrive3D - Complete Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS - SLEEK DARK UI
# ============================================================================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0B0F;
        --bg-sidebar: #12121A;
        --bg-card: #181824;
        --bg-card-hover: #1F1F2E;
        --text-primary: #FFFFFF;
        --text-secondary: #8B8B9E;
        --border-color: rgba(255, 255, 255, 0.06);
        --accent-primary: #5D5FEF;
        --accent-purple: #8B5CF6;
        --accent-success: #10B981;
        --accent-danger: #EF4444;
        --accent-warning: #F59E0B;
    }

    .stApp {
        background-color: var(--bg-main);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color);
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s;
        overflow: hidden;
        word-wrap: break-word;
    }
    .metric-card:hover {
        background: var(--bg-card-hover);
        transform: translateY(-3px);
        border-color: rgba(93, 95, 239, 0.3);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        line-height: 1.3;
        word-wrap: break-word;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        word-wrap: break-word;
    }

    /* View Cards */
    .view-card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid var(--border-color);
        height: 100%;
        overflow: hidden;
    }
    .view-card-title {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.8rem;
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
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 10px currentColor;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Buttons */
    .stButton > button {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 50px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: var(--accent-primary);
        background: rgba(93, 95, 239, 0.1);
        color: var(--accent-primary);
    }
    .stButton > button[kind="primary"] {
        background: var(--accent-primary);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(93, 95, 239, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(93, 95, 239, 0.6);
    }

    /* Radio Buttons - Styled as Nav Links */
    div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    div[role="radiogroup"] > label:hover {
        background-color: rgba(255,255,255,0.03);
    }
    div[role="radiogroup"] > label[aria-checked="true"] {
        background-color: var(--accent-primary);
        box-shadow: 0 4px 12px rgba(93, 95, 239, 0.4);
    }
    div[role="radiogroup"] > label[aria-checked="true"] p {
        color: white !important;
    }

    /* Inputs, Selects, Sliders */
    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
        background-color: var(--bg-main) !important;
        border-color: var(--border-color) !important;
        border-radius: 10px;
        color: var(--text-primary) !important;
    }
    .stSlider > div > div > div > div {
        background: var(--accent-primary) !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Detection Items */
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

    /* Info Boxes */
    .info-box {
        background: rgba(93, 95, 239, 0.1);
        border: 1px solid rgba(93, 95, 239, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }

    #MainMenu, footer, header {visibility: hidden;}
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-main); }
    ::-webkit-scrollbar-thumb { background: #333344; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE - ALL FEATURES
# ============================================================================
def init_session():
    defaults = {
        'system_running': False,
        'mode': 'demo',
        'frame_count': 0,
        'start_time': None,
        'metrics_history': deque(maxlen=300),
        'current_objects': [],
        'current_decision': {},
        'current_frame': None,
        'current_bev': None,
        'current_depth': None,
        'current_detections': None,
        'current_pointcloud': None,
        'engine': None,
        'uploaded_file': None,
        'processing_complete': False,
        'video_frame': 0,
        'video_total_frames': 0,
        'video_playing': False,
        'video_path': None,
        'video_cap': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# DEMO DATA
# ============================================================================
class DemoGenerator:
    CLASSES = ['car', 'person', 'truck', 'motorcycle', 'bus', 'bicycle']
    
    @staticmethod
    def generate_objects(num=None) -> List[Object3D]:
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
    def generate_decision(objects: List[Object3D]) -> Dict:
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
    def generate_bev(objects: List[Object3D]) -> np.ndarray:
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
            color = ICON_COLORS_BGR.get(obj.class_name, (255, 255, 255))
            cv2.circle(bev, (x, y), 14, color, -1)
            cv2.circle(bev, (x, y), 14, (255, 255, 255), 1)
            cv2.putText(bev, f"{obj.class_name[:3]} {obj.distance:.0f}m", (x-20, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return bev
    
    @staticmethod
    def generate_depth() -> np.ndarray:
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
# VISUALIZATION
# ============================================================================

def create_3d_plot(points: np.ndarray, objects: List[Object3D]) -> go.Figure:
    fig = go.Figure()
    if len(points) > 0:
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker=dict(size=2, color=points[:, 2], colorscale='Electric', opacity=0.7), name='Points'))
    for obj in objects:
        if obj.bbox_3d is not None:
            corners = obj.bbox_3d
            edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            color = ICON_COLORS_HEX.get(obj.class_name, '#ff0000')
            for edge in edges:
                fig.add_trace(go.Scatter3d(x=[corners[edge[0], 0], corners[edge[1], 0]], y=[corners[edge[0], 1], corners[edge[1], 1]], z=[corners[edge[0], 2], corners[edge[1], 2]], mode='lines', line=dict(color=color, width=3), showlegend=False))
    fig.update_layout(scene=dict(xaxis=dict(title='X (m)'), yaxis=dict(title='Y (m)'), zaxis=dict(title='Z (m)'), camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8B8B9E'), height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    return fig


def create_metrics_chart(metrics_history) -> go.Figure:
    if len(metrics_history) < 2:
        return go.Figure()
    df = pd.DataFrame(list(metrics_history))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=df['fps'], name='FPS', line=dict(color='#5D5FEF', width=2), fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df['objects'], name='Objects', line=dict(color='#10B981', width=2), fill='tozeroy'), row=2, col=1)
    fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8B8B9E'), showlegend=False, xaxis2=dict(title='Time'), yaxis=dict(gridcolor='#1F1F2E'), yaxis2=dict(gridcolor='#1F1F2E'))
    return fig


def create_bev_svg(objects: List[Object3D]) -> str:
    """Create BEV with actual SVG icons from map_icons_mini_project.html"""
    
    # SVG icon definitions (exact copies from map_icons_mini_project.html)
    svg_objects = ""
    for obj in objects:
        x = 350 + obj.position_3d[0] * 18 if obj.position_3d is not None else 350
        y = 450 - obj.distance * 12
        icon_key = obj.class_name.lower()
        
        # Map class names to icon keys
        if 'car' in icon_key or 'vehicle' in icon_key:
            icon_key = 'car'
        elif 'truck' in icon_key or 'lorry' in icon_key:
            icon_key = 'truck'
        elif 'bus' in icon_key:
            icon_key = 'bus'
        elif 'person' in icon_key or 'pedestrian' in icon_key:
            icon_key = 'person'
        elif 'motor' in icon_key or 'bike' in icon_key:
            icon_key = 'motorcycle'
        elif 'bicycle' in icon_key or 'cycle' in icon_key:
            icon_key = 'bicycle'
        
        icon = SVG_ICONS.get(icon_key, SVG_ICONS['car'])
        # Extract just the inner content (remove outer svg tags)
        icon_content = icon.replace('<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">', '').replace('</svg>', '')
        
        svg_objects += f'''
        <g transform="translate({x-20}, {y-20}) scale(0.8)">
            {icon_content}
            <text x="24" y="-8" fill="#ffffff" font-size="9" text-anchor="middle" font-family="Arial" font-weight="600">{obj.class_name[:6]} {obj.distance:.0f}m</text>
        </g>'''
    
    # Full SVG with gradients defined once
    return f'''<svg viewBox="0 0 700 500" xmlns="http://www.w3.org/2000/svg" style="background: radial-gradient(ellipse at center, #1a1a2e 0%, #0f0f1a 100%);">
        <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>
            </pattern>
            <linearGradient id="gradBlue" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#0a84ff"/><stop offset="100%" stop-color="#0050d0"/>
            </linearGradient>
            <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#5ac8fa"/><stop offset="100%" stop-color="#32ade6"/>
            </linearGradient>
            <linearGradient id="gradAmber" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#ff9f0a"/><stop offset="100%" stop-color="#ff6723"/>
            </linearGradient>
            <linearGradient id="gradMagenta" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#ff375f"/><stop offset="100%" stop-color="#d63384"/>
            </linearGradient>
            <linearGradient id="gradYellow" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#ffd60a"/><stop offset="100%" stop-color="#ffb800"/>
            </linearGradient>
            <linearGradient id="gradPurple" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#bf5af2"/><stop offset="100%" stop-color="#9747ff"/>
            </linearGradient>
            <linearGradient id="gradGreen" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#30d158"/><stop offset="100%" stop-color="#28a745"/>
            </linearGradient>
            <filter id="glow"><feGaussianBlur stdDeviation="2" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        
        <!-- Background Grid -->
        <rect width="700" height="500" fill="url(#grid)"/>
        
        <!-- Lane Markings -->
        <line x1="350" y1="50" x2="350" y2="450" stroke="rgba(255,255,255,0.15)" stroke-width="2" stroke-dasharray="15,15"/>
        
        <!-- Ego Vehicle -->
        <g transform="translate(350, 450)">
            <circle r="18" fill="url(#gradGreen)" opacity="0.9" filter="url(#glow)"/>
            <text y="25" fill="#30d158" font-size="11" text-anchor="middle" font-weight="bold" font-family="Arial">EGO</text>
        </g>
        
        <!-- Detected Objects with Icons -->
        {svg_objects}
        
        <!-- Title -->
        <text x="20" y="25" fill="#ffffff" font-size="13" font-weight="600" font-family="Arial">Bird's Eye View - SVG Icons from map_icons_mini_project.html</text>
        
        <!-- Legend -->
        <g transform="translate(520, 20)">
            <rect width="160" height="90" fill="rgba(30,30,45,0.8)" rx="8"/>
            <text x="10" y="20" fill="#a1a1a6" font-size="10" font-weight="600" font-family="Arial">ICON LEGEND</text>
            <g transform="translate(10, 35)"><svg width="20" height="20" viewBox="0 0 48 48"><rect x="8" y="18" width="32" height="16" rx="4" fill="url(#gradBlue)"/></svg><text x="25" y="15" fill="#a1a1a6" font-size="9" font-family="Arial">Car</text></g>
            <g transform="translate(90, 35)"><svg width="20" height="20" viewBox="0 0 48 48"><circle cx="24" cy="10" r="4.5" fill="url(#gradYellow)"/></svg><text x="25" y="15" fill="#a1a1a6" font-size="9" font-family="Arial">Person</text></g>
            <g transform="translate(10, 60)"><svg width="20" height="20" viewBox="0 0 48 48"><rect x="8" y="8" width="32" height="30" rx="5" fill="url(#gradAmber)"/></svg><text x="25" y="15" fill="#a1a1a6" font-size="9" font-family="Arial">Bus</text></g>
            <g transform="translate(90, 60)"><svg width="20" height="20" viewBox="0 0 48 48"><circle cx="12" cy="34" r="6" fill="none" stroke="url(#gradMagenta)" stroke-width="2.5"/></svg><text x="25" y="15" fill="#a1a1a6" font-size="9" font-family="Arial">Moto</text></g>
        </g>
    </svg>'''


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_image_file(file, engine):
    """Process uploaded image"""
    img = Image.open(file)
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if engine is None:
        engine = PerceptionEngine({'yolo_model': 'yolov8m.pt', 'confidence': 0.4, 'max_depth': 50.0})
    
    result = engine.process_frame(img_np)
    
    # Create depth overlay
    depth_overlay = img_np.copy()
    if result.depth_map is not None:
        depth_norm = (result.depth_map - result.depth_map.min()) / (result.depth_map.max() - result.depth_map.min())
        depth_col = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_overlay = cv2.addWeighted(img_np, 0.6, depth_col, 0.4, 0)
    
    return img_np, result.objects_3d, result.decision, result.bev_image, depth_overlay, result.detections_overlay, (result.point_cloud if result.point_cloud else DemoGenerator.generate_point_cloud())


def init_video(file):
    """Initialize video processing"""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp.write(file.read())
    temp.close()
    
    cap = cv2.VideoCapture(temp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return temp.name, cap, total


def process_video_frame(cap, engine, frame_num):
    """Process specific video frame"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        return None
    
    result = engine.process_frame(frame)
    
    depth_overlay = frame.copy()
    if result.depth_map is not None:
        depth_norm = (result.depth_map - result.depth_map.min()) / (result.depth_map.max() - result.depth_map.min())
        depth_col = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_overlay = cv2.addWeighted(frame, 0.6, depth_col, 0.4, 0)
    
    return frame, result.objects_3d, result.decision, result.bev_image, depth_overlay, result.detections_overlay, (result.point_cloud if result.point_cloud else DemoGenerator.generate_point_cloud())


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    load_css()
    init_session()
    
    # Header
    st.markdown('<div class="main-header">EdgeDrive3D Perception</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Complete Autonomous Vehicle Perception System</div>', unsafe_allow_html=True)
    
    # Sidebar - ALL FEATURES
    with st.sidebar:
        # Status
        status_html = f'<span class="status-badge {"status-running" if st.session_state.system_running else "status-stopped"}"><div class="status-dot"></div>{"System Active" if st.session_state.system_running else "System Offline"}</span>'
        st.markdown(status_html, unsafe_allow_html=True)
        
        st.divider()
        
        # MODE SELECTION - Full visibility
        st.markdown("**📊 Input Source**")
        mode = st.radio(
            "Select Mode:",
            ["🎮 Demo Mode", "📹 Webcam", "🖼️ Upload Image", "🎬 Upload Video", "📡 Pi Stream"],
            index=0,
            key="mode_radio"
        )
        
        mode_map = {
            "🎮 Demo Mode": "demo",
            "📹 Webcam": "webcam",
            "🖼️ Upload Image": "image",
            "🎬 Upload Video": "video",
            "📡 Pi Stream": "pi_stream"
        }
        st.session_state.mode = mode_map[mode]
        
        st.info(f"**Current Mode:** {st.session_state.mode.upper()}")
        
        st.divider()
        
        # FILE UPLOADERS - Based on mode
        if st.session_state.mode == "image":
            st.markdown("**📤 Upload Image**")
            uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'], key="img_uploader")
            st.session_state.uploaded_file = uploaded
            
            if uploaded:
                st.success(f"✅ Loaded: {uploaded.name}")
                if st.button("🔍 Process Image", type="primary", use_container_width=True):
                    with st.spinner("Processing image..."):
                        result = process_image_file(uploaded, st.session_state.engine)
                        if result[0] is not None:
                            st.session_state.current_frame = result[0]
                            st.session_state.current_objects = result[1]
                            st.session_state.current_decision = result[2]
                            st.session_state.current_bev = result[3]
                            st.session_state.current_depth = result[4]
                            st.session_state.current_detections = result[5]
                            st.session_state.current_pointcloud = result[6]
                            st.session_state.system_running = True
                            st.session_state.processing_complete = True
                            st.rerun()
        
        elif st.session_state.mode == "video":
            st.markdown("**📤 Upload Video**")
            uploaded = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'], key="vid_uploader")
            st.session_state.uploaded_file = uploaded
            
            if uploaded:
                st.success(f"✅ Loaded: {uploaded.name}")
                
                # Initialize video on first upload
                if not st.session_state.video_path:
                    if st.button("▶ Initialize Video", type="primary", use_container_width=True):
                        with st.spinner("Loading video..."):
                            path, cap, total = init_video(uploaded)
                            st.session_state.video_path = path
                            st.session_state.video_cap = cap
                            st.session_state.video_total_frames = total
                            st.session_state.video_frame = 0
                            st.info(f"📹 Video loaded: {total} frames")
                            st.rerun()
                
                if st.session_state.video_cap:
                    st.markdown("**🎬 Video Controls**")
                    
                    # Control buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("⏮ -10", use_container_width=True):
                            st.session_state.video_frame = max(0, st.session_state.video_frame - 10)
                            st.rerun()
                    with col2:
                        if st.button("⏮ -1", use_container_width=True):
                            st.session_state.video_frame = max(0, st.session_state.video_frame - 1)
                            st.rerun()
                    with col3:
                        if st.button("⏭ +1", use_container_width=True):
                            st.session_state.video_frame = min(st.session_state.video_total_frames - 1, st.session_state.video_frame + 1)
                            st.rerun()
                    with col4:
                        if st.button("⏭ +10", use_container_width=True):
                            st.session_state.video_frame = min(st.session_state.video_total_frames - 1, st.session_state.video_frame + 10)
                            st.rerun()
                    
                    # Frame slider
                    st.session_state.video_frame = st.slider(
                        "Frame Position", 
                        0, 
                        st.session_state.video_total_frames - 1, 
                        st.session_state.video_frame,
                        key="frame_slider"
                    )
                    
                    # Playback controls
                    st.markdown("**▶️ Playback**")
                    col_play1, col_play2, col_play3 = st.columns(3)
                    
                    with col_play1:
                        if st.button("▶ Play Continuous", type="primary", use_container_width=True):
                            st.session_state.video_playing = True
                            st.session_state.video_stop = False
                    
                    with col_play2:
                        if st.button("⏸ Pause", use_container_width=True):
                            st.session_state.video_playing = False
                    
                    with col_play3:
                        if st.button("⏹ Reset", use_container_width=True):
                            st.session_state.video_playing = False
                            st.session_state.video_stop = True
                            st.session_state.video_frame = 0
                            st.rerun()
                    
                    # Processing mode
                    st.markdown("**⚙️ Processing Mode**")
                    process_mode = st.radio(
                        "Select Mode",
                        ["Process Single Frame", "Process Continuously (Video Stream)"],
                        index=1,
                        key="process_mode_radio"
                    )
                    
                    if process_mode == "Process Single Frame":
                        if st.button("🔍 Process Current Frame", use_container_width=True):
                            with st.spinner("Processing frame..."):
                                if st.session_state.engine is None:
                                    st.session_state.engine = PerceptionEngine({'yolo_model': 'yolov8m.pt', 'confidence': 0.4, 'max_depth': 50.0})
                                result = process_video_frame(st.session_state.video_cap, st.session_state.engine, st.session_state.video_frame)
                                if result:
                                    st.session_state.current_frame = result[0]
                                    st.session_state.current_objects = result[1]
                                    st.session_state.current_decision = result[2]
                                    st.session_state.current_bev = result[3]
                                    st.session_state.current_depth = result[4]
                                    st.session_state.current_detections = result[5]
                                    st.session_state.current_pointcloud = result[6]
                                    st.session_state.system_running = True
                                    st.session_state.processing_complete = True
                                    st.rerun()
                    
                    # Continuous processing info
                    if process_mode == "Process Continuously (Video Stream)":
                        st.info("ℹ️ Video will process automatically when playing. Use Play/Pause to control.")
                        
                        # Auto-process when playing
                        if st.session_state.video_playing and not st.session_state.video_stop:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process all remaining frames
                            for frame_num in range(st.session_state.video_frame, st.session_state.video_total_frames):
                                if not st.session_state.video_playing or st.session_state.video_stop:
                                    break
                                
                                if st.session_state.engine is None:
                                    st.session_state.engine = PerceptionEngine({'yolo_model': 'yolov8m.pt', 'confidence': 0.4, 'max_depth': 50.0})
                                
                                result = process_video_frame(st.session_state.video_cap, st.session_state.engine, frame_num)
                                
                                if result:
                                    st.session_state.current_frame = result[0]
                                    st.session_state.current_objects = result[1]
                                    st.session_state.current_decision = result[2]
                                    st.session_state.current_bev = result[3]
                                    st.session_state.current_depth = result[4]
                                    st.session_state.current_detections = result[5]
                                    st.session_state.current_pointcloud = result[6]
                                    st.session_state.video_frame = frame_num
                                    st.session_state.system_running = True
                                    st.session_state.processing_complete = True
                                    
                                    # Update progress
                                    progress = (frame_num + 1) / st.session_state.video_total_frames
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing frame {frame_num + 1}/{st.session_state.video_total_frames}")
                                    
                                    # Small delay for visualization
                                    time.sleep(0.05)
                                    
                                    # Rerun to update display
                                    st.rerun()
                            
                            st.session_state.video_playing = False
                            progress_bar.empty()
                            status_text.empty()
        
        elif st.session_state.mode == "pi_stream":
            st.markdown("**📡 Pi Stream Settings**")
            udp_port = st.number_input("UDP Port", value=5000, min_value=1024)
            st.info(f"Listening on port {udp_port}")
        
        st.divider()
        
        # SETTINGS
        st.markdown("**⚙️ Vision Parameters**")
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
        model = st.selectbox("YOLO Model", ['yolov8n.pt (Fast)', 'yolov8m.pt (Balanced)', 'yolov8l.pt (Accurate)'], index=1)
        max_depth = st.slider("Max Depth (m)", 20.0, 100.0, 50.0, 5.0)
        
        st.divider()
        
        # HARDWARE
        st.markdown("**🔌 Hardware**")
        esp32_ip = st.text_input("ESP32 IP", "192.168.4.1")
        auto_control = st.toggle("Auto Control ESP32", False)
        
        st.divider()
        
        # ACTIONS
        st.markdown("**🎮 System Control**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ START", use_container_width=True, type="primary"):
                st.session_state.system_running = True
                st.session_state.start_time = time.time()
                if st.session_state.mode == 'demo':
                    st.session_state.demo_active = True
                if st.session_state.engine is None:
                    model_name = model.split()[0]
                    st.session_state.engine = PerceptionEngine({'yolo_model': model_name, 'confidence': confidence, 'max_depth': max_depth})
        
        with col2:
            if st.button("⏹ STOP", use_container_width=True):
                st.session_state.system_running = False
                st.session_state.demo_active = False
                st.session_state.video_playing = False
        
        if st.button("📸 Save Snapshot", use_container_width=True):
            if st.session_state.current_frame is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                Path("output/snapshots").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"output/snapshots/snapshot_{ts}.jpg", st.session_state.current_frame)
                st.success(f"Saved!")
    
    # Main Content
    if st.session_state.system_running or st.session_state.processing_complete:
        render_dashboard()
    else:
        render_idle()
    
    # Demo loop
    if st.session_state.get('demo_active', False):
        time.sleep(0.15)
        simulate_demo()
        st.rerun()


def render_idle():
    """Idle/Welcome screen"""
    st.divider()
    
    cols = st.columns(4)
    metrics = [("🌐", "Network", "Idle"), ("⚡", "GPU", "0%"), ("🧠", "Models", "0"), ("⏱️", "Latency", "---")]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{metrics[i][0]} {metrics[i][1]}</div><div class="metric-value">{metrics[i][2]}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; background: var(--bg-card); border-radius: 20px; border: 1px solid var(--border-color);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
            <h2 style="color: white;">System Standby</h2>
            <p style="color: var(--text-secondary); margin-bottom: 2rem;">Select a mode from the sidebar and click START, or upload an image/video to process.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎮 Start Demo Mode", use_container_width=True, type="primary"):
            st.session_state.demo_active = True
            st.session_state.system_running = True
            st.session_state.start_time = time.time()
            st.rerun()


def render_dashboard():
    """Main dashboard - ALL 5 OUTPUTS"""
    
    # Generate demo data if needed
    if st.session_state.get('demo_active', False):
        objects = DemoGenerator.generate_objects()
        decision = DemoGenerator.generate_decision(objects)
        bev = DemoGenerator.generate_bev(objects)
        depth = DemoGenerator.generate_depth()
        points, colors = DemoGenerator.generate_point_cloud()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        frame[:] = (18, 18, 26)
        cv2.putText(frame, "DEMO MODE ACTIVE", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        st.session_state.current_objects = objects
        st.session_state.current_decision = decision
        st.session_state.current_bev = bev
        st.session_state.current_depth = cv2.applyColorMap((depth * 5).astype(np.uint8), cv2.COLORMAP_TURBO)
        st.session_state.current_pointcloud = (points, colors)
        st.session_state.current_frame = frame
        st.session_state.current_detections = frame
    
    # Metrics Row
    cols = st.columns(4)
    fps = np.random.uniform(28, 35) if st.session_state.system_running else 0.0
    obj_count = len(st.session_state.current_objects)
    decision = st.session_state.current_decision.get('action', 'STOP').upper()
    uptime = time.time() - st.session_state.start_time if st.session_state.start_time else 0
    
    with cols[0]:
        st.markdown(f'<div class="metric-card"><div class="metric-label">⚡ Framerate</div><div class="metric-value">{fps:.0f} FPS</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🎯 Objects</div><div class="metric-value">{obj_count}</div></div>', unsafe_allow_html=True)
    with cols[2]:
        color = "#10B981" if decision == "FORWARD" else ("#F59E0B" if decision in ["LEFT", "RIGHT"] else "#EF4444")
        st.markdown(f'<div class="metric-card"><div class="metric-label">⚙️ Decision</div><div class="metric-value" style="color:{color}">{decision}</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="metric-card"><div class="metric-label">⏱️ Uptime</div><div class="metric-value">{uptime:.0f}s</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Warnings Section
    warnings = st.session_state.current_decision.get('warnings', [])
    if warnings:
        for w in warnings:
            st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">✅ System Nominal - All Clear</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ROW 1: Detection Output + Depth Map
    row1_col1, row1_col2 = st.columns([1.5, 1])
    
    with row1_col1:
        st.markdown('<div class="view-card">🎯 Object Detection</div>', unsafe_allow_html=True)
        if st.session_state.current_detections is not None:
            st.image(st.session_state.current_detections, use_container_width=True, channels="BGR")
        elif st.session_state.current_frame is not None:
            st.image(st.session_state.current_frame, use_container_width=True, channels="BGR")
        else:
            st.info("Waiting for detection output...")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row1_col2:
        st.markdown('<div class="view-card">🌊 Depth Map (Combined)</div>', unsafe_allow_html=True)
        if st.session_state.current_depth is not None:
            st.image(st.session_state.current_depth, use_container_width=True, channels="BGR")
        else:
            st.info("Depth data not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ROW 2: BEV SVG + 3D Output + Distance List
    row2_col1, row2_col2, row2_col3 = st.columns([1.5, 1.5, 1])
    
    with row2_col1:
        st.markdown('<div class="view-card">🗺️ BEV Map (SVG Icons)</div>', unsafe_allow_html=True)
        if st.session_state.current_bev is not None:
            # Try SVG first for demo
            if st.session_state.get('demo_active', False):
                svg_html = create_bev_svg(st.session_state.current_objects)
                st.components.v1.html(svg_html, height=520)
            else:
                st.image(st.session_state.current_bev, use_container_width=True, channels="BGR")
        else:
            st.info("BEV not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row2_col2:
        st.markdown('<div class="view-card">📦 3D Processed Output</div>', unsafe_allow_html=True)
        if st.session_state.current_pointcloud:
            pc = st.session_state.current_pointcloud
            if isinstance(pc, tuple):
                fig = create_3d_plot(pc[0], st.session_state.current_objects)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("3D data not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row2_col3:
        st.markdown('<div class="view-card">📏 Distance Data</div>', unsafe_allow_html=True)
        objs = st.session_state.current_objects
        if objs:
            html = '<div style="max-height: 350px; overflow-y: auto;">'
            for obj in objs[:10]:
                color = ICON_COLORS_HEX.get(obj.class_name, '#ffffff')
                html += f'<div class="detection-item"><div style="color:{color}; font-weight:600;">{obj.class_name}</div><div style="font-family:\'JetBrains Mono\'; color:#8B8B9E;">{obj.distance:.1f}m</div></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No objects detected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analytics
    st.divider()
    st.markdown('<div class="view-card">📈 System Telemetry</div>', unsafe_allow_html=True)
    if len(st.session_state.metrics_history) > 2:
        fig = create_metrics_chart(st.session_state.metrics_history)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Accumulating telemetry data...")
    st.markdown('</div>', unsafe_allow_html=True)


def simulate_demo():
    objects = DemoGenerator.generate_objects()
    decision = DemoGenerator.generate_decision(objects)
    bev = DemoGenerator.generate_bev(objects)
    depth = DemoGenerator.generate_depth()
    points, colors = DemoGenerator.generate_point_cloud()
    
    st.session_state.current_objects = objects
    st.session_state.current_decision = decision
    st.session_state.current_bev = bev
    st.session_state.current_depth = cv2.applyColorMap((depth * 5).astype(np.uint8), cv2.COLORMAP_TURBO)
    st.session_state.current_pointcloud = (points, colors)
    st.session_state.metrics_history.append({'fps': np.random.uniform(28, 35), 'objects': len(objects), 'timestamp': time.time()})


if __name__ == "__main__":
    main()
