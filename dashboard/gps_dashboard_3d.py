"""
=============================================================================
EDGE DRIVE 3D - GPS 3D CAMPUS DASHBOARD
=============================================================================
Beautiful 3D GPS + Perception dashboard with CIT Campus visualization

Features:
- Real-time 3D map of CIT Campus
- Live GPS position tracking
- 3D buildings and landmarks
- Detected objects in 3D space
- Perception camera feed
- Interactive camera controls
- Session recording and replay

Author: EdgeDrive3D Team
=============================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from core.perception_engine_gps import GPSPerceptionEngine, GPSPerceptionResult
from dashboard.components.map_3d import render_3d_map, get_object_color
from hardware.gps_reader import GPSReading


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EdgeDrive3D 3D Campus Map",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container */
    .main > div {padding: 1rem 2rem;}

    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .dashboard-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    .metric-card .unit {
        font-size: 0.9rem;
        color: #888;
        margin-left: 0.3rem;
    }

    /* Status indicators */
    .status-good {color: #22c55e;}
    .status-warning {color: #f59e0b;}
    .status-error {color: #ef4444;}

    /* Object list */
    .object-item {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 3D map container */
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'objects_detected_total' not in st.session_state:
    st.session_state.objects_detected_total = 0


# ============================================================================
# HEADER
# ============================================================================

def render_header():
    """Render beautiful header"""
    st.markdown("""
        <div class="dashboard-header">
            <h1>🏫 EdgeDrive3D 3D Campus Map</h1>
            <p>Coimbatore Institute of Technology - Real-time 3D Visualization</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.markdown("### ⚙️ System Controls")

        # GPS Settings
        st.markdown("#### 📍 GPS Configuration")
        gps_mode = st.radio(
            "GPS Mode",
            ["Simulated", "Hardware (USB)", "Hardware (Serial)"],
            index=0,
            help="Select GPS source"
        )

        gps_port = "auto"
        baudrate = 9600

        if gps_mode == "Hardware (Serial)":
            gps_port = st.text_input("Serial Port", value="/dev/ttyUSB0")
            baudrate = st.selectbox("Baud Rate", [9600, 19200, 115200], index=0)

        simulate_gps = gps_mode == "Simulated"

        # Perception Settings
        st.markdown("#### 🎯 Perception")
        yolo_model = st.selectbox(
            "YOLO Model",
            ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"],
            index=1,
            help="Select detection model"
        )

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9, value=0.4, step=0.05
        )

        max_depth = st.slider(
            "Max Depth (meters)",
            min_value=10, max_value=100, value=50, step=5
        )

        # 3D Map Settings
        st.markdown("#### 🗺️ 3D Map Display")
        map_zoom = st.slider("Zoom Level", 14.0, 18.0, 16.5, 0.1)
        map_pitch = st.slider("Camera Pitch", 0, 60, 45)
        map_bearing = st.slider("Camera Bearing", 0, 360, 0)
        map_style = st.selectbox(
            "Map Style",
            ["dark", "light", "satellite"],
            index=0
        )
        show_3d_buildings = st.checkbox("Show 3D Buildings", value=True)
        show_trajectory = st.checkbox("Show Trajectory", value=True)
        show_objects = st.checkbox("Show Detected Objects", value=True)

        # Recording
        st.markdown("#### 📼 Recording")
        record_session = st.checkbox("Record Session", value=False)

        # Start/Stop
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            start_btn = st.button("▶️ Start", type="primary", use_container_width=True)
            if start_btn and not st.session_state.is_running:
                st.session_state.is_running = True
                st.session_state.start_time = time.time()
                st.session_state.frame_count = 0
                st.session_state.objects_detected_total = 0
                st.rerun()

        with col2:
            stop_btn = st.button("⏹️ Stop", type="secondary", use_container_width=True)
            if stop_btn and st.session_state.is_running:
                st.session_state.is_running = False
                if st.session_state.engine:
                    st.session_state.engine.stop_gps()
                st.rerun()

        # Clear trajectory
        if st.button("🗑️ Clear Trajectory", use_container_width=True):
            if st.session_state.engine:
                st.session_state.engine.clear_trajectory()
            st.rerun()

        # Export data
        st.markdown("---")
        if st.button("📥 Export Session Data", use_container_width=True):
            if st.session_state.engine:
                filepath = f"output/gps_logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.session_state.engine.save_trajectory(filepath)
                st.success(f"Data saved to {filepath}")

        # Status
        st.markdown("---")
        st.markdown("### 📊 System Status")

        if st.session_state.engine:
            stats = st.session_state.engine.get_statistics()

            gps_status = "✅ Connected" if stats.get('gps_connected') else "❌ Disconnected"
            if stats.get('simulate_mode'):
                gps_status = "🔸 Simulated"

            st.markdown(f"**GPS:** {gps_status}")
            st.markdown(f"**Updates:** {stats.get('gps_updates', 0)}")
            st.markdown(f"**Geolocated:** {stats.get('objects_geolocated', 0)}")
            st.markdown(f"**Trajectory:** {stats.get('trajectory_points', 0)} pts")
        else:
            st.markdown("**GPS:** ⏳ Not initialized")

        # Session stats
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            st.markdown(f"**Session:** {elapsed:.0f}s")
            st.markdown(f"**Frames:** {st.session_state.frame_count}")
            st.markdown(f"**Objects:** {st.session_state.objects_detected_total}")

        # Help
        with st.expander("❓ 3D Map Controls"):
            st.markdown("""
            ### Mouse Controls
            - **Left Drag**: Pan map
            - **Scroll**: Zoom in/out
            - **Right Drag**: Rotate view
            - **Shift + Drag**: Adjust pitch
            
            ### Buttons
            - **Reset View**: Return to initial position
            - **Toggle 3D**: Show/hide buildings
            - **Animate**: Auto-rotate camera
            - **Top View**: Switch to overhead view
            
            ### Keyboard Shortcuts
            - `S`: Save snapshot
            - `R`: Toggle recording
            - `T`: Toggle trajectory
            - `B`: Toggle buildings
            """)

    return {
        'simulate_gps': simulate_gps,
        'gps_port': gps_port,
        'baudrate': baudrate,
        'yolo_model': yolo_model,
        'confidence': confidence,
        'max_depth': max_depth,
        'map_zoom': map_zoom,
        'map_pitch': map_pitch,
        'map_bearing': map_bearing,
        'map_style': map_style,
        'show_3d_buildings': show_3d_buildings,
        'show_trajectory': show_trajectory,
        'show_objects': show_objects,
        'record_session': record_session
    }


# ============================================================================
# METRIC CARDS
# ============================================================================

def render_metrics(gps_data: GPSReading = None, fps: float = 0, objects: list = None):
    """Render metric cards"""
    cols = st.columns(4)

    with cols[0]:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📍 GPS Status</h3>
                <div class="value {'status-good' if gps_data and gps_data.is_valid else 'status-warning'}">
                    {'FIX' if gps_data and gps_data.is_valid else 'NO FIX'}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        lat_lon = f"{gps_data.latitude:.4f}, {gps_data.longitude:.4f}" if gps_data and gps_data.is_valid else "--"
        st.markdown(f"""
            <div class="metric-card">
                <h3>🌍 Position</h3>
                <div class="value" style="font-size: 1.2rem;">{lat_lon}</div>
            </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        speed = f"{gps_data.speed:.1f}" if gps_data and gps_data.is_valid else "--"
        heading = f"{gps_data.heading:.0f}°" if gps_data and gps_data.is_valid else "--"
        st.markdown(f"""
            <div class="metric-card">
                <h3>🚗 Speed / Heading</h3>
                <div class="value" style="font-size: 1.4rem;">{speed} <span class="unit">m/s</span></div>
                <div style="color: #666; font-size: 0.9rem;">{heading}</div>
            </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        obj_count = len(objects) if objects else 0
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <h3>🎯 Objects</h3>
                <div class="value" style="color: #f59e0b;">{obj_count}</div>
                <div style="color: #666; font-size: 0.9rem;">Detected</div>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# 3D MAP COMPONENT
# ============================================================================

def render_3d_map_view(gps_data: GPSReading = None, objects: list = None, trajectory: list = None, config: dict = None):
    """Render 3D map with CIT Campus"""

    # Default center (CIT Campus coordinates)
    center_lat = gps_data.latitude if gps_data and gps_data.is_valid else 11.0293
    center_lon = gps_data.longitude if gps_data and gps_data.is_valid else 76.9382

    # Vehicle position
    vehicle_position = None
    if gps_data and gps_data.is_valid:
        vehicle_position = {
            'lat': gps_data.latitude,
            'lon': gps_data.longitude,
            'heading': gps_data.heading,
            'speed': gps_data.speed
        }
    else:
        vehicle_position = {
            'lat': center_lat,
            'lon': center_lon,
            'heading': 0,
            'speed': 0
        }

    # Render 3D map
    render_3d_map(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom=config.get('map_zoom', 16.5) if config else 16.5,
        vehicle_position=vehicle_position,
        objects=objects if config.get('show_objects', True) else [],
        trajectory=trajectory if config.get('show_trajectory', True) else [],
        map_style=config.get('map_style', 'dark') if config else 'dark',
        height=650
    )


# ============================================================================
# PERCEPTION VIEW
# ============================================================================

def render_perception_view(result: GPSPerceptionResult = None):
    """Render perception visualizations"""

    tabs = st.tabs(["📷 Camera Feed", "🗺️ BEV", "🌈 Depth", "📊 Objects"])

    with tabs[0]:
        if result and result.detections_overlay is not None:
            # Convert BGR to RGB for Streamlit
            image_rgb = cv2.cvtColor(result.detections_overlay, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
        else:
            st.info("Waiting for camera feed...")

    with tabs[1]:
        if result and result.bev_image is not None:
            image_rgb = cv2.cvtColor(result.bev_image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
        else:
            st.info("BEV not available")

    with tabs[2]:
        if result and result.depth_colored is not None:
            image_rgb = cv2.cvtColor(result.depth_colored, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
        else:
            st.info("Depth map not available")

    with tabs[3]:
        if result and result.objects_with_gps:
            for i, obj in enumerate(result.objects_with_gps):
                color = get_object_color(obj.get('class_name', 'unknown'))

                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"""
                            <div class="object-item" style="border-color: {color};">
                                <strong>{obj['class_name'].title()}</strong>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.metric("Distance", f"{obj['distance']:.2f}m")

                    with col3:
                        conf = obj['confidence'] * 100
                        st.metric("Confidence", f"{conf:.1f}%")

                    if 'gps_coordinates' in obj:
                        gps = obj['gps_coordinates']
                        st.caption(f"📍 {gps['latitude']:.6f}, {gps['longitude']:.6f}")

                    st.divider()
        else:
            st.info("No objects detected")


# ============================================================================
# DECISION PANEL
# ============================================================================

def render_decision_panel(result: GPSPerceptionResult = None):
    """Render decision/action panel"""

    st.markdown("### 🧠 Decision Engine")

    if result and result.decision:
        decision = result.decision

        # Action indicator
        action = decision.get('action', 'unknown').upper()

        action_colors = {
            'FORWARD': '#22c55e',
            'STOP': '#ef4444',
            'LEFT': '#f59e0b',
            'RIGHT': '#f59e0b',
            'BACKWARD': '#6b7280'
        }

        color = action_colors.get(action, '#888888')

        st.markdown(f"""
            <div style="
                background: {color}20;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                margin: 1rem 0;
            ">
                <div style="font-size: 2.5rem; font-weight: 700; color: {color};">
                    {action}
                </div>
                <div style="color: #666; margin-top: 0.5rem;">
                    {decision.get('reason', 'N/A')}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Speed recommendation
        speed = decision.get('speed', 0)
        st.progress(speed / 255)
        st.caption(f"Recommended Speed: {speed} / 255")

        # Warnings
        if decision.get('warnings'):
            st.warning("**Warnings:** " + " | ".join(decision['warnings']))

        # Closest object
        if 'closest_object' in decision:
            closest = decision['closest_object']
            st.markdown(f"""
                **Closest Object:** {closest.get('class', 'Unknown')} at {closest.get('distance', 0):.2f}m
            """)
    else:
        st.info("Waiting for decisions...")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def initialize_engine(config: dict):
    """Initialize GPS perception engine"""
    engine_config = {
        'yolo_model': config['yolo_model'],
        'confidence': config['confidence'],
        'max_depth': config['max_depth'],
        'gps_simulate': config['simulate_gps'],
        'gps_port': config['gps_port'],
        'baudrate': config['baudrate']
    }

    engine = GPSPerceptionEngine(engine_config)
    engine.start_gps()

    return engine


def main():
    """Main application"""

    # Render header
    render_header()

    # Render sidebar and get config
    config = render_sidebar()

    # Initialize engine if needed
    if st.session_state.engine is None and st.session_state.is_running:
        with st.spinner("Initializing GPS Perception Engine..."):
            st.session_state.engine = initialize_engine(config)
            st.success("Engine initialized!")

    # Create main layout
    if st.session_state.engine and st.session_state.is_running:
        # Main loop
        status_placeholder = st.empty()
        map_container = st.container()

        with map_container:
            st.markdown("### 🗺️ 3D Campus Map View")

            # Process frame placeholder
            frame_placeholder = st.empty()

            # Main loop
            while st.session_state.is_running:
                # Get camera frame (replace with actual camera capture)
                # For demo, create a blank frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "EdgeDrive3D 3D", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Process with GPS engine
                result = st.session_state.engine.process_frame_gps(frame)

                # Update metrics
                render_metrics(
                    gps_data=result.gps_reading,
                    fps=result.fps,
                    objects=result.objects_with_gps
                )

                # Render 3D map
                render_3d_map_view(
                    gps_data=result.gps_reading,
                    objects=result.objects_with_gps,
                    trajectory=st.session_state.engine.get_trajectory(),
                    config=config
                )

                # Render perception views
                render_perception_view(result)

                # Render decision panel
                render_decision_panel(result)

                # Update stats
                st.session_state.frame_count += 1
                st.session_state.objects_detected_total += len(result.objects_with_gps)

                # Status
                gps_status = "GPS" if result.gps_reading and result.gps_reading.is_valid else "No GPS"
                status_placeholder.info(
                    f"FPS: {result.fps:.1f} | "
                    f"{gps_status} | "
                    f"Objects: {len(result.objects_with_gps)}"
                )

                # Small delay
                time.sleep(0.05)

    else:
        # Not running - show demo 3D map
        st.info("Click 'Start' to begin GPS perception with 3D map")

        # Demo vehicle position
        demo_vehicle = {
            'lat': 11.0293,
            'lon': 76.9382,
            'heading': 45,
            'speed': 0
        }

        # Demo objects
        demo_objects = [
            {
                'class_name': 'person',
                'distance': 15.0,
                'confidence': 0.92,
                'gps_coordinates': {
                    'latitude': 11.0295,
                    'longitude': 76.9385,
                    'altitude': 420.0
                }
            },
            {
                'class_name': 'car',
                'distance': 25.0,
                'confidence': 0.88,
                'gps_coordinates': {
                    'latitude': 11.0298,
                    'longitude': 76.9388,
                    'altitude': 420.0
                }
            }
        ]

        # Demo trajectory
        demo_trajectory = [
            (11.0290, 76.9378),
            (11.0291, 76.9380),
            (11.0292, 76.9381),
            (11.0293, 76.9382)
        ]

        # Render demo 3D map
        render_3d_map(
            center_lat=11.0293,
            center_lon=76.9382,
            zoom=16.5,
            vehicle_position=demo_vehicle,
            objects=demo_objects,
            trajectory=demo_trajectory,
            map_style='dark',
            height=650
        )

        # Show info
        st.markdown("""
        ### 🎮 3D Map Features
        
        **CIT Campus Landmarks:**
        - Administrative Block
        - Main Academic Block
        - Engineering Block
        - Central Library
        - Auditorium
        - Workshop Complex
        - Hostel Block
        - Sports Complex
        - Cafeteria
        - Parking Area
        
        **Interactive Controls:**
        - Drag to pan the map
        - Scroll to zoom
        - Right-click drag to rotate
        - Shift + drag to adjust pitch
        
        **Visualization:**
        - Real-time vehicle position (green marker)
        - Detected objects with colors
        - Trajectory path (green line)
        - 3D buildings with heights
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "EdgeDrive3D 3D Campus Map | Built with ❤️ for Coimbatore Institute of Technology"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
