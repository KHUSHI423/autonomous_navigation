"""
Live Dashboard Module
Streamlit-based analytics dashboard for real-time traffic visualization
Displays metrics, charts, and controls for the visualization system
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import time
from datetime import datetime, timedelta
from loguru import logger
import json

# Try to import plotly for charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. Charts will use fallback.")

# Page config
st.set_page_config(
    page_title="Real-Time Traffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .status-connected {
        color: #00ff00;
        font-weight: bold;
    }
    .status-disconnected {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manage dashboard state"""
    
    def __init__(self):
        self.is_connected = False
        self.last_update = None
        self.objects = []
        self.statistics = {}
        self.fps = 0
        self.historical_data = []
        self.start_time = datetime.now()
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update state with new data"""
        self.objects = data.get('objects', [])
        self.statistics = data.get('statistics', {})
        self.fps = data.get('fps', 0)
        self.last_update = datetime.now()
        self.is_connected = True
        
        # Store historical data
        self.historical_data.append({
            'timestamp': self.last_update,
            'total_objects': len(self.objects),
            'by_class': self._count_by_class()
        })
        
        # Keep last 5 minutes of data
        cutoff = self.last_update - timedelta(minutes=5)
        self.historical_data = [
            d for d in self.historical_data 
            if d['timestamp'] > cutoff
        ]
    
    def _count_by_class(self) -> Dict[str, int]:
        """Count objects by class"""
        counts = {}
        for obj in self.objects:
            cls = obj.get('class_name', 'unknown')
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    def get_counts(self) -> Dict[str, int]:
        """Get current object counts by class"""
        return self._count_by_class()


class TrafficDashboard:
    """Main dashboard class"""
    
    def __init__(self, config_path: str = "config_realtime.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.state = DashboardState()
        
        # Session state initialization
        if 'dashboard_state' not in st.session_state:
            st.session_state.dashboard_state = self.state
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def render(self):
        """Render the dashboard"""
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("🚦 Traffic Dashboard")
            
            # Connection status
            st.subheader("Connection Status")
            
            if self.state.is_connected:
                st.markdown('<p class="status-connected">● Connected</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-disconnected">● Disconnected</p>', 
                           unsafe_allow_html=True)
            
            if self.state.last_update:
                st.write(f"Last update: {self.state.last_update.strftime('%H:%M:%S')}")
            
            st.divider()
            
            # Controls
            st.subheader("Controls")
            
            # Camera source selection
            camera_source = st.selectbox(
                "Camera Source",
                options=["Webcam (0)", "Webcam (1)", "Video File", "RTSP Stream"],
                index=0
            )
            
            if camera_source == "Video File":
                video_file = st.file_uploader(
                    "Upload Video",
                    type=["mp4", "avi", "mov"]
                )
            
            if camera_source == "RTSP Stream":
                rtsp_url = st.text_input("RTSP URL")
            
            st.divider()
            
            # Detection settings
            st.subheader("Detection Settings")
            
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            st.divider()
            
            # View options
            st.subheader("View Options")
            
            show_cars = st.checkbox("Cars", value=True)
            show_buses = st.checkbox("Buses", value=True)
            show_trucks = st.checkbox("Trucks", value=True)
            show_motorcycles = st.checkbox("Motorcycles", value=True)
            show_pedestrians = st.checkbox("Pedestrians", value=True)
            
            st.divider()
            
            # Session info
            st.subheader("Session Info")
            uptime = datetime.now() - self.state.start_time
            st.write(f"Uptime: {str(uptime).split('.')[0]}")
            st.write(f"Data points: {len(self.state.historical_data)}")
    
    def _render_main(self):
        """Render main dashboard content"""
        st.title("📊 Real-Time Traffic Analytics")
        
        if not self.state.is_connected:
            st.info("🔌 Connecting to visualization server...")
            st.markdown("""
            ### Quick Start
            
            1. **Start the WebSocket server:**
               ```bash
               python websocket_server.py
               ```
            
            2. **Open the 3D viewer:**
               - Open `viewer_realtime.html` in your browser
               - Or navigate to http://localhost:8765
            
            3. **Run the detection pipeline:**
               ```bash
               python main.py
               ```
            """)
            return
        
        # Metrics row
        self._render_metrics()
        
        st.divider()
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_time_series()
        
        with col2:
            self._render_pie_chart()
        
        st.divider()
        
        # Object table
        self._render_object_table()
    
    def _render_metrics(self):
        """Render metrics cards"""
        counts = self.state.get_counts()
        total = sum(counts.values())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Objects",
                value=total,
                delta=None
            )
        
        with col2:
            st.metric(
                label="Vehicles",
                value=counts.get('car', 0) + counts.get('bus', 0) + counts.get('truck', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Pedestrians",
                value=counts.get('person', 0),
                delta=None
            )
        
        with col4:
            st.metric(
                label="FPS",
                value=f"{self.state.fps:.1f}",
                delta=None
            )
    
    def _render_time_series(self):
        """Render time series chart"""
        if not PLOTLY_AVAILABLE or len(self.state.historical_data) < 2:
            st.write("📈 Time series data not available yet...")
            return
        
        # Prepare data
        df = pd.DataFrame(self.state.historical_data)
        df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['total_objects'],
            mode='lines+markers',
            name='Total Objects',
            line=dict(color='#4A90D9', width=2)
        ))
        
        fig.update_layout(
            title="Object Count Over Time",
            xaxis_title="Time",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_pie_chart(self):
        """Render pie chart of object distribution"""
        counts = self.state.get_counts()
        
        if not counts:
            st.write("No data available")
            return
        
        if not PLOTLY_AVAILABLE:
            # Fallback without plotly
            for cls, count in counts.items():
                st.write(f"**{cls}**: {count}")
            return
        
        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Object Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_object_table(self):
        """Render table of detected objects"""
        st.subheader("📋 Detected Objects")
        
        if not self.state.objects:
            st.info("No objects detected")
            return
        
        # Create dataframe
        data = []
        for obj in self.state.objects[:50]:  # Limit to 50
            data.append({
                'ID': obj.get('track_id', 'N/A'),
                'Class': obj.get('class_name', 'unknown'),
                'Position': f"({obj.get('position', {}).get('x', 0):.1f}, "
                           f"{obj.get('position', {}).get('z', 0):.1f})",
                'Speed': f"{obj.get('velocity', {}).get('x', 0):.2f}",
                'Confidence': f"{obj.get('confidence', 0):.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def simulate_data(state: DashboardState) -> None:
    """Simulate incoming data for testing"""
    import random
    
    # Simulate objects
    classes = ['car', 'bus', 'truck', 'motorcycle', 'person']
    objects = []
    
    for i in range(random.randint(5, 20)):
        cls = random.choice(classes)
        objects.append({
            'track_id': i,
            'class_name': cls,
            'position': {
                'x': random.uniform(-25, 25),
                'y': 0,
                'z': random.uniform(0, 50)
            },
            'velocity': {
                'x': random.uniform(-1, 1),
                'y': 0,
                'z': random.uniform(0, 2)
            },
            'confidence': random.uniform(0.5, 1.0)
        })
    
    # Update state
    state.update({
        'objects': objects,
        'statistics': {
            'tracks': {'total_tracks': len(objects)},
            'detections': {'total': len(objects)}
        },
        'fps': random.uniform(25, 35)
    })


def main():
    """Main dashboard entry point"""
    st.title("🚦 Real-Time Traffic Dashboard")
    
    # Initialize dashboard
    dashboard = TrafficDashboard()
    
    # Sidebar options
    mode = st.sidebar.radio(
        "Mode",
        ["Live", "Simulation", "Playback"],
        index=1  # Default to simulation for demo
    )
    
    if mode == "Simulation":
        # Simulation mode
        st.info("🔮 Running in simulation mode")
        
        # Create placeholder for metrics
        placeholder = st.empty()
        
        # Simulate data updates
        for _ in range(100):
            simulate_data(dashboard.state)
            dashboard.render()
            time.sleep(0.5)
            
            if st.sidebar.button("Stop Simulation"):
                break
    
    elif mode == "Live":
        # Live mode - connect to WebSocket
        st.info("🔴 Attempting to connect to live server...")
        
        # Try to connect to WebSocket
        try:
            import websockets
            import asyncio
            
            async def connect_and_listen():
                try:
                    async with websockets.connect("ws://localhost:8765/ws") as ws:
                        dashboard.state.is_connected = True
                        
                        async for message in ws:
                            data = json.loads(message)
                            dashboard.state.update(data)
                            st.rerun()
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    dashboard.state.is_connected = False
            
            # Run async connection
            # asyncio.run(connect_and_listen())
            
        except ImportError:
            st.warning("websockets not installed. Run: pip install websockets")
        
        dashboard.render()
    
    else:
        # Playback mode
        st.info("📼 Playback mode - select a recording file")
        # TODO: Implement playback functionality


if __name__ == "__main__":
    main()
