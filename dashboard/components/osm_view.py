"""
=============================================================================
EDGE DRIVE 3D - OPENSTREETMAP VISUALIZATION COMPONENTS
=============================================================================
Beautiful, interactive OpenStreetMap visualization using Folium

Features:
- Multiple map tile layers (OSM, Satellite, Terrain, Dark)
- Custom markers for detected objects
- Real-time vehicle position tracking
- Trajectory path visualization
- Perception overlay with icons
- Beautiful popups with object details

Author: EdgeDrive3D Team
=============================================================================
"""

import folium
from folium import plugins
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time
import hashlib
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.coordinate_transform import destination_point


# ============================================================================
# CUSTOM MAP ICONS (SVG)
# ============================================================================

OBJECT_ICONS = {
    'car': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#00ff00" stroke="white" stroke-width="1">
            <path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"/>
        </svg>
    ''',
    'person': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ffff00" stroke="white" stroke-width="1">
            <circle cx="12" cy="4" r="3"/>
            <path d="M12 9c-2.5 0-4 2-4 4v5h2v-5c0-1 .5-2 2-2s2 1 2 2v5h2v-5c0-2-1.5-4-4-4z"/>
            <path d="M12 11L9 20h6l-3-9z"/>
        </svg>
    ''',
    'bicycle': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ffa500" stroke="white" stroke-width="1">
            <circle cx="5.5" cy="17.5" r="3.5"/>
            <circle cx="18.5" cy="17.5" r="3.5"/>
            <path d="M15 6h-5a1 1 0 00-1 1v4h-3V9H4v2h2v3H4v2h2.5l2.5 4h2l-2-4h5l1 2h2l-1.5-3L20 11V9l-5-3z"/>
        </svg>
    ''',
    'motorcycle': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ff00ff" stroke="white" stroke-width="1">
            <circle cx="5" cy="17" r="3"/>
            <circle cx="19" cy="17" r="3"/>
            <path d="M9 17h6M12 14l3-6h-4l-2 4H7l2-5h6l3 7"/>
        </svg>
    ''',
    'bus': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ff0000" stroke="white" stroke-width="1">
            <path d="M4 6h16v12H4z"/>
            <circle cx="7" cy="18" r="2"/>
            <circle cx="17" cy="18" r="2"/>
            <path d="M4 10h16M6 6V4h4v2M14 4h4v2"/>
        </svg>
    ''',
    'truck': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ff8c00" stroke="white" stroke-width="1">
            <path d="M2 7h14v11H2z"/>
            <path d="M16 9h4l2 3v6h-6V9z"/>
            <circle cx="6" cy="18" r="2"/>
            <circle cx="18" cy="18" r="2"/>
        </svg>
    ''',
    'traffic_light': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ff0000" stroke="white" stroke-width="1">
            <rect x="9" y="2" width="6" height="14" rx="1"/>
            <circle cx="12" cy="6" r="1.5" fill="#ff0000"/>
            <circle cx="12" cy="10" r="1.5" fill="#ffff00"/>
            <circle cx="12" cy="14" r="1.5" fill="#00ff00"/>
            <path d="M11 16h2v4h-2z"/>
        </svg>
    ''',
    'animal': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#8b4513" stroke="white" stroke-width="1">
            <path d="M12 4c-1.5 0-3 1-3 2.5S10.5 9 12 9s3-1 3-2.5S13.5 4 12 4z"/>
            <path d="M12 9c-3 0-5 2-5 5v4h10v-4c0-3-2-5-5-5z"/>
            <circle cx="10" cy="12" r="1" fill="white"/>
            <circle cx="14" cy="12" r="1" fill="white"/>
        </svg>
    ''',
    'obstacle': '''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ff4444" stroke="white" stroke-width="1">
            <path d="M12 2L2 22h20L12 2z"/>
            <path d="M12 8v6M12 18v2"/>
        </svg>
    ''',
}

# Object colors for markers
OBJECT_COLORS = {
    'car': '#00ff00',
    'person': '#ffff00',
    'bicycle': '#ffa500',
    'motorcycle': '#ff00ff',
    'bus': '#ff0000',
    'truck': '#ff8c00',
    'train': '#8b0000',
    'traffic_light': '#ff0000',
    'stop_sign': '#ff0000',
    'animal': '#8b4513',
    'dog': '#8b4513',
    'cow': '#654321',
    'default': '#00ffff',
}


# ============================================================================
# MAP CREATION
# ============================================================================

def create_osm_map(
    center_lat: float = 0.0,
    center_lon: float = 0.0,
    zoom: int = 15,
    width: str = "100%",
    height: str = "600"
) -> folium.Map:
    """
    Create a beautiful OpenStreetMap with multiple tile layers
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Initial zoom level
        width: Map width (CSS)
        height: Map height (pixels)
    
    Returns:
        Folium map object
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        control_scale=True,
        zoom_control=True,
        dragging=True,
        scroll_wheel_zoom=True,
    )
    
    # Add multiple tile layers
    tile_layers = {
        'OpenStreetMap': folium.TileLayer(
            tiles='OpenStreetMap',
            name='Street Map',
            attr='© OpenStreetMap contributors',
            show=True
        ),
        'Satellite': folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            name='Satellite',
            attr='© ESRI',
            show=False
        ),
        'Terrain': folium.TileLayer(
            tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
            name='Terrain',
            attr='© Stamen Design',
            show=False
        ),
        'Dark Matter': folium.TileLayer(
            tiles='https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png',
            name='Dark Mode',
            attr='© CartoDB',
            show=False
        ),
        'Hybrid': folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            name='Hybrid',
            attr='© ESRI',
            overlay=True,
            show=False
        ),
    }
    
    for layer in tile_layers.values():
        layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(position='topleft').add_to(m)
    
    # Add minimap
    plugins.MiniMap(
        tile_layer='OpenStreetMap',
        position='bottomright',
        toggle_display='on',
        minimized_width=100,
        minimized_height=100
    ).add_to(m)
    
    # Add measure control
    plugins.MeasureControl(
        position='bottomleft',
        primary_length_unit='meters',
        secondary_area_unit='square_meters'
    ).add_to(m)
    
    # Add draw control for annotations
    plugins.Draw(
        export=True,
        filename='perception_data.geojson',
        position='topleft',
        draw_options={
            'polyline': True,
            'polygon': True,
            'circle': True,
            'marker': True,
            'rect': True,
        },
        edit_options={
            'edit': True,
            'remove': True
        }
    ).add_to(m)
    
    return m


# ============================================================================
# VEHICLE MARKER
# ============================================================================

def create_vehicle_marker(
    latitude: float,
    longitude: float,
    heading: float = 0.0,
    speed: float = 0.0,
    popup_info: Dict = None
) -> folium.Marker:
    """
    Create a custom vehicle marker with heading indicator
    
    Args:
        latitude: Vehicle latitude
        longitude: Vehicle longitude
        heading: Vehicle heading in degrees (0=North)
        speed: Vehicle speed in m/s
        popup_info: Additional info for popup
    
    Returns:
        Folium marker
    """
    # Create vehicle icon (arrow pointing in heading direction)
    vehicle_icon = f'''
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" 
             style="filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.5));"
             transform="rotate({heading}, 20, 20)">
            <defs>
                <linearGradient id="vehicleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#00ff00;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#008800;stop-opacity:1" />
                </linearGradient>
            </defs>
            <circle cx="20" cy="20" r="18" fill="#006400" stroke="#00ff00" stroke-width="2"/>
            <polygon points="20,8 28,28 20,24 12,28" fill="url(#vehicleGrad)" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="20" cy="20" r="4" fill="#ffffff"/>
        </svg>
    '''
    
    # Create popup content
    popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 150px;">
            <h4 style="margin: 0 0 10px 0; color: #006400; border-bottom: 2px solid #00ff00; padding-bottom: 5px;">
                🚗 Ego Vehicle
            </h4>
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="color: #666;">Position:</td>
                    <td style="text-align: right; font-weight: bold;">{latitude:.6f}, {longitude:.6f}</td>
                </tr>
                <tr>
                    <td style="color: #666;">Heading:</td>
                    <td style="text-align: right; font-weight: bold;">{heading:.1f}°</td>
                </tr>
                <tr>
                    <td style="color: #666;">Speed:</td>
                    <td style="text-align: right; font-weight: bold;">{speed:.1f} m/s ({speed*3.6:.1f} km/h)</td>
                </tr>
                <tr>
                    <td style="color: #666;">Timestamp:</td>
                    <td style="text-align: right; font-weight: bold;">{time.strftime('%H:%M:%S')}</td>
                </tr>
            </table>
        </div>
    """
    
    if popup_info:
        popup_html += f"""
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
                {popup_info.get('extra_html', '')}
            </div>
        """
    
    popup = folium.Popup(popup_html, max_width=300)
    
    # Create custom icon
    icon = folium.DivIcon(
        html=vehicle_icon,
        icon_size=(40, 40),
        icon_anchor=(20, 20)
    )
    
    marker = folium.Marker(
        location=[latitude, longitude],
        popup=popup,
        icon=icon,
        tooltip=f"Vehicle: {speed:.1f} m/s"
    )
    
    return marker


# ============================================================================
# OBJECT MARKERS
# ============================================================================

def create_object_marker(
    latitude: float,
    longitude: float,
    object_class: str,
    distance: float,
    confidence: float,
    position_3d: Tuple[float, float, float] = None,
    extra_info: Dict = None
) -> folium.CircleMarker:
    """
    Create a marker for a detected object
    
    Args:
        latitude: Object latitude
        longitude: Object longitude
        object_class: Class name (car, person, etc.)
        distance: Distance from vehicle (meters)
        confidence: Detection confidence (0-1)
        position_3d: 3D position (x, y, z) in camera frame
        extra_info: Additional info for popup
    
    Returns:
        Folium circle marker
    """
    # Get color for object class
    color = OBJECT_COLORS.get(object_class.lower(), OBJECT_COLORS['default'])
    
    # Size based on distance (closer = bigger)
    radius = max(8, min(20, 30 - distance * 0.5))
    
    # Opacity based on confidence
    fill_opacity = 0.5 + 0.5 * confidence
    
    # Create popup content
    popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 180px;">
            <h4 style="margin: 0 0 10px 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                🎯 {object_class.replace('_', ' ').title()}
            </h4>
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="color: #666;">Distance:</td>
                    <td style="text-align: right; font-weight: bold; color: {color};">{distance:.2f} m</td>
                </tr>
                <tr>
                    <td style="color: #666;">Confidence:</td>
                    <td style="text-align: right; font-weight: bold;">{confidence*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="color: #666;">Position:</td>
                    <td style="text-align: right; font-weight: bold;">{latitude:.6f}, {longitude:.6f}</td>
                </tr>
    """
    
    if position_3d:
        popup_html += f"""
                <tr>
                    <td style="color: #666;">3D Coords:</td>
                    <td style="text-align: right; font-weight: bold;">({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f})m</td>
                </tr>
        """
    
    if extra_info:
        for key, value in extra_info.items():
            popup_html += f"""
                <tr>
                    <td style="color: #666;">{key}:</td>
                    <td style="text-align: right; font-weight: bold;">{value}</td>
                </tr>
            """
    
    popup_html += """
            </table>
        </div>
    """
    
    popup = folium.Popup(popup_html, max_width=250)
    
    marker = folium.CircleMarker(
        location=[latitude, longitude],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        weight=2,
        popup=popup,
        tooltip=f"{object_class}: {distance:.1f}m"
    )
    
    return marker


def create_object_icon_marker(
    latitude: float,
    longitude: float,
    object_class: str,
    distance: float,
    confidence: float,
    heading: float = 0.0
) -> folium.Marker:
    """
    Create a marker with custom SVG icon for detected object
    """
    # Get icon SVG
    icon_svg = OBJECT_ICONS.get(object_class.lower(), OBJECT_ICONS['obstacle'])
    
    # Get color
    color = OBJECT_COLORS.get(object_class.lower(), OBJECT_COLORS['default'])
    
    # Create custom icon with rotation
    icon_html = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" 
             style="filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.5));"
             transform="rotate({heading}, 16, 16)">
            <circle cx="16" cy="16" r="15" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="2"/>
            <g transform="translate(2, 2) scale(0.85)">
                {icon_svg}
            </g>
        </svg>
    """
    
    icon = folium.DivIcon(
        html=icon_html,
        icon_size=(32, 32),
        icon_anchor=(16, 16)
    )
    
    popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 150px;">
            <h4 style="margin: 0 0 8px 0; color: {color};">
                {object_class.replace('_', ' ').title()}
            </h4>
            <p style="margin: 5px 0; font-size: 13px;">
                <strong>Distance:</strong> {distance:.2f}m<br>
                <strong>Confidence:</strong> {confidence*100:.1f}%
            </p>
        </div>
    """
    
    return folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup(popup_html, max_width=200),
        icon=icon,
        tooltip=f"{object_class}: {distance:.1f}m"
    )


# ============================================================================
# TRAJECTORY & PATH VISUALIZATION
# ============================================================================

def create_trajectory_polyline(
    gps_points: List[Tuple[float, float]],
    color: str = '#00ff00',
    weight: int = 4,
    opacity: float = 0.7,
    dashed: bool = False
) -> folium.PolyLine:
    """
    Create a polyline for vehicle trajectory
    
    Args:
        gps_points: List of (latitude, longitude) tuples
        color: Line color
        weight: Line width
        opacity: Line opacity
        dashed: If True, create dashed line
    
    Returns:
        Folium PolyLine
    """
    return folium.PolyLine(
        locations=gps_points,
        color=color,
        weight=weight,
        opacity=opacity,
        dash_array='5, 5' if dashed else None,
        tooltip=f'Trajectory ({len(gps_points)} points)'
    )


def create_detection_ray(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    color: str = '#ff0000',
    weight: int = 2,
    opacity: float = 0.5
) -> folium.PolyLine:
    """
    Create a ray from vehicle to detected object
    """
    return folium.PolyLine(
        locations=[[start_lat, start_lon], [end_lat, end_lon]],
        color=color,
        weight=weight,
        opacity=opacity,
        dash_array='3, 3',
        interactive=False
    )


def create_fov_sector(
    center_lat: float,
    center_lon: float,
    heading: float,
    fov_degrees: float = 70.0,
    range_meters: float = 50.0,
    color: str = '#00ff00',
    opacity: float = 0.2
) -> folium.Polygon:
    """
    Create a field-of-view sector polygon
    
    Args:
        center_lat: Center latitude (vehicle position)
        center_lon: Center longitude
        heading: Heading in degrees
        fov_degrees: Field of view angle
        range_meters: Detection range
        color: Sector color
        opacity: Fill opacity
    
    Returns:
        Folium Polygon
    """
    from utils.coordinate_transform import destination_point
    
    # Calculate sector points
    left_bearing = heading - fov_degrees / 2
    right_bearing = heading + fov_degrees / 2
    
    left_lat, left_lon = destination_point(center_lat, center_lon, left_bearing, range_meters)
    right_lat, right_lon = destination_point(center_lat, center_lon, right_bearing, range_meters)
    
    # Create sector polygon
    points = [
        [center_lat, center_lon],
        [left_lat, left_lon],
        [right_lat, right_lon],
        [center_lat, center_lon]
    ]
    
    return folium.Polygon(
        locations=points,
        color=color,
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=opacity,
        interactive=False
    )


# ============================================================================
# PERCEPTION OVERLAY MANAGER
# ============================================================================

class PerceptionOverlayManager:
    """
    Manage perception overlays on the map
    
    Handles adding/removing/updating detected objects
    """
    
    def __init__(self, map_obj: folium.Map):
        self.map = map_obj
        self.object_markers = {}  # id -> marker
        self.trajectory_points = []
        self.trajectory_line = None
        self.vehicle_marker = None
        self.fov_sector = None
    
    def update_vehicle_position(
        self,
        latitude: float,
        longitude: float,
        heading: float = 0.0,
        speed: float = 0.0,
        add_to_trajectory: bool = True
    ):
        """Update vehicle position on map"""
        # Remove old vehicle marker
        if self.vehicle_marker:
            self.vehicle_marker.remove_from(self.map)
        
        # Create new marker
        self.vehicle_marker = create_vehicle_marker(
            latitude, longitude, heading, speed
        )
        self.vehicle_marker.add_to(self.map)
        
        # Update trajectory
        if add_to_trajectory:
            self.trajectory_points.append((latitude, longitude))
            
            # Keep last 1000 points
            if len(self.trajectory_points) > 1000:
                self.trajectory_points = self.trajectory_points[-1000:]
            
            # Update trajectory line
            if self.trajectory_line:
                self.trajectory_line.remove_from(self.map)
            
            self.trajectory_line = create_trajectory_polyline(
                self.trajectory_points,
                color='#00ff00',
                weight=3,
                opacity=0.8
            )
            self.trajectory_line.add_to(self.map)
        
        # Update FOV sector
        if self.fov_sector:
            self.fov_sector.remove_from(self.map)
        
        self.fov_sector = create_fov_sector(
            latitude, longitude, heading,
            fov_degrees=70.0,
            range_meters=50.0
        )
        self.fov_sector.add_to(self.map)
    
    def update_objects(self, objects: List[Dict], vehicle_lat: float, vehicle_lon: float):
        """
        Update detected objects on map
        
        Args:
            objects: List of detected objects with gps_coordinates
            vehicle_lat: Vehicle latitude (for detection rays)
            vehicle_lon: Vehicle longitude
        """
        # Remove old markers
        for marker in self.object_markers.values():
            marker.remove_from(self.map)
        self.object_markers.clear()
        
        # Add new markers
        for i, obj in enumerate(objects):
            if 'gps_coordinates' not in obj:
                continue
            
            gps = obj['gps_coordinates']
            marker = create_object_icon_marker(
                latitude=gps['latitude'],
                longitude=gps['longitude'],
                object_class=obj.get('class_name', 'unknown'),
                distance=obj.get('distance', 0),
                confidence=obj.get('confidence', 0)
            )
            marker.add_to(self.map)
            self.object_markers[f"obj_{i}"] = marker
            
            # Add detection ray
            ray = create_detection_ray(
                vehicle_lat, vehicle_lon,
                gps['latitude'], gps['longitude'],
                color=OBJECT_COLORS.get(obj.get('class_name', 'unknown').lower(), '#ff0000'),
                weight=1,
                opacity=0.3
            )
            ray.add_to(self.map)
    
    def clear_objects(self):
        """Clear all object markers"""
        for marker in self.object_markers.values():
            marker.remove_from(self.map)
        self.object_markers.clear()
    
    def clear_trajectory(self):
        """Clear trajectory line"""
        if self.trajectory_line:
            self.trajectory_line.remove_from(self.map)
        self.trajectory_points.clear()
    
    def save_map(self, filepath: str):
        """Save map to HTML file"""
        self.map.save(filepath)
        print(f"  ✓ Map saved to: {filepath}")


# ============================================================================
# HTML LEGEND COMPONENT
# ============================================================================

def create_map_legend() -> str:
    """Create HTML legend for the map"""
    return """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #006400;
        border-radius: 10px;
        padding: 15px;
        font-family: 'Segoe UI', Arial, sans-serif;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 10px 0; color: #006400; border-bottom: 2px solid #00ff00; padding-bottom: 5px;">
            🎯 Perception Legend
        </h4>
        <div style="font-size: 12px;">
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background: #00ff00; border-radius: 50%; margin-right: 10px;"></div>
                <span>🚗 Ego Vehicle</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background: #00ff00; opacity: 0.3; border: 2px solid #00ff00; margin-right: 10px;"></div>
                <span>Field of View</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background: #ffff00; border-radius: 50%; margin-right: 10px;"></div>
                <span>🚶 Person</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background: #00ff00; border-radius: 50%; margin-right: 10px;"></div>
                <span>🚗 Vehicle</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background: #ffa500; border-radius: 50%; margin-right: 10px;"></div>
                <span>🚲 Bicycle</span>
            </div>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background: #ff0000; border-radius: 50%; margin-right: 10px;"></div>
                <span>🚌 Bus/Truck</span>
            </div>
        </div>
    </div>
    """


def add_legend_to_map(map_obj: folium.Map):
    """Add legend to map as HTML overlay"""
    legend_html = folium.Element(create_map_legend())
    map_obj.get_root().html.add_child(legend_html)


# ============================================================================
# MAIN (Test)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  OpenStreetMap Visualization Test")
    print("=" * 60)
    
    # Create map centered on Bangalore
    print("\nCreating map...")
    map_view = create_osm_map(
        center_lat=12.9716,
        center_lon=77.5946,
        zoom=16
    )
    
    # Add vehicle marker
    print("Adding vehicle marker...")
    vehicle = create_vehicle_marker(
        12.9716, 77.5946,
        heading=45.0,
        speed=5.0
    )
    vehicle.add_to(map_view)
    
    # Add some object markers
    print("Adding object markers...")
    test_objects = [
        {'class': 'car', 'lat': 12.9720, 'lon': 77.5950, 'dist': 15.0, 'conf': 0.92},
        {'class': 'person', 'lat': 12.9718, 'lon': 77.5948, 'dist': 8.0, 'conf': 0.88},
        {'class': 'bicycle', 'lat': 12.9714, 'lon': 77.5944, 'dist': 12.0, 'conf': 0.75},
    ]
    
    for obj in test_objects:
        marker = create_object_icon_marker(
            obj['lat'], obj['lon'],
            obj['class'], obj['dist'], obj['conf']
        )
        marker.add_to(map_view)
    
    # Add trajectory
    print("Adding trajectory...")
    trajectory = [
        (12.9710, 77.5940),
        (12.9712, 77.5942),
        (12.9714, 77.5944),
        (12.9716, 77.5946),
    ]
    traj_line = create_trajectory_polyline(trajectory)
    traj_line.add_to(map_view)
    
    # Add FOV sector
    print("Adding FOV sector...")
    fov = create_fov_sector(12.9716, 77.5946, heading=45.0)
    fov.add_to(map_view)
    
    # Add legend
    print("Adding legend...")
    add_legend_to_map(map_view)
    
    # Save map
    print("Saving map...")
    map_view.save("test_osm_map.html")
    print("\n✓ Test map saved to: test_osm_map.html")
    print("  Open in browser to view")
