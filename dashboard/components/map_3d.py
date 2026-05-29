"""
=============================================================================
EDGE DRIVE 3D - MODERN 3D CAMPUS VISUALIZATION
=============================================================================
Stunning 3D map with modern effects, animations, and visual polish
"""

import streamlit.components.v1 as components
import streamlit as st
import json
from typing import List, Dict, Optional, Tuple
import math


CIT_BUILDINGS = [
    {'name': 'Administrative Block', 'height': 15, 'color': '#D4A574', 'lat': 11.0293, 'lon': 76.9382, 'width': 40, 'depth': 25, 'floors': 4},
    {'name': 'Main Academic Block', 'height': 20, 'color': '#C49A6C', 'lat': 11.0295, 'lon': 76.9385, 'width': 60, 'depth': 30, 'floors': 5},
    {'name': 'Engineering Block', 'height': 18, 'color': '#B8956A', 'lat': 11.0298, 'lon': 76.9388, 'width': 50, 'depth': 28, 'floors': 4},
    {'name': 'Central Library', 'height': 12, 'color': '#A68B5C', 'lat': 11.0290, 'lon': 76.9390, 'width': 35, 'depth': 35, 'floors': 3},
    {'name': 'Auditorium', 'height': 10, 'color': '#9A7B4F', 'lat': 11.0287, 'lon': 76.9378, 'width': 45, 'depth': 30, 'floors': 2},
    {'name': 'Workshop Complex', 'height': 8, 'color': '#8B7355', 'lat': 11.0300, 'lon': 76.9380, 'width': 55, 'depth': 35, 'floors': 2},
    {'name': 'Hostel Block', 'height': 25, 'color': '#C9A961', 'lat': 11.0305, 'lon': 76.9392, 'width': 40, 'depth': 50, 'floors': 6},
    {'name': 'Sports Complex', 'height': 6, 'color': '#7B8B6F', 'lat': 11.0285, 'lon': 76.9395, 'width': 70, 'depth': 40, 'floors': 1},
    {'name': 'Cafeteria', 'height': 6, 'color': '#D4B896', 'lat': 11.0292, 'lon': 76.9375, 'width': 25, 'depth': 20, 'floors': 1},
    {'name': 'Parking Area', 'height': 4, 'color': '#6B6B6B', 'lat': 11.0288, 'lon': 76.9370, 'width': 30, 'depth': 40, 'floors': 1}
]


def generate_3d_map_html(center_lat, center_lon, zoom, vehicle_position, objects, trajectory, map_style, height):
    if vehicle_position is None:
        vehicle_position = {'lat': center_lat, 'lon': center_lon, 'heading': 0, 'speed': 0}
    if objects is None:
        objects = []
    if trajectory is None:
        trajectory = []

    buildings_geojson = []
    for building in CIT_BUILDINGS:
        half_w = building['width'] / 2
        half_d = building['depth'] / 2
        meters_to_deg = 0.000009
        corners = [
            [building['lon'] - half_w * meters_to_deg, building['lat'] - half_d * meters_to_deg],
            [building['lon'] + half_w * meters_to_deg, building['lat'] - half_d * meters_to_deg],
            [building['lon'] + half_w * meters_to_deg, building['lat'] + half_d * meters_to_deg],
            [building['lon'] - half_w * meters_to_deg, building['lat'] + half_d * meters_to_deg],
            [building['lon'] - half_w * meters_to_deg, building['lat'] - half_d * meters_to_deg]
        ]
        buildings_geojson.append({
            'type': 'Feature',
            'properties': {
                'height': building['height'],
                'color': building['color'],
                'name': building['name'],
                'floors': building['floors']
            },
            'geometry': {'type': 'Polygon', 'coordinates': [corners]}
        })

    trajectory_geojson = []
    if trajectory:
        trajectory_geojson = [{
            'type': 'Feature',
            'properties': {'stroke': '#00ffff', 'stroke-width': 4},
            'geometry': {'type': 'LineString', 'coordinates': [[lon, lat] for lat, lon in trajectory]}
        }]

    objects_geojson = []
    for obj in objects:
        if 'gps_coordinates' in obj:
            gps = obj['gps_coordinates']
            color = get_object_color(obj.get('class_name', 'unknown'))
            objects_geojson.append({
                'type': 'Feature',
                'properties': {'class': obj.get('class_name', 'unknown'), 'distance': obj.get('distance', 0), 'confidence': obj.get('confidence', 0), 'color': color},
                'geometry': {'type': 'Point', 'coordinates': [gps['longitude'], gps['latitude']]}
            })

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CIT Campus 3D - EdgeDrive</title>
    <script src="https://unpkg.com/deck.gl@8.9.3/dist.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', 'Roboto', sans-serif; overflow: hidden; background: linear-gradient(135deg, #0c0c1e 0%, #1a1a3e 50%, #0f0f2d 100%); }}
        #container {{ width: 100%; height: {height}px; }}
        
        #info-panel {{
            position: absolute; top: 20px; left: 20px;
            background: linear-gradient(135deg, rgba(15,15,45,0.95), rgba(30,30,60,0.9));
            backdrop-filter: blur(20px);
            padding: 25px; border-radius: 20px; color: white;
            z-index: 1000; min-width: 300px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 30px rgba(102,126,234,0.3);
            border: 1px solid rgba(102,126,234,0.4);
        }}
        #info-panel h2 {{
            font-size: 18px; margin-bottom: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 700; text-shadow: 0 0 30px rgba(102,126,234,0.5);
        }}
        .info-row {{ display: flex; justify-content: space-between; padding: 8px 0; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
        .info-row:last-child {{ border-bottom: none; }}
        .info-label {{ color: #8888aa; }}
        .info-value {{ font-weight: 600; color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.3); }}
        
        #controls {{
            position: absolute; bottom: 25px; left: 20px;
            background: linear-gradient(135deg, rgba(15,15,45,0.95), rgba(30,30,60,0.9));
            padding: 15px; border-radius: 16px; z-index: 1000;
            display: flex; gap: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid rgba(102,126,234,0.3);
        }}
        .control-btn {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none; color: white; padding: 12px 18px;
            border-radius: 10px; cursor: pointer; font-size: 13px;
            transition: all 0.3s ease; font-weight: 600;
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
            position: relative; overflow: hidden;
        }}
        .control-btn::before {{
            content: ''; position: absolute; top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }}
        .control-btn:hover::before {{ left: 100%; }}
        .control-btn:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102,126,234,0.6); }}
        .control-btn:active {{ transform: translateY(-1px); }}
        
        #legend {{
            position: absolute; top: 20px; right: 20px;
            background: linear-gradient(135deg, rgba(15,15,45,0.95), rgba(30,30,60,0.9));
            backdrop-filter: blur(20px);
            padding: 20px; border-radius: 16px; z-index: 1000;
            color: white; max-height: 80vh; overflow-y: auto;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid rgba(102,126,234,0.3);
            min-width: 200px;
        }}
        #legend h3 {{ font-size: 14px; margin-bottom: 12px; color: #667eea; font-weight: 600; }}
        .legend-item {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 12px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 4px; box-shadow: 0 0 10px currentColor; }}
        
        #loading {{
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, rgba(15,15,45,0.98), rgba(30,30,60,0.98));
            padding: 40px 60px; border-radius: 20px; color: white;
            z-index: 2000; text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.6);
            border: 1px solid rgba(102,126,234,0.4);
        }}
        .spinner {{
            width: 50px; height: 50px;
            border: 3px solid rgba(102,126,234,0.3);
            border-top-color: #667eea; border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto 20px;
            box-shadow: 0 0 30px rgba(102,126,234,0.5);
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        
        #tooltip {{
            position: absolute;
            background: linear-gradient(135deg, rgba(20,20,50,0.98), rgba(40,40,80,0.98));
            color: white; padding: 15px 18px; border-radius: 12px;
            font-size: 13px; z-index: 1000; display: none;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
            border: 1px solid rgba(102,126,234,0.5);
            backdrop-filter: blur(10px);
            max-width: 280px;
        }}
        #tooltip strong {{ color: #667eea; font-size: 14px; }}
        
        .pulse-ring {{
            position: absolute; border-radius: 50%;
            border: 2px solid rgba(0,255,255,0.6);
            animation: pulse 2s ease-out infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            100% {{ transform: scale(3); opacity: 0; }}
        }}
        
        #title-overlay {{
            position: absolute; top: 20px; left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, rgba(15,15,45,0.9), rgba(30,30,60,0.9));
            padding: 15px 40px; border-radius: 30px;
            z-index: 1000; text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid rgba(102,126,234,0.3);
        }}
        #title-overlay h1 {{
            font-size: 20px; font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 0;
        }}
        #title-overlay p {{ font-size: 11px; color: #8888aa; margin: 5px 0 0 0; }}
        
        #stats-bar {{
            position: absolute; bottom: 25px; right: 20px;
            background: linear-gradient(135deg, rgba(15,15,45,0.95), rgba(30,30,60,0.9));
            padding: 15px 20px; border-radius: 16px; z-index: 1000;
            display: flex; gap: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border: 1px solid rgba(102,126,234,0.3);
        }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #667eea; text-shadow: 0 0 15px rgba(102,126,234,0.5); }}
        .stat-label {{ font-size: 11px; color: #8888aa; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="title-overlay">
        <h1>🏫 CIT CAMPUS 3D</h1>
        <p>Coimbatore Institute of Technology • Real-time Visualization</p>
    </div>
    
    <div id="info-panel">
        <h2>🛰️ Vehicle Status</h2>
        <div class="info-row"><span class="info-label">Latitude</span><span class="info-value" id="lat">{vehicle_position['lat']:.6f}</span></div>
        <div class="info-row"><span class="info-label">Longitude</span><span class="info-value" id="lon">{vehicle_position['lon']:.6f}</span></div>
        <div class="info-row"><span class="info-label">Heading</span><span class="info-value" id="heading">{vehicle_position['heading']:.1f}°</span></div>
        <div class="info-row"><span class="info-label">Speed</span><span class="info-value" id="speed">{vehicle_position['speed']:.2f} m/s</span></div>
        <div class="info-row"><span class="info-label">Objects</span><span class="info-value" id="objects-count">{len(objects)}</span></div>
    </div>
    
    <div id="legend">
        <h3>🎯 Detected Objects</h3>
        <div class="legend-item"><div class="legend-color" style="background:#00ff00;box-shadow:0 0 10px #00ff00;"></div><span>Vehicle (You)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#ffff00;box-shadow:0 0 10px #ffff00;"></div><span>Person</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#00ffff;box-shadow:0 0 10px #00ffff;"></div><span>Car</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#ff9500;box-shadow:0 0 10px #ff9500;"></div><span>Bicycle</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#ff4444;box-shadow:0 0 10px #ff4444;"></div><span>Bus/Truck</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#00ffff;opacity:0.7;box-shadow:0 0 10px #00ffff;"></div><span>Trajectory</span></div>
        <h3 style="margin-top:15px;">🏢 Campus Buildings</h3>
        <div class="legend-item"><div class="legend-color" style="background:#D4A574;"></div><span>Admin ({CIT_BUILDINGS[0]['floors']}F)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#C49A6C;"></div><span>Main Academic ({CIT_BUILDINGS[1]['floors']}F)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#C9A961;"></div><span>Hostel ({CIT_BUILDINGS[6]['floors']}F)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#A68B5C;"></div><span>Library ({CIT_BUILDINGS[3]['floors']}F)</span></div>
    </div>
    
    <div id="controls">
        <button class="control-btn" onclick="resetView()">🔄 Reset</button>
        <button class="control-btn" onclick="toggleBuildings()">🏢 Buildings</button>
        <button class="control-btn" onclick="toggleAnimation()">🌀 Orbit</button>
        <button class="control-btn" onclick="setTopView()">🔝 Top</button>
        <button class="control-btn" onclick="toggleNightMode()">🌙 Night</button>
    </div>
    
    <div id="stats-bar">
        <div class="stat-item">
            <div class="stat-value" id="stat-buildings">{len(CIT_BUILDINGS)}</div>
            <div class="stat-label">Buildings</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="stat-objects">{len(objects)}</div>
            <div class="stat-label">Objects</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="stat-zoom">{zoom}</div>
            <div class="stat-label">Zoom</div>
        </div>
    </div>
    
    <div id="loading"><div class="spinner"></div><div>Loading 3D Campus...</div></div>
    <div id="tooltip"></div>

    <script>
        const CENTER_LON = {center_lon};
        const CENTER_LAT = {center_lat};
        const INITIAL_ZOOM = {zoom};
        const INITIAL_PITCH = 55;
        const INITIAL_BEARING = 30;
        const buildingsData = {json.dumps(buildings_geojson)};
        const trajectoryData = {json.dumps(trajectory_geojson)};
        const objectsData = {json.dumps(objects_geojson)};
        const vehicleLon = {vehicle_position['lon']};
        const vehicleLat = {vehicle_position['lat']};
        
        let deckgl = null;
        let showBuildings = true;
        let isAnimating = false;
        let animFrame = null;
        let isNightMode = true;
        
        function hexToRgb(hex) {{
            const r = parseInt(hex.slice(1,3), 16);
            const g = parseInt(hex.slice(3,5), 16);
            const b = parseInt(hex.slice(5,7), 16);
            return [r, g, b];
        }}
        
        function createGridLayer() {{
            const gridSize = 0.0004;
            const gridLines = [];
            for (let i = -5; i <= 5; i++) {{
                gridLines.push({{
                    type: 'Feature',
                    properties: {{ type: 'grid' }},
                    geometry: {{ type: 'LineString', coordinates: [[CENTER_LON - 0.003, CENTER_LAT + i * gridSize], [CENTER_LON + 0.003, CENTER_LAT + i * gridSize]] }}
                }});
                gridLines.push({{
                    type: 'Feature',
                    properties: {{ type: 'grid' }},
                    geometry: {{ type: 'LineString', coordinates: [[CENTER_LON + i * gridSize, CENTER_LAT - 0.003], [CENTER_LON + i * gridSize, CENTER_LAT + 0.003]] }}
                }});
            }}
            return new deck.GeoJsonLayer({{
                id: 'grid',
                data: gridLines,
                stroked: true,
                getWidth: 1,
                getColor: isNightMode ? [100, 100, 150, 100] : [150, 150, 180, 150],
                opacity: 0.4
            }});
        }}
        
        function createBuildingLayer() {{
            if (!showBuildings || buildingsData.length === 0) return null;
            return new deck.GeoJsonLayer({{
                id: 'buildings',
                data: buildingsData,
                extruded: true,
                wireframe: true,
                getElevation: f => f.properties.height,
                getFillColor: f => {{
                    const c = hexToRgb(f.properties.color);
                    return isNightMode ? [c[0]*0.7, c[1]*0.7, c[2]*0.9, 220] : [c[0], c[1], c[2], 200];
                }},
                getLineColor: isNightMode ? [255, 255, 255, 255] : [255, 255, 255, 180],
                lineWidthScale: 2,
                opacity: 0.95,
                pickable: true,
                autoHighlight: true,
                highlightColor: [255, 255, 100, 150],
                onHover: info => {{
                    const tip = document.getElementById('tooltip');
                    if (info.object) {{
                        tip.innerHTML = '<strong>' + info.object.properties.name + '</strong><br>Height: ' + info.object.properties.height + 'm<br>Floors: ' + info.object.properties.floors;
                        tip.style.left = (info.x + 20) + 'px';
                        tip.style.top = (info.y + 20) + 'px';
                        tip.style.display = 'block';
                    }} else {{
                        tip.style.display = 'none';
                    }}
                }}
            }});
        }}
        
        function createTrajectoryLayer() {{
            if (trajectoryData.length === 0) return null;
            return new deck.PathLayer({{
                id: 'trajectory',
                data: trajectoryData,
                getPath: d => d.geometry.coordinates,
                getColor: [0, 255, 255],
                getWidth: 6,
                widthMinPixels: 5,
                opacity: 0.9,
                capRounded: true,
                jointRounded: true
            }});
        }}
        
        function createObjectLayer() {{
            if (objectsData.length === 0) return null;
            return new deck.ScatterplotLayer({{
                id: 'objects',
                data: objectsData,
                getPosition: d => d.geometry.coordinates,
                getFillColor: d => hexToRgb(d.properties.color),
                getRadius: 20,
                radiusMinPixels: 12,
                radiusMaxPixels: 30,
                opacity: 0.9,
                pickable: true,
                autoHighlight: true,
                stroked: true,
                getLineColor: [255, 255, 255],
                getLineWidth: 2,
                onHover: info => {{
                    const tip = document.getElementById('tooltip');
                    if (info.object) {{
                        tip.innerHTML = '<strong>' + info.object.properties.class.toUpperCase() + '</strong><br>Distance: ' + info.object.properties.distance.toFixed(1) + 'm<br>Confidence: ' + (info.object.properties.confidence * 100).toFixed(0) + '%';
                        tip.style.left = (info.x + 20) + 'px';
                        tip.style.top = (info.y + 20) + 'px';
                        tip.style.display = 'block';
                    }} else {{
                        tip.style.display = 'none';
                    }}
                }}
            }});
        }}
        
        function createVehicleLayer() {{
            return new deck.ScatterplotLayer({{
                id: 'vehicle',
                data: [{{position: [vehicleLon, vehicleLat]}}],
                getPosition: d => d.position,
                getFillColor: [0, 255, 100],
                getRadius: 25,
                radiusMinPixels: 18,
                opacity: 1,
                stroked: true,
                getLineColor: [255, 255, 255],
                getLineWidth: 3
            }});
        }}
        
        function createGlowLayer() {{
            return new deck.ScatterplotLayer({{
                id: 'vehicle-glow',
                data: [{{position: [vehicleLon, vehicleLat]}}],
                getPosition: d => d.position,
                getFillColor: [0, 255, 100, 80],
                getRadius: 40,
                radiusMinPixels: 25,
                opacity: 0.4
            }});
        }}
        
        function getAllLayers() {{
            const layers = [createGridLayer()];
            const trajLayer = createTrajectoryLayer();
            if (trajLayer) layers.push(trajLayer);
            const objLayer = createObjectLayer();
            if (objLayer) layers.push(objLayer);
            layers.push(createGlowLayer());
            layers.push(createVehicleLayer());
            const buildingLayer = createBuildingLayer();
            if (buildingLayer) layers.push(buildingLayer);
            return layers;
        }}
        
        function updateStats(viewState) {{
            document.getElementById('lat').textContent = viewState.latitude.toFixed(6);
            document.getElementById('lon').textContent = viewState.longitude.toFixed(6);
            const hdg = ((viewState.bearing % 360) + 360) % 360;
            document.getElementById('heading').textContent = hdg.toFixed(1) + '°';
            document.getElementById('stat-zoom').textContent = viewState.zoom.toFixed(1);
        }}
        
        function initDeck() {{
            console.log('Initializing Deck.gl...');
            try {{
                deckgl = new deck.DeckGL({{
                    container: 'container',
                    layers: getAllLayers(),
                    initialViewState: {{
                        longitude: CENTER_LON,
                        latitude: CENTER_LAT,
                        zoom: INITIAL_ZOOM,
                        pitch: INITIAL_PITCH,
                        bearing: INITIAL_BEARING
                    }},
                    controller: {{
                        dragRotate: true,
                        dragPan: true,
                        scrollZoom: true,
                        touchZoom: true,
                        keyboard: true
                    }},
                    onViewStateChange: e => {{
                        updateStats(e.viewState);
                    }},
                    onWebGLInitialized: gl => {{
                        gl.clearColor(0.05, 0.05, 0.12, 1.0);
                        gl.enable(gl.DEPTH_TEST);
                        gl.enable(gl.BLEND);
                    }}
                }});
                document.getElementById('loading').style.display = 'none';
                console.log('Deck.gl ready!');
            }} catch (err) {{
                console.error('Error:', err);
                document.getElementById('loading').innerHTML = '<div style="color:#ff4444;">Error: ' + err.message + '</div>';
            }}
        }}
        
        function resetView() {{
            if (!deckgl) return;
            deckgl.setProps({{
                initialViewState: {{
                    longitude: CENTER_LON,
                    latitude: CENTER_LAT,
                    zoom: INITIAL_ZOOM,
                    pitch: INITIAL_PITCH,
                    bearing: INITIAL_BEARING
                }},
                transitionDuration: 1500,
                transitionInterpolator: new deck.LinearInterpolator()
            }});
        }}
        
        function toggleBuildings() {{
            showBuildings = !showBuildings;
            if (deckgl) {{
                deckgl.setProps({{layers: getAllLayers()}});
            }}
        }}
        
        function toggleAnimation() {{
            isAnimating = !isAnimating;
            if (isAnimating) animate();
            else if (animFrame) cancelAnimationFrame(animFrame);
        }}
        
        function animate() {{
            if (!isAnimating || !deckgl) return;
            const t = Date.now() * 0.0002;
            deckgl.setProps({{
                viewState: {{
                    bearing: Math.sin(t) * 60,
                    pitch: 45 + Math.cos(t * 0.5) * 15
                }},
                transitionDuration: 800
            }});
            animFrame = requestAnimationFrame(animate);
        }}
        
        function setTopView() {{
            if (!deckgl) return;
            deckgl.setProps({{
                initialViewState: {{
                    longitude: CENTER_LON,
                    latitude: CENTER_LAT,
                    zoom: INITIAL_ZOOM + 2,
                    pitch: 0,
                    bearing: 0
                }},
                transitionDuration: 1500
            }});
        }}
        
        function toggleNightMode() {{
            isNightMode = !isNightMode;
            if (deckgl) {{
                deckgl.setProps({{layers: getAllLayers()}});
            }}
            document.body.style.background = isNightMode ?
                'linear-gradient(135deg, #0c0c1e 0%, #1a1a3e 50%, #0f0f2d 100%)' :
                'linear-gradient(135deg, #1a2a4a 0%, #2a3a5a 50%, #1a2a4a 100%)';
        }}
        
        window.addEventListener('load', initDeck);
    </script>
</body>
</html>'''
    return html


def get_object_color(class_name):
    colors = {'person': '#ffff00', 'car': '#00ffff', 'bicycle': '#ff9500', 'bus': '#ff4444', 'truck': '#ff4444', 'motorcycle': '#ff69b4', 'unknown': '#888888'}
    return colors.get(class_name.lower(), colors['unknown'])


def render_3d_map(center_lat=11.0293, center_lon=76.9382, zoom=15.5, vehicle_position=None, objects=None, trajectory=None, map_style='dark', height=650, key=None):
    html = generate_3d_map_html(center_lat, center_lon, zoom, vehicle_position, objects, trajectory, map_style, height)
    components.html(html, height=height + 50, scrolling=False)


if __name__ == "__main__":
    st.set_page_config(page_title="3D Campus Map", layout="wide")
    st.title("🏫 CIT Campus 3D")
    vehicle_pos = {'lat': 11.0293, 'lon': 76.9382, 'heading': 45, 'speed': 5.0}
    test_objects = [
        {'class_name': 'person', 'distance': 15.0, 'confidence': 0.92, 'gps_coordinates': {'latitude': 11.0295, 'longitude': 76.9385}},
        {'class_name': 'car', 'distance': 25.0, 'confidence': 0.88, 'gps_coordinates': {'latitude': 11.0298, 'longitude': 76.9388}}
    ]
    test_trajectory = [(11.0290, 76.9378), (11.0291, 76.9380), (11.0292, 76.9381), (11.0293, 76.9382)]
    render_3d_map(vehicle_position=vehicle_pos, objects=test_objects, trajectory=test_trajectory, height=700)
    st.success("3D Campus loaded!")
