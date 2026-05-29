/**
 * ADAS 3D Navigation Dashboard - Tesla Style
 * 3D buildings, dark map theme, road-following simulation
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_BASE: '',
    GPS_POLL_INTERVAL: 1000,
    DEFAULT_LAT: 28.6139,
    DEFAULT_LON: 77.2090,
    DEFAULT_ZOOM: 15.5,
    MAP_TILT: 60,
    MAP_BEARING: 0,
    BUILDING_COLOR: '#1a1a1a',
    BUILDING_OUTLINE_COLOR: '#333333'
};

// ============================================
// Global State
// ============================================
let map = null;
let carMarker = null;
let routeLine = null;
let gpsData = {
    latitude: CONFIG.DEFAULT_LAT,
    longitude: CONFIG.DEFAULT_LON,
    altitude: 0,
    speed: 0,
    heading: 0,
    satellites: 0,
    accuracy: 0,
    status: 'disconnected'
};
let routeCoordinates = [];
let tripStartTime = null;
let totalDistance = 0;
let lastPosition = null;
let simulationMode = false;
let isSimulationRunning = false;
let is3DEnabled = true;

// ============================================
// Map Initialization with 3D Buildings
// ============================================
function initMap() {
    // Dark matter-style map style with 3D buildings
    const darkStyle = {
        version: 8,
        sources: {
            'osm': {
                type: 'raster',
                tiles: [
                    'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
                    'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',
                    'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png'
                ],
                tileSize: 256,
                attribution: '© OpenStreetMap'
            }
        },
        layers: [
            {
                id: 'background',
                type: 'background',
                paint: {
                    'background-color': '#0d0d0d'
                }
            },
            {
                id: 'osm-tiles',
                type: 'raster',
                source: 'osm',
                minzoom: 0,
                maxzoom: 19,
                paint: {
                    'raster-opacity': 0.6,
                    'raster-brightness-min': 0,
                    'raster-brightness-max': 0.5
                }
            }
        ]
    };

    map = new maplibregl.Map({
        container: 'map',
        style: darkStyle,
        center: [CONFIG.DEFAULT_LON, CONFIG.DEFAULT_LAT],
        zoom: CONFIG.DEFAULT_ZOOM,
        pitch: CONFIG.MAP_TILT,
        bearing: CONFIG.MAP_BEARING,
        antialias: true,
        preserveDrawingBuffer: true
    });

    map.on('load', () => {
        console.log('3D Map loaded');
        hideLoadingOverlay();
        add3DBuildings();
        addCarMarker();
        addRouteLine();
        setupMapEvents();
    });

    map.on('error', (e) => {
        console.error('Map error:', e);
        hideLoadingOverlay();
    });
}

function add3DBuildings() {
    // Get OSM building data and extrude them
    map.addSource('mapbox://mapbox.mapbox-streets-v8', {
        'type': 'vector',
        'url': 'mapbox://mapbox.mapbox-streets-v8'
    }).catch(() => {
        // Fallback: Use OSM buildings via overpass or skip
        console.log('Using fallback building layer');
        addFallbackBuildings();
    });

    // Add 3D building layer using OSM data
    map.on('style.load', () => {
        // Add building extrusion layer
        const layers = map.getStyle().layers;
        const labelLayerId = layers.find(
            (layer) => layer.type === 'symbol' && layer.layout['text-field']
        )?.id;

        map.addLayer(
            {
                'id': '3d-buildings',
                'source': 'composite',
                'source-layer': 'building',
                'filter': ['==', 'extrude', 'true'],
                'type': 'fill-extrusion',
                'minzoom': 14,
                'paint': {
                    'fill-extrusion-color': CONFIG.BUILDING_COLOR,
                    'fill-extrusion-height': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        15,
                        0,
                        15.05,
                        ['get', 'height']
                    ],
                    'fill-extrusion-base': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        15,
                        0,
                        15.05,
                        ['get', 'min_height']
                    ],
                    'fill-extrusion-opacity': 0.9
                }
            },
            labelLayerId
        ).catch(() => {
            console.log('3D buildings layer added with fallback');
        });
    });
}

function addFallbackBuildings() {
    // Fallback: Create uniform height buildings using fill-extrusion
    // This creates a grid of buildings for demonstration
    const buildingFeatures = [];
    const centerLat = CONFIG.DEFAULT_LAT;
    const centerLon = CONFIG.DEFAULT_LON;
    const gridSize = 0.002;
    const buildingSize = 0.0008;
    const uniformHeight = 50; // Uniform building height

    for (let lat = -5; lat <= 5; lat++) {
        for (let lon = -5; lon <= 5; lon++) {
            // Skip some buildings for variation
            if (Math.random() > 0.7) continue;

            const buildingLat = centerLat + lat * gridSize;
            const buildingLon = centerLon + lon * gridSize;

            buildingFeatures.push({
                type: 'Feature',
                geometry: {
                    type: 'Polygon',
                    coordinates: [[
                        [buildingLon - buildingSize/2, buildingLat - buildingSize/2],
                        [buildingLon + buildingSize/2, buildingLat - buildingSize/2],
                        [buildingLon + buildingSize/2, buildingLat + buildingSize/2],
                        [buildingLon - buildingSize/2, buildingLat + buildingSize/2],
                        [buildingLon - buildingSize/2, buildingLat - buildingSize/2]
                    ]]
                },
                properties: {
                    height: uniformHeight + Math.random() * 20,
                    color: CONFIG.BUILDING_COLOR
                }
            });
        }
    }

    map.addSource('buildings', {
        type: 'geojson',
        data: {
            type: 'FeatureCollection',
            features: buildingFeatures
        }
    });

    map.addLayer({
        id: '3d-buildings-fallback',
        type: 'fill-extrusion',
        source: 'buildings',
        minzoom: 13,
        paint: {
            'fill-extrusion-color': [
                'interpolate',
                ['linear'],
                ['zoom'],
                13,
                '#1a1a1a',
                16,
                '#2a2a2a'
            ],
            'fill-extrusion-height': ['get', 'height'],
            'fill-extrusion-base': 0,
            'fill-extrusion-opacity': 0.9
        }
    });
}

function addCarMarker() {
    const el = document.createElement('div');
    el.className = 'car-marker';
    el.innerHTML = `
        <svg viewBox="0 0 50 50" fill="none">
            <defs>
                <linearGradient id="carGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#00d4ff"/>
                    <stop offset="100%" style="stop-color:#0066ff"/>
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <!-- Outer glow ring -->
            <circle cx="25" cy="25" r="20" stroke="url(#carGrad)" stroke-width="2" fill="none" opacity="0.5" filter="url(#glow)"/>
            <!-- Car body -->
            <ellipse cx="25" cy="28" rx="14" ry="9" fill="url(#carGrad)" opacity="0.4"/>
            <!-- Direction arrow -->
            <path d="M25 10 L30 22 L25 20 L20 22 Z" fill="url(#carGrad)" filter="url(#glow)"/>
            <!-- Center point -->
            <circle cx="25" cy="25" r="5" fill="url(#carGrad)"/>
            <!-- Inner ring -->
            <circle cx="25" cy="25" r="12" stroke="url(#carGrad)" stroke-width="1.5" fill="none" opacity="0.6"/>
        </svg>
    `;

    carMarker = new maplibregl.Marker({ element: el, anchor: 'center' })
        .setLngLat([CONFIG.DEFAULT_LON, CONFIG.DEFAULT_LAT])
        .addTo(map);
}

function addRouteLine() {
    map.addSource('route', {
        type: 'geojson',
        data: {
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: []
            }
        }
    });

    map.addLayer({
        id: 'route',
        type: 'line',
        source: 'route',
        layout: {
            'line-join': 'round',
            'line-cap': 'round'
        },
        paint: {
            'line-color': '#00d4ff',
            'line-width': 4,
            'line-opacity': 0.8,
            'line-dasharray': [3, 2],
            'line-blur': 2
        }
    });
}

function updateCarPosition(lng, lat, heading) {
    if (!carMarker) return;

    carMarker.setLngLat([lng, lat]);

    const markerElement = carMarker.getElement();
    const svg = markerElement.querySelector('svg');
    if (svg) {
        svg.style.transform = `rotate(${heading}deg)`;
    }

    // Smooth camera follow
    if (!map.isMoving() && !map.isZooming()) {
        map.easeTo({
            center: [lng, lat],
            bearing: heading,
            pitch: CONFIG.MAP_TILT,
            duration: 1500,
            easing: (t) => t
        });
    }

    addToRoute(lng, lat);
}

function addToRoute(lng, lat) {
    routeCoordinates.push([lng, lat]);

    if (routeCoordinates.length > 200) {
        routeCoordinates.shift();
    }

    if (map.getSource('route')) {
        map.getSource('route').setData({
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: routeCoordinates
            }
        });
    }
}

function setupMapEvents() {
    map.on('move', () => {
        // Update bearing display if needed
    });
}

// ============================================
// GPS Data Handling
// ============================================
async function fetchGPSData() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/gps`);
        const data = await response.json();
        updateGPSData(data);
    } catch (error) {
        console.error('GPS fetch error:', error);
    }
}

function updateGPSData(data) {
    gpsData = { ...gpsData, ...data };
    updateDashboard();
    updateMap();
    updateConnectionStatus();
    calculateTripData();
}

function updateDashboard() {
    // Speed
    const speed = gpsData.speed || 0;
    document.getElementById('speedValue').textContent = speed.toFixed(0);

    // Speed bar
    const maxSpeed = 180;
    const speedPercent = Math.min((speed / maxSpeed) * 100, 100);
    document.getElementById('speedBar').style.width = speedPercent + '%';

    // Heading
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const index = Math.round(gpsData.heading / 45) % 8;
    document.getElementById('headingValue').textContent = directions[index];

    // Compass rotation
    const compassArrow = document.querySelector('.compass-arrow');
    if (compassArrow) {
        compassArrow.style.transform = `translate(-50%, 0) rotate(${gpsData.heading}deg)`;
    }

    // GPS data
    document.getElementById('latitude').textContent = gpsData.latitude.toFixed(6);
    document.getElementById('longitude').textContent = gpsData.longitude.toFixed(6);
    document.getElementById('altitude').textContent = gpsData.altitude.toFixed(0) + ' m';
    document.getElementById('satellites').textContent = gpsData.satellites;
    document.getElementById('accuracy').textContent = gpsData.accuracy.toFixed(1) + ' m';

    // Footer coords
    document.getElementById('footerCoords').textContent = 
        `${gpsData.latitude.toFixed(4)}, ${gpsData.longitude.toFixed(4)}`;

    // Signal strength
    updateSignalStrength(gpsData.satellites);

    // Time
    updateTime();
}

function updateSignalStrength(count) {
    const bars = document.querySelectorAll('.signal-bar');
    const activeCount = Math.min(Math.ceil(count / 3), 4);
    bars.forEach((bar, index) => {
        bar.classList.toggle('active', index < activeCount);
    });
}

function updateTime() {
    const now = new Date();
    document.getElementById('currentTime').textContent = now.toLocaleTimeString('en-US', { hour12: false });
}

function updateMap() {
    const lng = gpsData.longitude;
    const lat = gpsData.latitude;
    const heading = gpsData.heading || 0;
    updateCarPosition(lng, lat, heading);
}

function updateConnectionStatus() {
    const indicator = document.getElementById('connectionStatus');
    const statusText = indicator.querySelector('.status-text');

    if (gpsData.status?.includes('error')) {
        indicator.className = 'status-indicator error';
        statusText.textContent = 'ERROR';
    } else if (['connected', 'active', 'simulated'].includes(gpsData.status)) {
        indicator.className = 'status-indicator connected';
        statusText.textContent = gpsData.status.toUpperCase();
    } else {
        indicator.className = 'status-indicator';
        statusText.textContent = gpsData.status?.toUpperCase() || 'DISCONNECTED';
    }
}

function calculateTripData() {
    if (!lastPosition) {
        lastPosition = { lat: gpsData.latitude, lng: gpsData.longitude };
        tripStartTime = new Date();
        return;
    }

    const distance = calculateDistance(
        lastPosition.lat, lastPosition.lng,
        gpsData.latitude, gpsData.longitude
    );

    totalDistance += distance;
    lastPosition = { lat: gpsData.latitude, lng: gpsData.longitude };

    document.getElementById('tripDistance').textContent = totalDistance.toFixed(2) + ' km';

    if (tripStartTime) {
        const elapsed = new Date() - tripStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('tripTime').textContent =
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

        // Average speed
        const hoursElapsed = elapsed / 3600000;
        const avgSpeed = hoursElapsed > 0 ? totalDistance / hoursElapsed : 0;
        document.getElementById('avgSpeed').textContent = avgSpeed.toFixed(0) + ' km/h';
    }
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

function toRad(deg) {
    return deg * (Math.PI / 180);
}

// ============================================
// Simulation Controls
// ============================================
async function startSimulation() {
    try {
        await fetch(`${CONFIG.API_BASE}/gps/start_simulation`, { method: 'POST' });
        isSimulationRunning = true;
        addAlert('Navigation started - Following roads', 'info');
        updateSimulationButtons(true);
    } catch (error) {
        console.error('Start simulation error:', error);
        addAlert('Failed to start navigation', 'error');
    }
}

async function stopSimulation() {
    try {
        await fetch(`${CONFIG.API_BASE}/gps/stop_simulation`, { method: 'POST' });
        isSimulationRunning = false;
        addAlert('Navigation stopped', 'info');
        updateSimulationButtons(false);
    } catch (error) {
        console.error('Stop simulation error:', error);
    }
}

function updateSimulationButtons(running) {
    document.getElementById('btnStartSim').disabled = running;
    document.getElementById('btnStopSim').disabled = !running;
}

// ============================================
// Map Controls
// ============================================
function setupMapControls() {
    document.getElementById('btnTiltUp').addEventListener('click', () => {
        map.easeTo({ pitch: Math.min(map.getPitch() + 15, 75), duration: 500 });
    });

    document.getElementById('btnTiltDown').addEventListener('click', () => {
        map.easeTo({ pitch: Math.max(map.getPitch() - 15, 0), duration: 500 });
    });

    document.getElementById('btnZoomIn').addEventListener('click', () => {
        map.zoomTo(map.getZoom() + 0.5);
    });

    document.getElementById('btnZoomOut').addEventListener('click', () => {
        map.zoomTo(map.getZoom() - 0.5);
    });

    document.getElementById('btnCenter').addEventListener('click', () => {
        map.easeTo({
            center: [gpsData.longitude, gpsData.latitude],
            bearing: gpsData.heading,
            pitch: CONFIG.MAP_TILT,
            duration: 1000
        });
    });

    document.getElementById('btn3D').addEventListener('click', () => {
        is3DEnabled = !is3DEnabled;
        const buildingLayer = map.getLayer('3d-buildings-fallback');
        if (buildingLayer) {
            map.setLayoutProperty('3d-buildings-fallback', 'visibility', is3DEnabled ? 'visible' : 'none');
        }
        addAlert(is3DEnabled ? '3D buildings enabled' : '3D buildings disabled', 'info');
    });
}

// ============================================
// Simulation Toggle
// ============================================
function setupSimulationToggle() {
    const toggle = document.getElementById('simulationMode');
    toggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            startSimulation();
        } else {
            stopSimulation();
        }
    });
}

// ============================================
// Button Handlers
// ============================================
function setupButtons() {
    document.getElementById('btnStartSim').addEventListener('click', startSimulation);
    document.getElementById('btnStopSim').addEventListener('click', stopSimulation);
}

// ============================================
// Alerts
// ============================================
function addAlert(message, type = 'info') {
    const alertList = document.getElementById('alertList');
    const alertItem = document.createElement('div');
    alertItem.className = `alert-item ${type}`;
    alertItem.textContent = message;

    alertList.insertBefore(alertItem, alertList.firstChild);

    while (alertList.children.length > 5) {
        alertList.removeChild(alertList.lastChild);
    }
}

// ============================================
// Loading Overlay
// ============================================
function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('hidden');
    setTimeout(() => {
        overlay.style.display = 'none';
    }, 500);
}

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('ADAS 3D Navigation initializing...');

    initMap();
    setupMapControls();
    setupSimulationToggle();
    setupButtons();

    setInterval(fetchGPSData, CONFIG.GPS_POLL_INTERVAL);
    fetchGPSData();

    console.log('ADAS 3D Navigation ready');
});
