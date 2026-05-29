/**
 * ADAS Navigation Dashboard - Frontend JavaScript
 * Real-time GPS visualization with MapLibre GL
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_BASE: '',
    GPS_POLL_INTERVAL: 500,   // 500ms for live updates
    DEFAULT_LAT: 28.6139,     // New Delhi
    DEFAULT_LON: 77.2090,
    DEFAULT_ZOOM: 15,
    SIMULATION_SPEED: 0.0001,  // Degrees per update
    SIMULATION_MAX_SPEED: 120, // km/h
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
let simulationInterval = null;
let isConnected = false;

// ============================================
// Map Initialization
// ============================================
function initMap() {
    map = new maplibregl.Map({
        container: 'map',
        style: {
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
                    attribution: '© OpenStreetMap Contributors'
                }
            },
            layers: [
                {
                    id: 'osm-tiles',
                    type: 'raster',
                    source: 'osm',
                    minzoom: 0,
                    maxzoom: 19
                }
            ]
        },
        center: [CONFIG.DEFAULT_LON, CONFIG.DEFAULT_LAT],
        zoom: CONFIG.DEFAULT_ZOOM,
        pitch: 45,
        bearing: 0,
        antialias: true
    });

    map.on('load', () => {
        console.log('Map loaded successfully');
        addCarMarker();
        updateMapStyle();
    });

    map.on('error', (e) => {
        console.error('Map error:', e);
        addAlert('Map loading error', 'error');
    });
}

function updateMapStyle() {
    // Apply dark theme overlay
    const mapContainer = document.getElementById('map');
    mapContainer.style.filter = 'brightness(0.85) contrast(1.1) hue-rotate(10deg)';
}

function addCarMarker() {
    // Create custom car marker element
    const el = document.createElement('div');
    el.className = 'car-marker';
    el.innerHTML = `
        <svg viewBox="0 0 40 40" fill="none">
            <defs>
                <linearGradient id="carGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#00d4ff"/>
                    <stop offset="100%" style="stop-color:#00ff88"/>
                </linearGradient>
            </defs>
            <!-- Car body -->
            <ellipse cx="20" cy="22" rx="12" ry="8" fill="url(#carGradient)" opacity="0.3"/>
            <!-- Car direction arrow -->
            <path d="M20 8 L24 18 L20 16 L16 18 Z" fill="url(#carGradient)"/>
            <!-- Center point -->
            <circle cx="20" cy="20" r="4" fill="url(#carGradient)"/>
            <!-- Outer ring -->
            <circle cx="20" cy="20" r="14" stroke="url(#carGradient)" stroke-width="1.5" fill="none" opacity="0.5"/>
        </svg>
    `;

    carMarker = new maplibregl.Marker({ element: el, anchor: 'center' })
        .setLngLat([CONFIG.DEFAULT_LON, CONFIG.DEFAULT_LAT])
        .addTo(map);
}

function updateCarPosition(lng, lat, heading) {
    if (!carMarker) return;

    // Update marker position
    carMarker.setLngLat([lng, lat]);

    // Rotate marker based on heading
    const markerElement = carMarker.getElement();
    const svg = markerElement.querySelector('svg');
    if (svg) {
        svg.style.transform = `rotate(${heading}deg)`;
    }

    // Center map on car if not user-interacted
    if (!map.isMoving()) {
        map.easeTo({
            center: [lng, lat],
            bearing: heading,
            duration: 1000,
            easing: (t) => t
        });
    }

    // Add to route
    addToRoute(lng, lat);
}

function addToRoute(lng, lat) {
    routeCoordinates.push([lng, lat]);

    // Keep only last 100 points
    if (routeCoordinates.length > 100) {
        routeCoordinates.shift();
    }

    updateRouteLine();
}

function updateRouteLine() {
    if (routeCoordinates.length < 2) return;

    // Remove existing route line
    if (map.getSource('route')) {
        map.getSource('route').setData({
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: routeCoordinates
            }
        });
    } else {
        map.addSource('route', {
            type: 'geojson',
            data: {
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'LineString',
                    coordinates: routeCoordinates
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
                'line-width': 3,
                'line-opacity': 0.8,
                'line-dasharray': [2, 2]
            }
        });
    }
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
        console.error('Failed to fetch GPS data:', error);
        updateConnectionStatus('error');
    }
}

function updateGPSData(data) {
    gpsData = { ...gpsData, ...data };

    // Update UI elements
    updateDashboard();
    updateMap();
    updateConnectionStatus();
    calculateTripData();
}

function updateDashboard() {
    // GPS Data Panel
    document.getElementById('latitude').textContent = gpsData.latitude.toFixed(6);
    document.getElementById('longitude').textContent = gpsData.longitude.toFixed(6);
    document.getElementById('altitude').textContent = gpsData.altitude.toFixed(1);
    document.getElementById('heading').textContent = gpsData.heading.toFixed(1);
    document.getElementById('satellites').textContent = gpsData.satellites;
    document.getElementById('accuracy').textContent = gpsData.accuracy.toFixed(1);

    // Map Overlay
    document.getElementById('mapLat').textContent = gpsData.latitude.toFixed(6);
    document.getElementById('mapLon').textContent = gpsData.longitude.toFixed(6);

    // Speed Display
    const speed = gpsData.speed || 0;
    document.getElementById('speedValue').textContent = speed.toFixed(0);
    document.getElementById('gaugeSpeed').textContent = speed.toFixed(0);

    // Update speed gauge arc
    const maxSpeed = 180;
    const circumference = 445 * 0.75; // 270 degrees
    const offset = circumference - (speed / maxSpeed) * circumference;
    const speedArc = document.querySelector('.speed-arc');
    if (speedArc) {
        speedArc.style.strokeDashoffset = offset;
    }

    // Heading Direction
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const index = Math.round(gpsData.heading / 45) % 8;
    document.getElementById('vehicleHeading').textContent = directions[index];

    // Compass Ring
    const compassArrow = document.getElementById('compassArrow');
    if (compassArrow) {
        compassArrow.style.transform = `translate(-50%, 0) rotate(${gpsData.heading}deg)`;
    }

    // Satellite Bars
    updateSatelliteBars(gpsData.satellites);

    // Signal Strength
    updateSignalStrength(gpsData.satellites);

    // Time & Date
    updateTime();
}

function updateSatelliteBars(count) {
    const container = document.getElementById('satelliteBars');
    container.innerHTML = '';

    const maxBars = 8;
    for (let i = 0; i < maxBars; i++) {
        const bar = document.createElement('div');
        bar.className = 'satellite-bar' + (i < count ? ' active' : '');
        container.appendChild(bar);
    }
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
    document.getElementById('currentDate').textContent = now.toLocaleDateString('en-US');
}

function updateMap() {
    const lng = gpsData.longitude;
    const lat = gpsData.latitude;
    const heading = gpsData.heading || 0;

    updateCarPosition(lng, lat, heading);
}

function updateConnectionStatus(status) {
    const indicator = document.getElementById('connectionStatus');
    const statusText = indicator.querySelector('.status-text');
    const gpsSourceStatus = document.getElementById('gpsSourceStatus');

    if (status === 'error' || gpsData.status?.includes('error')) {
        indicator.className = 'status-indicator error';
        statusText.textContent = 'ERROR';
        if (gpsSourceStatus) gpsSourceStatus.textContent = 'DISCONNECTED';
    } else if (gpsData.status === 'active' || gpsData.status === 'simulated') {
        indicator.className = 'status-indicator connected';
        statusText.textContent = 'CONNECTED';
        if (gpsSourceStatus) gpsSourceStatus.textContent = 'CONNECTED';
    } else {
        indicator.className = 'status-indicator';
        statusText.textContent = 'DISCONNECTED';
        if (gpsSourceStatus) gpsSourceStatus.textContent = 'DISCONNECTED';
    }
}

function calculateTripData() {
    if (!lastPosition) {
        lastPosition = { lat: gpsData.latitude, lng: gpsData.longitude };
        tripStartTime = new Date();
        return;
    }

    // Calculate distance from last position
    const distance = calculateDistance(
        lastPosition.lat, lastPosition.lng,
        gpsData.latitude, gpsData.longitude
    );

    totalDistance += distance;
    lastPosition = { lat: gpsData.latitude, lng: gpsData.longitude };

    // Update trip display
    document.getElementById('tripDistance').textContent = totalDistance.toFixed(2) + ' km';

    if (tripStartTime) {
        const elapsed = new Date() - tripStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('tripTime').textContent =
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
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
// Simulation Mode
// ============================================
function startSimulation() {
    simulationMode = true;
    let simLat = CONFIG.DEFAULT_LAT;
    let simLon = CONFIG.DEFAULT_LON;
    let simHeading = 0;
    let simSpeed = 0;

    simulationInterval = setInterval(() => {
        // Simulate movement in a pattern
        simHeading = (simHeading + (Math.random() - 0.5) * 10) % 360;
        simSpeed = Math.min(Math.abs(Math.sin(Date.now() / 5000)) * CONFIG.SIMULATION_MAX_SPEED, CONFIG.SIMULATION_MAX_SPEED);

        simLat += Math.cos(toRad(simHeading)) * CONFIG.SIMULATION_SPEED;
        simLon += Math.sin(toRad(simHeading)) * CONFIG.SIMULATION_SPEED;

        const simData = {
            latitude: simLat,
            longitude: simLon,
            altitude: 216 + Math.sin(Date.now() / 2000) * 10,
            speed: simSpeed,
            heading: simHeading,
            satellites: 8 + Math.floor(Math.random() * 4),
            accuracy: 1 + Math.random() * 2,
            status: 'simulated'
        };

        updateGPSData(simData);
    }, CONFIG.GPS_POLL_INTERVAL);

    addAlert('Simulation mode started', 'info');
}

function stopSimulation() {
    simulationMode = false;
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
    addAlert('Simulation mode stopped', 'info');
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

    // Keep only last 5 alerts
    while (alertList.children.length > 5) {
        alertList.removeChild(alertList.lastChild);
    }
}

// ============================================
// Map Controls
// ============================================
function setupMapControls() {
    document.getElementById('btnZoomIn').addEventListener('click', () => {
        map.zoomTo(map.getZoom() + 1);
    });

    document.getElementById('btnZoomOut').addEventListener('click', () => {
        map.zoomTo(map.getZoom() - 1);
    });

    document.getElementById('btnCenter').addEventListener('click', () => {
        map.easeTo({
            center: [gpsData.longitude, gpsData.latitude],
            bearing: gpsData.heading,
            duration: 1000
        });
    });

    document.getElementById('btnFullscreen').addEventListener('click', () => {
        const container = document.querySelector('.map-container');
        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                console.error('Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
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
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('ADAS Dashboard initializing...');

    // Initialize map
    initMap();

    // Setup controls
    setupMapControls();
    setupSimulationToggle();

    // Start GPS polling (500ms for live updates)
    setInterval(fetchGPSData, CONFIG.GPS_POLL_INTERVAL);

    // Initial fetch
    fetchGPSData();

    console.log('ADAS Dashboard ready');
});
