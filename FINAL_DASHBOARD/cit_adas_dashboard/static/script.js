/**
 * AUTOMOTIVE NAVIGATION DASHBOARD
 * Real Map Data - OpenStreetMap/Mapbox Style
 * CIT Coimbatore → Coimbatore Airport
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_BASE: '',
    POLL_INTERVAL: 1000,
    
    // CIT Coimbatore - Main Gate
    START_LAT: 11.0286,
    START_LON: 77.0269,
    
    // Destination: Coimbatore Airport
    END_LAT: 11.0300,
    END_LON: 77.0440,
    
    // Camera - Dashboard Navigation Style
    CAMERA_DISTANCE: 30,
    CAMERA_HEIGHT: 18,
    CAMERA_TILT: 60,
    
    // Map Settings
    DEFAULT_ZOOM: 17.5,
    MIN_ZOOM: 15,
    MAX_ZOOM: 19,
    
    // Route Style - Wide glowing path
    ROUTE_COLOR: '#00ff88',
    ROUTE_GLOW_COLOR: '#00ffff',
    ROUTE_WIDTH: 10,
    ROUTE_GLOW_WIDTH: 25,
    
    // Vehicle
    ARROW_COLOR: '#00ff88'
};

// ============================================
// Global State
// ============================================
let map = null;
let vehicleArrow = null;
let routeLine = null;
let routeGlowLine = null;
let routeGlowOuter = null;

let gpsData = {
    latitude: CONFIG.START_LAT,
    longitude: CONFIG.START_LON,
    altitude: 420,
    speed: 0,
    heading: 0,
    satellites: 0,
    accuracy: 0,
    status: 'initializing'
};

// Trip Data
let tripStartTime = null;
let totalDistance = 0;
let maxSpeed = 0;
let speedHistory = [];
let lastPosition = null;
let routeCoordinates = [];
let isSimulationRunning = false;
let routeCompleted = false;

// ============================================
// Map Initialization - Real Map Tiles
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
                    attribution: '© OpenStreetMap'
                }
            },
            layers: [
                {
                    id: 'background',
                    type: 'background',
                    paint: { 'background-color': '#0a0a0f' }
                },
                {
                    id: 'osm-tiles',
                    type: 'raster',
                    source: 'osm',
                    minzoom: 0,
                    maxzoom: 19,
                    paint: {
                        'raster-opacity': 0.65,
                        'raster-brightness-min': 0.15,
                        'raster-brightness-max': 0.5,
                        'raster-contrast': 0.3,
                        'raster-saturation': -0.4
                    }
                }
            ]
        },
        center: [CONFIG.START_LON, CONFIG.START_LAT],
        zoom: CONFIG.DEFAULT_ZOOM,
        pitch: CONFIG.CAMERA_TILT,
        bearing: 0,
        antialias: true,
        preserveDrawingBuffer: true,
        maxBounds: [
            [CONFIG.START_LON - 0.03, CONFIG.START_LAT - 0.03],
            [CONFIG.START_LON + 0.03, CONFIG.START_LAT + 0.03]
        ]
    });

    map.on('load', () => {
        console.log('Navigation map loaded with real tiles');
        addVehicleArrow3D();
        addGlowingRoute();
        addDestinationMarker();
        hideLoadingScreen();
        addNotification('System Ready - Click START', 'info');
    });

    map.on('error', (e) => {
        console.error('Map error:', e);
        hideLoadingScreen();
    });
}

function addDestinationMarker() {
    // Add airport destination marker
    const destEl = document.createElement('div');
    destEl.className = 'destination-marker';
    destEl.innerHTML = `
        <svg viewBox="0 0 40 40" fill="none">
            <circle cx="20" cy="20" r="18" fill="#ff8800" opacity="0.3"/>
            <circle cx="20" cy="20" r="12" fill="#ff8800" opacity="0.5"/>
            <circle cx="20" cy="20" r="6" fill="#ff8800"/>
            <text x="20" y="24" text-anchor="middle" fill="white" font-size="8" font-weight="bold">✈</text>
        </svg>
    `;
    
    new maplibregl.Marker({ element: destEl, anchor: 'center' })
        .setLngLat([CONFIG.END_LON, CONFIG.END_LAT])
        .addTo(map);
}

function addVehicleArrow3D() {
    const el = document.createElement('div');
    el.className = 'vehicle-arrow-3d';
    el.innerHTML = `
        <svg viewBox="0 0 100 100" fill="none">
            <defs>
                <filter id="arrowGlow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
                <linearGradient id="arrowGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#00ff88;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00ffff;stop-opacity:1" />
                </linearGradient>
                <radialGradient id="pulseGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" style="stop-color:#00ff88;stop-opacity:0.7" />
                    <stop offset="100%" style="stop-color:#00ff88;stop-opacity:0" />
                </radialGradient>
            </defs>
            
            <!-- Pulse ring -->
            <circle cx="50" cy="50" r="42" fill="url(#pulseGrad)" opacity="0.5">
                <animate attributeName="r" values="38;46;38" dur="2.5s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.6;0.3;0.6" dur="2.5s" repeatCount="indefinite" />
            </circle>
            
            <!-- Shadow -->
            <ellipse cx="50" cy="55" rx="25" ry="15" fill="#000000" opacity="0.5"/>
            
            <!-- 3D Arrow -->
            <path d="M50 18 L59 42 L55 42 L55 68 L45 68 L45 42 L41 42 Z" 
                  fill="url(#arrowGrad)" 
                  filter="url(#arrowGlow)"
                  stroke="#00ff88" 
                  stroke-width="2"
                  stroke-linejoin="round"/>
            
            <!-- Center glow -->
            <circle cx="50" cy="50" r="7" fill="#00ff88" filter="url(#arrowGlow)"/>
            
            <!-- Direction lines -->
            <line x1="50" y1="12" x2="50" y2="6" stroke="#00ff88" stroke-width="2.5" opacity="0.7"/>
            <line x1="50" y1="82" x2="50" y2="88" stroke="#00ff88" stroke-width="2.5" opacity="0.7"/>
        </svg>
    `;
    
    vehicleArrow = new maplibregl.Marker({ 
        element: el, 
        anchor: 'center',
        rotationAlignment: 'map'
    })
    .setLngLat([CONFIG.START_LON, CONFIG.START_LAT])
    .addTo(map);
    
    console.log('3D vehicle arrow added');
}

function addGlowingRoute() {
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
    
    // Outer glow (wide, soft)
    map.addLayer({
        id: 'route-glow-outer',
        type: 'line',
        source: 'route',
        layout: {
            'line-join': 'round',
            'line-cap': 'round'
        },
        paint: {
            'line-color': CONFIG.ROUTE_GLOW_COLOR,
            'line-width': CONFIG.ROUTE_GLOW_WIDTH,
            'line-opacity': 0.3,
            'line-blur': 10
        }
    });
    
    // Inner glow
    map.addLayer({
        id: 'route-glow-inner',
        type: 'line',
        source: 'route',
        layout: {
            'line-join': 'round',
            'line-cap': 'round'
        },
        paint: {
            'line-color': CONFIG.ROUTE_COLOR,
            'line-width': CONFIG.ROUTE_WIDTH + 8,
            'line-opacity': 0.5,
            'line-blur': 5
        }
    });
    
    // Main route (solid)
    map.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        layout: {
            'line-join': 'round',
            'line-cap': 'round'
        },
        paint: {
            'line-color': CONFIG.ROUTE_COLOR,
            'line-width': CONFIG.ROUTE_WIDTH,
            'line-opacity': 0.95
        }
    });
    
    console.log('Wide glowing route added');
}

function updateCameraFollow(lng, lat, heading) {
    if (!map) return;
    
    map.easeTo({
        center: [lng, lat],
        bearing: heading,
        pitch: CONFIG.CAMERA_TILT,
        zoom: CONFIG.DEFAULT_ZOOM,
        duration: 1000,
        easing: (t) => t * (2 - t)
    });
}

function updateVehiclePosition(lng, lat, heading) {
    if (!vehicleArrow) return;
    
    vehicleArrow.setLngLat([lng, lat]);
    vehicleArrow.setRotation(heading);
    
    updateCameraFollow(lng, lat, heading);
    addToRoute(lng, lat);
}

function addToRoute(lng, lat) {
    routeCoordinates.push([lng, lat]);
    
    if (routeCoordinates.length > 500) {
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
    updateNavigation();
    updateSystemStatus();
    calculateTripData();
    updateSpeedGraph();
}

function updateDashboard() {
    const speed = gpsData.speed || 0;
    document.getElementById('speedValue').textContent = speed.toFixed(0);
    document.getElementById('headingValue').textContent = gpsData.heading.toFixed(0);
    
    const cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const cardinalIndex = Math.round(gpsData.heading / 45) % 8;
    document.getElementById('cardinalDir').textContent = cardinals[cardinalIndex];
    
    const needle = document.getElementById('compassRing').querySelector('.compass-needle');
    if (needle) {
        needle.style.transform = `translate(-50%, 0) rotate(${gpsData.heading}deg)`;
    }
    
    document.getElementById('latValue').textContent = gpsData.latitude.toFixed(6);
    document.getElementById('lonValue').textContent = gpsData.longitude.toFixed(6);
    document.getElementById('altValue').textContent = gpsData.altitude.toFixed(0) + ' M';
    document.getElementById('satelliteCount').textContent = gpsData.satellites;
    document.getElementById('accuracyValue').textContent = gpsData.accuracy.toFixed(1);
    
    updateSignalBars(gpsData.satellites);
    
    if (speed > maxSpeed) {
        maxSpeed = speed;
        document.getElementById('maxSpeed').textContent = maxSpeed.toFixed(0) + ' KM/H';
    }
    
    updateTime();
}

function updateSignalBars(count) {
    const bars = document.querySelectorAll('.signal-bars .bar');
    const activeCount = Math.min(Math.ceil(count / 3), 4);
    bars.forEach((bar, index) => {
        bar.classList.toggle('active', index < activeCount);
    });
}

function updateTime() {
    const now = new Date();
    document.getElementById('currentTime').textContent = 
        now.toLocaleTimeString('en-US', { hour12: false });
    document.getElementById('currentDate').textContent = 
        now.toISOString().split('T')[0];
}

function updateNavigation() {
    updateVehiclePosition(gpsData.longitude, gpsData.latitude, gpsData.heading || 0);
}

function updateSystemStatus() {
    const statusEl = document.getElementById('systemStatus');
    const label = statusEl.querySelector('.status-label');
    
    if (gpsData.status === 'completed') {
        statusEl.classList.add('completed');
        statusEl.classList.remove('active', 'error');
        label.textContent = 'ARRIVED';
        routeCompleted = true;
        isSimulationRunning = false;
        updateControlButtons();
        addNotification('Arrived at Coimbatore Airport!', 'completed');
    } else if (gpsData.status === 'simulated') {
        statusEl.classList.add('active');
        statusEl.classList.remove('error', 'completed');
        label.textContent = 'NAVIGATING';
    } else if (gpsData.status === 'serial') {
        statusEl.classList.add('active');
        label.textContent = 'GPS';
    } else if (gpsData.status?.includes('error')) {
        statusEl.classList.add('error');
        label.textContent = 'ERROR';
    } else if (gpsData.status === 'stopped') {
        statusEl.classList.remove('active', 'completed', 'error');
        label.textContent = 'PAUSED';
    } else if (gpsData.status === 'reset') {
        statusEl.classList.remove('active', 'completed', 'error');
        label.textContent = 'READY';
    } else {
        statusEl.classList.remove('active', 'completed', 'error');
        label.textContent = gpsData.status?.toUpperCase() || 'STANDBY';
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
    
    document.getElementById('tripDistance').textContent = totalDistance.toFixed(2) + ' KM';
    
    if (tripStartTime) {
        const elapsed = new Date() - tripStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('elapsedTime').textContent = 
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        
        const hoursElapsed = elapsed / 3600000;
        const avgSpeed = hoursElapsed > 0 ? totalDistance / hoursElapsed : 0;
        document.getElementById('avgSpeed').textContent = avgSpeed.toFixed(0) + ' KM/H';
    }
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

function toRad(deg) {
    return deg * (Math.PI / 180);
}

function updateSpeedGraph() {
    const canvas = document.getElementById('speedGraph');
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    speedHistory.push(gpsData.speed || 0);
    if (speedHistory.length > 30) speedHistory.shift();
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.15)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = (canvas.height / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    const maxSpeedGraph = 120;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    speedHistory.forEach((speed, index) => {
        const x = (index / (speedHistory.length - 1)) * canvas.width;
        const y = canvas.height - (speed / maxSpeedGraph) * canvas.height;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, 'rgba(0, 255, 136, 0.5)');
    gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
    ctx.lineTo(canvas.width, canvas.height);
    ctx.lineTo(0, canvas.height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
}

async function startSimulation() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/gps/start`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            isSimulationRunning = true;
            routeCompleted = false;
            updateControlButtons();
            addNotification('Starting Navigation to Airport', 'success');
            
            tripStartTime = null;
            totalDistance = 0;
            maxSpeed = 0;
            speedHistory = [];
            routeCoordinates = [];
        }
    } catch (error) {
        addNotification('Failed to start', 'error');
    }
}

async function stopSimulation() {
    await fetch(`${CONFIG.API_BASE}/gps/stop`, { method: 'POST' });
    isSimulationRunning = false;
    updateControlButtons();
    addNotification('Navigation Paused', 'warning');
}

async function resetGPS() {
    await fetch(`${CONFIG.API_BASE}/gps/reset`, { method: 'POST' });
    
    isSimulationRunning = false;
    routeCompleted = false;
    routeCoordinates = [];
    
    if (map.getSource('route')) {
        map.getSource('route').setData({
            type: 'Feature',
            properties: {},
            geometry: { type: 'LineString', coordinates: [] }
        });
    }
    
    tripStartTime = null;
    totalDistance = 0;
    maxSpeed = 0;
    speedHistory = [];
    
    updateControlButtons();
    addNotification('Reset to CIT Gate', 'info');
}

function updateControlButtons() {
    document.getElementById('btnStart').disabled = isSimulationRunning || routeCompleted;
    document.getElementById('btnStop').disabled = !isSimulationRunning;
}

function setupMapControls() {
    document.getElementById('btnTiltUp').addEventListener('click', () => {
        map.easeTo({ pitch: Math.min(map.getPitch() + 10, 75), duration: 500 });
    });
    
    document.getElementById('btnTiltDown').addEventListener('click', () => {
        map.easeTo({ pitch: Math.max(map.getPitch() - 10, 0), duration: 500 });
    });
    
    document.getElementById('btnZoomIn').addEventListener('click', () => {
        map.zoomTo(map.getZoom() + 0.5);
    });
    
    document.getElementById('btnZoomOut').addEventListener('click', () => {
        map.zoomTo(map.getZoom() - 0.5);
    });
    
    document.getElementById('btnCenter').addEventListener('click', () => {
        updateCameraFollow(gpsData.longitude, gpsData.latitude, gpsData.heading);
    });
    
    document.getElementById('btn3D').addEventListener('click', () => {
        const layer = map.getLayer('3d-buildings');
        if (layer) {
            const visibility = map.getLayoutProperty('3d-buildings', 'visibility');
            map.setLayoutProperty('3d-buildings', 'visibility', 
                visibility === 'visible' ? 'none' : 'visible');
        }
    });
}

function setupButtons() {
    document.getElementById('btnStart').addEventListener('click', startSimulation);
    document.getElementById('btnStop').addEventListener('click', stopSimulation);
    document.getElementById('btnReset').addEventListener('click', resetGPS);
}

function addNotification(message, type = 'info') {
    const container = document.getElementById('notifications');
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.innerHTML = `
        <span class="notif-icon">${type === 'info' ? 'ℹ' : type === 'success' ? '✓' : type === 'warning' ? '⚠' : '✓'}</span>
        <span class="notif-text">${message}</span>
    `;
    container.appendChild(notif);
    setTimeout(() => {
        notif.style.opacity = '0';
        notif.style.transform = 'translateX(20px)';
        setTimeout(() => notif.remove(), 300);
    }, 5000);
}

function hideLoadingScreen() {
    const screen = document.getElementById('loadingScreen');
    screen.classList.add('hidden');
    setTimeout(() => { screen.style.display = 'none'; }, 800);
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('Automotive Navigation Initializing...');
    console.log(`Route: CIT → Airport (${CONFIG.START_LAT}, ${CONFIG.START_LON})`);
    
    initMap();
    setupButtons();
    setupMapControls();
    
    setInterval(fetchGPSData, CONFIG.POLL_INTERVAL);
    fetchGPSData();
    
    console.log('Navigation System Ready');
});
