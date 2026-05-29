/* ==========================================================================
   AMBULANCE MODE - 3D VISUALIZATION WITH THREE.JS
   Emergency Response with Priority Signal Broadcasting
   ========================================================================== */

class AmbulanceMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.ambulance = null;
        this.emergencyActive = false;
        this.clock = new THREE.Clock();
        this.hospitalPosition = new THREE.Vector3(50, 0, 50);
        this.trafficLights = [];
        this.clearedVehicles = 0;
        this.signalsSent = 0;

        this.init();
        this.animate();
        this.setupEventListeners();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f2027);
        this.scene.fog = new THREE.FogExp2(0x0f2027, 0.002);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 15, 25);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Lighting
        this.setupLighting();

        // Environment
        this.createEnvironment();

        // Ambulance Vehicle
        this.createAmbulance();

        // Traffic Lights
        this.createTrafficLights();

        // Other Vehicles
        this.createTraffic();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(50, 50, 50);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        // Emergency lights will be added to ambulance
    }

    createEnvironment() {
        // Ground
        const groundGeometry = new THREE.PlaneGeometry(200, 200, 50, 50);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x0a151a,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid
        const gridHelper = new THREE.GridHelper(200, 50, 0x00d2ff, 0x006688);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        // Roads with lanes
        this.createRoads();

        // Buildings
        this.createBuildings();

        // Hospital building (destination)
        this.createHospital();
    }

    createRoads() {
        // Main road (horizontal)
        const road1Geometry = new THREE.PlaneGeometry(15, 200);
        const road1Material = new THREE.MeshStandardMaterial({ color: 0x1a1a2e, roughness: 0.9 });
        const road1 = new THREE.Mesh(road1Geometry, road1Material);
        road1.rotation.x = -Math.PI / 2;
        road1.position.y = 0.02;
        this.scene.add(road1);

        // Main road (vertical)
        const road2Geometry = new THREE.PlaneGeometry(200, 15);
        const road2 = new THREE.Mesh(road2Geometry, road1Material);
        road2.rotation.x = -Math.PI / 2;
        road2.position.y = 0.02;
        this.scene.add(road2);

        // Road markings
        this.createRoadMarkings();
    }

    createRoadMarkings() {
        const markingMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        // Dashed center lines
        for (let i = -95; i < 95; i += 8) {
            // Horizontal road
            const mark1 = new THREE.Mesh(new THREE.PlaneGeometry(0.3, 2), markingMaterial);
            mark1.rotation.x = -Math.PI / 2;
            mark1.position.set(0, 0.03, i);
            this.scene.add(mark1);

            // Vertical road
            const mark2 = new THREE.Mesh(new THREE.PlaneGeometry(2, 0.3), markingMaterial);
            mark2.rotation.x = -Math.PI / 2;
            mark2.position.set(i, 0.03, 0);
            this.scene.add(mark2);
        }
    }

    createBuildings() {
        const colors = [0x1a2a3a, 0x2a3a4a, 0x1a3a4a];
        
        for (let i = 0; i < 15; i++) {
            const height = Math.random() * 12 + 6;
            const width = Math.random() * 6 + 4;
            const depth = Math.random() * 6 + 4;

            const geometry = new THREE.BoxGeometry(width, height, depth);
            const material = new THREE.MeshStandardMaterial({
                color: colors[Math.floor(Math.random() * colors.length)],
                roughness: 0.3,
                metalness: 0.7,
                emissive: 0x00d2ff,
                emissiveIntensity: 0.1
            });

            const building = new THREE.Mesh(geometry, material);
            
            const angle = Math.random() * Math.PI * 2;
            const radius = 40 + Math.random() * 40;
            building.position.set(
                Math.cos(angle) * radius,
                height / 2,
                Math.sin(angle) * radius
            );

            building.castShadow = true;
            building.receiveShadow = true;
            this.scene.add(building);
        }
    }

    createHospital() {
        const hospitalGroup = new THREE.Group();

        // Main building
        const mainGeometry = new THREE.BoxGeometry(20, 8, 15);
        const mainMaterial = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            roughness: 0.5,
            emissive: 0x00d2ff,
            emissiveIntensity: 0.2
        });
        const main = new THREE.Mesh(mainGeometry, mainMaterial);
        main.position.y = 4;
        main.castShadow = true;
        hospitalGroup.add(main);

        // Red cross sign
        const crossGeometry = new THREE.BoxGeometry(2, 3, 0.3);
        const crossMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        
        const crossV = new THREE.Mesh(crossGeometry, crossMaterial);
        crossV.position.set(0, 6, 7.6);
        hospitalGroup.add(crossV);

        const crossH = new THREE.Mesh(new THREE.BoxGeometry(3, 2, 0.3), crossMaterial);
        crossH.position.set(0, 6, 7.6);
        hospitalGroup.add(crossH);

        // Helipad
        const padGeometry = new THREE.CylinderGeometry(5, 5, 0.3, 32);
        const padMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });
        const pad = new THREE.Mesh(padGeometry, padMaterial);
        pad.position.set(0, 8.15, -5);
        hospitalGroup.add(pad);

        // H marking
        const hGeometry = new THREE.BoxGeometry(0.5, 0.1, 2);
        const hMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const h1 = new THREE.Mesh(hGeometry, hMaterial);
        h1.position.set(-1, 8.3, -5);
        hospitalGroup.add(h1);
        
        const h2 = new THREE.Mesh(hGeometry, hMaterial);
        h2.position.set(1, 8.3, -5);
        hospitalGroup.add(h2);
        
        const h3 = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.1, 0.5), hMaterial);
        h3.position.set(0, 8.3, -5);
        hospitalGroup.add(h3);

        hospitalGroup.position.set(50, 0, 50);
        this.scene.add(hospitalGroup);
        this.hospitalPosition.set(50, 4, 50);
    }

    createAmbulance() {
        this.ambulance = new THREE.Group();

        // Main body
        const bodyGeometry = new THREE.BoxGeometry(2.2, 1.2, 4);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            roughness: 0.3,
            emissive: 0xffffff,
            emissiveIntensity: 0.1
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 1;
        body.castShadow = true;
        this.ambulance.add(body);

        // Cab
        const cabGeometry = new THREE.BoxGeometry(2, 0.8, 1.5);
        const cab = new THREE.Mesh(cabGeometry, bodyMaterial);
        cab.position.set(0, 1, 1.25);
        this.ambulance.add(cab);

        // Red cross markings
        const crossGeometry = new THREE.PlaneGeometry(0.8, 1);
        const crossMaterial = new THREE.MeshBasicMaterial({ 
            color: 0xff0000,
            side: THREE.DoubleSide
        });
        
        const crossSide = new THREE.Mesh(crossGeometry, crossMaterial);
        crossSide.position.set(1.11, 1, 0);
        crossSide.rotation.y = -Math.PI / 2;
        this.ambulance.add(crossSide);

        const crossBack = new THREE.Mesh(crossGeometry, crossMaterial);
        crossBack.position.set(0, 1, -2.01);
        this.ambulance.add(crossBack);

        // Emergency light bar
        const barGeometry = new THREE.BoxGeometry(1.8, 0.2, 0.4);
        const barMaterial = new THREE.MeshStandardMaterial({
            color: 0x333333,
            roughness: 0.5
        });
        const lightBar = new THREE.Mesh(barGeometry, barMaterial);
        lightBar.position.set(0, 1.7, 0);
        this.ambulance.add(lightBar);

        // Emergency lights (blue and red)
        this.createEmergencyLights(lightBar);

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e });

        [[-1.1, 0.4, 1.2], [1.1, 0.4, 1.2], [-1.1, 0.4, -1.2], [1.1, 0.4, -1.2]].forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            this.ambulance.add(wheel);
        });

        this.ambulance.position.set(0, 0, 0);
        this.scene.add(this.ambulance);
    }

    createEmergencyLights(lightBar) {
        // Blue lights
        const blueLightLeft = new THREE.PointLight(0x0000ff, 2, 10);
        blueLightLeft.position.set(-0.6, 1.9, 0);
        this.ambulance.add(blueLightLeft);

        const blueLightRight = new THREE.PointLight(0x0000ff, 2, 10);
        blueLightRight.position.set(0.6, 1.9, 0);
        this.ambulance.add(blueLightRight);

        // Red lights
        const redLightLeft = new THREE.PointLight(0xff0000, 2, 10);
        redLightLeft.position.set(-0.3, 1.9, 0);
        this.ambulance.add(redLightLeft);

        const redLightRight = new THREE.PointLight(0xff0000, 2, 10);
        redLightRight.position.set(0.3, 1.9, 0);
        this.ambulance.add(redLightRight);

        // Store lights for animation
        this.emergencyLights = [blueLightLeft, blueLightRight, redLightLeft, redLightRight];
        
        // Light meshes for visual effect
        const lightMeshGeo = new THREE.SphereGeometry(0.15, 8, 8);
        const blueMat = new THREE.MeshBasicMaterial({ color: 0x0000ff });
        const redMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });

        const blueMesh1 = new THREE.Mesh(lightMeshGeo, blueMat);
        blueMesh1.position.set(-0.6, 1.9, 0);
        this.ambulance.add(blueMesh1);

        const blueMesh2 = new THREE.Mesh(lightMeshGeo, blueMat);
        blueMesh2.position.set(0.6, 1.9, 0);
        this.ambulance.add(blueMesh2);

        const redMesh1 = new THREE.Mesh(lightMeshGeo, redMat);
        redMesh1.position.set(-0.3, 1.9, 0);
        this.ambulance.add(redMesh1);

        const redMesh2 = new THREE.Mesh(lightMeshGeo, redMat);
        redMesh2.position.set(0.3, 1.9, 0);
        this.ambulance.add(redMesh2);

        this.lightMeshes = [blueMesh1, blueMesh2, redMesh1, redMesh2];
    }

    createTrafficLights() {
        const positions = [
            { x: 10, z: 5 },
            { x: -10, z: 5 },
            { x: 10, z: -5 },
            { x: -10, z: -5 },
            { x: 5, z: 10 },
            { x: -5, z: 10 }
        ];

        positions.forEach(pos => {
            const lightGroup = new THREE.Group();

            // Pole
            const poleGeometry = new THREE.CylinderGeometry(0.1, 0.1, 4, 8);
            const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });
            const pole = new THREE.Mesh(poleGeometry, poleMaterial);
            pole.position.y = 2;
            lightGroup.add(pole);

            // Light box
            const boxGeometry = new THREE.BoxGeometry(0.6, 1.5, 0.4);
            const box = new THREE.Mesh(boxGeometry, poleMaterial);
            box.position.y = 3.75;
            lightGroup.add(box);

            // Lights
            const colors = [0xff0000, 0xffaa00, 0x00ff00];
            const positions_y = [4.2, 3.75, 3.3];
            
            colors.forEach((color, i) => {
                const lightGeo = new THREE.CircleGeometry(0.15, 16);
                const lightMat = new THREE.MeshBasicMaterial({ 
                    color: i === 2 ? 0x00ff00 : 0x333333,
                    transparent: true,
                    opacity: i === 2 ? 0.8 : 0.3
                });
                const light = new THREE.Mesh(lightGeo, lightMat);
                light.position.set(0, positions_y[i], 3.95);
                lightGroup.add(light);
            });

            lightGroup.position.set(pos.x, 0, pos.z);
            this.scene.add(lightGroup);
            this.trafficLights.push(lightGroup);
        });
    }

    createTraffic() {
        for (let i = 0; i < 8; i++) {
            const car = this.createCar();
            
            const angle = Math.random() * Math.PI * 2;
            const radius = 20 + Math.random() * 40;
            car.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
            car.rotation.y = Math.random() * Math.PI * 2;
            
            car.userData.originalColor = car.children[0].material.color.getHex();
            this.scene.add(car);
            
            // Store for animation
            if (!this.trafficCars) this.trafficCars = [];
            this.trafficCars.push(car);
        }
    }

    createCar() {
        const carGroup = new THREE.Group();

        const bodyGeometry = new THREE.BoxGeometry(2.5, 1.2, 4.5);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: Math.random() * 0xffffff,
            roughness: 0.3,
            metalness: 0.7
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        carGroup.add(body);

        const roofGeometry = new THREE.BoxGeometry(2, 0.8, 2.5);
        const roof = new THREE.Mesh(roofGeometry, bodyMaterial);
        roof.position.y = 1.8;
        roof.position.z = -0.5;
        carGroup.add(roof);

        return carGroup;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        // Animate emergency lights
        if (this.emergencyActive && this.emergencyLights) {
            this.emergencyLights.forEach((light, i) => {
                light.intensity = 2 + Math.sin(time * 10 + i) * 2;
            });

            this.lightMeshes.forEach((mesh, i) => {
                mesh.material.opacity = 0.5 + Math.sin(time * 10 + i) * 0.5;
            });
        }

        // Move ambulance
        if (this.emergencyActive && this.ambulance) {
            this.moveAmbulance(delta);
        }

        // Turn traffic lights green for ambulance
        if (this.emergencyActive) {
            this.updateTrafficLights(time);
        }

        // Clear vehicles from path
        if (this.emergencyActive && this.trafficCars) {
            this.clearTraffic(delta);
        }

        // Broadcast signal visualization
        if (this.emergencyActive) {
            this.broadcastSignal(time);
        }

        // Update telemetry
        this.updateTelemetry();

        this.renderer.render(this.scene, this.camera);
    }

    moveAmbulance(delta) {
        const direction = new THREE.Vector3()
            .subVectors(this.hospitalPosition, this.ambulance.position)
            .normalize();

        const speed = 8 * delta;
        this.ambulance.position.add(direction.multiplyScalar(speed));
        this.ambulance.lookAt(this.hospitalPosition);

        // Update camera to follow
        const cameraOffset = new THREE.Vector3(0, 12, 16);
        const targetPos = this.ambulance.position.clone().add(cameraOffset);
        this.camera.position.lerp(targetPos, 0.05);
        this.camera.lookAt(this.ambulance.position);

        // Check if reached hospital
        if (this.ambulance.position.distanceTo(this.hospitalPosition) < 5) {
            this.emergencyActive = false;
            this.showToast('🏥 Arrived at hospital!', 'success');
            document.getElementById('emergencyLevel').textContent = 'STANDBY';
            document.getElementById('emergencyLevel').style.color = '#00ff88';
        }
    }

    updateTrafficLights(time) {
        this.trafficLights.forEach((lightGroup, index) => {
            const lights = lightGroup.children.slice(2); // Skip pole and box
            lights.forEach((light, i) => {
                const opacity = i === 2 ? 0.8 : 0.3; // Green light on
                light.material.opacity = opacity;
                light.material.color.setHex(i === 2 ? 0x00ff00 : 0x333333);
            });
        });
    }

    clearTraffic(delta) {
        if (!this.trafficCars) return;

        this.trafficCars.forEach(car => {
            const dist = this.ambulance.position.distanceTo(car.position);
            
            if (dist < 20) {
                // Move car aside
                const direction = new THREE.Vector3()
                    .subVectors(car.position, this.ambulance.position)
                    .normalize();
                direction.y = 0;
                direction.normalize();
                
                car.position.add(direction.multiplyScalar(5 * delta));
                car.rotation.y = Math.atan2(direction.x, direction.z);

                // Change color to indicate awareness
                if (car.children[0]) {
                    car.children[0].material.color.setHex(0xffaa00);
                }

                this.clearedVehicles++;
            }
        });
    }

    broadcastSignal(time) {
        // Update signal counter
        if (Math.floor(time) > this.signalsSent) {
            this.signalsSent = Math.floor(time);
        }
    }

    updateTelemetry() {
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const gpsEl = document.getElementById('gpsSignal');
        const ultraEl = document.getElementById('ultrasonicValue');
        const distanceEl = document.getElementById('distanceToGoal');
        const etaEl = document.getElementById('eta');
        const vehiclesEl = document.querySelector('[data-stat="vehicles"]');

        if (speedEl) speedEl.textContent = this.emergencyActive ? '8.5' : '0.0';
        if (batteryEl) batteryEl.textContent = (92 - this.clock.getElapsedTime() * 0.1).toFixed(0);
        if (gpsEl) gpsEl.textContent = 12 + Math.floor(Math.random() * 4);
        if (ultraEl) ultraEl.textContent = (30 + Math.random() * 20).toFixed(1);

        if (this.ambulance) {
            const dist = this.ambulance.position.distanceTo(this.hospitalPosition);
            if (distanceEl) distanceEl.textContent = `${(dist * 10).toFixed(1)} m`;
            if (etaEl) etaEl.textContent = `${(dist / 8 * 60).toFixed(0)}s`;
        }
    }

    setupEventListeners() {
        document.getElementById('btnStartAmbulance')?.addEventListener('click', () => {
            this.startEmergency();
        });

        document.getElementById('btnPause')?.addEventListener('click', () => {
            this.togglePause();
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.emergencyStop();
        });

        document.getElementById('btnCompleteMission')?.addEventListener('click', () => {
            this.completeMission();
        });
    }

    startEmergency() {
        this.emergencyActive = true;
        this.clearedVehicles = 0;
        this.signalsSent = 0;
        this.clock.start();
        this.showToast('🚨 Emergency response initiated!', 'success');
        this.showToast('📡 Broadcasting priority signal...', 'info');
        
        document.getElementById('emergencyLevel').textContent = 'CRITICAL';
        document.getElementById('emergencyLevel').style.color = '#ff4444';
        document.getElementById('broadcastStatus').textContent = 'Active';
    }

    togglePause() {
        this.emergencyActive = !this.emergencyActive;
        this.showToast(this.emergencyActive ? '▶️ Emergency resumed' : '⏸️ Holding position', 'info');
    }

    emergencyStop() {
        this.emergencyActive = false;
        this.showToast('🛑 EMERGENCY STOP', 'danger');
    }

    completeMission() {
        this.emergencyActive = false;
        this.showToast('✅ Emergency response complete!', 'success');
        
        const modal = document.getElementById('deliveryModal');
        if (modal) modal.classList.add('active');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `
            <span class="toast-icon">${type === 'success' ? '✅' : type === 'danger' ? '⚠️' : 'ℹ️'}</span>
            <span class="toast-message">${message}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    onResize() {
        if (!this.container) return;

        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

// Global function
window.closeModal = function() {
    const modal = document.getElementById('deliveryModal');
    if (modal) modal.classList.remove('active');
    
    if (window.ambulanceMode3D) {
        window.ambulanceMode3D.ambulance.position.set(0, 0, 0);
        window.ambulanceMode3D.emergencyActive = false;
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.ambulanceMode3D = new AmbulanceMode3D();
});
