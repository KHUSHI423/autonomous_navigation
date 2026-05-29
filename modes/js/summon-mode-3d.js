/* ==========================================================================
   SUMMON MODE - 3D VISUALIZATION
   Vehicle comes to your exact location autonomously
   ========================================================================== */

class SummonMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.vehicle = null;
        this.userLocation = null;
        this.isSummoning = false;
        this.clock = new THREE.Clock();
        this.startPosition = new THREE.Vector3(0, 0, 0);
        this.targetPosition = new THREE.Vector3(40, 0, 40);

        this.init();
        this.animate();
        this.setupEventListeners();
    }

    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f2027);
        this.scene.fog = new THREE.FogExp2(0x0f2027, 0.0015);

        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(-20, 20, -20);
        this.camera.lookAt(20, 0, 20);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        this.setupLighting();
        this.createEnvironment();
        this.createVehicle();
        this.createUserLocation();
        this.createPathVisualization();

        window.addEventListener('resize', () => this.onResize());
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
        dirLight.position.set(50, 50, 50);
        dirLight.castShadow = true;
        this.scene.add(dirLight);
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

        // Parking lot markings
        this.createParkingLot();

        // Buildings
        this.createBuildings();
    }

    createParkingLot() {
        const lotGeometry = new THREE.PlaneGeometry(30, 40);
        const lotMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a1a2e,
            roughness: 0.9
        });
        const lot = new THREE.Mesh(lotGeometry, lotMaterial);
        lot.rotation.x = -Math.PI / 2;
        lot.position.y = 0.02;
        lot.position.set(0, 0, 0);
        this.scene.add(lot);

        // Parking space lines
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 });
        for (let i = -12; i <= 12; i += 4) {
            const points = [
                new THREE.Vector3(i, 0.03, -18),
                new THREE.Vector3(i, 0.03, 18)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, lineMaterial);
            this.scene.add(line);
        }
    }

    createBuildings() {
        const colors = [0x1a2a3a, 0x2a3a4a, 0x1a3a4a];
        const positions = [
            { x: -30, z: -30 }, { x: 30, z: -30 },
            { x: -30, z: 30 }, { x: 30, z: 30 }
        ];

        positions.forEach(pos => {
            const height = 10 + Math.random() * 10;
            const geometry = new THREE.BoxGeometry(15, height, 15);
            const material = new THREE.MeshStandardMaterial({
                color: colors[Math.floor(Math.random() * colors.length)],
                roughness: 0.3,
                metalness: 0.7,
                emissive: 0x00d2ff,
                emissiveIntensity: 0.1
            });
            const building = new THREE.Mesh(geometry, material);
            building.position.set(pos.x, height / 2, pos.z);
            building.castShadow = true;
            this.scene.add(building);
        });
    }

    createVehicle() {
        this.vehicle = new THREE.Group();

        // Car body
        const bodyGeometry = new THREE.BoxGeometry(2.5, 1.2, 5);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x00d2ff,
            roughness: 0.3,
            metalness: 0.8,
            emissive: 0x00d2ff,
            emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 1;
        body.castShadow = true;
        this.vehicle.add(body);

        // Car roof
        const roofGeometry = new THREE.BoxGeometry(2, 0.8, 2.5);
        const roof = new THREE.Mesh(roofGeometry, bodyMaterial);
        roof.position.set(0, 2, -0.5);
        this.vehicle.add(roof);

        // Headlights
        const lightGeometry = new THREE.SphereGeometry(0.3, 8, 8);
        const lightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
        
        const headlightLeft = new THREE.Mesh(lightGeometry, lightMaterial);
        headlightLeft.position.set(-0.8, 1, 2.51);
        this.vehicle.add(headlightLeft);

        const headlightRight = new THREE.Mesh(lightGeometry, lightMaterial);
        headlightRight.position.set(0.8, 1, 2.51);
        this.vehicle.add(headlightRight);

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.5, 0.5, 0.4, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e });

        [[-1.3, 0.5, 1.8], [1.3, 0.5, 1.8], [-1.3, 0.5, -1.8], [1.3, 0.5, -1.8]].forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            this.vehicle.add(wheel);
        });

        this.vehicle.position.copy(this.startPosition);
        this.scene.add(this.vehicle);
    }

    createUserLocation() {
        this.userLocation = new THREE.Group();

        // Person
        const bodyGeometry = new THREE.CylinderGeometry(0.4, 0.4, 1.6, 8);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0xff6b6b,
            roughness: 0.8
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        this.userLocation.add(body);

        const headGeometry = new THREE.SphereGeometry(0.35, 16, 16);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.y = 1.8;
        this.userLocation.add(head);

        // GPS marker
        const markerGeometry = new THREE.OctahedronGeometry(0.5);
        const markerMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.8
        });
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.y = 2.8;
        this.userLocation.add(marker);

        // Pulsing ring on ground
        const ringGeometry = new THREE.TorusGeometry(1.5, 0.05, 8, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.6
        });
        this.locationRing = new THREE.Mesh(ringGeometry, ringMaterial);
        this.locationRing.rotation.x = Math.PI / 2;
        this.locationRing.position.y = 0.05;
        this.userLocation.add(this.locationRing);

        this.userLocation.position.copy(this.targetPosition);
        this.scene.add(this.userLocation);
    }

    createPathVisualization() {
        // Dotted line from vehicle start to user
        const points = [this.startPosition.clone(), this.targetPosition.clone()];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineDashedMaterial({
            color: 0x00d2ff,
            dashSize: 2,
            gapSize: 1,
            transparent: true,
            opacity: 0.6
        });
        this.pathLine = new THREE.Line(geometry, material);
        this.pathLine.computeLineDistances();
        this.scene.add(this.pathLine);

        // Waypoint markers
        const waypointGeometry = new THREE.SphereGeometry(0.3, 8, 8);
        const waypointMaterial = new THREE.MeshBasicMaterial({ color: 0x00d2ff });
        
        const midPoint = this.startPosition.clone().lerp(this.targetPosition, 0.5);
        const waypoint = new THREE.Mesh(waypointGeometry, waypointMaterial);
        waypoint.position.set(midPoint.x, 0.5, midPoint.z);
        this.scene.add(waypoint);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        // Animate location ring
        if (this.locationRing) {
            this.locationRing.rotation.z += delta;
            this.locationRing.scale.setScalar(1 + Math.sin(time * 2) * 0.1);
        }

        // Move vehicle to user
        if (this.isSummoning && this.vehicle) {
            this.moveVehicle(delta);
        }

        // Update telemetry
        this.updateTelemetry();

        this.renderer.render(this.scene, this.camera);
    }

    moveVehicle(delta) {
        const direction = new THREE.Vector3()
            .subVectors(this.targetPosition, this.vehicle.position);
        
        const distance = direction.length();
        
        if (distance > 1) {
            direction.normalize();
            const speed = 4 * delta;
            this.vehicle.position.add(direction.multiplyScalar(speed));
            this.vehicle.lookAt(this.targetPosition);
        } else {
            this.isSummoning = false;
            this.showToast('🎉 Vehicle has arrived!', 'success');
            this.showArrivalModal();
        }

        // Update camera to follow
        const cameraOffset = new THREE.Vector3(-15, 15, -15);
        const targetPos = this.vehicle.position.clone().add(cameraOffset);
        this.camera.position.lerp(targetPos, 0.05);
        this.camera.lookAt(this.vehicle.position);
    }

    updateTelemetry() {
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const gpsEl = document.getElementById('gpsSignal');
        const ultraEl = document.getElementById('ultrasonicValue');
        const distanceEl = document.getElementById('distanceToGoal');
        const etaEl = document.getElementById('eta');

        if (this.vehicle && this.userLocation) {
            const distance = this.vehicle.position.distanceTo(this.targetPosition);
            
            if (speedEl) speedEl.textContent = this.isSummoning ? '4.0' : '0.0';
            if (distanceEl) distanceEl.textContent = `${distance.toFixed(1)} m`;
            if (etaEl) etaEl.textContent = `${(distance / 4 * 60).toFixed(0)}s`;
        }

        if (batteryEl) batteryEl.textContent = (88 - this.clock.getElapsedTime() * 0.1).toFixed(0);
        if (gpsEl) gpsEl.textContent = 12 + Math.floor(Math.random() * 3);
        if (ultraEl) ultraEl.textContent = (40 + Math.random() * 20).toFixed(1);
    }

    setupEventListeners() {
        document.getElementById('btnStartSummon')?.addEventListener('click', () => {
            this.startSummon();
        });

        document.getElementById('btnPause')?.addEventListener('click', () => {
            this.isSummoning = !this.isSummoning;
            this.showToast(this.isSummoning ? '▶️ Resumed' : '⏸️ Paused', 'info');
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.isSummoning = false;
            this.showToast('🛑 Emergency stop', 'danger');
        });

        document.getElementById('btnSendBack')?.addEventListener('click', () => {
            this.sendBack();
        });
    }

    startSummon() {
        this.isSummoning = true;
        this.vehicle.position.copy(this.startPosition);
        this.clock.start();
        this.showToast('📍 Vehicle summoned!', 'success');
        document.getElementById('navStatus').textContent = 'Active';
    }

    sendBack() {
        this.showToast('↩️ Sending vehicle back...', 'info');
        this.isSummoning = true;
        // Swap positions temporarily
        const temp = this.vehicle.position.clone();
        this.vehicle.position.copy(this.targetPosition);
        this.targetPosition.copy(temp);
    }

    showArrivalModal() {
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
    
    if (window.summonMode3D) {
        window.summonMode3D.vehicle.position.copy(window.summonMode3D.startPosition);
        window.summonMode3D.isSummoning = false;
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.summonMode3D = new SummonMode3D();
});
