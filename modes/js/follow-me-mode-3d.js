/* ==========================================================================
   FOLLOW-ME MODE - 3D VISUALIZATION
   GPS Target Tracking with Safe Following Distance
   ========================================================================== */

class FollowMeMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.robot = null;
        this.target = null;
        this.isFollowing = false;
        this.clock = new THREE.Clock();
        this.safeDistance = 3.0;
        this.targetPosition = new THREE.Vector3(0, 0, 10);

        this.init();
        this.animate();
        this.setupEventListeners();
    }

    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f2027);
        this.scene.fog = new THREE.FogExp2(0x0f2027, 0.002);

        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 15, 25);
        this.camera.lookAt(0, 0, 0);

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
        this.createRobot();
        this.createTarget();
        this.createConnectionLine();

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
        const groundGeometry = new THREE.PlaneGeometry(200, 200, 50, 50);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x0a151a,
            roughness: 0.8
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        const gridHelper = new THREE.GridHelper(200, 50, 0x00d2ff, 0x006688);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        // Add paths/walkways
        this.createPaths();
    }

    createPaths() {
        const pathGeometry = new THREE.PlaneGeometry(4, 200);
        const pathMaterial = new THREE.MeshStandardMaterial({
            color: 0x2a2a3a,
            roughness: 0.9
        });
        const path = new THREE.Mesh(pathGeometry, pathMaterial);
        path.rotation.x = -Math.PI / 2;
        path.position.y = 0.02;
        this.scene.add(path);
    }

    createRobot() {
        this.robot = new THREE.Group();

        const bodyGeometry = new THREE.BoxGeometry(2, 0.8, 3);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x00d2ff,
            roughness: 0.3,
            metalness: 0.8,
            emissive: 0x00d2ff,
            emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        this.robot.add(body);

        // GPS antenna
        const antennaGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.5, 8);
        const antennaMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff88,
            emissive: 0x00ff88,
            emissiveIntensity: 0.5
        });
        const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
        antenna.position.set(0, 1.5, 0);
        this.robot.add(antenna);

        // GPS signal rings
        const ringGeometry = new THREE.TorusGeometry(1, 0.05, 8, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.6
        });
        this.gpsRing = new THREE.Mesh(ringGeometry, ringMaterial);
        this.gpsRing.rotation.x = Math.PI / 2;
        this.gpsRing.position.y = 1.5;
        this.robot.add(this.gpsRing);

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e });

        [[-1.1, 0.4, 1], [1.1, 0.4, 1], [-1.1, 0.4, -1], [1.1, 0.4, -1]].forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            this.robot.add(wheel);
        });

        this.robot.position.set(0, 0, 0);
        this.scene.add(this.robot);
    }

    createTarget() {
        // Person representation
        this.target = new THREE.Group();

        // Body
        const bodyGeometry = new THREE.CylinderGeometry(0.4, 0.4, 1.6, 8);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0xff6b6b,
            roughness: 0.8
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        this.target.add(body);

        // Head
        const headGeometry = new THREE.SphereGeometry(0.35, 16, 16);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.y = 1.8;
        this.target.add(head);

        // GPS marker above head
        const markerGeometry = new THREE.OctahedronGeometry(0.3);
        const markerMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.8
        });
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.y = 2.5;
        this.target.add(marker);

        // GPS glow
        const glowGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.3
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.y = 2.5;
        this.target.add(glow);

        this.target.position.copy(this.targetPosition);
        this.scene.add(this.target);
    }

    createConnectionLine() {
        const points = [new THREE.Vector3(0, 0.5, 0), new THREE.Vector3(0, 0.5, 10)];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineDashedMaterial({
            color: 0x00ff88,
            dashSize: 1,
            gapSize: 0.5,
            transparent: true,
            opacity: 0.6
        });
        this.connectionLine = new THREE.Line(geometry, material);
        this.connectionLine.computeLineDistances();
        this.scene.add(this.connectionLine);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        // Animate GPS ring
        if (this.gpsRing) {
            this.gpsRing.rotation.z += delta * 2;
            this.gpsRing.scale.setScalar(1 + Math.sin(time * 3) * 0.1);
        }

        // Move target (person walking)
        if (this.target) {
            this.moveTarget(delta, time);
        }

        // Follow target
        if (this.isFollowing && this.robot) {
            this.followTarget(delta);
        }

        // Update connection line
        this.updateConnectionLine();

        // Update telemetry
        this.updateTelemetry();

        this.renderer.render(this.scene, this.camera);
    }

    moveTarget(delta, time) {
        // Simulate person walking in a pattern
        const speed = 1.5;
        const radius = 15;
        
        this.targetPosition.x = Math.sin(time * 0.3) * radius;
        this.targetPosition.z = Math.cos(time * 0.3) * radius + 10;
        
        this.target.position.lerp(this.targetPosition, delta * 2);
        this.target.lookAt(
            Math.sin((time + 0.1) * 0.3) * radius,
            0,
            Math.cos((time + 0.1) * 0.3) * radius + 10
        );
    }

    followTarget(delta) {
        if (!this.target) return;

        const direction = new THREE.Vector3()
            .subVectors(this.target.position, this.robot.position);
        direction.y = 0;
        
        const distance = direction.length();
        
        if (distance > this.safeDistance) {
            direction.normalize();
            const moveSpeed = 2 * delta;
            this.robot.position.add(direction.multiplyScalar(moveSpeed));
            this.robot.lookAt(this.target.position);
        }

        // Update camera to follow robot
        const cameraOffset = new THREE.Vector3(0, 12, 16);
        const targetPos = this.robot.position.clone().add(cameraOffset);
        this.camera.position.lerp(targetPos, 0.05);
        this.camera.lookAt(this.robot.position);
    }

    updateConnectionLine() {
        if (!this.connectionLine || !this.robot || !this.target) return;

        const points = [
            new THREE.Vector3(this.robot.position.x, 0.5, this.robot.position.z),
            new THREE.Vector3(this.target.position.x, 2, this.target.position.z)
        ];
        this.connectionLine.geometry.setFromPoints(points);
        this.connectionLine.computeLineDistances();
    }

    updateTelemetry() {
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const gpsEl = document.getElementById('gpsSignal');
        const ultraEl = document.getElementById('ultrasonicValue');
        const distanceEl = document.getElementById('distanceToGoal');
        const followSpeedEl = document.getElementById('followSpeed');
        const targetGPSEl = document.getElementById('targetGPS');

        if (this.robot && this.target) {
            const distance = this.robot.position.distanceTo(this.target.position);
            
            if (speedEl) speedEl.textContent = this.isFollowing ? '1.5' : '0.0';
            if (distanceEl) distanceEl.textContent = `${distance.toFixed(1)} m`;
            if (followSpeedEl) followSpeedEl.textContent = this.isFollowing ? '1.5 m/s' : '0.0 m/s';
        }

        if (batteryEl) batteryEl.textContent = (85 - this.clock.getElapsedTime() * 0.05).toFixed(0);
        if (gpsEl) gpsEl.textContent = 10 + Math.floor(Math.random() * 4);
        if (ultraEl) ultraEl.textContent = (35 + Math.random() * 15).toFixed(1);

        if (this.target && targetGPSEl) {
            const lat = 12.9716 + this.target.position.x * 0.0001;
            const lon = 77.5946 + this.target.position.z * 0.0001;
            targetGPSEl.textContent = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
        }
    }

    setupEventListeners() {
        document.getElementById('btnStartFollow')?.addEventListener('click', () => {
            this.startFollowing();
        });

        document.getElementById('btnPause')?.addEventListener('click', () => {
            this.isFollowing = !this.isFollowing;
            this.showToast(this.isFollowing ? '▶️ Following resumed' : '⏸️ Holding position', 'info');
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.isFollowing = false;
            this.showToast('🛑 Emergency stop', 'danger');
        });

        document.getElementById('btnReturnHome')?.addEventListener('click', () => {
            this.returnToBase();
        });
    }

    startFollowing() {
        this.isFollowing = true;
        this.showToast('🧭 Following target...', 'success');
        document.getElementById('targetStatus').textContent = 'Locked';
        document.getElementById('targetStatus').style.color = '#00ff88';
    }

    returnToBase() {
        this.isFollowing = false;
        this.showToast('🏠 Returning to base...', 'info');
        
        // Animate back to origin
        const originalPos = new THREE.Vector3(0, 0, 0);
        const animateReturn = () => {
            if (this.robot.position.distanceTo(originalPos) > 0.5) {
                const direction = new THREE.Vector3()
                    .subVectors(originalPos, this.robot.position)
                    .normalize();
                this.robot.position.add(direction.multiplyScalar(0.2));
                this.robot.lookAt(originalPos);
                requestAnimationFrame(animateReturn);
            }
        };
        animateReturn();
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.followMeMode3D = new FollowMeMode3D();
});
