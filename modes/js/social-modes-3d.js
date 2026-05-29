/* ==========================================================================
   SOCIAL IMPACT MODES - 3D VISUALIZATION
   Elder Assist, Guidance, Medical Delivery, Hospital Assist
   ========================================================================== */

class SocialModes3D {
    constructor(modeType) {
        this.modeType = modeType;
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.robot = null;
        this.target = null;
        this.isRunning = false;
        this.clock = new THREE.Clock();

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

        this.setupLighting();
        this.createEnvironment();
        this.createRobot();
        this.createTarget();

        window.addEventListener('resize', () => this.onResize());
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
        dirLight.position.set(50, 50, 50);
        this.scene.add(dirLight);
    }

    createEnvironment() {
        const groundGeometry = new THREE.PlaneGeometry(200, 200);
        const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x0a151a, roughness: 0.8 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        const gridHelper = new THREE.GridHelper(200, 50, 0x00d2ff, 0x006688);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        // Mode-specific environment
        if (this.modeType === 'hospital') {
            this.createHospitalEnvironment();
        } else if (this.modeType === 'medical') {
            this.createCityEnvironment();
        } else {
            this.createParkEnvironment();
        }
    }

    createHospitalEnvironment() {
        // Hospital floor
        const floorGeometry = new THREE.PlaneGeometry(60, 40);
        const floorMaterial = new THREE.MeshStandardMaterial({ color: 0x2a3a4a, roughness: 0.5 });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0.02;
        this.scene.add(floor);

        // Walls
        const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7 });
        
        // Left wall
        const leftWall = new THREE.Mesh(new THREE.BoxGeometry(1, 8, 40), wallMaterial);
        leftWall.position.set(-30, 4, 0);
        this.scene.add(leftWall);

        // Right wall
        const rightWall = new THREE.Mesh(new THREE.BoxGeometry(1, 8, 40), wallMaterial);
        rightWall.position.set(30, 4, 0);
        this.scene.add(rightWall);

        // Rooms
        this.createHospitalRooms();
    }

    createHospitalRooms() {
        const rooms = [
            { name: 'Pharmacy', pos: [-20, 4.1, -15], color: 0x00ff88 },
            { name: 'ICU Ward', pos: [20, 4.1, 15], color: 0xff0080 },
            { name: 'Emergency', pos: [-20, 4.1, 15], color: 0xff4444 }
        ];

        rooms.forEach(room => {
            const roomGeo = new THREE.BoxGeometry(12, 0.2, 8);
            const roomMat = new THREE.MeshBasicMaterial({ color: room.color, transparent: true, opacity: 0.3 });
            const roomSign = new THREE.Mesh(roomGeo, roomMat);
            roomSign.position.set(...room.pos);
            this.scene.add(roomSign);
        });
    }

    createCityEnvironment() {
        // Buildings
        const colors = [0x1a2a3a, 0x2a3a4a, 0x1a3a4a];
        for (let i = 0; i < 8; i++) {
            const height = 8 + Math.random() * 12;
            const geometry = new THREE.BoxGeometry(8, height, 8);
            const material = new THREE.MeshStandardMaterial({
                color: colors[Math.floor(Math.random() * colors.length)],
                roughness: 0.3,
                metalness: 0.7,
                emissive: 0x00d2ff,
                emissiveIntensity: 0.1
            });
            const building = new THREE.Mesh(geometry, material);
            const angle = (i / 8) * Math.PI * 2;
            building.position.set(Math.cos(angle) * 35, height / 2, Math.sin(angle) * 35);
            building.castShadow = true;
            this.scene.add(building);
        }

        // Roads
        const roadGeometry = new THREE.PlaneGeometry(10, 200);
        const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e, roughness: 0.9 });
        const road = new THREE.Mesh(roadGeometry, roadMaterial);
        road.rotation.x = -Math.PI / 2;
        road.position.y = 0.02;
        this.scene.add(road);
    }

    createParkEnvironment() {
        // Trees
        for (let i = 0; i < 12; i++) {
            const treeGroup = new THREE.Group();
            
            const trunkGeometry = new THREE.CylinderGeometry(0.3, 0.5, 2, 8);
            const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x4a3728 });
            const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
            trunk.position.y = 1;
            treeGroup.add(trunk);

            const leavesGeometry = new THREE.ConeGeometry(2, 4, 8);
            const leavesMaterial = new THREE.MeshStandardMaterial({
                color: 0x00ff88,
                emissive: 0x00ff88,
                emissiveIntensity: 0.2
            });
            const leaves = new THREE.Mesh(leavesGeometry, leavesMaterial);
            leaves.position.y = 3;
            treeGroup.add(leaves);

            const angle = (i / 12) * Math.PI * 2;
            treeGroup.position.set(Math.cos(angle) * 30, 0, Math.sin(angle) * 30);
            this.scene.add(treeGroup);
        }

        // Path
        const pathGeometry = new THREE.PlaneGeometry(4, 200);
        const pathMaterial = new THREE.MeshStandardMaterial({ color: 0x2a2a3a });
        const path = new THREE.Mesh(pathGeometry, pathMaterial);
        path.rotation.x = -Math.PI / 2;
        path.position.y = 0.02;
        this.scene.add(path);
    }

    createRobot() {
        this.robot = new THREE.Group();

        const bodyGeometry = new THREE.BoxGeometry(2, 0.8, 3);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: this.modeType === 'medical' ? 0xff4444 : 0x00d2ff,
            roughness: 0.3,
            metalness: 0.8,
            emissive: this.modeType === 'medical' ? 0xff4444 : 0x00d2ff,
            emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        this.robot.add(body);

        // Mode-specific top
        if (this.modeType === 'medical') {
            // Medical package
            const packageGeometry = new THREE.BoxGeometry(1.5, 1, 2);
            const packageMaterial = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                emissive: 0xff4444,
                emissiveIntensity: 0.3
            });
            const pkg = new THREE.Mesh(packageGeometry, packageMaterial);
            pkg.position.set(0, 1.8, 0);
            this.robot.add(pkg);

            // Red cross
            const crossGeometry = new THREE.BoxGeometry(0.3, 0.8, 0.1);
            const crossMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const crossV = new THREE.Mesh(crossGeometry, crossMaterial);
            crossV.position.set(0, 1.8, 1.01);
            this.robot.add(crossV);
            
            const crossH = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.3, 0.1), crossMaterial);
            crossH.position.set(0, 1.8, 1.01);
            this.robot.add(crossH);
        } else {
            // Sensor dome
            const domeGeometry = new THREE.SphereGeometry(0.5, 16, 16);
            const domeMaterial = new THREE.MeshStandardMaterial({
                color: 0x00ff88,
                emissive: 0x00ff88,
                emissiveIntensity: 0.4
            });
            const dome = new THREE.Mesh(domeGeometry, domeMaterial);
            dome.position.set(0, 1.5, 0);
            this.robot.add(dome);
        }

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
        if (this.modeType === 'elder' || this.modeType === 'guidance') {
            // Person
            this.target = new THREE.Group();
            const bodyGeometry = new THREE.CylinderGeometry(0.4, 0.4, 1.6, 8);
            const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xff6b6b });
            const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
            body.position.y = 0.8;
            this.target.add(body);

            const headGeometry = new THREE.SphereGeometry(0.35, 16, 16);
            const head = new THREE.Mesh(headGeometry, bodyMaterial);
            head.position.y = 1.8;
            this.target.add(head);

            this.target.position.set(0, 0, 5);
            this.scene.add(this.target);
        } else if (this.modeType === 'medical') {
            // Destination marker
            const markerGeometry = new THREE.OctahedronGeometry(1);
            const markerMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ff88,
                transparent: true,
                opacity: 0.6
            });
            this.target = new THREE.Mesh(markerGeometry, markerMaterial);
            this.target.position.set(30, 1, 30);
            this.scene.add(this.target);

            // Glow
            const glowGeometry = new THREE.SphereGeometry(1.5, 16, 16);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ff88,
                transparent: true,
                opacity: 0.2
            });
            const glow = new THREE.Mesh(glowGeometry, glowMaterial);
            glow.position.set(30, 1, 30);
            this.scene.add(glow);
        } else if (this.modeType === 'hospital') {
            // ICU destination
            const markerGeometry = new THREE.BoxGeometry(8, 0.2, 6);
            const markerMaterial = new THREE.MeshBasicMaterial({
                color: 0xff0080,
                transparent: true,
                opacity: 0.4
            });
            this.target = new THREE.Mesh(markerGeometry, markerMaterial);
            this.target.position.set(20, 0.03, 15);
            this.scene.add(this.target);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        // Animate target
        if (this.target) {
            if (this.modeType === 'elder' || this.modeType === 'guidance') {
                // Person walking
                this.target.position.x = Math.sin(time * 0.3) * 5;
                this.target.position.z = 5 + Math.cos(time * 0.3) * 3;
            } else {
                // Pulsing marker
                this.target.scale.setScalar(1 + Math.sin(time * 2) * 0.1);
            }
        }

        // Move robot if running
        if (this.isRunning && this.robot && this.target) {
            this.moveRobot(delta);
        }

        this.updateTelemetry();
        this.renderer.render(this.scene, this.camera);
    }

    moveRobot(delta) {
        const targetPos = this.target.position.clone();
        const direction = new THREE.Vector3()
            .subVectors(targetPos, this.robot.position);
        direction.y = 0;
        
        const distance = direction.length();
        
        if (distance > 1) {
            direction.normalize();
            this.robot.position.add(direction.multiplyScalar(1.5 * delta));
            this.robot.lookAt(targetPos);
        }
    }

    updateTelemetry() {
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const ultraEl = document.getElementById('ultrasonicValue');
        const distanceEl = document.getElementById('distanceToGoal');
        const progressEl = document.getElementById('progress');

        if (speedEl) speedEl.textContent = this.isRunning ? '1.5' : '0.0';
        if (batteryEl) batteryEl.textContent = (88 - this.clock.getElapsedTime() * 0.05).toFixed(0);
        if (ultraEl) ultraEl.textContent = (35 + Math.random() * 15).toFixed(1);

        if (this.robot && this.target) {
            const distance = this.robot.position.distanceTo(this.target.position);
            if (distanceEl) distanceEl.textContent = `${distance.toFixed(1)} m`;
            
            // Calculate progress
            const maxDistance = 50;
            const progress = Math.min(100, Math.floor((1 - distance / maxDistance) * 100));
            if (progressEl) progressEl.textContent = `${progress}%`;
        }
    }

    setupEventListeners() {
        const startBtn = document.getElementById(`btnStart${this.modeType.charAt(0).toUpperCase() + this.modeType.slice(1)}`) ||
                         document.getElementById('btnStartAssist') ||
                         document.getElementById('btnStartGuidance') ||
                         document.getElementById('btnStartDelivery') ||
                         document.getElementById('btnStartTransport');

        startBtn?.addEventListener('click', () => {
            this.isRunning = true;
            this.showToast('🚀 Mode activated!', 'success');
        });

        document.getElementById('btnPause')?.addEventListener('click', () => {
            this.isRunning = !this.isRunning;
            this.showToast(this.isRunning ? '▶️ Resumed' : '⏸️ Paused', 'info');
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.isRunning = false;
            this.showToast('🛑 Stopped', 'danger');
        });

        document.getElementById(`btnComplete${this.modeType.charAt(0).toUpperCase() + this.modeType.slice(1)}`)?.addEventListener('click', () =>
        this.completeMission());
    }

    completeMission() {
        this.isRunning = false;
        this.showToast('✅ Mission complete!', 'success');
        
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
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

// Global function
window.closeModal = function() {
    const modal = document.getElementById('deliveryModal');
    if (modal) modal.classList.remove('active');
};

// Initialize based on page
document.addEventListener('DOMContentLoaded', () => {
    const script = document.currentScript || document.querySelector('script[data-mode]');
    const modeType = script?.getAttribute('data-mode') || 'elder';
    window.socialModes3D = new SocialModes3D(modeType);
});
