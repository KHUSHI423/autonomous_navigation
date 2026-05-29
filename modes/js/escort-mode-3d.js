/* ==========================================================================
   ESCORT MODE - 3D VISUALIZATION
   Follows a person at safe distance with obstacle detection
   ========================================================================== */

class EscortMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.robot = null;
        this.person = null;
        this.isEscorting = false;
        this.clock = new THREE.Clock();
        this.safeDistance = 2.5;
        this.personPath = [];
        this.personPathIndex = 0;

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
        this.createPerson();
        this.createPersonPath();

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

        // Sidewalk
        const sidewalkGeometry = new THREE.PlaneGeometry(6, 200);
        const sidewalkMaterial = new THREE.MeshStandardMaterial({ color: 0x2a2a3a });
        const sidewalk = new THREE.Mesh(sidewalkGeometry, sidewalkMaterial);
        sidewalk.rotation.x = -Math.PI / 2;
        sidewalk.position.y = 0.02;
        this.scene.add(sidewalk);
    }

    createRobot() {
        this.robot = new THREE.Group();

        const bodyGeometry = new THREE.BoxGeometry(2, 0.8, 3);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x00d2ff, roughness: 0.3, metalness: 0.8,
            emissive: 0x00d2ff, emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        this.robot.add(body);

        const domeGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const domeMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.4
        });
        const dome = new THREE.Mesh(domeGeometry, domeMaterial);
        dome.position.set(0, 1.5, 0);
        this.robot.add(dome);

        const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e });
        [[-1.1, 0.4, 1], [1.1, 0.4, 1], [-1.1, 0.4, -1], [1.1, 0.4, -1]].forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            this.robot.add(wheel);
        });

        this.robot.position.set(0, 0, -5);
        this.scene.add(this.robot);
    }

    createPerson() {
        this.person = new THREE.Group();

        const bodyGeometry = new THREE.CylinderGeometry(0.4, 0.4, 1.6, 8);
        const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xff6b6b });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        this.person.add(body);

        const headGeometry = new THREE.SphereGeometry(0.35, 16, 16);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.y = 1.8;
        this.person.add(head);

        // Safety zone ring
        const ringGeometry = new THREE.TorusGeometry(3, 0.05, 8, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88, transparent: true, opacity: 0.4
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.05;
        this.person.add(ring);
        this.safetyRing = ring;

        this.person.position.set(0, 0, 5);
        this.scene.add(this.person);
    }

    createPersonPath() {
        // Create a walking path
        for (let i = 0; i < 10; i++) {
            this.personPath.push({
                x: Math.sin(i * 0.5) * 3,
                z: 5 + i * 3
            });
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        // Animate safety ring
        if (this.safetyRing) {
            this.safetyRing.rotation.z += delta;
        }

        // Move person along path
        if (this.person && this.personPath.length > 0) {
            this.movePerson(delta, time);
        }

        // Robot follows person
        if (this.isEscorting && this.robot) {
            this.escortPerson(delta);
        }

        this.updateTelemetry();
        this.renderer.render(this.scene, this.camera);
    }

    movePerson(delta, time) {
        const pathIndex = Math.floor(time * 0.3) % this.personPath.length;
        const nextIndex = (pathIndex + 1) % this.personPath.length;
        
        const targetPos = this.personPath[pathIndex];
        this.person.position.lerp(new THREE.Vector3(targetPos.x, 0, targetPos.z), delta * 2);
        this.person.lookAt(
            this.personPath[nextIndex]?.x || 0,
            0,
            this.personPath[nextIndex]?.z || 10
        );
    }

    escortPerson(delta) {
        if (!this.person) return;

        const direction = new THREE.Vector3()
            .subVectors(this.person.position, this.robot.position);
        direction.y = 0;
        
        const distance = direction.length();
        
        if (distance > this.safeDistance) {
            direction.normalize();
            this.robot.position.add(direction.multiplyScalar(2 * delta));
            this.robot.lookAt(this.person.position);
        }

        // Update camera
        const cameraOffset = new THREE.Vector3(0, 12, 16);
        const targetPos = this.robot.position.clone().add(cameraOffset);
        this.camera.position.lerp(targetPos, 0.05);
        this.camera.lookAt(this.robot.position);
    }

    updateTelemetry() {
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const ultraEl = document.getElementById('ultrasonicValue');
        const followEl = document.getElementById('followDistance');

        if (speedEl) speedEl.textContent = this.isEscorting ? '1.4' : '0.0';
        if (batteryEl) batteryEl.textContent = (86 - this.clock.getElapsedTime() * 0.05).toFixed(0);
        if (ultraEl) ultraEl.textContent = (35 + Math.random() * 15).toFixed(1);

        if (this.robot && this.person) {
            const distance = this.robot.position.distanceTo(this.person.position);
            if (followEl) followEl.textContent = `${distance.toFixed(1)} m`;
        }
    }

    setupEventListeners() {
        document.getElementById('btnStartEscort')?.addEventListener('click', () => {
            this.isEscorting = true;
            this.showToast('🚶 Escorting person...', 'success');
        });

        document.getElementById('btnPause')?.addEventListener('click', () => {
            this.isEscorting = !this.isEscorting;
            this.showToast(this.isEscorting ? '▶️ Resumed' : '⏸️ Paused', 'info');
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.isEscorting = false;
            this.showToast('🛑 Stopped', 'danger');
        });

        document.getElementById('btnReturnHome')?.addEventListener('click', () => {
            this.showToast('🏠 Returning to base...', 'info');
            this.isEscorting = false;
            this.robot.position.set(0, 0, -5);
        });
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

document.addEventListener('DOMContentLoaded', () => {
    window.escortMode3D = new EscortMode3D();
});
