/* ==========================================================================
   CRASH RESPONSE MODE - 3D VISUALIZATION
   Impact Detection + GPS + Image Capture + Severity Analysis
   ========================================================================== */

class CrashResponseMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.robot = null;
        this.isMonitoring = true;
        this.clock = new THREE.Clock();
        this.gForceData = { x: 0.02, y: 0.01, z: 1.0 };
        this.impactThreshold = 2.5;
        this.alertsSent = 0;
        this.lastImpact = null;

        this.init();
        this.animate();
        this.setupEventListeners();
        this.simulateSensorData();
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
        this.camera.position.set(0, 10, 20);
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
        this.createImpactZones();

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
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        const gridHelper = new THREE.GridHelper(200, 50, 0x00d2ff, 0x006688);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        // Add some obstacles/objects that could cause impacts
        this.createObstacles();
    }

    createObstacles() {
        const colors = [0x4a4a4a, 0x3a3a4a, 0x2a2a3a];
        
        for (let i = 0; i < 10; i++) {
            const geometry = new THREE.BoxGeometry(2, 2, 2);
            const material = new THREE.MeshStandardMaterial({
                color: colors[Math.floor(Math.random() * colors.length)],
                roughness: 0.7
            });
            const obstacle = new THREE.Mesh(geometry, material);
            
            const angle = Math.random() * Math.PI * 2;
            const radius = 15 + Math.random() * 30;
            obstacle.position.set(
                Math.cos(angle) * radius,
                1,
                Math.sin(angle) * radius
            );
            
            obstacle.castShadow = true;
            obstacle.receiveShadow = true;
            this.scene.add(obstacle);
        }
    }

    createRobot() {
        this.robot = new THREE.Group();

        // Main body with accelerometer visualization
        const bodyGeometry = new THREE.BoxGeometry(2, 1, 3);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x00d2ff,
            roughness: 0.3,
            metalness: 0.8,
            emissive: 0x00d2ff,
            emissiveIntensity: 0.1
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 1;
        body.castShadow = true;
        this.robot.add(body);

        // Sensor dome (accelerometer)
        const domeGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const domeMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff88,
            emissive: 0x00ff88,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.8
        });
        const dome = new THREE.Mesh(domeGeometry, domeMaterial);
        dome.position.set(0, 1.8, 0);
        this.robot.add(dome);

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e });

        [[-1.1, 0.4, 1], [1.1, 0.4, 1], [-1.1, 0.4, -1], [1.1, 0.4, -1]].forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            this.robot.add(wheel);
        });

        // Camera mounts (4 directions)
        this.createCameraMounts();

        this.robot.position.set(0, 0, 0);
        this.scene.add(this.robot);
    }

    createCameraMounts() {
        const cameraPositions = [
            { pos: [0, 1.2, 1.6], rot: 0, name: 'front' },
            { pos: [0, 1.2, -1.6], rot: Math.PI, name: 'rear' },
            { pos: [1.1, 1.2, 0], rot: Math.PI / 2, name: 'right' },
            { pos: [-1.1, 1.2, 0], rot: -Math.PI / 2, name: 'left' }
        ];

        cameraPositions.forEach(cam => {
            const camGeometry = new THREE.BoxGeometry(0.3, 0.3, 0.2);
            const camMaterial = new THREE.MeshStandardMaterial({
                color: 0x333333,
                emissive: 0x00ff88,
                emissiveIntensity: 0.3
            });
            const camera = new THREE.Mesh(camGeometry, camMaterial);
            camera.position.set(...cam.pos);
            camera.rotation.y = cam.rot;
            camera.userData.name = cam.name;
            this.robot.add(camera);
        });
    }

    createImpactZones() {
        // Visual rings showing impact detection zones
        const ringGeometry = new THREE.RingGeometry(4.9, 5.1, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.05;
        this.scene.add(ring);
        this.impactRing = ring;

        const ring2Geometry = new THREE.RingGeometry(9.9, 10.1, 32);
        const ring2 = new THREE.Mesh(ring2Geometry, ringMaterial.clone());
        ring2.rotation.x = Math.PI / 2;
        ring2.position.y = 0.05;
        this.scene.add(ring2);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        // Animate robot dome (sensor monitoring)
        if (this.robot) {
            const dome = this.robot.children.find(c => c.geometry.type === 'SphereGeometry');
            if (dome) {
                dome.scale.setScalar(1 + Math.sin(time * 5) * 0.1);
                
                // Change color based on G-force
                const gMagnitude = Math.sqrt(
                    this.gForceData.x ** 2 + 
                    this.gForceData.y ** 2 + 
                    (this.gForceData.z - 1) ** 2
                );
                
                if (gMagnitude > this.impactThreshold) {
                    dome.material.emissive.setHex(0xff0000);
                    dome.material.emissiveIntensity = 1;
                } else if (gMagnitude > 1) {
                    dome.material.emissive.setHex(0xffaa00);
                    dome.material.emissiveIntensity = 0.5;
                } else {
                    dome.material.emissive.setHex(0x00ff88);
                    dome.material.emissiveIntensity = 0.3;
                }
            }
        }

        // Animate impact ring
        if (this.impactRing) {
            this.impactRing.rotation.z += delta * 0.5;
            this.impactRing.material.opacity = 0.3 + Math.sin(time * 2) * 0.1;
        }

        this.updateTelemetry();
        this.renderer.render(this.scene, this.camera);
    }

    simulateSensorData() {
        setInterval(() => {
            if (this.isMonitoring) {
                // Normal sensor noise
                this.gForceData.x = (Math.random() - 0.5) * 0.1;
                this.gForceData.y = (Math.random() - 0.5) * 0.1;
                this.gForceData.z = 1 + (Math.random() - 0.5) * 0.05;
            }
        }, 500);
    }

    simulateImpact(severity = 'MEDIUM') {
        const severityLevels = {
            'LOW': { g: 1.5, shake: 0.5 },
            'MEDIUM': { g: 3.0, shake: 1.5 },
            'HIGH': { g: 5.0, shake: 3.0 }
        };

        const impact = severityLevels[severity];
        
        // Simulate G-force spike
        this.gForceData.x = (Math.random() - 0.5) * impact.g;
        this.gForceData.y = (Math.random() - 0.5) * impact.g;
        this.gForceData.z = 1 + (Math.random() - 0.5) * impact.g;

        // Shake robot
        if (this.robot) {
            const originalPos = this.robot.position.clone();
            let shakeCount = 0;
            const shakeInterval = setInterval(() => {
                this.robot.position.x = originalPos.x + (Math.random() - 0.5) * impact.shake;
                this.robot.position.z = originalPos.z + (Math.random() - 0.5) * impact.shake;
                shakeCount++;
                
                if (shakeCount > 10) {
                    clearInterval(shakeInterval);
                    this.robot.position.copy(originalPos);
                }
            }, 50);
        }

        // Show impact overlay
        const overlay = document.getElementById('impactOverlay');
        if (overlay) {
            overlay.style.display = 'flex';
            document.getElementById('impactSeverity').textContent = `Severity: ${severity}`;
            
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 2000);
        }

        // Update status
        this.lastImpact = {
            time: new Date().toLocaleTimeString(),
            severity: severity,
            gForce: Math.sqrt(this.gForceData.x ** 2 + this.gForceData.y ** 2 + (this.gForceData.z - 1) ** 2).toFixed(2)
        };

        document.getElementById('systemStatus').textContent = 'IMPACT DETECTED';
        document.getElementById('systemStatus').style.color = '#ff4444';
        document.getElementById('lastImpact').textContent = `${severity} - ${this.lastImpact.time}`;

        // Capture images (simulated)
        this.captureImages();

        this.showToast(`💥 ${severity} impact detected!`, 'danger');
    }

    captureImages() {
        // Simulate image capture from 4 cameras
        const slots = document.querySelectorAll('.image-slot');
        slots.forEach((slot, index) => {
            slot.innerHTML = `
                <span class="slot-icon">✅</span>
                <span class="slot-text">Captured</span>
                <span class="slot-time" style="font-size: 0.7rem; color: #00ff88;">${new Date().toLocaleTimeString()}</span>
            `;
        });
    }

    sendAlerts() {
        this.alertsSent += 3; // SMS, Email, Emergency Services
        document.getElementById('alertsSent').textContent = this.alertsSent;
        
        this.showToast('📡 Alerts sent to emergency contacts!', 'success');
        this.showToast('🚑 Emergency services notified', 'info');
        this.showToast('📍 GPS location shared', 'info');
    }

    updateTelemetry() {
        const gXEl = document.getElementById('gforceX');
        const gYEl = document.getElementById('gforceY');
        const gZEl = document.getElementById('gforceZ');
        const thresholdEl = document.getElementById('impactThreshold');

        if (gXEl) gXEl.textContent = this.gForceData.x.toFixed(2);
        if (gYEl) gYEl.textContent = this.gForceData.y.toFixed(2);
        if (gZEl) gZEl.textContent = this.gForceData.z.toFixed(2);
        if (thresholdEl) thresholdEl.textContent = this.impactThreshold.toFixed(1);

        // Update GPS with slight variation
        const gpsEl = document.getElementById('gpsLocation');
        if (gpsEl) {
            const lat = 12.9716 + (Math.random() - 0.5) * 0.0001;
            const lon = 77.5946 + (Math.random() - 0.5) * 0.0001;
            gpsEl.textContent = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
        }
    }

    setupEventListeners() {
        document.getElementById('btnSimulateImpact')?.addEventListener('click', () => {
            const severities = ['LOW', 'MEDIUM', 'HIGH'];
            const severity = severities[Math.floor(Math.random() * severities.length)];
            this.simulateImpact(severity);
        });

        document.getElementById('btnSendAlert')?.addEventListener('click', () => {
            this.sendAlerts();
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.isMonitoring = false;
            this.showToast('🛑 Monitoring paused', 'warning');
        });

        document.getElementById('btnResetSystem')?.addEventListener('click', () => {
            this.isMonitoring = true;
            this.gForceData = { x: 0.02, y: 0.01, z: 1.0 };
            this.lastImpact = null;
            this.alertsSent = 0;
            
            document.getElementById('systemStatus').textContent = 'Monitoring';
            document.getElementById('systemStatus').style.color = '#00ff88';
            document.getElementById('lastImpact').textContent = 'None';
            document.getElementById('alertsSent').textContent = '0';
            
            // Reset camera slots
            const slots = document.querySelectorAll('.image-slot');
            slots.forEach(slot => {
                slot.innerHTML = `
                    <span class="slot-icon">📹</span>
                    <span class="slot-text">Front View</span>
                `;
            });
            
            this.showToast('🔄 System reset complete', 'success');
        });
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `
            <span class="toast-icon">${type === 'success' ? '✅' : type === 'danger' ? '⚠️' : type === 'warning' ? '⏸️' : 'ℹ️'}</span>
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
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.crashMode3D = new CrashResponseMode3D();
});
