/* ==========================================================================
   AUTO-PARK MODE - 3D VISUALIZATION
   Autonomous Parking with Precision Maneuvering
   ========================================================================== */

class AutoParkMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.vehicle = null;
        this.parkingSpots = [];
        this.selectedSpot = null;
        this.isParking = false;
        this.clock = new THREE.Clock();
        this.parkingPath = [];
        this.parkingProgress = 0;

        this.init();
        this.animate();
        this.setupEventListeners();
    }

    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f2027);

        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(30, 25, 30);
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.setupLighting();
        this.createParkingLot();
        this.createVehicle();
        this.createParkingSpots();

        window.addEventListener('resize', () => this.onResize());
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(50, 50, 50);
        dirLight.castShadow = true;
        this.scene.add(dirLight);
    }

    createParkingLot() {
        const groundGeometry = new THREE.PlaneGeometry(100, 100);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a1a2e,
            roughness: 0.9
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0.01;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid lines
        const gridHelper = new THREE.GridHelper(100, 20, 0x00d2ff, 0x006688);
        this.scene.add(gridHelper);
    }

    createVehicle() {
        this.vehicle = new THREE.Group();

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

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.5, 0.5, 0.4, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1a2e });

        [[-1.3, 0.5, 1.8], [1.3, 0.5, 1.8], [-1.3, 0.5, -1.8], [1.3, 0.5, -1.8]].forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            this.vehicle.add(wheel);
        });

        this.vehicle.position.set(-20, 0, 0);
        this.scene.add(this.vehicle);
    }

    createParkingSpots() {
        const spotMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });

        for (let i = 0; i < 6; i++) {
            const spotGeometry = new THREE.PlaneGeometry(4, 6);
            const spot = new THREE.Mesh(spotGeometry, spotMaterial.clone());
            spot.rotation.x = -Math.PI / 2;
            spot.position.set(-10 + i * 6, 0.02, 15);
            spot.userData.index = i;
            spot.userData.occupied = false;
            this.scene.add(spot);
            this.parkingSpots.push(spot);

            // Spot number
            const text = this.createTextSprite(`${i + 1}`);
            text.position.set(-10 + i * 6, 0.5, 15);
            this.scene.add(text);
        }
    }

    createTextSprite(text) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 128;
        canvas.height = 128;
        ctx.fillStyle = '#00d2ff';
        ctx.font = 'Bold 64px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 64);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(2, 2, 1);
        return sprite;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        this.controls.update();

        if (this.isParking && this.vehicle) {
            this.executeParkingManeuver(delta);
        }

        this.updateTelemetry();
        this.renderer.render(this.scene, this.camera);
    }

    executeParkingManeuver(delta) {
        if (!this.selectedSpot) return;

        this.parkingProgress += delta * 0.3;
        
        if (this.parkingProgress >= 1) {
            this.isParking = false;
            this.parkingProgress = 1;
            this.showToast('✅ Parked successfully!', 'success');
            this.showParkingModal();
            return;
        }

        // Simple reverse parking animation
        const startPos = this.vehicle.position.clone();
        const endPos = this.selectedSpot.position.clone();
        endPos.y = 0;
        endPos.z -= 3;

        // Lerp with rotation
        this.vehicle.position.lerpVectors(startPos, endPos, delta * 0.5);
        this.vehicle.rotation.y += delta * 0.3;

        // Update progress display
        document.getElementById('maneuverProgress').textContent = `${Math.floor(this.parkingProgress * 100)}%`;
    }

    findAvailableSpot() {
        const available = this.parkingSpots.filter(spot => !spot.userData.occupied);
        if (available.length > 0) {
            this.selectedSpot = available[0];
            this.selectedSpot.material.opacity = 0.8;
            this.selectedSpot.material.color.setHex(0x00d2ff);
            
            document.getElementById('spotDetection').textContent = `Spot ${this.selectedSpot.userData.index + 1} Found`;
            document.getElementById('spotDetection').style.color = '#00ff88';
            
            this.showToast(`🅿 Parking spot ${this.selectedSpot.userData.index + 1} found!`, 'success');
        }
    }

    updateTelemetry() {
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const ultraFEl = document.getElementById('ultrasonicF');
        const ultraREl = document.getElementById('ultrasonicR');
        const distanceEl = document.getElementById('distanceToGoal');

        if (speedEl) speedEl.textContent = this.isParking ? '0.8' : '0.0';
        if (batteryEl) batteryEl.textContent = (90 - this.clock.getElapsedTime() * 0.05).toFixed(0);
        if (ultraFEl) ultraFEl.textContent = (40 + Math.random() * 10).toFixed(1);
        if (ultraREl) ultraREl.textContent = (35 + Math.random() * 10).toFixed(1);

        if (this.vehicle && this.selectedSpot) {
            const distance = this.vehicle.position.distanceTo(this.selectedSpot.position);
            if (distanceEl) distanceEl.textContent = `${distance.toFixed(1)} m`;
        }
    }

    setupEventListeners() {
        document.getElementById('btnFindSpot')?.addEventListener('click', () => {
            this.findAvailableSpot();
        });

        document.getElementById('btnStartParking')?.addEventListener('click', () => {
            if (this.selectedSpot) {
                this.isParking = true;
                this.showToast('🅿 Starting parking maneuver...', 'info');
                document.getElementById('parkingStatus').textContent = 'Parking...';
            } else {
                this.showToast('⚠️ Please find a spot first!', 'warning');
            }
        });

        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.isParking = false;
            this.showToast('🛑 Emergency stop', 'danger');
        });

        document.getElementById('btnExitSpot')?.addEventListener('click', () => {
            this.exitParking();
        });
    }

    exitParking() {
        this.showToast('↗️ Exiting parking spot...', 'info');
        this.vehicle.position.set(-20, 0, 0);
        this.vehicle.rotation.set(0, 0, 0);
        this.isParking = false;
        this.parkingProgress = 0;
        
        if (this.selectedSpot) {
            this.selectedSpot.material.opacity = 0.3;
            this.selectedSpot.material.color.setHex(0x00ff88);
            this.selectedSpot = null;
        }
        
        document.getElementById('parkingStatus').textContent = 'Ready';
        document.getElementById('maneuverProgress').textContent = '0%';
    }

    showParkingModal() {
        const modal = document.getElementById('parkingModal');
        if (modal) modal.classList.add('active');
        
        if (this.selectedSpot) {
            this.selectedSpot.userData.occupied = true;
            this.selectedSpot.material.color.setHex(0xff4444);
        }
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
    const modal = document.getElementById('parkingModal');
    if (modal) modal.classList.remove('active');
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.autoParkMode3D = new AutoParkMode3D();
});
