/* ==========================================================================
   DELIVERY MODE - 3D VISUALIZATION WITH THREE.JS
   Hackathon 2026 - Autonomous Robot Car
   ========================================================================== */

class DeliveryMode3D {
    constructor() {
        this.container = document.getElementById('viewport3d');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.robot = null;
        this.waypoints = [];
        this.currentWaypointIndex = 0;
        this.obstacles = [];
        this.detectionRays = [];
        this.isDeliveryActive = false;
        this.clock = new THREE.Clock();
        this.telemetryData = {
            speed: 0,
            battery: 87,
            gpsSatellites: 12,
            ultrasonicDistance: 45.2
        };

        this.init();
        this.animate();
        this.setupEventListeners();
        this.startTelemetrySimulation();
    }

    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================
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
        this.camera.position.set(0, 10, 20);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.1;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 50;

        // Lighting
        this.setupLighting();

        // Environment
        this.createEnvironment();

        // Robot Car
        this.createRobotCar();

        // Waypoints
        this.createWaypoints();

        // Obstacles
        this.createObstacles();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);

        // Directional light (sun)
        const dirLight = new THREE.DirectionalLight(0x00d2ff, 0.8);
        dirLight.position.set(50, 50, 50);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 200;
        dirLight.shadow.camera.left = -50;
        dirLight.shadow.camera.right = 50;
        dirLight.shadow.camera.top = 50;
        dirLight.shadow.camera.bottom = -50;
        this.scene.add(dirLight);

        // Point lights for cyberpunk effect
        const pointLight1 = new THREE.PointLight(0x00d2ff, 1, 50);
        pointLight1.position.set(10, 5, 10);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xff0080, 1, 50);
        pointLight2.position.set(-10, 5, -10);
        this.scene.add(pointLight2);
    }

    createEnvironment() {
        // Ground plane with grid
        const groundGeometry = new THREE.PlaneGeometry(200, 200, 50, 50);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x0a151a,
            roughness: 0.8,
            metalness: 0.2,
            wireframe: false
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid helper
        const gridHelper = new THREE.GridHelper(200, 50, 0x00d2ff, 0x006688);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);

        // Buildings (simple boxes)
        this.createBuildings();

        // Trees (cones)
        this.createTrees();

        // Roads
        this.createRoads();
    }

    createBuildings() {
        const buildingColors = [0x1a2a3a, 0x2a3a4a, 0x1a3a4a, 0x2a2a3a];
        
        for (let i = 0; i < 20; i++) {
            const height = Math.random() * 15 + 5;
            const width = Math.random() * 8 + 4;
            const depth = Math.random() * 8 + 4;
            
            const geometry = new THREE.BoxGeometry(width, height, depth);
            const material = new THREE.MeshStandardMaterial({
                color: buildingColors[Math.floor(Math.random() * buildingColors.length)],
                roughness: 0.3,
                metalness: 0.7,
                emissive: 0x00d2ff,
                emissiveIntensity: 0.1
            });
            
            const building = new THREE.Mesh(geometry, material);
            
            // Position buildings around the scene
            const angle = Math.random() * Math.PI * 2;
            const radius = 30 + Math.random() * 50;
            building.position.set(
                Math.cos(angle) * radius,
                height / 2,
                Math.sin(angle) * radius
            );
            
            building.castShadow = true;
            building.receiveShadow = true;
            this.scene.add(building);

            // Add windows (glowing points)
            this.addBuildingWindows(building, width, height, depth);
        }
    }

    addBuildingWindows(building, width, height, depth) {
        const windowGeometry = new THREE.PlaneGeometry(0.5, 0.8);
        const windowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.8
        });

        const windowsPerRow = Math.floor(width / 1.5);
        const numRows = Math.floor(height / 2);

        for (let row = 0; row < numRows; row++) {
            for (let col = 0; col < windowsPerRow; col++) {
                if (Math.random() > 0.3) { // 70% chance of window
                    const windowMesh = new THREE.Mesh(windowGeometry, windowMaterial);
                    windowMesh.position.set(
                        -width/2 + 1 + col * 1.5,
                        -height/2 + 1.5 + row * 2,
                        depth/2 + 0.01
                    );
                    building.add(windowMesh);
                }
            }
        }
    }

    createTrees() {
        for (let i = 0; i < 15; i++) {
            const treeGroup = new THREE.Group();

            // Trunk
            const trunkGeometry = new THREE.CylinderGeometry(0.3, 0.5, 2, 8);
            const trunkMaterial = new THREE.MeshStandardMaterial({
                color: 0x4a3728,
                roughness: 0.9
            });
            const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
            trunk.castShadow = true;
            treeGroup.add(trunk);

            // Leaves (cone)
            const leavesGeometry = new THREE.ConeGeometry(2, 4, 8);
            const leavesMaterial = new THREE.MeshStandardMaterial({
                color: 0x00ff88,
                roughness: 0.8,
                emissive: 0x00ff88,
                emissiveIntensity: 0.2
            });
            const leaves = new THREE.Mesh(leavesGeometry, leavesMaterial);
            leaves.position.y = 3;
            leaves.castShadow = true;
            treeGroup.add(leaves);

            // Position trees
            const angle = Math.random() * Math.PI * 2;
            const radius = 25 + Math.random() * 40;
            treeGroup.position.set(
                Math.cos(angle) * radius,
                1,
                Math.sin(angle) * radius
            );

            this.scene.add(treeGroup);
        }
    }

    createRoads() {
        // Main road
        const roadGeometry = new THREE.PlaneGeometry(10, 200);
        const roadMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a1a2e,
            roughness: 0.9
        });
        const road = new THREE.Mesh(roadGeometry, roadMaterial);
        road.rotation.x = -Math.PI / 2;
        road.position.y = 0.02;
        road.receiveShadow = true;
        this.scene.add(road);

        // Road markings
        const markingGeometry = new THREE.PlaneGeometry(0.3, 2);
        const markingMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        for (let z = -95; z < 95; z += 8) {
            const marking = new THREE.Mesh(markingGeometry, markingMaterial);
            marking.rotation.x = -Math.PI / 2;
            marking.position.set(0, 0.03, z);
            this.scene.add(marking);
        }
    }

    createRobotCar() {
        this.robot = new THREE.Group();

        // Main body
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

        // Top sensor dome
        const domeGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const domeMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff88,
            roughness: 0.2,
            metalness: 0.5,
            emissive: 0x00ff88,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.8
        });
        const dome = new THREE.Mesh(domeGeometry, domeMaterial);
        dome.position.set(0, 1.5, 0);
        this.robot.add(dome);

        // Wheels
        const wheelGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16);
        const wheelMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a1a2e,
            roughness: 0.9
        });

        const wheelPositions = [
            [-1.1, 0.4, 1],
            [1.1, 0.4, 1],
            [-1.1, 0.4, -1],
            [1.1, 0.4, -1]
        ];

        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            wheel.castShadow = true;
            this.robot.add(wheel);
        });

        // Headlights
        const headlightGeometry = new THREE.SphereGeometry(0.2, 8, 8);
        const headlightMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.9
        });

        const headlightLeft = new THREE.Mesh(headlightGeometry, headlightMaterial);
        headlightLeft.position.set(-0.6, 0.8, 1.51);
        this.robot.add(headlightLeft);

        const headlightRight = new THREE.Mesh(headlightGeometry, headlightMaterial);
        headlightRight.position.set(0.6, 0.8, 1.51);
        this.robot.add(headlightRight);

        // Headlight beams
        this.createHeadlightBeams();

        // Ultrasonic sensor visualization (rotating ring)
        const ringGeometry = new THREE.TorusGeometry(1.5, 0.05, 8, 64);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.6
        });
        this.ultrasonicRing = new THREE.Mesh(ringGeometry, ringMaterial);
        this.ultrasonicRing.rotation.x = Math.PI / 2;
        this.ultrasonicRing.position.y = 0.1;
        this.robot.add(this.ultrasonicRing);

        this.robot.position.set(0, 0, 0);
        this.robot.castShadow = true;
        this.scene.add(this.robot);
    }

    createHeadlightBeams() {
        const beamGeometry = new THREE.ConeGeometry(1, 10, 32, 1, true);
        const beamMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });

        const beamLeft = new THREE.Mesh(beamGeometry, beamMaterial);
        beamLeft.rotation.x = Math.PI / 2;
        beamLeft.position.set(-0.6, 0.8, 6);
        beamLeft.scale.y = 1;
        this.robot.add(beamLeft);

        const beamRight = new THREE.Mesh(beamGeometry, beamMaterial);
        beamRight.rotation.x = Math.PI / 2;
        beamRight.position.set(0.6, 0.8, 6);
        this.robot.add(beamRight);
    }

    createWaypoints() {
        const waypointPositions = [
            { x: 0, y: 0.5, z: 0 },
            { x: 20, y: 0.5, z: 30 },
            { x: -15, y: 0.5, z: 50 },
            { x: -30, y: 0.5, z: 20 },
            { x: -20, y: 0.5, z: -20 },
            { x: 10, y: 0.5, z: -40 },
            { x: 30, y: 0.5, z: -10 },
            { x: 0, y: 0.5, z: 0 } // Return to start
        ];

        waypointPositions.forEach((pos, index) => {
            // Waypoint marker
            const geometry = new THREE.OctahedronGeometry(0.8);
            const material = new THREE.MeshStandardMaterial({
                color: index === 0 ? 0x00ff88 : 0x00d2ff,
                emissive: index === 0 ? 0x00ff88 : 0x00d2ff,
                emissiveIntensity: 0.8,
                transparent: true,
                opacity: 0.9
            });
            const waypoint = new THREE.Mesh(geometry, material);
            waypoint.position.set(pos.x, pos.y, pos.z);
            
            // Add glow
            const glowGeometry = new THREE.SphereGeometry(1.2, 16, 16);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: index === 0 ? 0x00ff88 : 0x00d2ff,
                transparent: true,
                opacity: 0.3
            });
            const glow = new THREE.Mesh(glowGeometry, glowMaterial);
            waypoint.add(glow);

            this.scene.add(waypoint);
            this.waypoints.push({ mesh: waypoint, position: pos, index });

            // Add waypoint number
            this.addWaypointLabel(pos, index + 1);
        });

        // Connect waypoints with lines
        this.createWaypointLines();
    }

    addWaypointLabel(position, number) {
        // Simple visual indicator (could be enhanced with CSS2DRenderer)
        const labelGeometry = new THREE.TextGeometry ? 
            new THREE.TextGeometry(number.toString(), { size: 1, height: 0.1 }) :
            new THREE.BoxGeometry(0.5, 0.5, 0.1);
        const labelMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const label = new THREE.Mesh(labelGeometry, labelMaterial);
        label.position.set(position.x, position.y + 2, position.z);
        this.scene.add(label);
    }

    createWaypointLines() {
        const points = this.waypoints.map(wp => new THREE.Vector3(
            wp.position.x, wp.position.y, wp.position.z
        ));

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x00d2ff,
            transparent: true,
            opacity: 0.5
        });

        const line = new THREE.Line(geometry, material);
        line.frustumCulled = false; // Prevent flickering
        this.scene.add(line);
        this.waypointLine = line;
    }

    createObstacles() {
        // Create various obstacles (cars, pedestrians, barriers)
        const obstacleTypes = ['car', 'pedestrian', 'barrier'];

        for (let i = 0; i < 10; i++) {
            const type = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
            let obstacle;

            if (type === 'car') {
                obstacle = this.createCarObstacle();
            } else if (type === 'pedestrian') {
                obstacle = this.createPedestrianObstacle();
            } else {
                obstacle = this.createBarrierObstacle();
            }

            // Random position
            const angle = Math.random() * Math.PI * 2;
            const radius = 10 + Math.random() * 40;
            obstacle.position.set(
                Math.cos(angle) * radius,
                0,
                Math.sin(angle) * radius
            );

            obstacle.userData.type = type;
            obstacle.userData.distance = 0;
            this.obstacles.push(obstacle);
            this.scene.add(obstacle);
        }
    }

    createCarObstacle() {
        const carGroup = new THREE.Group();

        // Car body
        const bodyGeometry = new THREE.BoxGeometry(2.5, 1.2, 4);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: Math.random() * 0xffffff,
            roughness: 0.3,
            metalness: 0.7
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.8;
        body.castShadow = true;
        carGroup.add(body);

        // Car roof
        const roofGeometry = new THREE.BoxGeometry(2, 0.8, 2);
        const roof = new THREE.Mesh(roofGeometry, bodyMaterial);
        roof.position.y = 1.8;
        roof.position.z = -0.3;
        carGroup.add(roof);

        return carGroup;
    }

    createPedestrianObstacle() {
        const personGroup = new THREE.Group();

        // Body
        const bodyGeometry = new THREE.CylinderGeometry(0.3, 0.3, 1.5, 8);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0xff6b6b,
            roughness: 0.8
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.75;
        body.castShadow = true;
        personGroup.add(body);

        // Head
        const headGeometry = new THREE.SphereGeometry(0.3, 8, 8);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.y = 1.7;
        personGroup.add(head);

        return personGroup;
    }

    createBarrierObstacle() {
        const barrierGroup = new THREE.Group();

        // Barrier posts
        const postGeometry = new THREE.CylinderGeometry(0.1, 0.1, 1, 8);
        const postMaterial = new THREE.MeshStandardMaterial({
            color: 0xffaa00,
            roughness: 0.5
        });

        const post1 = new THREE.Mesh(postGeometry, postMaterial);
        post1.position.set(-1, 0.5, 0);
        barrierGroup.add(post1);

        const post2 = new THREE.Mesh(postGeometry, postMaterial);
        post2.position.set(1, 0.5, 0);
        barrierGroup.add(post2);

        // Barrier bar
        const barGeometry = new THREE.BoxGeometry(2, 0.15, 0.15);
        const barMaterial = new THREE.MeshStandardMaterial({
            color: 0xffaa00,
            emissive: 0xffaa00,
            emissiveIntensity: 0.3
        });
        const bar = new THREE.Mesh(barGeometry, barMaterial);
        bar.position.y = 0.8;
        barrierGroup.add(bar);

        return barrierGroup;
    }

    // ==========================================================================
    // ANIMATION LOOP
    // ==========================================================================
    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        // Update controls
        this.controls.update();

        // Animate robot
        if (this.robot) {
            // Move robot along waypoints if delivery is active
            if (this.isDeliveryActive && this.currentWaypointIndex < this.waypoints.length) {
                this.moveRobotAlongPath(delta);
            }

            // Animate ultrasonic ring
            if (this.ultrasonicRing) {
                this.ultrasonicRing.rotation.z += delta * 2;
            }

            // Update headlight beams
            this.updateHeadlightBeams();
        }

        // Update obstacles (animate pedestrians walking)
        this.obstacles.forEach(obstacle => {
            if (obstacle.userData.type === 'pedestrian') {
                obstacle.rotation.y += delta * 0.5;
            }

            // Calculate distance from robot
            if (this.robot) {
                const distance = this.robot.position.distanceTo(obstacle.position);
                obstacle.userData.distance = distance;
            }
        });

        // Update detection rays
        this.updateDetectionRays();

        // Update telemetry display
        this.updateTelemetryDisplay();

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    moveRobotAlongPath(delta) {
        if (this.currentWaypointIndex >= this.waypoints.length) {
            this.isDeliveryActive = false;
            this.currentWaypointIndex = 0;
            this.showToast('🎉 Delivery route completed!', 'success');
            return;
        }

        const target = this.waypoints[this.currentWaypointIndex].position;
        const direction = new THREE.Vector3()
            .subVectors(target, this.robot.position)
            .normalize();

        const speed = 5 * delta; // meters per second
        this.robot.position.add(direction.multiplyScalar(speed));
        this.robot.lookAt(target);

        // Update telemetry speed
        this.telemetryData.speed = (speed / delta).toFixed(1);

        // Check if reached waypoint
        if (this.robot.position.distanceTo(target) < 1) {
            this.currentWaypointIndex++;
            this.showToast(`✅ Waypoint ${this.currentWaypointIndex} reached`, 'info');
            
            // Update mission display
            this.updateMissionDisplay();
        }

        // Update camera to follow robot
        const cameraOffset = new THREE.Vector3(0, 8, 12);
        const targetCameraPos = this.robot.position.clone().add(cameraOffset);
        this.camera.position.lerp(targetCameraPos, 0.05);
        this.camera.lookAt(this.robot.position);
    }

    updateHeadlightBeams() {
        // Flicker effect
        if (Math.random() > 0.95) {
            this.robot.children.forEach(child => {
                if (child.material && child.material.opacity !== undefined) {
                    child.material.opacity = 0.25 + Math.random() * 0.15;
                }
            });
        }
    }

    updateDetectionRays() {
        // Visualize ultrasonic detection
        if (!this.robot) return;

        const raycaster = new THREE.Raycaster();
        const maxDistance = 5;

        // Cast rays in multiple directions
        for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2 + this.robot.rotation.y;
            const direction = new THREE.Vector3(
                Math.sin(angle),
                0,
                Math.cos(angle)
            );

            raycaster.set(this.robot.position, direction);
            const intersects = raycaster.intersectObjects(this.obstacles, true);

            if (intersects.length > 0 && intersects[0].distance < maxDistance) {
                this.telemetryData.ultrasonicDistance = intersects[0].distance * 100; // Convert to cm
                
                // Draw detection line
                this.drawDetectionLine(
                    this.robot.position,
                    intersects[0].point,
                    intersects[0].distance < 1.5 ? 0xff4444 : 0x00ff88
                );
            }
        }
    }

    drawDetectionLine(start, end, color) {
        const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
        const material = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.8
        });
        const line = new THREE.Line(geometry, material);
        this.scene.add(line);

        // Remove after short time
        setTimeout(() => {
            this.scene.remove(line);
            geometry.dispose();
            material.dispose();
        }, 100);
    }

    updateTelemetryDisplay() {
        // Update DOM elements
        const speedEl = document.getElementById('speedValue');
        const batteryEl = document.getElementById('batteryValue');
        const gpsEl = document.getElementById('gpsSignal');
        const ultraEl = document.getElementById('ultrasonicValue');

        if (speedEl) speedEl.textContent = this.telemetryData.speed;
        if (batteryEl) batteryEl.textContent = this.telemetryData.battery.toFixed(0);
        if (gpsEl) gpsEl.textContent = this.telemetryData.gpsSatellites;
        if (ultraEl) ultraEl.textContent = this.telemetryData.ultrasonicDistance.toFixed(1);

        // Update obstacle display
        this.updateObstacleDisplay();
    }

    updateObstacleDisplay() {
        const display = document.getElementById('obstacleDisplay');
        if (!display) return;

        const nearbyObstacles = this.obstacles
            .filter(obs => obs.userData.distance < 15 && obs.userData.distance > 0)
            .sort((a, b) => a.userData.distance - b.userData.distance)
            .slice(0, 3);

        if (nearbyObstacles.length > 0) {
            display.innerHTML = nearbyObstacles.map(obs => `
                <div class="obstacle-item">
                    <span class="obstacle-icon">${obs.userData.type === 'car' ? '🚗' : obs.userData.type === 'pedestrian' ? '🚶' : '🚧'}</span>
                    <span class="obstacle-name">${obs.userData.type}</span>
                    <span class="obstacle-dist">${obs.userData.distance.toFixed(1)}m</span>
                </div>
            `).join('');
        }
    }

    updateMissionDisplay() {
        const objectiveEl = document.getElementById('currentObjective');
        const distanceEl = document.getElementById('distanceToGoal');
        const etaEl = document.getElementById('eta');

        if (this.currentWaypointIndex < this.waypoints.length) {
            const target = this.waypoints[this.currentWaypointIndex].position;
            const distance = this.robot.position.distanceTo(target);
            const eta = (distance / 5).toFixed(0); // Assuming 5 m/s

            if (objectiveEl) {
                objectiveEl.textContent = this.currentWaypointIndex === 0 ? 
                    'Starting Delivery' : 
                    `Navigate to Waypoint ${this.currentWaypointIndex + 1}`;
            }

            if (distanceEl) distanceEl.textContent = `${distance.toFixed(1)} m`;
            if (etaEl) etaEl.textContent = `${eta}s`;
        }
    }

    // ==========================================================================
    // EVENT HANDLERS
    // ==========================================================================
    setupEventListeners() {
        // Start Delivery button
        document.getElementById('btnStartDelivery')?.addEventListener('click', () => {
            this.startDelivery();
        });

        // Pause button
        document.getElementById('btnPause')?.addEventListener('click', () => {
            this.togglePause();
        });

        // Emergency Stop button
        document.getElementById('btnEmergencyStop')?.addEventListener('click', () => {
            this.emergencyStop();
        });

        // Complete Delivery button
        document.getElementById('btnCompleteDelivery')?.addEventListener('click', () => {
            this.completeDelivery();
        });
    }

    startDelivery() {
        this.isDeliveryActive = true;
        this.showToast('🚀 Delivery started! Navigating to waypoint 1', 'success');
        document.getElementById('currentObjective').textContent = 'Navigate to Waypoint 1';
    }

    togglePause() {
        this.isDeliveryActive = !this.isDeliveryActive;
        this.showToast(this.isDeliveryActive ? '▶️ Delivery resumed' : '⏸️ Delivery paused', 'info');
    }

    emergencyStop() {
        this.isDeliveryActive = false;
        this.showToast('🛑 EMERGENCY STOP ACTIVATED', 'danger');
        
        // Flash robot red
        this.robot.children.forEach(child => {
            if (child.material && child.material.emissive) {
                const originalEmissive = child.material.emissive.getHex();
                child.material.emissive.setHex(0xff0000);
                setTimeout(() => {
                    child.material.emissive.setHex(originalEmissive);
                }, 2000);
            }
        });
    }

    completeDelivery() {
        this.isDeliveryActive = false;
        this.showToast('✅ Delivery completed successfully!', 'success');
        
        // Show modal
        const modal = document.getElementById('deliveryModal');
        if (modal) {
            modal.classList.add('active');
        }
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

    startTelemetrySimulation() {
        // Simulate battery drain
        setInterval(() => {
            if (this.isDeliveryActive) {
                this.telemetryData.battery -= 0.1;
                if (this.telemetryData.battery <= 0) {
                    this.telemetryData.battery = 100;
                }
            }
        }, 1000);

        // Simulate GPS satellite variation
        setInterval(() => {
            this.telemetryData.gpsSatellites = 10 + Math.floor(Math.random() * 5);
        }, 2000);

        // Simulate ultrasonic noise
        setInterval(() => {
            if (this.telemetryData.ultrasonicDistance < 999) {
                this.telemetryData.ultrasonicDistance += (Math.random() - 0.5) * 2;
            }
        }, 500);
    }

    onResize() {
        if (!this.container) return;

        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

// ==========================================================================
// GLOBAL FUNCTIONS
// ==========================================================================
window.closeModal = function() {
    const modal = document.getElementById('deliveryModal');
    if (modal) {
        modal.classList.remove('active');
    }
    // Reset delivery
    if (window.deliveryMode3D) {
        window.deliveryMode3D.currentWaypointIndex = 0;
        window.deliveryMode3D.robot.position.set(0, 0, 0);
    }
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.deliveryMode3D = new DeliveryMode3D();
});
