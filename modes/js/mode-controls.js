/* ==========================================================================
   MODE CONTROLS - Shared control logic for all mode pages
   ========================================================================== */

class ModeControls {
    constructor(modeType) {
        this.modeType = modeType;
        this.esp32IP = '10.17.122.207';
        this.isRunning = false;
        this.init();
    }

    init() {
        this.setupButtonListeners();
        this.startHardwareMonitor();
        this.startRadarVisualization();
    }

    setupButtonListeners() {
        // Generic button handlers
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = btn.id;
                this.handleAction(action);
            });
        });
    }

    handleAction(action) {
        console.log(`Action: ${action}`);
        
        switch(action) {
            case 'btnStartDelivery':
            case 'btnStartAmbulance':
            case 'btnStartFollow':
            case 'btnStartSummon':
            case 'btnStartParking':
            case 'btnStartEscort':
            case 'btnStartElder':
            case 'btnStartGuidance':
            case 'btnStartMedical':
            case 'btnStartHospital':
                this.startMode();
                break;
            case 'btnPause':
                this.pauseMode();
                break;
            case 'btnEmergencyStop':
                this.emergencyStop();
                break;
            case 'btnCompleteDelivery':
            case 'btnCompleteMission':
                this.completeMode();
                break;
        }
    }

    async startMode() {
        this.isRunning = true;
        this.showToast(`🚀 ${this.modeType} mode activated!`, 'success');
        
        // Send start command to ESP32
        try {
            await fetch(`http://${this.esp32IP}:8080/mode?m=auto`, {
                mode: 'no-cors'
            });
        } catch (error) {
            console.log('ESP32 not reachable, using simulation mode');
        }
    }

    pauseMode() {
        this.isRunning = !this.isRunning;
        this.showToast(this.isRunning ? '▶️ Resumed' : '⏸️ Paused', 'info');
    }

    async emergencyStop() {
        this.isRunning = false;
        this.showToast('🛑 EMERGENCY STOP', 'danger');
        
        try {
            await fetch(`http://${this.esp32IP}:8080/stop`, {
                mode: 'no-cors'
            });
        } catch (error) {
            console.log('Emergency stop command sent (simulated)');
        }
    }

    completeMode() {
        this.isRunning = false;
        this.showToast('✅ Mission completed successfully!', 'success');
        
        const modal = document.getElementById('deliveryModal');
        if (modal) {
            modal.classList.add('active');
        }
    }

    async startHardwareMonitor() {
        setInterval(async () => {
            try {
                const response = await fetch(`http://${this.esp32IP}:8080/status`, {
                    mode: 'no-cors',
                    signal: AbortSignal.timeout(2000)
                });
                const data = await response.json();
                this.updateHardwareStatus(data);
            } catch (error) {
                // Use simulated data
                this.simulateHardwareStatus();
            }
        }, 1000);
    }

    updateHardwareStatus(data) {
        // Update ultrasonic value
        const ultraEl = document.getElementById('ultrasonicValue');
        if (ultraEl && data.ultrasonic !== undefined) {
            ultraEl.textContent = parseFloat(data.ultrasonic).toFixed(1);
        }
    }

    simulateHardwareStatus() {
        // Simulate sensor readings
        const ultraEl = document.getElementById('ultrasonicValue');
        if (ultraEl) {
            const distance = 30 + Math.random() * 50;
            ultraEl.textContent = distance.toFixed(1);
        }
    }

    startRadarVisualization() {
        const canvas = document.getElementById('radarCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 150;

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const maxRadius = Math.min(centerX, centerY) - 10;
        let angle = 0;

        const drawRadar = () => {
            // Clear canvas
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw radar circles
            ctx.strokeStyle = 'rgba(0, 210, 255, 0.3)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 4; i++) {
                ctx.beginPath();
                ctx.arc(centerX, centerY, (maxRadius / 4) * i, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Draw radar lines
            for (let i = 0; i < 8; i++) {
                const lineAngle = (i / 8) * Math.PI * 2;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(
                    centerX + Math.cos(lineAngle) * maxRadius,
                    centerY + Math.sin(lineAngle) * maxRadius
                );
                ctx.stroke();
            }

            // Draw sweeping line
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.8)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(
                centerX + Math.cos(angle) * maxRadius,
                centerY + Math.sin(angle) * maxRadius
            );
            ctx.stroke();

            // Draw gradient behind sweep
            const gradient = ctx.createConicGradient(angle, centerX, centerY);
            gradient.addColorStop(0, 'rgba(0, 255, 136, 0.3)');
            gradient.addColorStop(0.2, 'rgba(0, 255, 136, 0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw detected obstacles
            this.drawRadarObstacles(ctx, centerX, centerY, maxRadius);

            angle += 0.05;
            if (angle > Math.PI * 2) angle = 0;

            requestAnimationFrame(drawRadar);
        };

        drawRadar();
    }

    drawRadarObstacles(ctx, centerX, centerY, maxRadius) {
        // Simulate obstacle detection
        const obstacles = [
            { angle: 0.5, distance: 0.6 },
            { angle: 2.1, distance: 0.4 },
            { angle: 4.2, distance: 0.8 }
        ];

        obstacles.forEach(obs => {
            const x = centerX + Math.cos(obs.angle) * maxRadius * obs.distance;
            const y = centerY + Math.sin(obs.angle) * maxRadius * obs.distance;

            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#ff4444';
            ctx.fill();
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
}

// ==========================================================================
// CAMERA FEED INTEGRATION
// ==========================================================================
class CameraFeedManager {
    constructor() {
        this.container = document.getElementById('cameraFeed');
        this.videoElement = null;
        this.esp32IP = '10.17.122.207';
        this.init();
    }

    init() {
        this.connectToCamera();
    }

    async connectToCamera() {
        // Try to connect to Raspberry Pi camera stream
        try {
            // This would connect to the Pi's camera stream
            // For now, show placeholder
            this.showPlaceholder();
        } catch (error) {
            this.showPlaceholder();
        }
    }

    showPlaceholder() {
        if (!this.container) return;

        this.container.innerHTML = `
            <div class="feed-placeholder">
                <span class="feed-icon">📹</span>
                <span class="feed-text">Waiting for video stream...</span>
                <span class="feed-subtext">Port: 5000</span>
            </div>
        `;

        // Try to fetch actual frame periodically
        setInterval(() => this.tryFetchFrame(), 1000);
    }

    async tryFetchFrame() {
        try {
            // In production, this would fetch from the Pi
            // const response = await fetch(`http://${this.esp32IP}:5000/frame`);
            // const blob = await response.blob();
            // this.displayFrame(blob);
        } catch (error) {
            // Keep placeholder
        }
    }

    displayFrame(blob) {
        const url = URL.createObjectURL(blob);
        this.container.innerHTML = `<img src="${url}" style="width:100%;height:100%;object-fit:cover;" />`;
    }
}

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Determine mode type from page
    const pagePath = window.location.pathname;
    const modeType = pagePath.replace('_mode.html', '').split('/').pop();
    
    // Initialize controls
    window.modeControls = new ModeControls(modeType);
    window.cameraFeed = new CameraFeedManager();
});
