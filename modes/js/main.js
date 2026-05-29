/* ==========================================================================
   MAIN JAVASCRIPT - Animations, Counter, Hardware Integration
   ========================================================================== */

// ==========================================================================
// COUNTER ANIMATION FOR STATS
// ==========================================================================
class CounterAnimation {
    constructor() {
        this.counters = document.querySelectorAll('.stat-number');
        this.duration = 2000;
        this.init();
    }

    init() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        this.counters.forEach(counter => observer.observe(counter));
    }

    animateCounter(element) {
        const target = parseInt(element.getAttribute('data-count'));
        const start = 0;
        const startTime = performance.now();

        const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);

        const updateCounter = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / this.duration, 1);
            const easedProgress = easeOutQuart(progress);
            const current = Math.floor(start + (target - start) * easedProgress);
            
            element.textContent = current + (target === 100 ? '%' : '');

            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        };

        requestAnimationFrame(updateCounter);
    }
}

// ==========================================================================
// SMOOTH SCROLL FOR NAVIGATION
// ==========================================================================
class SmoothScroll {
    constructor() {
        this.links = document.querySelectorAll('a[href^="#"]');
        this.init();
    }

    init() {
        this.links.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href');
                const target = document.querySelector(targetId);
                
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
}

// ==========================================================================
// CARD TILT EFFECT (3D)
// ==========================================================================
class CardTiltEffect {
    constructor() {
        this.cards = document.querySelectorAll('.mode-card, .hardware-card');
        this.init();
    }

    init() {
        this.cards.forEach(card => {
            card.addEventListener('mousemove', (e) => this.handleMouseMove(e, card));
            card.addEventListener('mouseleave', () => this.handleMouseLeave(card));
        });
    }

    handleMouseMove(e, card) {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const rotateX = (y - centerY) / 10;
        const rotateY = (centerX - x) / 10;
        
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-10px) scale(1.02)`;
    }

    handleMouseLeave(card) {
        card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0) scale(1)';
    }
}

// ==========================================================================
// HARDWARE STATUS MONITOR (Integration with Robot Car)
// ==========================================================================
class HardwareMonitor {
    constructor() {
        this.esp32IP = '10.17.122.207'; // Default ESP32 IP
        this.updateInterval = 1000;
        this.statusElements = {};
        this.init();
    }

    init() {
        this.cacheElements();
        this.startMonitoring();
    }

    cacheElements() {
        // Cache DOM elements for hardware status
        this.statusElements = {
            camera: document.querySelector('.hw-status[data-hw="camera"]'),
            ultrasonic: document.querySelector('.hw-status[data-hw="ultrasonic"]'),
            yolo: document.querySelector('.hw-status[data-hw="yolo"]'),
            gps: document.querySelector('.hw-status[data-hw="gps"]'),
            motor: document.querySelector('.hw-status[data-hw="motor"]'),
            network: document.querySelector('.hw-status[data-hw="network"]')
        };
    }

    async startMonitoring() {
        try {
            // Try to fetch status from ESP32
            const response = await fetch(`http://${this.esp32IP}:8080/status`);
            const data = await response.json();
            this.updateStatus(data);
        } catch (error) {
            console.log('Hardware monitor: Using simulated data (ESP32 not reachable)');
            this.simulateStatus();
        }
        
        // Continue monitoring
        setInterval(() => this.checkStatus(), this.updateInterval);
    }

    async checkStatus() {
        try {
            const response = await fetch(`http://${this.esp32IP}:8080/status`, {
                method: 'GET',
                mode: 'cors',
                signal: AbortSignal.timeout(2000)
            });
            const data = await response.json();
            this.updateStatus(data);
        } catch (error) {
            // Silently fail, keep last known status
        }
    }

    updateStatus(data) {
        // Update ultrasonic status
        if (this.statusElements.ultrasonic && data.ultrasonic !== undefined) {
            const distance = parseFloat(data.ultrasonic);
            const statusEl = this.statusElements.ultrasonic;
            
            if (distance < 999) {
                statusEl.innerHTML = `
                    <span class="status-indicator"></span>
                    <span>${distance.toFixed(1)}cm</span>
                `;
                
                // Change color based on distance
                if (distance < 15) {
                    statusEl.style.color = '#ff4444'; // Danger
                } else if (distance < 30) {
                    statusEl.style.color = '#ffaa00'; // Warning
                } else {
                    statusEl.style.color = '#00ff88'; // OK
                }
            }
        }

        // Update mode status
        if (data.mode) {
            console.log(`Robot Mode: ${data.mode}`);
        }
    }

    simulateStatus() {
        // Simulate hardware status for demo purposes
        const distances = [45.2, 52.1, 38.7, 67.3, 41.9];
        let index = 0;
        
        setInterval(() => {
            if (this.statusElements.ultrasonic) {
                const distance = distances[index % distances.length] + (Math.random() * 10 - 5);
                this.statusElements.ultrasonic.innerHTML = `
                    <span class="status-indicator"></span>
                    <span>${distance.toFixed(1)}cm</span>
                `;
                this.statusElements.ultrasonic.style.color = '#00ff88';
            }
            index++;
        }, 2000);
    }
}

// ==========================================================================
// MODE SELECTION AND NAVIGATION
// ==========================================================================
class ModeNavigation {
    constructor() {
        this.modeCards = document.querySelectorAll('.mode-card');
        this.currentMode = null;
        this.init();
    }

    init() {
        this.modeCards.forEach(card => {
            card.addEventListener('click', () => {
                const mode = card.getAttribute('data-mode');
                this.navigateToMode(mode);
            });
        });
    }

    navigateToMode(mode) {
        // Store the selected mode in sessionStorage
        sessionStorage.setItem('selectedMode', mode);
        
        // Add visual feedback
        event.currentTarget.style.transform = 'scale(0.95)';
        setTimeout(() => {
            // Navigation happens via href in the HTML
        }, 200);
    }
}

// ==========================================================================
// PARALLAX SCROLL EFFECT
// ==========================================================================
class ParallaxScroll {
    constructor() {
        this.hero = document.querySelector('.hero');
        this.init();
    }

    init() {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            
            if (this.hero) {
                this.hero.style.transform = `translateY(${scrolled * 0.5}px)`;
            }
        });
    }
}

// ==========================================================================
// GLITCH TEXT EFFECT
// ==========================================================================
class GlitchEffect {
    constructor() {
        this.glitchElements = document.querySelectorAll('.glitch');
        this.init();
    }

    init() {
        this.glitchElements.forEach(el => {
            const originalText = el.getAttribute('data-text') || el.textContent;
            const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&';
            
            el.addEventListener('mouseover', () => {
                let iterations = 0;
                const maxIterations = 10;
                
                const interval = setInterval(() => {
                    el.textContent = el.textContent
                        .split('')
                        .map((char, index) => {
                            if (index < iterations) {
                                return originalText[index];
                            }
                            return chars[Math.floor(Math.random() * chars.length)];
                        })
                        .join('');
                    
                    iterations += 1/3;
                    
                    if (iterations >= originalText.length) {
                        clearInterval(interval);
                        el.textContent = originalText;
                    }
                }, 30);
            });
        });
    }
}

// ==========================================================================
// INITIALIZATION
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all modules
    new CounterAnimation();
    new SmoothScroll();
    new CardTiltEffect();
    new HardwareMonitor();
    new ModeNavigation();
    new ParallaxScroll();
    new GlitchEffect();
    
    console.log('🤖 Autonomous Modes Dashboard Initialized');
    console.log('📡 Hardware Monitor Active');
    console.log('🎨 Visual Effects Loaded');
});

// ==========================================================================
// UTILITY FUNCTIONS FOR MODE PAGES
// ==========================================================================
window.ModeUtils = {
    // Format distance for display
    formatDistance: (meters) => {
        if (meters >= 1000) {
            return `${(meters / 1000).toFixed(2)} km`;
        }
        return `${meters.toFixed(1)} m`;
    },

    // Calculate ETA
    calculateETA: (distance, speed) => {
        const timeInSeconds = distance / speed;
        const minutes = Math.floor(timeInSeconds / 60);
        const seconds = Math.floor(timeInSeconds % 60);
        return `${minutes}m ${seconds}s`;
    },

    // Get status color based on value
    getStatusColor: (value, thresholds) => {
        if (value < thresholds.danger) return '#ff4444';
        if (value < thresholds.warning) return '#ffaa00';
        return '#00ff88';
    },

    // Animate number
    animateNumber: (element, start, end, duration) => {
        const startTime = performance.now();
        const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);

        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easedProgress = easeOutQuart(progress);
            const current = start + (end - start) * easedProgress;
            
            element.textContent = current.toFixed(1);

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        };

        requestAnimationFrame(update);
    }
};
