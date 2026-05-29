/* ==========================================================================
   PARTICLES SYSTEM - Cyberpunk Background Effect
   ========================================================================== */

class ParticleSystem {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.particles = [];
        this.particleCount = 100;
        this.colors = ['#00d2ff', '#3a7bd5', '#ff0080', '#00ff88'];
        
        if (this.container) {
            this.init();
            this.animate();
        }
    }

    init() {
        for (let i = 0; i < this.particleCount; i++) {
            this.createParticle();
        }
    }

    createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 4 + 2;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const duration = Math.random() * 20 + 10;
        const delay = Math.random() * 5;
        const color = this.colors[Math.floor(Math.random() * this.colors.length)];
        
        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            border-radius: 50%;
            left: ${x}%;
            top: ${y}%;
            opacity: ${Math.random() * 0.5 + 0.3};
            box-shadow: 0 0 ${size * 2}px ${color};
            animation: particleFloat ${duration}s ease-in-out ${delay}s infinite;
        `;
        
        this.container.appendChild(particle);
        this.particles.push({
            element: particle,
            x: x,
            y: y,
            vx: (Math.random() - 0.5) * 0.1,
            vy: (Math.random() - 0.5) * 0.1,
            size: size
        });
    }

    animate() {
        this.particles.forEach((particle, index) => {
            particle.x += particle.vx;
            particle.y += particle.vy;

            if (particle.x < 0 || particle.x > 100) particle.vx *= -1;
            if (particle.y < 0 || particle.y > 100) particle.vy *= -1;

            particle.element.style.left = `${particle.x}%`;
            particle.element.style.top = `${particle.y}%`;
        });

        requestAnimationFrame(() => this.animate());
    }
}

// Add particle float animation to document
const style = document.createElement('style');
style.textContent = `
    @keyframes particleFloat {
        0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.3;
        }
        25% {
            transform: translate(10px, -10px) scale(1.2);
            opacity: 0.6;
        }
        50% {
            transform: translate(-5px, 10px) scale(0.8);
            opacity: 0.4;
        }
        75% {
            transform: translate(15px, 5px) scale(1.1);
            opacity: 0.5;
        }
    }
    
    .particle {
        pointer-events: none;
        will-change: transform, opacity;
    }
`;
document.head.appendChild(style);

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    new ParticleSystem('particles');
});
