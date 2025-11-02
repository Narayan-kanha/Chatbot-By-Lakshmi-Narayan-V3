// A combination of scripts inspired by the template provided.

document.addEventListener('DOMContentLoaded', () => {
    
    // --- Animated Particle Background ---
    function createParticles() {
        const particlesContainer = document.getElementById('particles-js'); // Assuming a container with this ID exists
        if (!particlesContainer) return;
        
        const particleCount = 30;
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 15 + 's';
            particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
            
            // Randomly assign one of the accent colors
            if (Math.random() > 0.5) {
                particle.style.background = 'var(--accent-color-1)';
            } else {
                 particle.style.background = 'var(--accent-color-2)';
            }
            
            particlesContainer.appendChild(particle);
        }
    }

    // You would need a <div id="particles-js"></div> in your base.html for this to work
    // createParticles();

    // --- Navbar Scroll Effect ---
    window.addEventListener('scroll', function() {
        const navbar = document.getElementById('navbar');
        if (navbar) {
             if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        }
    });
});