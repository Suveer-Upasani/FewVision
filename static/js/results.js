/* static/js/results.js — Results page logic */

(function () {
    'use strict';

    // -------------------------------------------------------------------------
    // Download button — provide visual feedback on click
    // -------------------------------------------------------------------------
    const downloadBtn = document.getElementById('downloadBtn');

    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            // Visual feedback
            const original = downloadBtn.innerHTML;
            downloadBtn.innerHTML = `
                <svg viewBox="0 0 20 20" fill="none" width="18">
                    <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5"
                            stroke-dasharray="44" stroke-dashoffset="0"
                            style="animation: spin 0.8s linear infinite; transform-origin: center"/>
                </svg>
                Preparing Download…
            `;
            downloadBtn.style.pointerEvents = 'none';

            setTimeout(() => {
                downloadBtn.innerHTML = original;
                downloadBtn.style.pointerEvents = '';
            }, 3000);
        });
    }

    // -------------------------------------------------------------------------
    // Animate stat counters
    // -------------------------------------------------------------------------
    function animateCounters() {
        const counters = document.querySelectorAll('.rstat-value');
        counters.forEach(counter => {
            const target = parseInt(counter.textContent, 10);
            if (isNaN(target) || target === 0) return;

            let current = 0;
            const increment = Math.max(1, Math.floor(target / 40));
            const interval = setInterval(() => {
                current = Math.min(current + increment, target);
                counter.textContent = current;
                if (current >= target) clearInterval(interval);
            }, 30);
        });
    }

    // -------------------------------------------------------------------------
    // Init
    // -------------------------------------------------------------------------
    setTimeout(animateCounters, 400);

})();
