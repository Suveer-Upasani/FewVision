/* static/js/dashboard.js — Dashboard page logic */

(function () {
    'use strict';

    // -------------------------------------------------------------------------
    // Animate score bars on load (they start at 0% in CSS)
    // -------------------------------------------------------------------------
    function animateBars() {
        // Bars already have the target width set via inline style from Jinja2.
        // We trigger a reflow then let CSS transition do the work.
        const bars = document.querySelectorAll('.score-bar');
        bars.forEach(bar => {
            const target = bar.style.width;
            bar.style.width = '0%';
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    bar.style.width = target;
                });
            });
        });
    }

    // -------------------------------------------------------------------------
    // Stagger card appearance
    // -------------------------------------------------------------------------
    function staggerCards() {
        const cards = document.querySelectorAll('.image-card');
        cards.forEach((card, i) => {
            card.style.animationDelay = `${i * 0.07}s`;
        });
    }

    // -------------------------------------------------------------------------
    // Generate augmented dataset
    // -------------------------------------------------------------------------
    const generateBtn  = document.getElementById('generateBtn');
    const genProgress  = document.getElementById('genProgress');

    if (generateBtn) {
        generateBtn.addEventListener('click', async () => {
            const sessionId = generateBtn.dataset.session;
            if (!sessionId) return;

            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating…';
            genProgress.classList.remove('hidden');

            try {
                const response = await fetch(`/api/generate/${sessionId}`, {
                    method: 'POST',
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Generation failed');
                }

                // Redirect to results page
                window.location.href = `/results/${sessionId}`;

            } catch (err) {
                genProgress.classList.add('hidden');
                generateBtn.disabled = false;
                generateBtn.innerHTML = `
                    <svg viewBox="0 0 20 20" fill="none" width="18">
                        <path d="M10 3v10M6 9l4 4 4-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M3 15h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                    </svg>
                    Generate Augmented Dataset
                `;
                showError(err.message);
            }
        });
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    function showError(message) {
        const existing = document.getElementById('errorToast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.id = 'errorToast';
        toast.style.cssText = `
            position: fixed; bottom: 24px; right: 24px; z-index: 999;
            background: hsl(355, 70%, 20%); border: 1px solid hsl(355, 75%, 40%);
            color: hsl(355, 80%, 80%); padding: 14px 20px; border-radius: 12px;
            font-size: 0.9rem; max-width: 380px; box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        `;
        toast.textContent = `⚠️ ${message}`;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 6000);
    }

    // -------------------------------------------------------------------------
    // Init
    // -------------------------------------------------------------------------
    staggerCards();
    // Small delay before triggering bar animations so they're visible
    setTimeout(animateBars, 200);

})();
