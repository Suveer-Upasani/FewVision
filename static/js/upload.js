/* static/js/upload.js — Upload page logic */

(function () {
    'use strict';

    const dropzone       = document.getElementById('dropzone');
    const dropIdle       = document.getElementById('dropIdle');
    const fileInput      = document.getElementById('fileInput');
    const browseBtn      = document.getElementById('browseBtn');
    const previewSection = document.getElementById('previewSection');
    const previewGrid    = document.getElementById('previewGrid');
    const previewCount   = document.getElementById('previewCount');
    const clearBtn       = document.getElementById('clearBtn');
    const actionBar      = document.getElementById('actionBar');
    const analyzeBtn     = document.getElementById('analyzeBtn');
    const processingState = document.getElementById('processingState');
    const processingStep  = document.getElementById('processingStep');

    const STEPS = [
        { el: document.getElementById('step1'), label: 'Running quality analysis…' },
        { el: document.getElementById('step2'), label: 'Analysing content…' },
        { el: document.getElementById('step3'), label: 'Generating adaptive policy…' },
        { el: document.getElementById('step4'), label: 'Producing reports…' },
    ];

    let selectedFiles = [];

    // -------------------------------------------------------------------------
    // Drag-and-drop
    // -------------------------------------------------------------------------
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
        dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false)
    );

    dropzone.addEventListener('dragenter', () => dropzone.classList.add('dragover'));
    dropzone.addEventListener('dragleave', (e) => {
        if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        dropzone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(isValidImage);
        addFiles(files);
    });

    // Click-to-browse
    dropzone.addEventListener('click', (e) => {
        if (e.target === browseBtn || e.target === dropzone || dropzone.contains(e.target)) {
            fileInput.click();
        }
    });
    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

    // Keyboard accessibility
    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files).filter(isValidImage);
        addFiles(files);
        fileInput.value = ''; // Reset so same file can be added again if needed
    });

    // -------------------------------------------------------------------------
    // File management
    // -------------------------------------------------------------------------
    function isValidImage(file) {
        return /\.(jpe?g|png|bmp|tiff?)$/i.test(file.name);
    }

    function addFiles(files) {
        if (!files.length) return;

        const remaining = 50 - selectedFiles.length;
        const toAdd = files.slice(0, remaining);

        toAdd.forEach(file => {
            // Avoid duplicates by name+size
            const exists = selectedFiles.some(f => f.name === file.name && f.size === file.size);
            if (!exists) selectedFiles.push(file);
        });

        renderPreviews();
    }

    function renderPreviews() {
        previewGrid.innerHTML = '';

        selectedFiles.forEach((file, idx) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const thumb = document.createElement('div');
                thumb.className = 'preview-thumb';
                thumb.setAttribute('role', 'listitem');
                thumb.style.animationDelay = `${idx * 0.04}s`;

                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = file.name;

                const label = document.createElement('div');
                label.className = 'preview-thumb-name';
                label.textContent = file.name;

                thumb.appendChild(img);
                thumb.appendChild(label);
                previewGrid.appendChild(thumb);
            };
            reader.readAsDataURL(file);
        });

        const count = selectedFiles.length;
        previewCount.textContent = `${count} image${count !== 1 ? 's' : ''} selected`;

        // Show/hide sections
        previewSection.classList.toggle('hidden', count === 0);
        actionBar.classList.toggle('hidden', count === 0);
        dropIdle.style.opacity = count > 0 ? '0.5' : '1';
    }

    clearBtn.addEventListener('click', () => {
        selectedFiles = [];
        renderPreviews();
    });

    // -------------------------------------------------------------------------
    // Upload + pipeline
    // -------------------------------------------------------------------------
    analyzeBtn.addEventListener('click', () => {
        if (selectedFiles.length === 0) return;
        uploadAndAnalyze();
    });

    async function uploadAndAnalyze() {
        // Switch to processing state
        previewSection.classList.add('hidden');
        actionBar.classList.add('hidden');
        dropzone.classList.add('hidden');
        processingState.classList.remove('hidden');

        const formData = new FormData();
        selectedFiles.forEach(file => formData.append('files', file));

        // Animate processing steps
        let stepIdx = 0;
        const stepInterval = setInterval(() => {
            if (stepIdx > 0 && STEPS[stepIdx - 1]) {
                STEPS[stepIdx - 1].el.classList.remove('active');
                STEPS[stepIdx - 1].el.classList.add('done');
                STEPS[stepIdx - 1].el.querySelector('.step-dot').textContent = '';
            }
            if (stepIdx < STEPS.length) {
                STEPS[stepIdx].el.classList.add('active');
                processingStep.textContent = STEPS[stepIdx].label;
                stepIdx++;
            }
        }, 1800);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            clearInterval(stepInterval);

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            // Mark all steps done
            STEPS.forEach(s => {
                s.el.classList.remove('active');
                s.el.classList.add('done');
            });
            processingStep.textContent = 'Complete! Redirecting…';

            // Short pause then redirect
            await delay(600);
            window.location.href = data.redirect;

        } catch (err) {
            clearInterval(stepInterval);
            processingState.classList.add('hidden');
            dropzone.classList.remove('hidden');
            previewSection.classList.remove('hidden');
            actionBar.classList.remove('hidden');
            showError(err.message);
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

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
            animation: fadeIn 0.3s ease;
        `;
        toast.textContent = `⚠️ ${message}`;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 6000);
    }

})();
