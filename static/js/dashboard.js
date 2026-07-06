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
    // Product Inspection (Workflow 2)
    // -------------------------------------------------------------------------
    const testDropzone = document.getElementById('testDropzone');
    const testFileInput = document.getElementById('testFileInput');
    const testBrowseBtn = document.getElementById('testBrowseBtn');
    const testClearBtn = document.getElementById('testClearBtn');
    const testPreviewSection = document.getElementById('testPreviewSection');
    const testPreviewGrid = document.getElementById('testPreviewGrid');
    const testPreviewCount = document.getElementById('testPreviewCount');
    const testActionSection = document.getElementById('testActionSection');
    const inspectBtn = document.getElementById('inspectBtn');
    const inspectProgress = document.getElementById('inspectProgress');

    const resultsSection = document.getElementById('inspectionResultsSection');
    const summaryNormalCount = document.getElementById('summaryNormalCount');
    const summarySuspiciousCount = document.getElementById('summarySuspiciousCount');
    const summaryAnomalousCount = document.getElementById('summaryAnomalousCount');
    const inspectionResultsList = document.getElementById('inspectionResultsList');

    let selectedTestFiles = [];

    if (testDropzone && testFileInput) {
        // Dropzone events
        ['dragenter', 'dragover'].forEach(eventName => {
            testDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                testDropzone.classList.add('drag-active');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            testDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                testDropzone.classList.remove('drag-active');
            }, false);
        });

        testDropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleTestFiles(files);
        });

        testBrowseBtn.addEventListener('click', () => {
            testFileInput.click();
        });

        testFileInput.addEventListener('change', (e) => {
            handleTestFiles(e.target.files);
        });

        testClearBtn.addEventListener('click', () => {
            selectedTestFiles = [];
            updateTestPreview();
        });
    }

    function handleTestFiles(files) {
        const valid = Array.from(files).filter(f => {
            const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
            return ['.jpg', '.jpeg', '.png', '.bmp'].includes(ext);
        });

        if (selectedTestFiles.length + valid.length > 20) {
            showError('Maximum limit of 20 test images reached.');
            return;
        }

        selectedTestFiles = [...selectedTestFiles, ...valid];
        updateTestPreview();
    }

    function updateTestPreview() {
        testPreviewGrid.innerHTML = '';
        if (selectedTestFiles.length === 0) {
            testPreviewSection.classList.add('hidden');
            testActionSection.classList.add('hidden');
            return;
        }

        testPreviewSection.classList.remove('hidden');
        testActionSection.classList.remove('hidden');
        testPreviewCount.textContent = `${selectedTestFiles.length} test image(s) selected`;

        selectedTestFiles.forEach((file, idx) => {
            const card = document.createElement('div');
            card.className = 'test-preview-card';
            card.innerHTML = `
                <div class="test-preview-thumb-wrap">
                    <img class="test-preview-thumb" src="" alt="">
                </div>
                <div class="test-preview-details">
                    <span class="test-preview-name">${file.name}</span>
                    <span class="test-preview-size">${(file.size / 1024).toFixed(1)} KB</span>
                </div>
                <button type="button" class="btn-remove" data-index="${idx}">&times;</button>
            `;

            // Draw thumbnail image
            const img = card.querySelector('.test-preview-thumb');
            const reader = new FileReader();
            reader.onload = (e) => { img.src = e.target.result; };
            reader.readAsDataURL(file);

            // Bind remove button
            card.querySelector('.btn-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                selectedTestFiles.splice(idx, 1);
                updateTestPreview();
            });

            testPreviewGrid.appendChild(card);
        });
    }

    if (inspectBtn) {
        inspectBtn.addEventListener('click', async () => {
            if (selectedTestFiles.length === 0) return;

            const sessionId = inspectBtn.dataset.session;
            const formData = new FormData();
            formData.append('session_id', sessionId);
            selectedTestFiles.forEach(file => {
                formData.append('files', file);
            });

            inspectBtn.disabled = true;
            inspectProgress.classList.remove('hidden');
            resultsSection.classList.add('hidden');

            try {
                const response = await fetch('/api/inspect', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Inspection failed');
                }

                // Render Results
                renderInspectionResults(data);
                
                // Clear selection
                selectedTestFiles = [];
                updateTestPreview();

            } catch (err) {
                showError(err.message);
            } finally {
                inspectBtn.disabled = false;
                inspectProgress.classList.add('hidden');
            }
        });
    }

    function renderInspectionResults(data) {
        const summary = data.summary;
        const results = data.results;

        summaryNormalCount.textContent = summary.normal_count;
        summarySuspiciousCount.textContent = summary.suspicious_count;
        summaryAnomalousCount.textContent = summary.anomalous_count;

        // Update download results link
        const btnDownloadResults = document.getElementById('btnDownloadResults');
        if (btnDownloadResults) {
            btnDownloadResults.href = `/api/inference/${data.session_id}/download`;
        }

        inspectionResultsList.innerHTML = '';

        results.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = `inspection-result-card border-${item.prediction.toLowerCase()}`;
            card.style.animationDelay = `${index * 0.08}s`;

            let neighborsHtml = '';
            item.top_k_neighbors.forEach(n => {
                neighborsHtml += `
                    <div class="neighbor-row">
                        <span class="neighbor-rank">Rank ${n.rank}</span>
                        <span class="neighbor-name" title="${n.filename}">${n.filename}</span>
                        <div class="neighbor-values">
                            <span class="neighbor-dist">Dist: ${n.distance.toFixed(4)}</span>
                            <span class="neighbor-sim">Sim: ${(n.similarity * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                `;
            });

            card.innerHTML = `
                <div class="irc-header">
                    <div class="irc-image-name">${item.image_name}</div>
                    <div class="irc-badge badge-${item.prediction.toLowerCase()}">
                        ${item.prediction === 'Normal' ? '✅' : item.prediction === 'Suspicious' ? '⚠️' : '❌'}
                        ${item.prediction}
                    </div>
                </div>

                <div class="irc-main">
                    <div class="irc-stats-panel">
                        <div class="irc-stat-row">
                            <span class="irc-stat-label">Anomaly Score</span>
                            <div class="irc-stat-progress">
                                <div class="irc-stat-bar bar-${item.prediction.toLowerCase()}" style="width: ${item.anomaly_score}%"></div>
                            </div>
                            <span class="irc-stat-val">${item.anomaly_score.toFixed(1)}/100</span>
                        </div>
                        <div class="irc-stat-row">
                            <span class="irc-stat-label">Confidence</span>
                            <div class="irc-stat-progress">
                                <div class="irc-stat-bar bar-confidence" style="width: ${item.confidence * 100}%"></div>
                            </div>
                            <span class="irc-stat-val">${(item.confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>

                    <div class="irc-metrics-grid">
                        <div class="irc-metric-cell">
                            <span class="irc-metric-lbl">Quality Score</span>
                            <span class="irc-metric-val">${item.quality_score.toFixed(1)}/100</span>
                        </div>
                        <div class="irc-metric-cell">
                            <span class="irc-metric-lbl">Content Score</span>
                            <span class="irc-metric-val">${item.content_score.toFixed(1)}/100</span>
                        </div>
                        <div class="irc-metric-cell cell-full">
                            <span class="irc-metric-lbl">Nearest Reference Image</span>
                            <span class="irc-metric-val font-mono" title="${item.nearest_reference}">${item.nearest_reference}</span>
                        </div>
                    </div>
                </div>

                <details class="irc-neighbors-details">
                    <summary class="irc-neighbors-toggle">View Top 5 Nearest Reference Neighbors</summary>
                    <div class="irc-neighbors-list">
                        ${neighborsHtml}
                    </div>
                </details>
            `;

            inspectionResultsList.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
