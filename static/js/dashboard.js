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
            card.style.animationDelay = `${index * 0.08}s`;

            // Backward compatibility fallbacks if product_grade is missing
            const hasGrade = item.product_grade && item.product_grade.grade;
            const grade = hasGrade ? item.product_grade.grade : (item.prediction === 'Normal' ? 'PASS' : (item.prediction === 'Anomalous' ? 'FAIL' : 'REVIEW'));
            const gradeConf = hasGrade ? item.product_grade.confidence : item.confidence;
            
            // Color themes based on grade
            let gradeClass = 'pass';
            let gradeColor = '#34d399'; // Emerald / green
            let gradeEmoji = '✅';
            if (grade === 'FAIL') {
                gradeClass = 'fail';
                gradeColor = '#f87171'; // Red
                gradeEmoji = '❌';
            } else if (grade === 'REVIEW') {
                gradeClass = 'review';
                gradeColor = '#fbbf24'; // Amber
                gradeEmoji = '⚠️';
            }

            // Set card class names
            card.className = `inspection-result-card border-${gradeClass}`;

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

            let patchcoreHtml = '';
            if (item.patchcore_enabled) {
                let patchMatchesHtml = '';
                if (item.top_5_patch_matches) {
                    item.top_5_patch_matches.forEach(pm => {
                        patchMatchesHtml += `
                            <div class="patch-match-row">
                                <span class="pm-rank">Rank ${pm.rank}</span>
                                <span class="pm-coords">Patch [Row ${pm.test_row}, Col ${pm.test_col}]</span>
                                <span class="pm-dist">Dist: ${pm.distance.toFixed(4)}</span>
                                <span class="pm-sim">Sim: ${(pm.similarity * 100).toFixed(1)}%</span>
                                <div class="pm-ref-info">
                                    Matched Ref: <strong title="${pm.reference_image}">${pm.reference_image}</strong> (Patch [Row ${pm.reference_row}, Col ${pm.reference_col}])
                                </div>
                            </div>
                        `;
                    });
                }

                let tabsHtml = '';
                if (item.padim && item.padim.enabled) {
                    tabsHtml = `
                        <div class="irc-tabs" style="display: flex; gap: 15px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                            <button class="irc-tab-btn active" data-tab="patchcore" style="background: none; border: none; color: #fff; border-bottom: 2px solid #3b82f6; padding: 4px 12px; cursor: pointer; font-weight: 500; font-size: 0.9rem; transition: all 0.2s;">PatchCore</button>
                            <button class="irc-tab-btn" data-tab="padim" style="background: none; border: none; color: #aaa; padding: 4px 12px; cursor: pointer; font-weight: 500; font-size: 0.9rem; transition: all 0.2s;">PaDiM</button>
                        </div>
                    `;
                }

                patchcoreHtml = `
                    <div class="irc-patchcore-panel">
                        ${tabsHtml}
                        <div class="irc-patchcore-header" style="margin-bottom: 12px; font-weight: 600; color: #fff;">Defect Localization</div>
                        
                        <div class="irc-patchcore-grid">
                            <div class="irc-pc-box">
                                <span class="irc-pc-lbl">Original</span>
                                <img src="${item.original_url}" alt="Original" class="irc-pc-img">
                            </div>
                            <div class="irc-pc-box">
                                <span class="irc-pc-lbl">Heatmap</span>
                                <img src="${item.heatmap_url}" alt="Heatmap" class="irc-loc-heatmap irc-pc-img">
                            </div>
                            <div class="irc-pc-box">
                                <span class="irc-pc-lbl">Overlay & BBox</span>
                                <img src="${item.overlay_url}" alt="Overlay & Bounding Box" class="irc-loc-overlay irc-pc-img">
                            </div>
                        </div>

                        <div class="irc-pc-metrics">
                            <div class="irc-pc-metric">
                                <span class="irc-loc-score-lbl irc-pc-metric-lbl">Max Patch Score</span>
                                <span class="irc-loc-score-val irc-pc-metric-val">${item.max_patch_score.toFixed(4)}</span>
                            </div>
                            <div class="irc-pc-metric">
                                <span class="irc-pc-metric-lbl">Anomaly Area %</span>
                                <span class="irc-loc-area-val irc-pc-metric-val">${item.anomaly_area_percent.toFixed(2)}%</span>
                            </div>
                            <div class="irc-pc-metric cell-full">
                                <span class="irc-pc-metric-lbl">Detected Region (Bounding Box)</span>
                                <span class="irc-loc-bbox-val irc-pc-metric-val font-mono">[${item.bounding_box.join(', ')}]</span>
                            </div>
                            <div class="irc-pc-metric cell-full">
                                <span class="irc-pc-metric-lbl">Centroid</span>
                                <span class="irc-loc-centroid-val irc-pc-metric-val font-mono">[${item.centroid.join(', ')}]</span>
                            </div>
                        </div>

                        <details class="irc-patch-matches-details">
                            <summary class="irc-patch-matches-toggle">View Top 5 Anomalous Patch Matches</summary>
                            <div class="irc-patch-matches-list">
                                ${patchMatchesHtml}
                            </div>
                        </details>
                    </div>
                `;
            }

            let reasonsHtml = '';
            if (hasGrade && item.product_grade.reasons && item.product_grade.reasons.length > 0) {
                reasonsHtml += `
                    <div class="irc-reasons-box" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08);">
                        <span style="font-size: 0.8rem; font-weight: 600; color: #aaa;">Decision Reasons:</span>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px; font-size: 0.82rem; color: #ddd; line-height: 1.4;">
                `;
                item.product_grade.reasons.forEach(r => {
                    reasonsHtml += `<li style="margin-bottom: 4px;">${r}</li>`;
                });
                reasonsHtml += `
                        </ul>
                    </div>
                `;
            }

            card.innerHTML = `
                <!-- Product Decision Header -->
                <div class="irc-grade-header" style="margin: 0 0 15px 0; padding: 14px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 0.72rem; text-transform: uppercase; color: #888; letter-spacing: 0.5px; font-weight: 600;">Product Grade Decision</span>
                            <div style="font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px; color: ${gradeColor}; margin-top: 2px;">
                                ${gradeEmoji} ${grade}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.72rem; text-transform: uppercase; color: #888; letter-spacing: 0.5px; font-weight: 600;">Decision Confidence</span>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-top: 4px;">
                                ${(gradeConf * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                    ${reasonsHtml}
                </div>

                <!-- Technical Evidence Subtitle -->
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #666; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 10px;">
                    Technical Inspection Evidence
                </div>

                <div class="irc-main" style="margin-top: 0;">
                    <!-- Raw Status Badge -->
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 0.85rem; color: #bbb;">
                        <span>Global Image-Level Status:</span>
                        <div class="irc-badge badge-${item.prediction.toLowerCase()}" style="margin: 0;">
                            ${item.prediction === 'Normal' ? '✅' : item.prediction === 'Suspicious' ? '⚠️' : '❌'}
                            ${item.prediction}
                        </div>
                    </div>

                    <div class="irc-stats-panel">
                        <div class="irc-stat-row">
                            <span class="irc-stat-label">Anomaly Score</span>
                            <div class="irc-stat-progress">
                                <div class="irc-stat-bar bar-${item.prediction.toLowerCase()}" style="width: ${item.anomaly_score}%"></div>
                            </div>
                            <span class="irc-stat-val">${item.anomaly_score.toFixed(1)}/100</span>
                        </div>
                    </div>

                    <!-- Compact Localization Evidence Summary -->
                    <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 6px; font-size: 0.82rem; color: #aaa; display: flex; flex-direction: column; gap: 6px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>PatchCore Localizer:</span>
                            <strong style="color: ${item.patchcore_enabled ? (item.anomaly_area_percent >= 1.0 ? '#f87171' : '#34d399') : '#888'};">
                                ${item.patchcore_enabled ? (item.anomaly_area_percent >= 1.0 ? 'Defective' : 'Normal') + ` (${item.anomaly_area_percent.toFixed(2)}%)` : 'Disabled'}
                            </strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>PaDiM Localizer:</span>
                            <strong style="color: ${item.padim && item.padim.enabled ? (item.padim.anomaly_area_percent >= 1.0 ? '#f87171' : '#34d399') : '#888'};">
                                ${item.padim && item.padim.enabled ? (item.padim.anomaly_area_percent >= 1.0 ? 'Defective' : 'Normal') + ` (${item.padim.anomaly_area_percent.toFixed(2)}%)` : 'Disabled'}
                            </strong>
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
                            <span class="irc-metric-val font-mono" title="${item.nearest_reference}" style="word-break: break-all; white-space: normal;">${item.nearest_reference}</span>
                        </div>
                    </div>
                </div>

                ${patchcoreHtml}

                <details class="irc-neighbors-details">
                    <summary class="irc-neighbors-toggle">View Top 5 Nearest Reference Neighbors</summary>
                    <div class="irc-neighbors-list">
                        ${neighborsHtml}
                    </div>
                </details>
            `;

            // Bind tab switching events
            const tabButtons = card.querySelectorAll('.irc-tab-btn');
            tabButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    tabButtons.forEach(b => {
                        b.classList.remove('active');
                        b.style.color = '#aaa';
                        b.style.borderBottom = 'none';
                    });
                    btn.classList.add('active');
                    btn.style.color = '#fff';
                    btn.style.borderBottom = '2px solid #3b82f6';

                    const tab = btn.dataset.tab;
                    const heatmapImg = card.querySelector('.irc-loc-heatmap');
                    const overlayImg = card.querySelector('.irc-loc-overlay');
                    const scoreLbl = card.querySelector('.irc-loc-score-lbl');
                    const scoreVal = card.querySelector('.irc-loc-score-val');
                    const areaVal = card.querySelector('.irc-loc-area-val');
                    const bboxVal = card.querySelector('.irc-loc-bbox-val');
                    const centroidVal = card.querySelector('.irc-loc-centroid-val');
                    const matchesDetails = card.querySelector('.irc-patch-matches-details');

                    if (tab === 'patchcore') {
                        heatmapImg.src = item.heatmap_url;
                        overlayImg.src = item.overlay_url;
                        scoreLbl.textContent = 'Max Patch Score';
                        scoreVal.textContent = item.max_patch_score.toFixed(4);
                        areaVal.textContent = `${item.anomaly_area_percent.toFixed(2)}%`;
                        bboxVal.textContent = `[${item.bounding_box.join(', ')}]`;
                        centroidVal.textContent = `[${item.centroid.join(', ')}]`;
                        if (matchesDetails) matchesDetails.style.display = 'block';
                    } else if (tab === 'padim') {
                        if (item.padim && item.padim.enabled) {
                            heatmapImg.src = item.padim.heatmap_url;
                            overlayImg.src = item.padim.overlay_url;
                            scoreLbl.textContent = 'PaDiM Score (Top-5% Mean)';
                            scoreVal.textContent = item.padim.image_score.toFixed(4);
                            areaVal.textContent = `${item.padim.anomaly_area_percent.toFixed(2)}%`;
                            bboxVal.textContent = `[${item.padim.bounding_box.join(', ')}]`;
                            centroidVal.textContent = `[${item.padim.centroid.join(', ')}]`;
                        } else {
                            scoreVal.textContent = 'N/A';
                            areaVal.textContent = 'N/A';
                            bboxVal.textContent = '[]';
                            centroidVal.textContent = '[]';
                        }
                        if (matchesDetails) matchesDetails.style.display = 'none';
                    }
                });
            });

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
