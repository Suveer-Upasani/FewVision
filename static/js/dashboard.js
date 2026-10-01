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

    // -------------------------------------------------------------------------
    // Mode tab switching — Image / Video / Live Webcam
    // -------------------------------------------------------------------------
    const tabImage = document.getElementById('tabImage');
    const tabVideo = document.getElementById('tabVideo');
    const tabWebcam = document.getElementById('tabWebcam');
    const imageModePanel = document.getElementById('imageModePanel');
    const videoModePanel = document.getElementById('videoModePanel');
    const webcamModePanel = document.getElementById('webcamModePanel');

    function switchMode(activeTab, activePanel) {
        [tabImage, tabVideo, tabWebcam].forEach(t => {
            if (t) {
                const isActive = (t === activeTab);
                t.classList.toggle('mode-tab-active', isActive);
                t.setAttribute('aria-selected', isActive ? 'true' : 'false');
            }
        });
        [imageModePanel, videoModePanel, webcamModePanel].forEach(p => {
            if (p) p.classList.toggle('hidden', p !== activePanel);
        });

        // Hide other result sections when switching
        const ir = document.getElementById('inspectionResultsSection');
        const vr = document.getElementById('videoResultsSection');
        if (activePanel !== imageModePanel && ir) ir.classList.add('hidden');
        if (activePanel !== videoModePanel && vr) vr.classList.add('hidden');
    }

    if (tabImage) tabImage.addEventListener('click', () => switchMode(tabImage, imageModePanel));
    if (tabVideo) tabVideo.addEventListener('click', () => switchMode(tabVideo, videoModePanel));
    if (tabWebcam) tabWebcam.addEventListener('click', () => {
        switchMode(tabWebcam, webcamModePanel);
        if (window.initWebcamDevices) window.initWebcamDevices();
    });


    // -------------------------------------------------------------------------
    // Video dropzone — file selection
    // -------------------------------------------------------------------------
    const videoDropzone = document.getElementById('videoDropzone');
    const videoFileInput = document.getElementById('videoFileInput');
    const videoBrowseBtn = document.getElementById('videoBrowseBtn');
    const videoPreviewSection = document.getElementById('videoPreviewSection');
    const videoPreviewName = document.getElementById('videoPreviewName');
    const videoClearBtn = document.getElementById('videoClearBtn');
    const videoConfigSection = document.getElementById('videoConfigSection');
    const videoActionSection = document.getElementById('videoActionSection');
    const inspectVideoBtn = document.getElementById('inspectVideoBtn');
    const videoInspectProgress = document.getElementById('videoInspectProgress');

    let selectedVideoFile = null;
    const VIDEO_EXTS = ['.mp4', '.mov', '.avi', '.mkv'];

    function isVideoFile(name) {
        const ext = name.substring(name.lastIndexOf('.')).toLowerCase();
        return VIDEO_EXTS.includes(ext);
    }

    function handleVideoFile(file) {
        if (!isVideoFile(file.name)) {
            showError('Unsupported file type. Please upload MP4, MOV, AVI, or MKV.');
            return;
        }
        selectedVideoFile = file;
        videoPreviewName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        videoPreviewSection.classList.remove('hidden');
        videoConfigSection.classList.remove('hidden');
        videoActionSection.classList.remove('hidden');
    }

    function clearVideoSelection() {
        selectedVideoFile = null;
        videoFileInput.value = '';
        videoPreviewSection.classList.add('hidden');
        videoConfigSection.classList.add('hidden');
        videoActionSection.classList.add('hidden');
    }

    if (videoDropzone && videoFileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            videoDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                videoDropzone.classList.add('drag-active');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            videoDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                videoDropzone.classList.remove('drag-active');
            }, false);
        });

        videoDropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) handleVideoFile(files[0]);
        });

        if (videoBrowseBtn) {
            videoBrowseBtn.addEventListener('click', () => videoFileInput.click());
        }

        videoFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleVideoFile(e.target.files[0]);
        });

        if (videoClearBtn) {
            videoClearBtn.addEventListener('click', clearVideoSelection);
        }
    }

    // -------------------------------------------------------------------------
    // Video inspect button
    // -------------------------------------------------------------------------
    if (inspectVideoBtn) {
        inspectVideoBtn.addEventListener('click', async () => {
            if (!selectedVideoFile) return;

            const sessionId = inspectVideoBtn.dataset.session;
            const formData = new FormData();
            formData.append('session_id', sessionId);
            formData.append('files', selectedVideoFile);

            // Optionally pass config params as custom headers (server uses config.py defaults).
            // These are informational only — actual behavior is driven by config.py env vars.
            const interval = document.getElementById('vcfgInterval');
            const maxFrames = document.getElementById('vcfgMaxFrames');
            const aggregation = document.getElementById('vcfgAggregation');
            const topK = document.getElementById('vcfgTopK');
            if (interval) formData.append('frame_interval', interval.value);
            if (maxFrames) formData.append('max_frames', maxFrames.value);
            if (aggregation) formData.append('aggregation', aggregation.value);
            if (topK) formData.append('top_k', topK.value);

            inspectVideoBtn.disabled = true;
            if (videoInspectProgress) videoInspectProgress.classList.remove('hidden');
            const vr = document.getElementById('videoResultsSection');
            if (vr) vr.classList.add('hidden');

            try {
                const response = await fetch('/api/inspect', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Video inspection failed');
                }

                if (data.input_type === 'video') {
                    renderVideoResults(data);
                } else {
                    // Fallback: should not happen if the file was a video
                    renderInspectionResults(data);
                }

                clearVideoSelection();

            } catch (err) {
                showError(err.message);
            } finally {
                inspectVideoBtn.disabled = false;
                if (videoInspectProgress) videoInspectProgress.classList.add('hidden');
            }
        });
    }

    // -------------------------------------------------------------------------
    // Render video results
    // -------------------------------------------------------------------------
    function renderVideoResults(data) {
        const section = document.getElementById('videoResultsSection');
        if (!section) return;

        // --- Summary stats ---
        const score = data.video_score;
        const vstScore = document.getElementById('vstVideoScore');
        const vstProcessed = document.getElementById('vstProcessed');
        const vstAnomalous = document.getElementById('vstAnomalous');
        const vstPersistence = document.getElementById('vstPersistence');
        const vstStatus = document.getElementById('vstStatus');
        const vstAggregation = document.getElementById('vstAggregation');

        if (vstScore) vstScore.textContent = typeof score === 'number' ? score.toFixed(2) : '—';
        if (vstProcessed) vstProcessed.textContent = data.processed_frames ?? '—';
        if (vstAnomalous) vstAnomalous.textContent = data.anomalous_frames ?? '—';
        if (vstPersistence) {
            const pct = typeof data.anomaly_persistence === 'number'
                ? (data.anomaly_persistence * 100).toFixed(1) + '%'
                : '—';
            vstPersistence.textContent = pct;
        }

        if (vstStatus) {
            const statusMap = {
                NORMAL: 'prediction-normal',
                SUSPICIOUS: 'prediction-suspicious',
                ANOMALY: 'prediction-anomalous',
            };
            vstStatus.textContent = data.status ?? '—';
            vstStatus.className = 'video-stat-val prediction-badge ' + (statusMap[data.status] || '');
        }

        if (vstAggregation) vstAggregation.textContent = data.aggregation_method ?? '—';

        // --- Frame timeline ---
        renderFrameTimeline(data.frame_results || []);

        // Show section
        section.classList.remove('hidden');
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // -------------------------------------------------------------------------
    // Frame timeline rendering
    // -------------------------------------------------------------------------
    function renderFrameTimeline(frames) {
        const strip = document.getElementById('timelineStrip');
        if (!strip) return;
        strip.innerHTML = '';

        frames.forEach((frame) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'timeline-frame' + (frame.is_anomaly ? ' timeline-frame-anomalous' : '');
            chip.setAttribute('role', 'listitem');
            chip.setAttribute('aria-label',
                `Frame ${frame.frame_index} — ${frame.is_anomaly ? 'Anomalous' : 'Normal'} — Score: ${frame.score.toFixed(3)}`
            );
            chip.innerHTML = `
                <span class="tf-index">${frame.frame_index}</span>
                <span class="tf-score">${frame.score.toFixed(2)}</span>
            `;
            chip.addEventListener('click', () => showFrameDetail(frame));
            strip.appendChild(chip);
        });
    }

    // -------------------------------------------------------------------------
    // Frame detail panel
    // -------------------------------------------------------------------------
    function showFrameDetail(frame) {
        const panel = document.getElementById('frameDetailPanel');
        const title = document.getElementById('frameDetailTitle');
        const imagesEl = document.getElementById('frameDetailImages');
        const metricsEl = document.getElementById('frameDetailMetrics');
        if (!panel) return;

        if (title) {
            title.textContent =
                `Frame ${frame.frame_index} @ ${frame.timestamp.toFixed(2)}s — ` +
                (frame.is_anomaly ? '🔴 Anomalous' : '✅ Normal');
        }

        // Images
        if (imagesEl) {
            imagesEl.innerHTML = '';

            const imgs = [
                { label: 'Original', url: frame.original_url },
                { label: 'Heatmap', url: frame.heatmap_url },
                { label: 'Overlay', url: frame.overlay_url },
            ].filter(i => i.url);

            if (imgs.length === 0) {
                imagesEl.innerHTML = '<p class="irc-no-images">No images available for this frame.</p>';
            } else {
                imgs.forEach(({ label, url }) => {
                    const wrap = document.createElement('div');
                    wrap.className = 'irc-pc-img-wrap';
                    wrap.innerHTML = `
                        <span class="irc-pc-img-lbl">${label}</span>
                        <img class="irc-pc-img" src="${url}" alt="${label} for frame ${frame.frame_index}" loading="lazy">
                    `;
                    imagesEl.appendChild(wrap);
                });
            }
        }

        // Metrics
        if (metricsEl) {
            const bbox = Array.isArray(frame.bounding_box) ? `[${frame.bounding_box.join(', ')}]` : '—';
            const centroid = Array.isArray(frame.centroid) ? `[${frame.centroid.join(', ')}]` : '—';
            metricsEl.innerHTML = `
                <div class="irc-pc-metric-row">
                    <span class="irc-pc-metric-lbl">Max Patch Score</span>
                    <span class="irc-pc-metric-val">${frame.score.toFixed(6)}</span>
                </div>
                <div class="irc-pc-metric-row">
                    <span class="irc-pc-metric-lbl">Anomaly Area</span>
                    <span class="irc-pc-metric-val">${frame.anomaly_area_percent.toFixed(2)}%</span>
                </div>
                <div class="irc-pc-metric-row">
                    <span class="irc-pc-metric-lbl">Bounding Box</span>
                    <span class="irc-pc-metric-val">${bbox}</span>
                </div>
                <div class="irc-pc-metric-row">
                    <span class="irc-pc-metric-lbl">Centroid</span>
                    <span class="irc-pc-metric-val">${centroid}</span>
                </div>
                ${frame.error ? `<div class="irc-pc-metric-row"><span class="irc-pc-metric-lbl">Error</span><span class="irc-pc-metric-val" style="color:hsl(355,70%,65%)">${frame.error}</span></div>` : ''}
            `;
        }

        panel.classList.remove('hidden');
    }

    // -------------------------------------------------------------------------
    // Live Webcam Inspection Studio Controller
    // -------------------------------------------------------------------------
    (function initWebcamStudio() {
        // Toolbar controls
        const webcamSourceSelect = document.getElementById('webcamSourceSelect');
        const cameraDeviceSelect = document.getElementById('cameraDeviceSelect');
        const cameraDeviceGroup = document.getElementById('cameraDeviceGroup');
        const webcamRoiMode = document.getElementById('webcamRoiMode');
        const webcamThresholdInput = document.getElementById('webcamThresholdInput');
        const webcamThreshVal = document.getElementById('webcamThreshVal');
        const webcamTemporalInput = document.getElementById('webcamTemporalInput');
        const webcamTemporalVal = document.getElementById('webcamTemporalVal');

        // Viewport elements
        const webcamViewport = document.getElementById('webcamViewport');
        const webcamVideo = document.getElementById('webcamVideo');
        const webcamServerStream = document.getElementById('webcamServerStream');
        const webcamOverlayCanvas = document.getElementById('webcamOverlayCanvas');
        const webcamPlaceholder = document.getElementById('webcamPlaceholder');
        const webcamVerdictBadge = document.getElementById('webcamVerdictBadge');
        const webcamWarningBanner = document.getElementById('webcamWarningBanner');

        // Action buttons
        const btnWebcamStart = document.getElementById('btnWebcamStart');
        const btnWebcamPause = document.getElementById('btnWebcamPause');
        const btnWebcamCapture = document.getElementById('btnWebcamCapture');
        const btnWebcamReset = document.getElementById('btnWebcamReset');

        // Telemetry elements
        const telemetryAnomalyScore = document.getElementById('telemetryAnomalyScore');
        const telemetryRawScore = document.getElementById('telemetryRawScore');
        const telemetryThresholdVal = document.getElementById('telemetryThresholdVal');
        const telemetryAreaPct = document.getElementById('telemetryAreaPct');
        const anomalyMeterFill = document.getElementById('anomalyMeterFill');
        const anomalyMeterThresholdMarker = document.getElementById('anomalyMeterThresholdMarker');
        const telemetryQualityChip = document.getElementById('telemetryQualityChip');
        const tqBlur = document.getElementById('tqBlur');
        const tqBrightness = document.getElementById('tqBrightness');
        const tqContrast = document.getElementById('tqContrast');
        const tqNoise = document.getElementById('tqNoise');
        const telemetryDevice = document.getElementById('telemetryDevice');
        const tpFPS = document.getElementById('tpFPS');
        const tpLatency = document.getElementById('tpLatency');
        const tpDinov2 = document.getElementById('tpDinov2');
        const tpPatchcore = document.getElementById('tpPatchcore');
        const tpTotalInspections = document.getElementById('tpTotalInspections');
        const tpTotalAnomalies = document.getElementById('tpTotalAnomalies');

        // Human Review elements
        const btnReviewAnomaly = document.getElementById('btnReviewAnomaly');
        const btnReviewNormal = document.getElementById('btnReviewNormal');
        const btnReviewFP = document.getElementById('btnReviewFP');
        const btnReviewFN = document.getElementById('btnReviewFN');
        const btnReviewUncertain = document.getElementById('btnReviewUncertain');
        const reviewNotesInput = document.getElementById('reviewNotesInput');
        const reviewFeedbackToast = document.getElementById('reviewFeedbackToast');

        // History elements
        const historyCountBadge = document.getElementById('historyCountBadge');
        const webcamHistoryTbody = document.getElementById('webcamHistoryTbody');
        const btnClearHistory = document.getElementById('btnClearHistory');

        // Modal elements
        const captureModal = document.getElementById('captureModal');
        const modalBody = document.getElementById('modalBody');
        const modalCloseBtn = document.getElementById('modalCloseBtn');

        if (!btnWebcamStart) return;

        // State
        let isInspecting = false;
        let isPaused = false;
        let mediaStream = null;
        let offscreenCanvas = null;
        let offscreenCtx = null;
        let loopActive = false;
        let isProcessingFrame = false;
        let lastResult = null;
        let lastRawB64 = null;
        let serverPollInterval = null;
        let inspectionCounter = 0;
        let anomaliesCounter = 0;
        let activeSessionId = btnWebcamStart.dataset.session || '';

        // Device enumeration helper
        window.initWebcamDevices = async function () {
            if (!cameraDeviceSelect) return;
            cameraDeviceSelect.innerHTML = '';

            const mode = webcamSourceSelect ? webcamSourceSelect.value : 'browser';
            if (mode === 'browser') {
                if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                    try {
                        const devices = await navigator.mediaDevices.enumerateDevices();
                        const videoDevices = devices.filter(d => d.kind === 'videoinput');
                        if (videoDevices.length > 0) {
                            videoDevices.forEach((dev, idx) => {
                                const opt = document.createElement('option');
                                opt.value = dev.deviceId;
                                opt.textContent = dev.label || `Camera ${idx + 1}`;
                                cameraDeviceSelect.appendChild(opt);
                            });
                            return;
                        }
                    } catch (e) {
                        console.warn('enumerateDevices error:', e);
                    }
                }
                const opt = document.createElement('option');
                opt.value = 'default';
                opt.textContent = 'Default Browser Camera';
                cameraDeviceSelect.appendChild(opt);
            } else {
                // Fetch server hardware cameras
                try {
                    const resp = await fetch('/api/webcam/cameras');
                    const data = await resp.json();
                    if (data.success && data.cameras && data.cameras.length > 0) {
                        data.cameras.forEach(c => {
                            const opt = document.createElement('option');
                            opt.value = c.index;
                            opt.textContent = c.name;
                            cameraDeviceSelect.appendChild(opt);
                        });
                        return;
                    }
                } catch (e) {
                    console.warn('fetch cameras error:', e);
                }
                const opt = document.createElement('option');
                opt.value = '0';
                opt.textContent = 'Default Camera Index 0';
                cameraDeviceSelect.appendChild(opt);
            }
        };

        // Source change handler
        if (webcamSourceSelect) {
            webcamSourceSelect.addEventListener('change', () => {
                if (isInspecting) stopInspection();
                window.initWebcamDevices();
            });
        }

        // Sliders
        if (webcamThresholdInput && webcamThreshVal) {
            webcamThresholdInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value).toFixed(2);
                webcamThreshVal.textContent = val;
                if (telemetryThresholdVal) telemetryThresholdVal.textContent = val;
                if (anomalyMeterThresholdMarker) {
                    anomalyMeterThresholdMarker.style.left = `${Math.min(100, (parseFloat(val) / 1.0) * 100)}%`;
                }
                // Send threshold update to server
                fetch('/api/webcam/threshold', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold: parseFloat(val) }),
                }).catch(() => {});
            });
        }

        if (webcamTemporalInput && webcamTemporalVal) {
            webcamTemporalInput.addEventListener('input', (e) => {
                webcamTemporalVal.textContent = e.target.value;
            });
        }

        // Start / Stop toggle
        btnWebcamStart.addEventListener('click', () => {
            if (!isInspecting) {
                startInspection();
            } else {
                stopInspection();
            }
        });

        // Pause button
        if (btnWebcamPause) {
            btnWebcamPause.addEventListener('click', () => {
                isPaused = !isPaused;
                btnWebcamPause.classList.toggle('btn-primary', isPaused);
                btnWebcamPause.innerHTML = isPaused
                    ? `<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume`
                    : `<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`;

                const mode = webcamSourceSelect.value;
                if (mode !== 'browser') {
                    fetch('/api/webcam/pause', { method: 'POST' }).catch(() => {});
                }
            });
        }

        // Capture snapshot button
        if (btnWebcamCapture) {
            btnWebcamCapture.addEventListener('click', async () => {
                btnWebcamCapture.disabled = true;
                btnWebcamCapture.textContent = 'Saving…';
                try {
                    let payload = {
                        result_meta: lastResult || {},
                    };

                    if (lastRawB64) {
                        payload.raw_frame = lastRawB64;
                    }
                    if (lastResult && lastResult.overlay_base64) {
                        payload.annotated_frame = lastResult.overlay_base64;
                    }
                    if (lastResult && lastResult.heatmap_base64) {
                        payload.heatmap_frame = lastResult.heatmap_base64;
                    }

                    const resp = await fetch('/api/webcam/capture', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                    const res = await resp.json();

                    if (res.success) {
                        showReviewToast(`Capture saved: #${res.capture_id}`, false);
                        loadHistory();
                    } else {
                        showReviewToast(res.error || 'Capture failed', true);
                    }
                } catch (err) {
                    showReviewToast(err.message, true);
                } finally {
                    btnWebcamCapture.disabled = false;
                    btnWebcamCapture.innerHTML = `
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><circle cx="12" cy="12" r="3"/><path d="M19 4h-3.5L14 2H10L8.5 4H5a2 2 0 00-2 2v12a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2z"/></svg>
                        Capture Snapshot
                    `;
                }
            });
        }

        // Reset counters
        if (btnWebcamReset) {
            btnWebcamReset.addEventListener('click', () => {
                inspectionCounter = 0;
                anomaliesCounter = 0;
                if (tpTotalInspections) tpTotalInspections.textContent = '0';
                if (tpTotalAnomalies) tpTotalAnomalies.textContent = '0';
            });
        }

        // Active ROI helper
        function getRoiSpec() {
            const mode = webcamRoiMode ? webcamRoiMode.value : 'center';
            if (mode === 'center') {
                return [0.10, 0.10, 0.90, 0.90]; // Centered 80% box
            }
            if (mode === 'auto') {
                return null;
            }
            return null; // Full frame
        }

        // Start Inspection implementation
        async function startInspection() {
            const source = webcamSourceSelect ? webcamSourceSelect.value : 'browser';

            btnWebcamStart.disabled = true;
            btnWebcamStart.textContent = 'Starting…';

            try {
                if (source === 'browser') {
                    // Browser WebRTC camera
                    const deviceId = cameraDeviceSelect ? cameraDeviceSelect.value : null;
                    const constraints = {
                        video: {
                            width: { ideal: 1280 },
                            height: { ideal: 720 },
                        },
                        audio: false,
                    };
                    if (deviceId && deviceId !== 'default') {
                        constraints.video.deviceId = { exact: deviceId };
                    }

                    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
                    webcamVideo.srcObject = mediaStream;
                    await webcamVideo.play();

                    webcamVideo.classList.remove('hidden');
                    webcamServerStream.classList.add('hidden');
                    webcamPlaceholder.classList.add('hidden');

                    // Setup offscreen canvas for capturing frames
                    offscreenCanvas = document.createElement('canvas');
                    offscreenCtx = offscreenCanvas.getContext('2d');

                    isInspecting = true;
                    isPaused = false;
                    loopActive = true;
                    runBrowserInspectionLoop();

                } else {
                    // Server Hardware Camera or Simulated Test Stream
                    const mode = source === 'simulated' ? 'simulated' : 'hardware';
                    const camIdx = cameraDeviceSelect ? parseInt(cameraDeviceSelect.value || '0', 10) : 0;

                    const resp = await fetch('/api/webcam/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: activeSessionId,
                            mode: mode,
                            camera_index: camIdx,
                            roi: getRoiSpec(),
                            auto_roi: webcamRoiMode && webcamRoiMode.value === 'auto',
                            target_fps: 15,
                        }),
                    });
                    const res = await resp.json();
                    if (!res.success) {
                        throw new Error(res.error || 'Failed to start camera on server.');
                    }

                    // Display MJPEG stream
                    webcamServerStream.src = `/api/webcam/stream?t=${Date.now()}`;
                    webcamServerStream.classList.remove('hidden');
                    webcamVideo.classList.add('hidden');
                    webcamPlaceholder.classList.add('hidden');

                    isInspecting = true;
                    isPaused = false;
                    startServerStatusPolling();
                }

                btnWebcamStart.classList.replace('btn-primary', 'btn-secondary');
                btnWebcamStart.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><rect x="5" y="5" width="14" height="14" rx="2"/></svg> Stop Inspection`;
                btnWebcamStart.disabled = false;
                if (btnWebcamPause) btnWebcamPause.disabled = false;
                if (btnWebcamCapture) btnWebcamCapture.disabled = false;
                updateVerdictBadge('NORMAL', 'Inspecting');

            } catch (err) {
                console.error('Start inspection error:', err);
                showError(err.message || 'Unable to access camera.');
                stopInspection();
            }
        }

        // Stop Inspection implementation
        function stopInspection() {
            loopActive = false;
            isInspecting = false;
            isPaused = false;

            if (serverPollInterval) {
                clearInterval(serverPollInterval);
                serverPollInterval = null;
            }

            if (mediaStream) {
                mediaStream.getTracks().forEach(t => t.stop());
                mediaStream = null;
            }
            if (webcamVideo) {
                webcamVideo.srcObject = null;
                webcamVideo.classList.add('hidden');
            }
            if (webcamServerStream) {
                webcamServerStream.src = '';
                webcamServerStream.classList.add('hidden');
            }
            if (webcamPlaceholder) {
                webcamPlaceholder.classList.remove('hidden');
            }

            // Clear overlay canvas
            if (webcamOverlayCanvas) {
                const ctx = webcamOverlayCanvas.getContext('2d');
                ctx.clearRect(0, 0, webcamOverlayCanvas.width, webcamOverlayCanvas.height);
            }

            // Stop server camera if running
            fetch('/api/webcam/stop', { method: 'POST' }).catch(() => {});

            btnWebcamStart.classList.replace('btn-secondary', 'btn-primary');
            btnWebcamStart.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Inspection`;
            btnWebcamStart.disabled = false;
            if (btnWebcamPause) {
                btnWebcamPause.disabled = true;
                btnWebcamPause.classList.remove('btn-primary');
                btnWebcamPause.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`;
            }
            if (btnWebcamCapture) btnWebcamCapture.disabled = true;

            updateVerdictBadge('IDLE', 'READY');
            if (webcamWarningBanner) webcamWarningBanner.classList.add('hidden');
        }

        // Controlled Frame Sampling Loop (Zero Queue Backlog)
        async function runBrowserInspectionLoop() {
            if (!loopActive) return;

            if (!isPaused && !isProcessingFrame && webcamVideo && webcamVideo.readyState >= 2) {
                isProcessingFrame = true;
                try {
                    const vw = webcamVideo.videoWidth || 640;
                    const vh = webcamVideo.videoHeight || 360;

                    // Match canvas dimensions to video
                    if (offscreenCanvas.width !== vw || offscreenCanvas.height !== vh) {
                        offscreenCanvas.width = vw;
                        offscreenCanvas.height = vh;
                    }

                    offscreenCtx.drawImage(webcamVideo, 0, 0, vw, vh);
                    const b64Frame = offscreenCanvas.toDataURL('image/jpeg', 0.85);
                    lastRawB64 = b64Frame;

                    const thresh = webcamThresholdInput ? parseFloat(webcamThresholdInput.value) : 0.50;
                    const roi = getRoiSpec();
                    const autoRoi = webcamRoiMode && webcamRoiMode.value === 'auto';

                    const resp = await fetch('/api/webcam/inspect_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            frame: b64Frame,
                            session_id: activeSessionId,
                            roi: roi,
                            auto_roi: autoRoi,
                            threshold: thresh,
                        }),
                    });

                    if (resp.ok) {
                        const res = await resp.json();
                        if (res.success) {
                            lastResult = res;
                            renderInspectionTelemetry(res);
                        }
                    }
                } catch (err) {
                    console.warn('Frame inspection network/compute error:', err);
                } finally {
                    isProcessingFrame = false;
                }
            }

            if (loopActive) {
                // Schedule next frame with a gentle timeout to allow browser UI thread to breathe
                setTimeout(() => requestAnimationFrame(runBrowserInspectionLoop), 30);
            }
        }

        // Poll server status when in hardware or simulated stream mode
        function startServerStatusPolling() {
            if (serverPollInterval) clearInterval(serverPollInterval);
            serverPollInterval = setInterval(async () => {
                if (!isInspecting) return;
                try {
                    const resp = await fetch('/api/webcam/status');
                    const data = await resp.json();
                    if (data.success && data.camera) {
                        if (tpFPS) tpFPS.textContent = data.camera.measured_fps ? data.camera.measured_fps.toFixed(1) : '—';
                        if (tpTotalInspections) tpTotalInspections.textContent = data.inspector.total_inspections;
                        if (tpTotalAnomalies) tpTotalAnomalies.textContent = data.inspector.total_anomalies;
                    }
                } catch (e) {
                    console.warn('Server status poll error:', e);
                }
            }, 800);
        }

        // Render Telemetry & Overlay Canvas
        function renderInspectionTelemetry(res) {
            inspectionCounter++;
            if (res.status === 'ANOMALY') anomaliesCounter++;

            // 1. Verdict badge
            updateVerdictBadge(res.status, res.status);

            // 2. Score & Meter
            if (telemetryAnomalyScore) telemetryAnomalyScore.textContent = res.stabilized_score.toFixed(3);
            if (telemetryRawScore) telemetryRawScore.textContent = res.score.toFixed(3);
            if (telemetryThresholdVal) telemetryThresholdVal.textContent = res.threshold.toFixed(2);
            if (telemetryAreaPct) telemetryAreaPct.textContent = `${res.anomaly_area_percent.toFixed(1)}%`;

            if (anomalyMeterFill) {
                const fillPct = Math.min(100, Math.max(0, (res.stabilized_score / 1.0) * 100));
                anomalyMeterFill.style.width = `${fillPct}%`;
            }

            // 3. Technical Quality
            if (res.quality) {
                if (telemetryQualityChip) {
                    telemetryQualityChip.textContent = res.quality.status;
                    telemetryQualityChip.className = `quality-chip chip-${res.quality.status.toLowerCase()}`;
                }
                if (tqBlur) tqBlur.textContent = res.quality.blur.toFixed(1);
                if (tqBrightness) tqBrightness.textContent = res.quality.brightness.toFixed(0);
                if (tqContrast) tqContrast.textContent = res.quality.contrast.toFixed(1);
                if (tqNoise) tqNoise.textContent = res.quality.noise.toFixed(2);

                if (webcamWarningBanner) {
                    if (res.quality.status === 'BAD' || res.status === 'QUALITY WARNING') {
                        webcamWarningBanner.textContent = res.quality.message || 'IMAGE QUALITY TOO LOW';
                        webcamWarningBanner.classList.remove('hidden');
                    } else {
                        webcamWarningBanner.classList.add('hidden');
                    }
                }
            }

            // 4. Performance & Hardware
            if (telemetryDevice) telemetryDevice.textContent = res.device || 'CPU';
            if (tpFPS) tpFPS.textContent = res.fps ? res.fps.toFixed(1) : '—';
            if (tpLatency) tpLatency.textContent = `${res.latency_ms.toFixed(0)} ms`;
            if (tpDinov2 && res.timings) tpDinov2.textContent = `${res.timings.dinov2_ms.toFixed(0)} ms`;
            if (tpPatchcore && res.timings) tpPatchcore.textContent = `${res.timings.patchcore_ms.toFixed(0)} ms`;
            if (tpTotalInspections) tpTotalInspections.textContent = res.total_inspections || inspectionCounter;
            if (tpTotalAnomalies) tpTotalAnomalies.textContent = res.total_anomalies || anomaliesCounter;

            // 5. Draw overlay onto Canvas
            if (webcamOverlayCanvas && res.overlay_base64) {
                const ctx = webcamOverlayCanvas.getContext('2d');
                const overlayImg = new Image();
                overlayImg.onload = () => {
                    const cw = webcamViewport.clientWidth || 640;
                    const ch = webcamViewport.clientHeight || 360;
                    if (webcamOverlayCanvas.width !== cw || webcamOverlayCanvas.height !== ch) {
                        webcamOverlayCanvas.width = cw;
                        webcamOverlayCanvas.height = ch;
                    }
                    ctx.clearRect(0, 0, cw, ch);
                    ctx.drawImage(overlayImg, 0, 0, cw, ch);
                };
                overlayImg.src = res.overlay_base64;
            }
        }

        function updateVerdictBadge(status, label) {
            if (!webcamVerdictBadge) return;
            webcamVerdictBadge.textContent = label;
            webcamVerdictBadge.className = 'webcam-verdict-badge';

            if (status === 'ANOMALY') {
                webcamVerdictBadge.classList.add('verdict-anomaly');
            } else if (status === 'QUALITY WARNING') {
                webcamVerdictBadge.classList.add('verdict-warning');
            } else if (status === 'NORMAL') {
                webcamVerdictBadge.classList.add('verdict-normal');
            } else {
                webcamVerdictBadge.classList.add('verdict-idle');
            }
        }

        function showReviewToast(msg, isError) {
            if (!reviewFeedbackToast) return;
            reviewFeedbackToast.textContent = msg;
            reviewFeedbackToast.style.borderColor = isError ? 'hsl(355, 75%, 45%)' : 'hsl(145, 70%, 45%)';
            reviewFeedbackToast.style.color = isError ? 'hsl(355, 80%, 75%)' : 'hsl(145, 70%, 75%)';
            reviewFeedbackToast.classList.remove('hidden');
            setTimeout(() => reviewFeedbackToast.classList.add('hidden'), 4000);
        }

        // Human Review action triggers
        async function submitReview(expertLabel) {
            const notes = reviewNotesInput ? reviewNotesInput.value.trim() : '';
            const pred = lastResult ? lastResult.status : 'NORMAL';
            const score = lastResult ? lastResult.score : 0.0;
            const qScore = (lastResult && lastResult.quality) ? lastResult.quality.quality_score : 0.0;

            const payload = {
                expert_label: expertLabel,
                model_prediction: pred,
                model_score: score,
                quality_score: qScore,
                notes: notes,
            };
            if (lastRawB64) {
                payload.frame = lastRawB64;
            }

            try {
                const resp = await fetch('/api/webcam/review', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await resp.json();
                if (data.success) {
                    showReviewToast(`Feedback saved: [${data.review_status}] in ${data.category}`, false);
                    if (reviewNotesInput) reviewNotesInput.value = '';
                    loadHistory();
                } else {
                    showReviewToast(data.error || 'Failed to save review', true);
                }
            } catch (err) {
                showReviewToast(err.message, true);
            }
        }

        if (btnReviewAnomaly) btnReviewAnomaly.addEventListener('click', () => submitReview('CONFIRM_ANOMALY'));
        if (btnReviewNormal) btnReviewNormal.addEventListener('click', () => submitReview('MARK_NORMAL'));
        if (btnReviewFP) btnReviewFP.addEventListener('click', () => submitReview('FALSE_POSITIVE'));
        if (btnReviewFN) btnReviewFN.addEventListener('click', () => submitReview('FALSE_NEGATIVE'));
        if (btnReviewUncertain) btnReviewUncertain.addEventListener('click', () => submitReview('UNCERTAIN'));

        // History Management
        async function loadHistory() {
            if (!webcamHistoryTbody) return;
            try {
                const resp = await fetch('/api/webcam/history');
                const data = await resp.json();
                if (data.success && data.history) {
                    renderHistoryTable(data.history);
                }
            } catch (e) {
                console.warn('Load history error:', e);
            }
        }

        function renderHistoryTable(items) {
            if (!webcamHistoryTbody) return;
            webcamHistoryTbody.innerHTML = '';

            if (historyCountBadge) {
                historyCountBadge.textContent = `${items.length} records`;
            }

            if (items.length === 0) {
                webcamHistoryTbody.innerHTML = `
                    <tr class="history-empty-row">
                        <td colspan="8">No inspections captured yet. Use "Capture Snapshot" or start continuous inspection.</td>
                    </tr>
                `;
                return;
            }

            items.forEach(item => {
                const tr = document.createElement('tr');
                const statusClass = item.status === 'ANOMALY' ? 'chip-anomalous' : (item.status === 'QUALITY WARNING' ? 'chip-suspicious' : 'chip-normal');
                const qualityClass = (item.quality || 'GOOD').toLowerCase();

                tr.innerHTML = `
                    <td><code>${item.id}</code></td>
                    <td>${item.timestamp}</td>
                    <td><span class="summary-chip ${statusClass}" style="padding:2px 8px;font-size:0.75rem">${item.status}</span></td>
                    <td><strong>${typeof item.score === 'number' ? item.score.toFixed(3) : item.score}</strong></td>
                    <td><span class="quality-chip chip-${qualityClass}">${item.quality || '—'}</span></td>
                    <td>${typeof item.latency_ms === 'number' ? item.latency_ms.toFixed(0) : item.latency_ms} ms</td>
                    <td><span style="font-size:0.8rem;color:hsl(220,15%,75%)">${item.expert_decision || 'Unreviewed'}</span></td>
                    <td>
                        <button type="button" class="btn-ghost btn-sm view-snap-btn" data-id="${item.id}" data-dir="${item.capture_dir || ''}">
                            View
                        </button>
                    </td>
                `;
                webcamHistoryTbody.appendChild(tr);
            });

            // Bind view buttons
            document.querySelectorAll('.view-snap-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const snapId = btn.dataset.id;
                    const snapDir = btn.dataset.dir;
                    openCaptureModal(snapId, snapDir);
                });
            });
        }

        // Clear History
        if (btnClearHistory) {
            btnClearHistory.addEventListener('click', async () => {
                if (!confirm('Clear all inspection history records?')) return;
                try {
                    await fetch('/api/webcam/history/clear', { method: 'POST' });
                    loadHistory();
                } catch (e) {
                    showError(e.message);
                }
            });
        }

        // Modal handling
        function openCaptureModal(snapId, snapDir) {
            if (!captureModal || !modalBody) return;
            modalBody.innerHTML = `
                <div style="display:flex;flex-direction:column;gap:18px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;">
                        <h4 style="margin:0;color:hsl(220,15%,90%)">Capture ID: <code>${snapId}</code></h4>
                        <span style="font-size:0.8rem;color:hsl(220,15%,60%)">Folder: ${snapDir}</span>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:16px;">
                        <div>
                            <div style="font-size:0.76rem;font-weight:700;margin-bottom:6px;color:hsl(220,15%,65%)">RAW CAPTURE</div>
                            <img src="/data/inspection_results/${snapDir}/raw.jpg" alt="Raw frame" style="width:100%;border-radius:8px;border:1px solid hsl(220,20%,20%)" onerror="this.src='/static/img/placeholder.png'">
                        </div>
                        <div>
                            <div style="font-size:0.76rem;font-weight:700;margin-bottom:6px;color:hsl(220,15%,65%)">ANNOTATED OVERLAY</div>
                            <img src="/data/inspection_results/${snapDir}/annotated.jpg" alt="Annotated frame" style="width:100%;border-radius:8px;border:1px solid hsl(220,20%,20%)" onerror="this.src='/static/img/placeholder.png'">
                        </div>
                        <div>
                            <div style="font-size:0.76rem;font-weight:700;margin-bottom:6px;color:hsl(220,15%,65%)">HEATMAP (JET)</div>
                            <img src="/data/inspection_results/${snapDir}/heatmap.jpg" alt="Heatmap" style="width:100%;border-radius:8px;border:1px solid hsl(220,20%,20%)" onerror="this.src='/static/img/placeholder.png'">
                        </div>
                    </div>
                </div>
            `;
            captureModal.classList.remove('hidden');
        }

        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', () => {
                if (captureModal) captureModal.classList.add('hidden');
            });
        }
        if (captureModal) {
            captureModal.addEventListener('click', (e) => {
                if (e.target === captureModal) captureModal.classList.add('hidden');
            });
        }

        // Initialize history on load
        loadHistory();
    })();

    // Close button for frame detail
    const frameDetailClose = document.getElementById('frameDetailClose');
    if (frameDetailClose) {
        frameDetailClose.addEventListener('click', () => {
            const panel = document.getElementById('frameDetailPanel');
            if (panel) panel.classList.add('hidden');
        });
    }

})();

