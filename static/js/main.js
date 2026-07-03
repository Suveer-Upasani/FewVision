document.addEventListener('DOMContentLoaded', () => {
    // Test Zone Elements
    const dropzoneTest = document.getElementById('dropzone');
    const fileInputTest = document.getElementById('fileInput');
    const loadingTest = document.getElementById('loading');
    
    // Train Zone Elements
    const dropzoneTrain = document.getElementById('dropzoneTrain');
    const fileInputTrain = document.getElementById('fileInputTrain');
    const loadingTrain = document.getElementById('loadingTrain');
    const trainSuccess = document.getElementById('trainSuccess');

    // Results Elements
    const resultsSection = document.getElementById('results');
    const mlPrediction = document.getElementById('mlPrediction');
    const mlConfidence = document.getElementById('mlConfidence');
    const suitabilityRating = document.getElementById('suitabilityRating');
    const qualityScore = document.getElementById('qualityScore');
    const contentScore = document.getElementById('contentScore');
    const blurValue = document.getElementById('blurValue');
    const noiseValue = document.getElementById('noiseValue');
    const barQuality = document.getElementById('barQuality');
    const barContent = document.getElementById('barContent');
    const reportImage = document.getElementById('reportImage');

    // Setup Drag and Drop for a specific zone
    function setupDropzone(dropzone, fileInput, handleDropCallback) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
        });

        dropzone.addEventListener('drop', handleDropCallback, false);
        dropzone.addEventListener('click', () => fileInput.click());
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Initialize both dropzones
    setupDropzone(dropzoneTest, fileInputTest, (e) => handleTestFiles(e.dataTransfer.files));
    setupDropzone(dropzoneTrain, fileInputTrain, (e) => handleTrainFiles(e.dataTransfer.files));
    
    fileInputTest.addEventListener('change', function() {
        if (this.files.length > 0) handleTestFiles(this.files);
    });
    
    fileInputTrain.addEventListener('change', function() {
        if (this.files.length > 0) handleTrainFiles(this.files);
    });

    // --- TRAINING FLOW ---
    function handleTrainFiles(files) {
        if (files.length === 0) return;
        uploadTrainFiles(files);
    }

    async function uploadTrainFiles(files) {
        dropzoneTrain.classList.add('hidden');
        trainSuccess.classList.add('hidden');
        loadingTrain.classList.remove('hidden');

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            if (files[i].type.startsWith('image/')) {
                formData.append('files', files[i]);
            }
        }

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error occurred');

            loadingTrain.classList.add('hidden');
            trainSuccess.classList.remove('hidden');
            dropzoneTrain.classList.remove('hidden');
            
            // Make dropzone smaller
            dropzoneTrain.style.padding = '20px';
            dropzoneTrain.querySelector('h3').style.fontSize = '1.2rem';
            dropzoneTrain.querySelector('.upload-icon').style.width = '32px';
            dropzoneTrain.querySelector('.upload-icon').style.height = '32px';
            dropzoneTrain.querySelector('.upload-icon').style.marginBottom = '10px';

        } catch (error) {
            alert('Training failed: ' + error.message);
            dropzoneTrain.classList.remove('hidden');
            loadingTrain.classList.add('hidden');
        }
    }

    // --- TESTING FLOW ---
    function handleTestFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        uploadTestFile(file);
    }

    async function uploadTestFile(file) {
        dropzoneTest.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingTest.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error occurred');

            populateResults(data);

        } catch (error) {
            alert('Analysis failed: ' + error.message);
            dropzoneTest.classList.remove('hidden');
            loadingTest.classList.add('hidden');
        }
    }

    function populateResults(data) {
        mlPrediction.textContent = data.ml_prediction;
        mlPrediction.className = 'verdict-text'; // Reset
        if (data.ml_prediction === 'PASS') mlPrediction.classList.add('verdict-pass');
        else if (data.ml_prediction === 'DEFECT') mlPrediction.classList.add('verdict-defect');
        mlConfidence.textContent = data.ml_confidence;

        const m = data.metrics;
        suitabilityRating.textContent = m.suitability_rating;
        qualityScore.textContent = m.quality_score;
        contentScore.textContent = m.content_score;
        blurValue.textContent = m.blur;
        noiseValue.textContent = m.noise;

        setTimeout(() => {
            barQuality.style.width = `${m.quality_score}%`;
            barContent.style.width = `${m.content_score}%`;
        }, 100);

        reportImage.classList.remove('loaded');
        reportImage.onload = () => reportImage.classList.add('loaded');
        reportImage.src = `${data.report_url}?t=${new Date().getTime()}`;

        loadingTest.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        dropzoneTest.classList.remove('hidden');
        
        dropzoneTest.style.padding = '20px';
        dropzoneTest.querySelector('h3').style.fontSize = '1.2rem';
        dropzoneTest.querySelector('.upload-icon').style.width = '32px';
        dropzoneTest.querySelector('.upload-icon').style.height = '32px';
        dropzoneTest.querySelector('.upload-icon').style.marginBottom = '10px';
    }
});
