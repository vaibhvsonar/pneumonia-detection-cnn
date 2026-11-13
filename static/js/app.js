// ============================================================================
// Configuration
// ============================================================================

// Base API URL:
// - Empty string '' when frontend & backend are hosted together (e.g., Render, local)
// - Or set your cloud backend URL (e.g., 'https://your-api.onrender.com') if frontend is on Vercel
const BACKEND_URL = window.BACKEND_API_URL || '';

const API_ENDPOINTS = {
    PREDICT: `${BACKEND_URL}/predict`,
    HEALTH: `${BACKEND_URL}/health`,
    MODEL_INFO: `${BACKEND_URL}/model-info`
};

const CLASSES = {
    0: 'NORMAL',
    1: 'PNEUMONIA'
};

const CLASS_ICONS = {
    0: '✓',
    1: '⚠️'
};

const CLASS_COLORS = {
    0: 'normal',
    1: 'pneumonia'
};

// ============================================================================
// DOM Elements
// ============================================================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const modelStatus = document.getElementById('modelStatus');
const emptyState = document.getElementById('emptyState');

// ============================================================================
// Event Listeners
// ============================================================================

browseBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];

    if (file) {
        handleFile(file);
    }
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();

    uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }
});

// ============================================================================
// Functions
// ============================================================================

function handleFile(file) {
    // Validate file
    if (!['image/png', 'image/jpeg'].includes(file.type)) {
        showError('Invalid file type. Only PNG and JPEG are supported.');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }

    // Show preview
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.classList.add('show');
        emptyState.style.display = 'none';

        // Make prediction
        makePrediction(file);
    };

    reader.readAsDataURL(file);
}

function makePrediction(file) {
    loadingSpinner.classList.add('show');
    resultsSection.classList.remove('show');

    clearError();

    const formData = new FormData();
    formData.append('file', file);

    fetch(API_ENDPOINTS.PREDICT, {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loadingSpinner.classList.remove('show');

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Failed to make prediction.');
            }
        })
        .catch(error => {
            loadingSpinner.classList.remove('show');
            showError('Connection error: ' + error.message);
        });
}

function displayResults(prediction) {
    const classIndex = prediction.class_index;
    const className = CLASSES[classIndex];
    const icon = CLASS_ICONS[classIndex];
    const colorClass = CLASS_COLORS[classIndex];

    const confidence = (prediction.confidence * 100).toFixed(2);
    const normalConf = (prediction.normal_confidence * 100).toFixed(2);
    const pneumoniaConf = (prediction.pneumonia_confidence * 100).toFixed(2);

    const resultHTML = `
        <div class="prediction-result ${colorClass}">
            <div class="prediction-header">
                <div class="prediction-icon">${icon}</div>

                <div class="prediction-text">
                    <h3>${className}</h3>
                    <p>Confidence: <strong>${confidence}%</strong></p>
                </div>
            </div>

            <div class="confidence-section">
                <h4>Confidence Breakdown:</h4>

                <div class="confidence-bars">
                    <div class="confidence-bar normal">
                        <div class="confidence-label">Normal</div>

                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${normalConf}%"></div>
                        </div>

                        <div class="progress-value">${normalConf}%</div>
                    </div>

                    <div class="confidence-bar pneumonia">
                        <div class="confidence-label">Pneumonia</div>

                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${pneumoniaConf}%"></div>
                        </div>

                        <div class="progress-value">${pneumoniaConf}%</div>
                    </div>
                </div>
            </div>
        </div>
    `;

    resultsContainer.innerHTML = resultHTML;
    resultsSection.classList.add('show');
}

function resetApp() {
    fileInput.value = '';

    previewSection.classList.remove('show');
    resultsSection.classList.remove('show');
    loadingSpinner.classList.remove('show');

    clearError();

    emptyState.style.display = 'block';
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
}

function clearError() {
    errorMessage.classList.remove('show');
    errorMessage.textContent = '';
}

// ============================================================================
// Initialization
// ============================================================================

window.addEventListener('load', () => {
    fetch(API_ENDPOINTS.HEALTH)
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                modelStatus.classList.add('warning');

                document.getElementById('modelBadge').textContent = 'WARNING';

                document.getElementById('modelBadge').classList.remove('ready');

                document.getElementById('statusText').innerHTML =
                    '⚠️ Model weights not found. Using untrained model for demonstration.';
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
        });
});