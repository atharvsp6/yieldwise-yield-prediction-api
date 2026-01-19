// ============================================================================
// YieldWise Frontend - JavaScript Logic
// ============================================================================

const API_BASE = '';  // Use relative paths
let predictions = [];
const MAX_RECENT = 5;
let formOptions = {}; // Store loaded form options

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Set current year
    document.getElementById('currentYear').textContent = new Date().getFullYear();
    
    // Load form options
    loadFormOptions();
    
    // Check API health
    checkHealth();
    
    // Load model info
    loadModelInfo();
    
    // Set up event listeners
    setupEventListeners();
    
    // Restore previous predictions from localStorage
    restorePredictions();
    
    // Refresh health every 30 seconds
    setInterval(checkHealth, 30000);
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Form submission
    document.getElementById('predictionForm').addEventListener('submit', handleFormSubmit);
    
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', handleTabClick);
    });
    
    // State change to update districts
    document.getElementById('state').addEventListener('change', updateDistricts);
}

// ============================================================================
// Form Options Loading
// ============================================================================

async function loadFormOptions() {
    try {
        const response = await fetch('/static/form_options.json');
        formOptions = await response.json();
        
        // Populate states
        populateDropdown('state', formOptions.states);
        
        // Populate seasons
        populateDropdown('season', formOptions.seasons);
        
        // Populate crops
        populateDropdown('crop', formOptions.crops);
        
        // Update districts for default state
        updateDistricts();
        
    } catch (error) {
        console.error('Failed to load form options:', error);
    }
}

function populateDropdown(elementId, options) {
    const select = document.getElementById(elementId);
    // Clear existing options except the first
    select.innerHTML = `<option value="">Select ${elementId.charAt(0).toUpperCase() + elementId.slice(1)}...</option>`;
    
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option.toLowerCase().replace(/\s+/g, ' ');
        opt.textContent = option;
        select.appendChild(opt);
    });
}

function updateDistricts() {
    const stateSelect = document.getElementById('state');
    const districtSelect = document.getElementById('district');
    const selectedState = stateSelect.value;
    
    // Clear existing districts
    districtSelect.innerHTML = '<option value="">Select District...</option>';
    
    if (selectedState && formOptions.state_districts[selectedState]) {
        formOptions.state_districts[selectedState].forEach(district => {
            const opt = document.createElement('option');
            opt.value = district.toLowerCase().replace(/\s+/g, ' ');
            opt.textContent = district;
            districtSelect.appendChild(opt);
        });
    }
}

function handleTabClick(event) {
    const tabName = event.target.getAttribute('data-tab');
    
    // Remove active class from all buttons and panes
            const summaryHtml = `
            <div class="result-summary-item">
                <span class="result-summary-label">🌍 Crop</span>
                <span class="result-summary-value">${input.crop.toUpperCase()} (${input.season})</span>
            </div>
            <div class="result-summary-item">
                <span class="result-summary-label">📍 Location</span>
                <span class="result-summary-value">${input.district.toUpperCase()}, ${input.state.toUpperCase()}</span>
            </div>
            <div class="result-summary-item">
                <span class="result-summary-label">📅 Year</span>
                <span class="result-summary-value">${input.year}</span>
            </div>
            <div class="result-summary-item">
                <span class="result-summary-label">🌱 Area</span>
                <span class="result-summary-value">${input.area.toLocaleString()} hectares</span>
            </div>
        `;
    document.getElementById('submitBtn').disabled = true;
    
    try {
        // Collect form data
        const formData = new FormData(document.getElementById('predictionForm'));
        
        // Validate all required fields
        const state = formData.get('state');
        const crop = formData.get('crop');
        const season = formData.get('season');
        
        if (!state) {
            throw new Error('State is required');
        }
        if (!crop) {
            throw new Error('Crop is required');
        }
        if (!season) {
            throw new Error('Season is required');
        }
        
        const payload = {
            state: state,
            district: formData.get('district'),
            crop: crop,
            season: season,
            year: parseInt(formData.get('year')),
            area: parseFloat(formData.get('area')),
            rainfall: parseFloat(formData.get('rainfall')),
            seasonal_rainfall: parseFloat(formData.get('seasonal_rainfall')),
            temperature: parseFloat(formData.get('temperature')),
            humidity: parseFloat(formData.get('humidity')),
            soil_ph: parseFloat(formData.get('soil_ph')),
            fertilizer: parseFloat(formData.get('fertilizer')),
            pesticide: parseFloat(formData.get('pesticide'))
        };
        
        console.log('Sending prediction request:', payload);
        
        // Send request
        const response = await fetch(`/predict-yield`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Display result
        displayResult(result, payload);
        
        // Add to recent predictions
        addRecentPrediction(payload, result);
        
    } catch (error) {
        showError(`Error: ${error.message}`);
        console.error('Prediction error:', error);
    } finally {
        document.getElementById('loadingContainer').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;
    }
}

function displayResult(result, input) {
    const container = document.getElementById('resultContainer');
    
    // Update yield value (convert kg/ha to tonnes/ha)
    const yieldTonnes = result.predicted_yield / 1000;
    document.getElementById('resultYield').textContent = yieldTonnes.toFixed(2);
    
    // Update summary
    const summaryHtml = `
        <div class="result-summary-item">
            <span class="result-summary-label">🌍 Crop</span>
            <span class="result-summary-value">${input.crop.toUpperCase()} (${input.season})</span>
        </div>
        <div class="result-summary-item">
            <span class="result-summary-label">📍 Location</span>
            <span class="result-summary-value">${input.district}, ${input.state}</span>
        </div>
        <div class="result-summary-item">
            <span class="result-summary-label">📅 Year</span>
            <span class="result-summary-value">${input.year}</span>
        </div>
        <div class="result-summary-item">
            <span class="result-summary-label">🌱 Area</span>
            <span class="result-summary-value">${input.area} hectares</span>
        </div>
    `;
    document.getElementById('resultSummary').innerHTML = summaryHtml;
    
    // Update metadata
    const metadataHtml = `
        <p><strong>${result.model_name}</strong> | Status: ${result.status} | Unit: tonnes/ha</p>
    `;
    document.getElementById('resultMetadata').innerHTML = metadataHtml;
    
    // Show container
    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    const container = document.getElementById('errorContainer');
    document.getElementById('errorText').textContent = message;
    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================================
// Health Check
// ============================================================================

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusText = document.getElementById('statusText');
        const statusDot = document.querySelector('.status-dot');
        
        if (data.status === 'healthy' && data.model_loaded) {
            statusText.textContent = 'API Healthy';
            statusDot.style.background = '#2ecc71';
        } else {
            statusText.textContent = 'API Warning';
            statusDot.style.background = '#f39c12';
        }
    } catch (error) {
        document.getElementById('statusText').textContent = 'API Offline';
        document.querySelector('.status-dot').style.background = '#e74c3c';
        console.error('Health check failed:', error);
    }
}

// ============================================================================
// Model Info
// ============================================================================

async function loadModelInfo() {
    try {
        const response = await fetch(`/model-info`);
        const data = await response.json();
        
        const modelType = data.model_type || 'Unknown';
        const features = data.features?.categorical?.length || 0;
        
        const infoHtml = `
            <p><strong>Model:</strong> ${modelType}</p>
            <p><strong>Features:</strong> ${features} inputs</p>
            <p><strong>Status:</strong> ✅ Ready</p>
        `;
        document.getElementById('modelInfo').innerHTML = infoHtml;
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

async function loadModelInfoFull() {
    try {
        const response = await fetch(`/model-info`);
        const data = await response.json();
        
        const infoHtml = `
            <h3>📊 Model Configuration</h3>
            <p><strong>Model Type:</strong> ${data.model_type}</p>
            <p><strong>Target:</strong> ${data.target}</p>
            
            <h3>🎯 Input Features</h3>
            
            <strong>Categorical Features:</strong>
            <ul>
                ${data.features.categorical.map(f => `<li>${f}</li>`).join('')}
            </ul>
            
            <strong>Numerical Features:</strong>
            <ul>
                ${data.features.numerical.map(f => `<li>${f}</li>`).join('')}
            </ul>
            
            <h3>📈 Validation Metrics</h3>
            <ul>
                ${Object.entries(data.validation_metrics).map(([key, value]) => 
                    `<li><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}</li>`
                ).join('')}
            </ul>
            
            <h3>ℹ️ Training Info</h3>
            <ul>
                ${Object.entries(data.training_info).map(([key, value]) => 
                    `<li><strong>${key}:</strong> ${value}</li>`
                ).join('')}
            </ul>
        `;
        document.getElementById('modelInfoFull').innerHTML = infoHtml;
    } catch (error) {
        console.error('Failed to load full model info:', error);
        document.getElementById('modelInfoFull').innerHTML = '<p>Failed to load model information</p>';
    }
}

// ============================================================================
// Recent Predictions
// ============================================================================

function addRecentPrediction(input, result) {
    const prediction = {
        timestamp: new Date().toLocaleTimeString(),
        crop: input.crop,
        yield: result.predicted_yield,
        location: `${input.district}, ${input.state}`
    };
    
    predictions.unshift(prediction);
    if (predictions.length > MAX_RECENT) {
        predictions.pop();
    }
    
    // Save to localStorage
    localStorage.setItem('yieldwise_predictions', JSON.stringify(predictions));
    
    // Update UI
    updateRecentPredictionsList();
}

function restorePredictions() {
    const stored = localStorage.getItem('yieldwise_predictions');
    if (stored) {
        try {
            predictions = JSON.parse(stored);
            updateRecentPredictionsList();
        } catch (error) {
            console.error('Failed to restore predictions:', error);
        }
    }
}

function updateRecentPredictionsList() {
    const container = document.getElementById('recentPredictions');
    
    if (predictions.length === 0) {
        container.innerHTML = '<p class="empty-state">No predictions yet</p>';
        return;
    }
    
    const html = predictions.map(pred => `
        <div class="recent-item">
            <div class="recent-item-crop">${pred.crop.toUpperCase()}</div>
            <div class="recent-item-yield">
                📊 ${pred.yield.toFixed(2)} kg/ha | ${pred.location}
            </div>
            <div class="recent-item-yield" style="font-size: 0.75rem; color: #bdc3c7;">
                ${pred.timestamp}
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

// ============================================================================
// Utilities
// ============================================================================

function formatNumber(num, decimals = 2) {
    return num.toFixed(decimals);
}
