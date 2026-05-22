// Food Trivia Array for Loading Experience
const FOOD_TRIVIA = [
    "Our model was trained on the massive Food101 dataset containing 101 distinct food classes.",
    "Baklava is a rich, sweet dessert pastry made of layers of filo filled with chopped nuts and sweetened with syrup, popularized in Ottoman Palace cuisine.",
    "The word 'pizza' was first documented in 997 AD in Gaeta, a city in southern Italy.",
    "Tacos are a traditional Mexican dish dating back long before the Spanish conquest.",
    "Sushi originated as an ancient method of preserving fish in fermented rice.",
    "Despite the name, French fries are widely believed to have originated in Belgium.",
    "The name 'hamburger' comes from Hamburg, Germany, brought over by immigrants who introduced the recipe to America.",
    "The earliest forms of ice cream date back to ancient China, where a frozen mixture of milk and rice was eaten."
];

// Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const selectBtn = document.getElementById('selectBtn');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const clearBtn = document.getElementById('clearBtn');
const uploadPrompt = document.getElementById('uploadPrompt');
const predictBtn = document.getElementById('predictBtn');

const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const triviaText = document.getElementById('triviaText');
const predictionResults = document.getElementById('predictionResults');

const topPredictionName = document.getElementById('topPredictionName');
const topPredictionConfidence = document.getElementById('topPredictionConfidence');
const predictionsList = document.getElementById('predictionsList');

let selectedFile = null;
let triviaInterval = null;

// Trigger file select
selectBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent trigger uploadZone click
    fileInput.click();
});

uploadZone.addEventListener('click', () => {
    if (!selectedFile) {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', handleFileSelect);

// Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith('image/')) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        } else {
            alert('Please upload a valid image file.');
        }
    }
});

// Clear selection
clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Reset function
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.style.display = 'none';
    uploadPrompt.style.display = 'block';
    predictBtn.disabled = true;
    
    // Reset results pane to empty state
    emptyState.style.display = 'flex';
    loadingState.style.display = 'none';
    predictionResults.style.display = 'none';
    clearInterval(triviaInterval);
}

// File Select handler
function handleFileSelect() {
    if (fileInput.files.length > 0) {
        selectedFile = fileInput.files[0];
        
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            uploadPrompt.style.display = 'none';
            previewContainer.style.display = 'flex';
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(selectedFile);
    }
}

// Predict Action
predictBtn.addEventListener('click', analyzeImage);

function startTriviaRotation() {
    let index = 0;
    triviaText.textContent = FOOD_TRIVIA[index];
    
    triviaInterval = setInterval(() => {
        index = (index + 1) % FOOD_TRIVIA.length;
        triviaText.style.opacity = 0;
        setTimeout(() => {
            triviaText.textContent = FOOD_TRIVIA[index];
            triviaText.style.opacity = 1;
        }, 300);
    }, 4500);
}

function analyzeImage() {
    if (!selectedFile) return;

    // Show loading state
    emptyState.style.display = 'none';
    predictionResults.style.display = 'none';
    loadingState.style.display = 'flex';
    
    startTriviaRotation();
    
    // Prepare data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // API Request
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('A server error occurred or the image could not be processed.');
        }
        return response.json();
    })
    .then(data => {
        clearInterval(triviaInterval);
        
        if (data.success && data.predictions && data.predictions.length > 0) {
            displayResults(data.predictions);
        } else {
            throw new Error(data.error || 'An unexpected error occurred.');
        }
    })
    .catch(error => {
        clearInterval(triviaInterval);
        loadingState.style.display = 'none';
        emptyState.style.display = 'flex';
        alert('Error: ' + error.message);
    });
}

function displayResults(predictions) {
    // Hide loading
    loadingState.style.display = 'none';
    
    // Display results panel
    predictionResults.style.display = 'flex';
    
    // Primary Prediction (Index 0)
    const primary = predictions[0];
    topPredictionName.textContent = primary.class_name;
    topPredictionConfidence.textContent = `${(primary.probability * 100).toFixed(1)}%`;
    
    // Other predictions list (Top-5 breakdown)
    predictionsList.innerHTML = '';
    
    predictions.forEach((pred, index) => {
        const percentage = (pred.probability * 100).toFixed(1);
        
        const row = document.createElement('div');
        row.className = 'prediction-row';
        
        row.innerHTML = `
            <div class="prediction-info">
                <span class="class-label">${index + 1}. ${pred.class_name}</span>
                <span class="confidence-val">${percentage}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
        `;
        
        predictionsList.appendChild(row);
        
        // Trigger reflow for progress bar animation
        setTimeout(() => {
            row.querySelector('.progress-fill').style.width = `${percentage}%`;
        }, 50 + (index * 50));
    });
}
