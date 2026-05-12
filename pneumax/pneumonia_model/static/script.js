document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnChange = document.getElementById('btn-change');
    const btnAnalyze = document.getElementById('btn-analyze');
    
    const resultSection = document.getElementById('result-section');
    const loader = document.getElementById('loader');
    const diagnosisCard = document.getElementById('diagnosis-card');
    
    const resultBadge = document.getElementById('result-badge');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const confidenceValue = document.getElementById('confidence-value');
    const progressBar = document.getElementById('progress-bar');
    
    let currentFile = null;

    // --- Drag and Drop Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    btnChange.addEventListener('click', () => {
        // Reset view
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        btnAnalyze.disabled = true;
        resultSection.classList.add('hidden');
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file (JPG, PNG).');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            btnAnalyze.disabled = false;
            
            // Hide previous results
            resultSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // --- Analysis Logic ---
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Updates for loading
        btnAnalyze.disabled = true;
        resultSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        diagnosisCard.classList.add('hidden');
        
        // Reset progress bar
        progressBar.style.width = '0%';
        resultBadge.className = 'result-badge'; // Reset classes

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showResult(data);
            } else {
                alert('Error analyzing image: ' + data.error);
                resultSection.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
            resultSection.classList.add('hidden');
        } finally {
            btnAnalyze.disabled = false;
            loader.classList.add('hidden');
        }
    });

    function showResult(data) {
        diagnosisCard.classList.remove('hidden');
        
        const isNormal = data.diagnosis === "NORMAL";
        const themeClass = isNormal ? 'normal' : 'pneumonia';
        
        // Update Classes
        resultBadge.classList.add(themeClass);
        diagnosisCard.classList.remove('normal', 'pneumonia'); // Clean up old
        diagnosisCard.classList.add(themeClass); // Add new class to parent for CSS targeting
        
        // Update Content
        resultText.textContent = data.diagnosis;
        confidenceValue.textContent = data.confidence.toFixed(1) + '%';
        resultIcon.className = isNormal ? 'ph-fill ph-check-circle' : 'ph-fill ph-warning-octagon';
        
        // Animate Progress Bar (small delay for transition to work)
        setTimeout(() => {
            progressBar.style.width = data.confidence + '%';
        }, 100);
    }
});
