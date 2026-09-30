const API_URL = "http://127.0.0.1:5000";

const imageInput = document.getElementById("imageInput");
const selectImageBtn = document.getElementById("selectImageBtn");
const changeImageBtn = document.getElementById("changeImageBtn");
const dropZone = document.getElementById("dropZone");

const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");

const loading = document.getElementById("loading");
const emptyResult = document.getElementById("emptyResult");
const resultContainer = document.getElementById("resultContainer");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

const normalProbability = document.getElementById("normalProbability");
const normalBar = document.getElementById("normalBar");
const abnormalProbability = document.getElementById("abnormalProbability");
const abnormalBar = document.getElementById("abnormalBar");

const statusBadge = document.getElementById("statusBadge");
const explanationSection = document.getElementById("explanationSection");
const gradcamImage = document.getElementById("gradcamImage");
const gradcamContainer = document.getElementById("gradcamContainer");

const errorMessage = document.getElementById("errorMessage");

let selectedFile = null;

// File Selection Handlers
selectImageBtn.addEventListener("click", () => imageInput.click());
changeImageBtn.addEventListener("click", () => imageInput.click());

// Drag and Drop Handlers
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

imageInput.addEventListener("change", function () {
    if (this.files && this.files.length > 0) {
        handleFile(this.files[0]);
    }
});

function handleFile(file) {
    const allowedTypes = ["image/bmp", "image/jpeg", "image/png"];
    if (!allowedTypes.includes(file.type)) {
        showError("Please select a BMP, JPG, JPEG or PNG image.");
        selectedFile = null;
        analyzeBtn.disabled = true;
        return;
    }

    selectedFile = file;
    clearError();

    const imageURL = URL.createObjectURL(file);
    imagePreview.src = imageURL;
    if(fileName) fileName.textContent = file.name;

    dropZone.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    analyzeBtn.disabled = false;

    resetResult();
}

// Analysis Handler
analyzeBtn.addEventListener("click", async function () {
    if (!selectedFile) {
        showError("Please select an image first.");
        return;
    }

    clearError();
    analyzeBtn.disabled = true;

    emptyResult.classList.add("hidden");
    resultContainer.classList.add("hidden");
    explanationSection.classList.add("hidden");
    loading.classList.remove("hidden");

    statusBadge.textContent = "Processing";
    statusBadge.className = "status-badge badge-waiting";
    statusBadge.classList.remove("hidden");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let message = "Prediction request failed.";
            try {
                const errorData = await response.json();
                if (errorData.error) message = errorData.error;
            } catch (e) {}
            throw new Error(message);
        }

        const data = await response.json();
        displayResult(data);

    } catch (error) {
        console.error(error);
        showError("Could not connect to the AI server. Make sure Flask is running on port 5000.");
        resetResult();
    } finally {
        loading.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});

// Display Result
function displayResult(data) {
    resultContainer.classList.remove("hidden");

    const predictedClass = data.prediction;
    const confidenceValue = Number(data.confidence);
    const normalValue = Number(data.normal_probability);
    const abnormalValue = Number(data.abnormal_probability);

    prediction.textContent = predictedClass;
    confidence.textContent = `${confidenceValue.toFixed(2)}%`;

    normalProbability.textContent = `${normalValue.toFixed(2)}%`;
    normalBar.style.width = `${Math.min(normalValue, 100)}%`;

    abnormalProbability.textContent = `${abnormalValue.toFixed(2)}%`;
    abnormalBar.style.width = `${Math.min(abnormalValue, 100)}%`;

    if (predictedClass === "Normal") {
        statusBadge.textContent = "Normal";
        statusBadge.className = "status-badge badge-normal";
        statusBadge.classList.remove("hidden");
        prediction.parentElement.className = "prediction-section normal-result";
    } else {
        statusBadge.textContent = "Abnormal";
        statusBadge.className = "status-badge badge-abnormal";
        statusBadge.classList.remove("hidden");
        prediction.parentElement.className = "prediction-section abnormal-result";
    }

    if (data.gradcam_url) {
        gradcamImage.src = `${API_URL}${data.gradcam_url}?t=${Date.now()}`;
        explanationSection.classList.remove("hidden");
    }
}

// Reset Result
function resetResult() {
    emptyResult.classList.remove("hidden");
    resultContainer.classList.add("hidden");
    loading.classList.add("hidden");

    statusBadge.textContent = "";
    statusBadge.className = "status-badge hidden";

    prediction.parentElement.className = "prediction-section";
    explanationSection.classList.add("hidden");
}

// Error Helpers
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}