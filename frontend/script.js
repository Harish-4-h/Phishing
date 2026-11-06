
const API_URL = "http://127.0.0.1:8000";

const messageInput = document.getElementById("message-input");
const predictButton = document.getElementById("predict-button");
const clearButton = document.getElementById("clear-button");

const buttonText = document.getElementById("button-text");
const loadingSpinner = document.getElementById("loading-spinner");

const apiStatus = document.getElementById("api-status");
const statusIndicator = document.getElementById("status-indicator");

const resultSection = document.getElementById("result-section");
const resultBanner = document.getElementById("result-banner");

const errorMessage = document.getElementById("error-message");
const charCount = document.getElementById("char-count");


// Check backend API connection
async function checkAPI() {
    try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API connection failed");
        }

        const data = await response.json();

        if (!data.model_loaded || !data.vectorizer_loaded) {
            throw new Error("Model or vectorizer not loaded");
        }

        apiStatus.textContent = "API Connected";
        statusIndicator.className = "status-dot online";

    } catch (error) {
        apiStatus.textContent = "API Disconnected";
        statusIndicator.className = "status-dot offline";
    }
}


// Update character count
messageInput.addEventListener("input", () => {
    charCount.textContent =
        `${messageInput.value.length} / 20000 characters`;
});


// Clear input and previous results
clearButton.addEventListener("click", () => {
    messageInput.value = "";
    charCount.textContent = "0 / 20000 characters";

    resultSection.classList.add("hidden");
    errorMessage.classList.add("hidden");

    messageInput.focus();
});


// Display error messages
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


// Control loading state
function setLoading(isLoading) {
    predictButton.disabled = isLoading;

    loadingSpinner.classList.toggle("hidden", !isLoading);

    buttonText.textContent = isLoading
        ? "Analyzing..."
        : "Analyze Message";
}


// Display prediction result
function displayResult(data) {
    resultSection.classList.remove("hidden");

    const isSpam = data.prediction === 1;

    resultBanner.className = isSpam
        ? "result-banner danger"
        : "result-banner safe";

    document.getElementById("result-icon").textContent =
        isSpam ? "⚠️" : "✅";

    document.getElementById("result-label").textContent =
        data.result;

    document.getElementById("result-description").textContent =
        isSpam
            ? "This message has been identified as Spam/Phishing."
            : "This message has been identified as Legitimate.";

    document.getElementById("confidence-value").textContent =
        `${data.confidence}%`;

    document.getElementById("confidence-bar").style.width =
        `${data.confidence}%`;

    document.getElementById("legitimate-value").textContent =
        `${data.probabilities.legitimate}%`;

    document.getElementById("legitimate-bar").style.width =
        `${data.probabilities.legitimate}%`;

    document.getElementById("spam-value").textContent =
        `${data.probabilities.spam_phishing}%`;

    document.getElementById("spam-bar").style.width =
        `${data.probabilities.spam_phishing}%`;

    document.getElementById("analyzed-text").textContent =
        data.message;

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// Send message to FastAPI prediction endpoint
async function predictMessage() {
    const message = messageInput.value.trim();

    errorMessage.classList.add("hidden");
    resultSection.classList.add("hidden");

    if (!message) {
        showError("Please enter a message before analyzing.");
        messageInput.focus();
        return;
    }

    setLoading(true);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Unable to analyze the message."
            );
        }

        if (
            typeof data.prediction !== "number" ||
            !data.probabilities ||
            typeof data.confidence !== "number"
        ) {
            throw new Error("Invalid prediction response from server.");
        }

        displayResult(data);

    } catch (error) {
        showError(
            error.message.includes("Failed to fetch")
                ? "Cannot connect to the backend. Make sure FastAPI is running."
                : error.message
        );

    } finally {
        setLoading(false);
    }
}


// Analyze button event
predictButton.addEventListener("click", predictMessage);


// Initial API connection check
checkAPI();
