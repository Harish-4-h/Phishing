import re
import html
import joblib

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --------------------------------
# 1. Initialize FastAPI
# --------------------------------

app = FastAPI(
    title="AI Phishing & Spam Detection API",
    description="Email and message classification using Multinomial Naive Bayes",
    version="1.0.0"
)

# --------------------------------
# 2. Configure CORS
# --------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------
# 3. Load trained model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

model_path = MODEL_DIR / "naive_bayes_model.pkl"
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# --------------------------------
# 4. Request schema
# --------------------------------

class MessageRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=20000,
        description="Email or message to classify"
    )

# --------------------------------
# 5. Text preprocessing
# --------------------------------

def clean_text(text):

    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# --------------------------------
# 6. Health endpoint
# --------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

# --------------------------------
# 7. Prediction endpoint
# --------------------------------

@app.post("/predict")
def predict(request: MessageRequest):

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Clean input
    cleaned_message = clean_text(message)

    # Convert text into TF-IDF features
    features = vectorizer.transform([cleaned_message])

    # Predict class
    prediction = int(model.predict(features)[0])

    # Get class probabilities
    probabilities = model.predict_proba(features)[0]

    legitimate_confidence = float(probabilities[0] * 100)
    spam_confidence = float(probabilities[1] * 100)

    # Classification result
    result = (
        "Spam/Phishing" if prediction == 1
        else "Legitimate"
    )

    return {
        "message": message,
        "prediction": prediction,
        "result": result,
        "confidence": round(
            float(probabilities[prediction] * 100), 2
        ),
        "probabilities": {
            "legitimate": round(legitimate_confidence, 2),
            "spam_phishing": round(spam_confidence, 2)
        }
    }

# --------------------------------
# 8. Serve Frontend
# --------------------------------

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend"
)