
import re
import html
import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------
# 1. Define project paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------
# 2. Load datasets
# --------------------------------

print("Loading datasets...")

train_df = pd.read_csv(DATASET_DIR / "spam_train.csv")
val_df = pd.read_csv(DATASET_DIR / "spam_val.csv")
test_df = pd.read_csv(DATASET_DIR / "spam_test.csv")

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

# --------------------------------
# 3. Remove training data leakage
# --------------------------------

val_texts = set(val_df["text"].astype(str))
test_texts = set(test_df["text"].astype(str))

original_size = len(train_df)

train_df = train_df.drop_duplicates(subset="text")

train_df = train_df[
    ~train_df["text"].astype(str).isin(val_texts | test_texts)
].copy()

print("\nTraining duplicate removal:")
print("Original records:", original_size)
print("Remaining records:", len(train_df))
print("Removed records:", original_size - len(train_df))

# --------------------------------
# 4. Text preprocessing
# --------------------------------

def clean_text(text):

    text = html.unescape(str(text))

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("\nCleaning text...")

train_text = train_df["text"].apply(clean_text)
val_text = val_df["text"].apply(clean_text)
test_text = test_df["text"].apply(clean_text)

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# --------------------------------
# 5. TF-IDF feature extraction
# --------------------------------

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=100000,
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)
X_test = vectorizer.transform(test_text)

print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)
print("Testing feature shape:", X_test.shape)

# --------------------------------
# 6. Train Multinomial Naive Bayes
# --------------------------------

print("\nTraining Multinomial Naive Bayes...")

model = MultinomialNB(alpha=0.5)

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------
# 7. Evaluation function
# --------------------------------

def evaluate_model(model, features, actual_labels, dataset_name):

    predictions = model.predict(features)

    accuracy = accuracy_score(actual_labels, predictions)

    precision = precision_score(
        actual_labels, predictions, zero_division=0
    )

    recall = recall_score(
        actual_labels, predictions, zero_division=0
    )

    f1 = f1_score(
        actual_labels, predictions, zero_division=0
    )

    cm = confusion_matrix(
        actual_labels, predictions, labels=[0, 1]
    )

    print("\n" + "=" * 50)
    print(dataset_name)
    print("=" * 50)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            actual_labels,
            predictions,
            target_names=["Legitimate", "Spam/Phishing"],
            zero_division=0
        )
    )

    return predictions


# --------------------------------
# 8. Evaluate validation dataset
# --------------------------------

evaluate_model(
    model,
    X_val,
    y_val,
    "VALIDATION RESULTS"
)

# --------------------------------
# 9. Evaluate testing dataset
# --------------------------------

evaluate_model(
    model,
    X_test,
    y_test,
    "FINAL TEST RESULTS"
)

# --------------------------------
# 10. Save model and vectorizer
# --------------------------------

joblib.dump(
    model,
    MODEL_DIR / "naive_bayes_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print("Model:", MODEL_DIR / "naive_bayes_model.pkl")
print("Vectorizer:", MODEL_DIR / "tfidf_vectorizer.pkl")

print("\nTraining pipeline completed!")
