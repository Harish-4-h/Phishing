
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

train = pd.read_csv(DATASET_DIR / "spam_train.csv")
val = pd.read_csv(DATASET_DIR / "spam_val.csv")
test = pd.read_csv(DATASET_DIR / "spam_test.csv")

train_texts = set(train["text"].astype(str))
val_texts = set(val["text"].astype(str))
test_texts = set(test["text"].astype(str))

print("Train - Validation overlap:", len(train_texts & val_texts))
print("Train - Test overlap:", len(train_texts & test_texts))
print("Validation - Test overlap:", len(val_texts & test_texts))

print("\nDuplicate overlap check completed!")

