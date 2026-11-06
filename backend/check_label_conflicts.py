
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

train = pd.read_csv(DATASET_DIR / "spam_train.csv")
val = pd.read_csv(DATASET_DIR / "spam_val.csv")
test = pd.read_csv(DATASET_DIR / "spam_test.csv")

train["split"] = "train"
val["split"] = "validation"
test["split"] = "test"

combined = pd.concat([train, val, test], ignore_index=True)

# Find messages appearing more than once
duplicates = combined[
    combined.duplicated(subset="text", keep=False)
]

# Check whether identical messages have different labels
conflicts = duplicates.groupby("text")["label"].nunique()

conflicting_messages = conflicts[conflicts > 1]

print("Total repeated messages:", duplicates["text"].nunique())
print("Messages with conflicting labels:", len(conflicting_messages))

print("\nConflicting examples:")

print(
    duplicates[
        duplicates["text"].isin(conflicting_messages.index)
    ].head(10)
)

print("\nAnalysis completed!")
