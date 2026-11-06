
import pandas as pd
from pathlib import Path

# Dataset folder path
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Load all three datasets
train_df = pd.read_csv(DATASET_DIR / "spam_train.csv")
val_df = pd.read_csv(DATASET_DIR / "spam_val.csv")
test_df = pd.read_csv(DATASET_DIR / "spam_test.csv")

# Display dataset information
datasets = {
    "Training Dataset": train_df,
    "Validation Dataset": val_df,
    "Testing Dataset": test_df
}

for name, df in datasets.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nLabel Distribution:")
    print(df["label"].value_counts())

    print("\nSample Records:")
    print(df.head(3))

print("\nDataset inspection completed!")

