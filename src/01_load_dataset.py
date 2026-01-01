import pandas as pd
import os

# Get the project root directory (parent of src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_root, "dataset", "NSL_KDD_READY.csv")

data = pd.read_csv(dataset_path)

# Remove rows with all NaN values or empty rows
data = data.dropna(how='all')

# Remove rows where label is NaN
data = data.dropna(subset=['label'])

print("=" * 60)
print("DATASET LOADING AND INSPECTION")
print("=" * 60)
print("\nFirst 5 rows:")
print(data.head())
print("\n" + "=" * 60)
print("\nDataset Info:")
print(data.info())
print("\n" + "=" * 60)
print("\nLabel Distribution:")
print(data["label"].value_counts().sort_index())
print("\n" + "=" * 60)
print(f"\nTotal samples: {len(data)}")
print(f"Total features: {len(data.columns) - 1}")  # Excluding label
print(f"Missing values per column:")
print(data.isnull().sum().sum(), "total missing values")
print("=" * 60)

