import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

# --------------------------------------------------
# Resolve project root
# --------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input dataset (already scaled)
input_path = os.path.join(project_root, "dataset", "NSL_KDD_SCALED.csv")

# Models directory (for all .pkl artifacts)
models_dir = os.path.join(project_root, "models")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading dataset...")
data = pd.read_csv(input_path)

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

# --------------------------------------------------
# Train / Test split
# --------------------------------------------------
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Save split datasets
# --------------------------------------------------
joblib.dump(X_train, os.path.join(models_dir, "xtrain.pkl"))
joblib.dump(X_test, os.path.join(models_dir, "xtest.pkl"))
joblib.dump(y_train, os.path.join(models_dir, "ytrain.pkl"))
joblib.dump(y_test, os.path.join(models_dir, "ytest.pkl"))

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\n" + "=" * 60)
print("DATA SPLIT SUMMARY")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
print(f"Testing samples : {X_test.shape[0]} | Features: {X_test.shape[1]}")
print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())
print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())
print("=" * 60)

print("\n✔ Data split completed and saved to /models directory")
