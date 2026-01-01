import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# ===============================
# Project Paths
# ===============================
# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(project_root, "dataset", "NSL_KDD_READY.csv")
output_path = os.path.join(project_root, "dataset", "NSL_KDD_ENCODED.csv")

# Updated: Save encoders in 'models' folder
MODELS_DIR = os.path.join(project_root, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")

# ===============================
# Load Dataset
# ===============================
data = pd.read_csv(input_path)

# ===============================
# Encode Categorical Features
# ===============================
categorical_cols = ["protocol_type", "service", "flag"]
encoders = {}

print("Encoding categorical features...")
for col in categorical_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col].astype(str))
    encoders[col] = encoder
    print(f"  - Encoded '{col}': {len(encoder.classes_)} unique values")

# ===============================
# Save Encoded Dataset and Encoders
# ===============================
data.to_csv(output_path, index=False)
joblib.dump(encoders, encoders_path)

print(f"\nCategorical features encoded successfully!")
print(f"Encoded dataset saved to: {output_path}")
print(f"Encoders saved to: {encoders_path}")
