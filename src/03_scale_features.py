import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ===============================
# Project Paths
# ===============================
# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(project_root, "dataset", "NSL_KDD_ENCODED.csv")
output_path = os.path.join(project_root, "dataset", "NSL_KDD_SCALED.csv")

# Updated: Save scaler in 'models' folder
MODELS_DIR = os.path.join(project_root, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

# ===============================
# Load Dataset
# ===============================
data = pd.read_csv(input_path)

X = data.drop("label", axis=1)
y = data["label"]

# ===============================
# Scale Numerical Features
# ===============================
print("Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, scaler_path)

# Convert back to DataFrame for easier handling
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df["label"] = y.values

# Save scaled dataset
X_scaled_df.to_csv(output_path, index=False)

print(f"\nFeature scaling completed!")
print(f"Scaled features shape: {X_scaled.shape}")
print(f"Scaled dataset saved to: {output_path}")
print(f"Scaler saved to: {scaler_path}")
