import joblib
import os

# ===============================
# Project Paths
# ===============================
# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(project_root, "models")
model_path = os.path.join(MODELS_DIR, "network_anomaly_model.pkl")

# ===============================
# Load and Verify Model
# ===============================
try:
    model = joblib.load(model_path)
    print("=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)
    print("\nModel loaded successfully.")
    print(f"Model type: {type(model)}")
    print(f"Model saved at: {model_path}")
    print(f"Number of features: {model.n_features_in_}")
    print(f"Number of estimators: {model.n_estimators}")
    print("\n" + "=" * 60)
except FileNotFoundError:
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print("\nModel file not found!")
    print(f"Expected location: {model_path}")
    print("\nPlease run 05_train_model.py first to train and save the model.")
    print("=" * 60)
