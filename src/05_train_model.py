from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --------------------------------------------------
# Resolve project root
# --------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models directory
models_dir = os.path.join(project_root, "models")

# --------------------------------------------------
# Load training data
# --------------------------------------------------
print("Loading training data...")

X_train = joblib.load(os.path.join(models_dir, "xtrain.pkl"))
y_train = joblib.load(os.path.join(models_dir, "ytrain.pkl"))

print(f"Training samples : {X_train.shape[0]}")
print(f"Feature count    : {X_train.shape[1]}")
print("Starting model training (RandomForest)...")
print("This may take a few minutes depending on system resources.\n")

# --------------------------------------------------
# Define model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,      # Use all CPU cores
    verbose=1
)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model.fit(X_train, y_train)

print("\n✔ Model training completed successfully")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
model_path = os.path.join(models_dir, "network_anomaly_model.pkl")
joblib.dump(model, model_path)

print(f"✔ Model saved at: {model_path}")
