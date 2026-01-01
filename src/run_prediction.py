"""
User-facing CLI script for network anomaly detection predictions.

Usage:
    python run_prediction.py <path_to_user_csv>

The input CSV must contain all 41 required features (without label column).
Missing values will be filled with 0.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# ===============================
# Project Paths
# ===============================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(project_root, "models")
OUTPUT_DIR = os.path.join(project_root, "results")

# Paths to saved preprocessing objects and model
MODEL_PATH = os.path.join(MODELS_DIR, "network_anomaly_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ===============================
# Required Features
# ===============================
REQUIRED_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# ===============================
# Load User CSV
# ===============================
def load_user_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERROR: File '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        print("WARNING: 'label' column found in input. It will be ignored.")
        df = df.drop("label", axis=1)

    missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing_features:
        print(f"\nERROR: Missing required features: {', '.join(missing_features)}")
        sys.exit(1)

    df = df[REQUIRED_FEATURES]

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"WARNING: Found {missing_count} missing values. Filling with 0.")
        df = df.fillna(0)

    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    return df

# ===============================
# Preprocess
# ===============================
def preprocess(df):
    print("\nPreprocessing data...")

    if not os.path.exists(ENCODER_PATH):
        print(f"ERROR: Encoders not found at {ENCODER_PATH}")
        sys.exit(1)

    if not os.path.exists(SCALER_PATH):
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        sys.exit(1)

    encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_processed = df.copy()

    print("  - Encoding categorical features...")
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        try:
            df_processed[col] = le.transform(df_processed[col].astype(str))
        except ValueError:
            print(f"  WARNING: Unknown category in '{col}'. Using default encoding 0.")
            df_processed[col] = df_processed[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    print("  - Scaling numerical features...")
    df_scaled = scaler.transform(df_processed)

    print("✓ Preprocessing completed")
    return df_scaled

# ===============================
# Predict
# ===============================
def predict(df_scaled):
    print("\nMaking predictions...")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(df_scaled)

    print(f"✓ Predictions completed for {len(predictions)} samples")
    return predictions

# ===============================
# Save Predictions
# ===============================
def save_predictions(predictions, original_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "user_predictions.csv")

    results_df = pd.DataFrame({
        "prediction": predictions,
        "prediction_label": ["Normal" if p == 0 else "Anomaly" for p in predictions]
    })

    results_df = pd.concat([original_df.reset_index(drop=True), results_df], axis=1)
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Predictions saved to: {output_file}")
    return output_file

# ===============================
# Print Summary
# ===============================
def print_summary(predictions):
    normal_count = np.sum(predictions == 0)
    anomaly_count = np.sum(predictions == 1)
    total = len(predictions)

    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total samples analyzed: {total}")
    print(f"Normal traffic:  {normal_count:5d} ({normal_count/total*100:.2f}%)")
    print(f"Anomaly detected: {anomaly_count:5d} ({anomaly_count/total*100:.2f}%)")
    print("=" * 60)

    print("\nFirst 10 predictions:")
    for i in range(min(10, len(predictions))):
        label = "Normal" if predictions[i] == 0 else "Anomaly"
        print(f"  Sample {i+1}: {label} ({predictions[i]})")

# ===============================
# Main
# ===============================
def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("Network Anomaly Detection - Prediction Script")
        print("=" * 60)
        print("\nUsage: python run_prediction.py <path_to_user_csv>")
        sys.exit(1)

    user_csv = sys.argv[1]

    try:
        df = load_user_csv(user_csv)
        df_scaled = preprocess(df)
        predictions = predict(df_scaled)
        save_predictions(predictions, df)
        print_summary(predictions)

        print("\n" + "=" * 60)
        print("SUCCESS: Prediction pipeline completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
