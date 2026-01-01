import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ===============================
# Project Paths
# ===============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "network_anomaly_model.pkl")
X_TEST_PATH = os.path.join(MODELS_DIR, "X_test.pkl")
Y_TEST_PATH = os.path.join(MODELS_DIR, "y_test.pkl")

# ===============================
# Load Model and Test Data
# ===============================
# Load all files with joblib to avoid unpickling errors
model = joblib.load(MODEL_PATH)
X_test = joblib.load(X_TEST_PATH)
y_test = joblib.load(Y_TEST_PATH)

print("✓ Model and test data loaded")

# ===============================
# Predictions
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# Evaluation Metrics
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ===============================
# Save Metrics to File
# ===============================
report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")

with open(report_path, "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=======================\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Anomaly"]
        )
    )

print("✓ Evaluation metrics saved")

# ===============================
# Confusion Matrix Plot
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Normal", "Anomaly"])
plt.yticks([0, 1], ["Normal", "Anomaly"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("✓ Confusion matrix saved")

# ===============================
# ROC Curve
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

print("✓ ROC curve saved")
print("\nSUCCESS: Model evaluation completed")
