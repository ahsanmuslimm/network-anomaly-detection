
# Network Traffic Anomaly Detection using Machine Learning

## 📌 Project Overview
This project implements an AI-driven Network Traffic Anomaly Detection System designed to identify abnormal or suspicious network traffic. The system applies Machine Learning techniques to classify network activities as normal or anomalous, which is a key requirement in modern cyber security systems.

This project is developed as part of an academic assignment focused on applying Artificial Intelligence (AI) and Machine Learning (ML) to real-world cyber security problems.

---

## 🎯 Objectives
- Detect anomalous patterns in network traffic
- Apply AI/ML techniques to a cyber security use case
- Build a complete end-to-end ML pipeline
- Evaluate model performance using standard metrics
- Provide a clean and understandable project structure

---

## 🧠 Problem Statement
Traditional rule-based intrusion detection systems struggle to detect new and evolving cyber threats. This project uses Machine Learning models that learn from network traffic data to automatically detect anomalies that may indicate malicious activity.

---

## Project Structure
```
network-anomaly-detection/
│
├── src/
│   ├── 01_load_dataset.py
│   ├── 02_encode_features.py
│   ├── 03_scale_features.py
│   ├── 04_split_data.py
│   ├── 05_train_model.py
│   ├── 06_evaluate_model.py
│   ├── 07_save_model.py
│   └── run_prediction.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── evaluation_report.txt
│   ├── evaluation_results.txt
│   └── user_predictions.csv
│
├── models/
│   └── network_anomaly_model.pkl   # Trained model
│   └─  encoders.pkl                # Categorical feature encoders
|   └─  scaler.pkl                  # Feature scaler
|   └─  X_train.pkl, X_test.pkl    # Train/test features
|   └─  y_train.pkl, y_test.pkl    # Train/test labels
│── report/
|   └── Report
| 
├── README.md
└── requirements.txt
```

---

## 🔄 Workflow Pipeline
1. Load raw network traffic dataset  
2. Encode categorical features  
3. Scale numerical features  
4. Split data into training and testing sets  
5. Train a machine learning model  
6. Evaluate model performance  
7. Save the trained model  
8. Predict anomalies on new data  

---

## 🤖 Machine Learning Model
- Type: Supervised Learning
- Task: Network Classification (Normal vs Anomaly)
- Libraries Used:
  - Python
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

All evaluation outputs are stored in the `results/` directory.

---

## ⚙️ How to Run the Project

### 1️⃣ Install Dependencies

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Run the Complete Pipeline
Run the scripts in order:   [Devloper Side] #Only run once 
```
python src/01_load_dataset.py
python src/02_encode_features.py
python src/03_scale_features.py
python src/04_split_data.py
python src/05_train_model.py
python src/06_evaluate_model.py
python src/07_save_model.py
```

### 3️⃣ Run Prediction on New Data/ 
User csv file   #csv file should contain all 41 feature as like main-dataset excluding: label no worry about order of columns 
User will save its input csv in dataset folder where the user prediction script will take and generate new user_prediction.csv file as output.
User can evalaute its result through the trained model named: network_anamoly_model.pkl and you can evlaute it through 06 script and see the output which will come in results folder.
```
python src/run_prediction.py
```

---

## 🛡️ Cyber Security Application
This project can be used as:
- Network Intrusion Detection System (IDS)
- Traffic monitoring tool
- Foundation for advanced cyber threat detection systems

---

## 📚 Learning Outcomes
- Practical application of AI in cyber security
- Understanding of anomaly detection techniques
- Experience with full ML lifecycle
- Model evaluation and result analysis

---


## ✅ Project Status
✔ Fully implemented  
✔ End-to-end pipeline working  
✔ Ready for evaluation  
