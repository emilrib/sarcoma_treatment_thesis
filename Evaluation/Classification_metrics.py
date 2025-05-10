import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from global_config import datasets_dir, model_dir
from Model.cf_config import covariate_cols

# ---------------------------
# Load data and preprocessor
# ---------------------------
df = pd.read_csv(os.path.join(datasets_dir, "cf_results.csv"))
df = df.dropna(subset=["chemo_status", "survival_status"] + covariate_cols)
preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))

# Preprocess features
X = preprocessor.transform(df[covariate_cols])
T = df["chemo_status"].astype(int).values
Y = df["survival_status"].astype(int).values

# ---------------------------
# Split treated and control groups
# ---------------------------
X_treated = X[T == 1]
Y_treated = Y[T == 1]

X_control = X[T == 0]
Y_control = Y[T == 0]

# ---------------------------
# Train separate outcome models
# ---------------------------
clf1 = RandomForestClassifier(random_state=42)
clf0 = RandomForestClassifier(random_state=42)

clf1.fit(X_treated, Y_treated)  # model for treated patients
clf0.fit(X_control, Y_control)  # model for untreated patients

# ---------------------------
# Predict for all patients
# ---------------------------
mu1 = clf1.predict_proba(X)[:, 1]
mu0 = clf0.predict_proba(X)[:, 1]

# Impute predicted outcome based on actual treatment
Y_pred = np.where(T == 1, mu1, mu0)
Y_pred_binary = (Y_pred >= 0.5).astype(int)

# ---------------------------
# Classification Metrics
# ---------------------------
accuracy = accuracy_score(Y, Y_pred_binary)
precision = precision_score(Y, Y_pred_binary)
recall = recall_score(Y, Y_pred_binary)
f1 = f1_score(Y, Y_pred_binary)
roc_auc = roc_auc_score(Y, Y_pred)

print("\nClassification Metrics (based on actual treatment assignment):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
