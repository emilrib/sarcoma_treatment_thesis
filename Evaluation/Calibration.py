import numpy as np
import pandas as pd
import os
import joblib
from global_config import datasets_dir, model_dir
from Model.cf_config import covariate_cols
from validate.calibration_test import test_calibration, get_last_model_reg

# ---------------------------
# Load Data and Model
# ---------------------------
df_test = pd.read_csv(os.path.join(datasets_dir, "cf_results.csv"))

# Diagnostic check for NaNs
#missing = df_test[["chemo_status", "survival_status"] + covariate_cols].isnull().sum()
#print("\nMissing value check:")
#print(missing[missing > 0])

# Drop rows with any NaNs in required columns
df_test = df_test.dropna(subset=["chemo_status", "survival_status"] + covariate_cols)
#df_test = df_test.dropna(subset=["chemo_status", "metastasis_label"] + covariate_cols)


# Filter again for strictly finite values
df_test = df_test[np.isfinite(df_test["chemo_status"]) & np.isfinite(df_test["survival_status"])]
#df_test = df_test[np.isfinite(df_test["chemo_status"]) & np.isfinite(df_test["metastasis_label"])]

# Redefine inputs after dropping rows
T_test = df_test["chemo_status"].astype(float).values
Y_test = df_test["survival_status"].astype(float).values
#Y_test = df_test["metastasis_label"].astype(float).values

# Load and apply preprocessor to align with training
preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
X_test = preprocessor.transform(df_test[covariate_cols])
cf_model = joblib.load(os.path.join(model_dir, "cf_model.pkl"))

# ---------------------------
# Run Calibration
# ---------------------------
calibration_result = test_calibration(model=cf_model, X=X_test, T=T_test, y=Y_test, n_bins=5)


