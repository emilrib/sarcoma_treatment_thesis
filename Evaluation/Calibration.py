import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import statsmodels.api as sm
import matplotlib.patches as mpatches
from global_config import datasets_dir, model_dir
from Model.cf_config import covariate_cols
from validate.results import CalibrationEvaluationResults
from validate.calibration_test import test_calibration, get_last_model_reg

# ---------------------------
# Load Data and Model
# ---------------------------
df_test = pd.read_csv(os.path.join(datasets_dir, "cf_results.csv"))

# Diagnostic check for NaNs
missing = df_test[["chemo_status", "survival_status"] + covariate_cols].isnull().sum()
print("\nMissing value check:")
print(missing[missing > 0])

# Drop rows with any NaNs in required columns
df_test = df_test.dropna(subset=["chemo_status", "survival_status"] + covariate_cols)

# Filter again for strictly finite values
df_test = df_test[np.isfinite(df_test["chemo_status"]) & np.isfinite(df_test["survival_status"])]

# Redefine inputs after dropping rows
T_test = df_test["chemo_status"].astype(float).values
Y_test = df_test["survival_status"].astype(float).values

# Load and apply preprocessor to align with training
preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
X_test = preprocessor.transform(df_test[covariate_cols])
cf_model = joblib.load(os.path.join(model_dir, "cf_model.pkl"))

# ---------------------------
# Run Calibration
# ---------------------------
print("\nRunning test_calibration from evaluate.calibration_test...")
calibration_result = test_calibration(model=cf_model, X=X_test, T=T_test, y=Y_test, n_bins=5)

# Print Calibration Summary
summary_df = calibration_result.summary()
print("\nCalibration Summary:")
print(summary_df)

# Save calibration summary as CSV
calibration_output_path = os.path.join(datasets_dir, "cate_calibration_summary.csv")
summary_df.to_csv(calibration_output_path, index=False)
print(f"\nCalibration summary saved to: {calibration_output_path}")

# ---------------------------
# Print detailed regression output similar to R-style
# ---------------------------
print("\nDetailed Calibration Coefficient Output:")
model_reg = get_last_model_reg()
if model_reg is not None:
    coef_names = ['Intercept', 'mean.forest.prediction', 'differential.forest.prediction']

    # Print header
    print(f"{'Coefficient':30s} {'Estimate':>10s} {'Std.Error':>10s} {'t-value':>10s} {'p-value':>12s} {'Signif':>6s}")
    print("-" * 82)

    # Print each row of the summary
    for idx, name in enumerate(coef_names):
        estimate = model_reg.params[idx]
        stderr = model_reg.bse[idx]
        tval = model_reg.tvalues[idx]
        pval = model_reg.pvalues[idx]
        signif = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '.' if pval < 0.1 else ''
        print(f"{name:30s} {estimate:10.6f} {stderr:10.6f} {tval:10.4f} {pval:12.4e} {signif:>6s}")
else:
    print("No regression model was returned from test_calibration().")
