import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from global_config import datasets_dir, model_dir
from validate.calibration_test import test_calibration  # Import from your local package
from Model.cf_config import covariate_cols
# ---------------------------
# Load Data and Model
# ---------------------------
df_test = pd.read_csv(os.path.join(datasets_dir, "cf_results.csv"))

# Diagnostic check for NaNs
missing = df_test[["Chemo_status", "survival_status"] + covariate_cols].isnull().sum()
print("\nMissing value check:")
print(missing[missing > 0])

# Drop rows with any NaNs in required columns
df_test = df_test.dropna(subset=["Chemo_status", "survival_status"] + covariate_cols)

# Redefine inputs after dropping rows
T_test = df_test["Chemo_status"].values
Y_test = df_test["survival_status"].values

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
# Plot Calibration for Treatment = 1
# ---------------------------
print("\nGenerating Calibration Plot for treatment = 1")
fig = calibration_result.plot_cal(tmt=1)
plt.tight_layout()
plt.savefig(os.path.join(datasets_dir, "calibration_plot_treatment1.png"))
plt.show()

# ---------------------------
# Explanation
# ---------------------------
print("""
This calibration test evaluates how well predicted CATEs align with observed treatment effects (GATEs).

- The R^2 in the summary indicates how well the predicted CATE explains group-level outcomes.
- The scatter plot shows each group's average predicted vs actual treatment effect, with a regression fit.
- Strong alignment and high R^2 indicates that the CATE model is well-calibrated.
""")
