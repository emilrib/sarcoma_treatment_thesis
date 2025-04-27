import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from validate import CalibrationEvaluationResults
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# ---------------------------
# Load Data and Model
# ---------------------------

if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

output_dir = os.path.join(current_dir)

df_test = pd.read_csv(os.path.join(output_dir, "dataset_with_cate.csv"))

X_test = np.load(os.path.join(output_dir, "X_test.npy"))
T_test = np.load(os.path.join(output_dir, "T_test.npy"))
Y_test = np.load(os.path.join(output_dir, "Y_test.npy"))

cf_model = joblib.load(os.path.join(output_dir, "cf_model.pkl"))
preprocessor = joblib.load(os.path.join(output_dir, "preprocessor.pkl"))  # if you want to re-transform X later

# ---------------------------
# Predict CATEs
# ---------------------------
print("\nPredicting CATEs...")
cate_preds = cf_model.effect(X_test)

# ---------------------------
# Calibration Evaluation
# ---------------------------
print("\nRunning Calibration Evaluation...")

# Calculate Mean and Differential predictions
mean_cate = np.mean(cate_preds)
differential_cate = cate_preds - mean_cate

# Build calibration dataframe
df_cal = pd.DataFrame({
    "mean_pred": np.full_like(cate_preds, mean_cate),  # constant for all
    "diff_pred": differential_cate,
    "outcome": Y_test,
    "treatment": T_test
})

# Bin into quantiles based on predicted CATE
df_cal['cate_bin'] = pd.qcut(cate_preds, q=5, labels=False)

# Compute group-level average outcome
summary_cal = df_cal.groupby('cate_bin').agg(
    mean_pred=('mean_pred', 'mean'),
    diff_pred=('diff_pred', 'mean'),
    gate=('outcome', 'mean'),
    se_gate=('outcome', 'sem')  # standard error
).reset_index()

# ---------------------------
# Regression: gate ~ mean_pred + diff_pred
# ---------------------------
X = sm.add_constant(summary_cal[['mean_pred', 'diff_pred']])  # add intercept
y = summary_cal['gate']

model = sm.OLS(y, X).fit(cov_type='HC3')  # Heteroskedasticity-robust SEs

print("\nCalibration Regression Results (Equivalent to test_calibration()):")
print(model.summary())

# ---------------------------
# Clean Printing of Estimates
# ---------------------------
print("\nCoefficients Summary:")
coef_names = ['Intercept', 'mean.forest.prediction', 'differential.forest.prediction']
for idx, name in enumerate(coef_names):
    print(f"{name:30s} Estimate: {model.params[idx]:.6f}  Std.Error: {model.bse[idx]:.6f}  "
          f"t-value: {model.tvalues[idx]:.4f}  p-value: {model.pvalues[idx]:.4e}")

# ---------------------------
# Optional: Calibration Plot (GATE vs Predicted)
# ---------------------------

# For simple visualization
plt.figure(figsize=(8,5))
plt.scatter(summary_cal['mean_pred'] + summary_cal['diff_pred'], summary_cal['gate'], label='Binned Data Points')
plt.plot(summary_cal['mean_pred'] + summary_cal['diff_pred'],
         model.predict(X), color='red', label='Calibration Fit')
plt.xlabel('Predicted CATE')
plt.ylabel('Observed GATE')
plt.title('Calibration Curve (Test Calibration Style)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



