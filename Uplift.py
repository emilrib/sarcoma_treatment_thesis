import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from validate import UpliftEvaluationResults
import joblib


# ---------------------------
# Setup paths
# ---------------------------
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

output_dir = os.path.join(current_dir)

X_test = np.load(os.path.join(output_dir, "X_test.npy"))
T_test = np.load(os.path.join(output_dir, "T_test.npy"))
Y_test = np.load(os.path.join(output_dir, "Y_test.npy"))
cf_model = joblib.load(os.path.join(output_dir, "cf_model.pkl"))
preprocessor = joblib.load(os.path.join(output_dir, "preprocessor.pkl"))

# Load full test dataframe
df_test = pd.read_csv(os.path.join(output_dir, "dataset_with_cate.csv"))
cate_preds = cf_model.effect(X_test)

# ---------------------------
# Run Uplift Evaluation
# ---------------------------
print("\nRunning Uplift Evaluation...")

# Sort individuals by predicted CATE
uplift_df = pd.DataFrame({
    "cate_pred": cate_preds,
    "outcome": Y_test,
    "treatment": T_test
}).sort_values("cate_pred", ascending=False).reset_index(drop=True)

# Create uplift curve (cumulative gain)
uplift_df['cumulative_treated'] = uplift_df['treatment'].cumsum()
uplift_df['cumulative_outcome'] = uplift_df['outcome'].cumsum()
uplift_df['percentage_treated'] = np.linspace(0, 1, len(uplift_df))

# Uplift = outcome gain over random
uplift_df['gain_over_random'] = uplift_df['cumulative_outcome'] - uplift_df['percentage_treated'] * uplift_df['outcome'].sum()

# Dummy error estimate (optional real bootstrap later)
uplift_df['err'] = 0.05  # fixed dummy error

# Build curve_data_dict expected by UpliftEvaluationResults
curve_data_dict = {
    1: uplift_df.rename(columns={
        "percentage_treated": "Percentage treated",
        "gain_over_random": "value"
    })
}

# Estimate simple stats
uplift_integral = uplift_df["gain_over_random"].mean()
uplift_std = uplift_df["gain_over_random"].std() / np.sqrt(len(uplift_df))
uplift_pval = 2 * (1 - stats.norm.cdf(np.abs(uplift_integral / uplift_std)))

params = [uplift_integral]
errs = [uplift_std]
pvals = [uplift_pval]
treatments = np.array([0, 1])  # 0 = control, 1 = treatment group

# Instantiate UpliftEvaluationResults
uplift_eval = UpliftEvaluationResults(
    params=params,
    errs=errs,
    pvals=pvals,
    treatments=treatments,
    curve_data_dict=curve_data_dict
)

# ---------------------------
# Print Uplift Summary
# ---------------------------
print("\n Uplift Evaluation Summary:")
print(uplift_eval.summary())

# ---------------------------
# Plot Uplift Curve
# ---------------------------
plt.figure(figsize=(8, 5))
uplift_eval.plot_uplift(tmt=1)
plt.title("Uplift Curve for Treated Patients")
plt.grid(True)
plt.tight_layout()
plt.show()