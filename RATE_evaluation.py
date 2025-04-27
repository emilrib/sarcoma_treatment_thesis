import os.path
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

"""
RATE assesses how well CATE rank observations on an evaluation set according to treatment profit 
A significant RATE suggests there is heterogeneity present in treatment effects.
"""


# ---------------------------
# Setup paths
# ---------------------------
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

output_dir = os.path.join(current_dir)

# ---------------------------
# Load Holdout Data and Trained Model
# ---------------------------

# Define where your outputs were saved
output_dir = current_dir  # Assuming current_dir is already defined

X_val = np.load(os.path.join(output_dir, "X_test.npy"))
T_val = np.load(os.path.join(output_dir, "T_test.npy"))
Y_val = np.load(os.path.join(output_dir, "Y_test.npy"))

# Load trained Causal Forest model (optional)
cf_model = joblib.load(os.path.join(output_dir, "cf_model.pkl"))

# Load dataset with CATEs
df_cf = pd.read_csv(os.path.join(output_dir, "dataset_with_cate.csv"))

# ---------------------------
# Priority Ranking Evaluation
# ---------------------------

# Priority scores = estimated CATEs from df_cf (already computed)
priority_cate = df_cf.loc[df_cf.index[-len(X_val):], "CATE"].values

# True CATE estimates on holdout can be approximated by cf_model.effect(X_val) if needed
true_cate = cf_model.effect(X_val)

# Rank by predicted priority (highest CATE first)
sorted_indices = np.argsort(-priority_cate)

# Evaluate Average Treatment Effect across top-k percentages
k_list = np.linspace(0.1, 1.0, 10)  # From top 10% to 100%
avg_treatment_effects = []

for k in k_list:
    top_k = int(k * len(priority_cate))
    selected = sorted_indices[:top_k]
    avg_te = np.mean(true_cate[selected])
    avg_treatment_effects.append(avg_te)

# Build a summary DataFrame
rate = pd.DataFrame({
    'Top Percentile': (k_list * 100).astype(int),
    'Average Treatment Effect': avg_treatment_effects
})

print("\nPriority Ranking Evaluation:")
print(rate)

# ---------------------------
# Calculate AUTOC and Standard Error
# ---------------------------

# Normalize x-axis to [0,1] before calculating area
normalized_top_percentile = rate['Top Percentile'] / 100
AUTOC = np.trapz(rate['Average Treatment Effect'], normalized_top_percentile)

# Estimate Std. Error using standard deviation of RATE and sample size
rate_std = np.std(rate['Average Treatment Effect'])
rate_n = len(rate['Average Treatment Effect'])
AUTOC_stderr = rate_std / np.sqrt(rate_n)

# Calculate 95% Confidence Interval bounds
AUTOC_lower = AUTOC - 1.96 * AUTOC_stderr
AUTOC_upper = AUTOC + 1.96 * AUTOC_stderr

print(f"\nAUTOC (Area Under TOC Curve): {AUTOC:.4f}")
print(f"Standard Error of AUTOC: {AUTOC_stderr:.4f}")
print(f"95% Confidence Interval for AUTOC: ({AUTOC_lower:.4f}, {AUTOC_upper:.4f})")

# ---------------------------
# Combined Plot: TOC and RATE (with shaded background and 95% CI)
# ---------------------------

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# TOC Curve with Confidence Band
axs[0].plot(rate['Top Percentile'], rate['Average Treatment Effect'], marker='o', label='TOC Curve')
axs[0].fill_between(
    rate['Top Percentile'],
    rate['Average Treatment Effect'] - 1.96 * AUTOC_stderr,
    rate['Average Treatment Effect'] + 1.96 * AUTOC_stderr,
    color='blue', alpha=0.2, label='95% CI'
)
for i, txt in enumerate(rate['Average Treatment Effect']):
    axs[0].annotate(f"{txt:.2f}", (rate['Top Percentile'][i], rate['Average Treatment Effect'][i]),
                    textcoords="offset points", xytext=(0,10), ha='center')
axs[0].set_title('TOC Curve: ATE by Priority Percentile')
axs[0].set_xlabel('Top Percentile of Patients')
axs[0].set_ylabel('Average Treatment Effect')
axs[0].legend()
axs[0].grid(True)

# RATE Curve with Zero Line and Background Shading
axs[1].plot(rate['Top Percentile'], rate['Average Treatment Effect'], marker='o', linestyle='-')
axs[1].axhline(0, color='black', linestyle='--', label='Zero Effect')
axs[1].fill_between(rate['Top Percentile'], rate['Average Treatment Effect'], 0, where=(rate['Average Treatment Effect']>=0), color='green', alpha=0.3)
axs[1].fill_between(rate['Top Percentile'], rate['Average Treatment Effect'], 0, where=(rate['Average Treatment Effect']<0), color='red', alpha=0.3)
axs[1].set_title('RATE Curve (Simple Check with Shading)')
axs[1].set_xlabel('Top Percentile of Patients')
axs[1].set_ylabel('Average Treatment Effect')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# ---------------------------
# Optional: Cumulative Gain Plot
# ---------------------------

cumulative_gain = np.cumsum(true_cate[sorted_indices]) / np.arange(1, len(true_cate) + 1)
percentiles = np.linspace(1, 100, len(cumulative_gain))

plt.figure(figsize=(8, 5))
plt.plot(percentiles, cumulative_gain)
plt.title('Cumulative Gain Curve')
plt.xlabel('Top Percentile of Patients')
plt.ylabel('Cumulative Average Treatment Effect')
plt.grid(True)
plt.tight_layout()
plt.show()