import os.path
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from global_config import  datasets_dir, model_dir
"""
RATE assesses how well CATE rank observations on an evaluation set according to treatment profit 
A significant RATE suggests there is heterogeneity present in treatment effects.
"""

# ---------------------------
# Load Holdout Data and Trained Model
# ---------------------------


X_val = np.load(os.path.join(datasets_dir, "X_test_corrected.npy"))
T_val = np.load(os.path.join(datasets_dir, "T_test.npy"))
Y_val = np.load(os.path.join(datasets_dir, "Y_test.npy"))

# Load trained Causal Forest model (optional)
cf_model = joblib.load(os.path.join(model_dir, "cf_model.pkl"))

# Load dataset with CATEs
df_cf = pd.read_csv(os.path.join(datasets_dir, "dataset_with_cate.csv"))

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


print("\n📢 AUTOC Summary:")
print(f" Area Under TOC Curve (AUTOC): {AUTOC:.4f}")
print(f" Standard Error: {AUTOC_stderr:.4f}")
print(f" 95% Confidence Interval: ({AUTOC_lower:.4f}, {AUTOC_upper:.4f})")

plt.figure(figsize=(10, 4))

# TOC Curve with 95% Confidence Band
plt.plot(rate['Top Percentile'], rate['Average Treatment Effect'], marker='o', label='TOC Curve')
plt.fill_between(
    rate['Top Percentile'],
    rate['Average Treatment Effect'] - 1.96 * AUTOC_stderr,
    rate['Average Treatment Effect'] + 1.96 * AUTOC_stderr,
    color='blue', alpha=0.2, label='95% Confidence Interval'
)

# Add annotations for each point (optional but helpful)
for i, txt in enumerate(rate['Average Treatment Effect']):
    plt.annotate(f"{txt:.2f}", (rate['Top Percentile'][i], rate['Average Treatment Effect'][i]),
                 textcoords="offset points", xytext=(0,10), ha='center')

# Axis labels and title
plt.title('TOC Curve: ATE by Priority Percentile')
plt.xlabel('Top Percentile of Patients')
plt.ylabel('Average Treatment Effect')

# Vertical and horizontal reference lines
plt.axhline(0, color='black', linestyle='--')

# Legend
plt.legend()

# Grid and tight layout
plt.grid(True)
plt.tight_layout()

# Show
plt.show()