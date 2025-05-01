import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from global_config import datasets_dir, model_dir
from Model.cf_config import covariate_cols

# ---------------------------
# Load model and dataset with CATEs
# ---------------------------
df_cf = pd.read_csv(os.path.join(datasets_dir, "cf_results.csv"))
cf_model = joblib.load(os.path.join(model_dir, "cf_model.pkl"))
preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))

# ---------------------------
# Define evaluation subset (last 20% of patients)
# ---------------------------
test_frac = 0.2
test_size = int(test_frac * df_cf.shape[0])
df_test = df_cf.iloc[-test_size:].copy()

# Preprocess test data
X_test = preprocessor.transform(df_test[covariate_cols])
pred_cate = df_test["CATE"].values
true_cate = cf_model.effect(X_test)

# ---------------------------
# Sort by predicted CATE (priority ranking)
# ---------------------------
sorted_indices = np.argsort(-pred_cate)

df_test_ranked = df_test.copy()
df_test_ranked['rank'] = 0
df_test_ranked.iloc[sorted_indices, df_test_ranked.columns.get_loc('rank')] = np.arange(1, len(df_test_ranked) + 1)

# ---------------------------
# Evaluate ATE at top-K quantiles
# ---------------------------
k_list = np.linspace(0.1, 1.0, 10)
rate = []

for k in k_list:
    top_k = int(k * len(df_test_ranked))
    selected = sorted_indices[:top_k]
    avg_te = np.mean(true_cate[selected])
    rate.append(avg_te)

rate_df = pd.DataFrame({
    "Top Percentile": (k_list * 100).astype(int),
    "Average Treatment Effect": rate
})

# ---------------------------
# Compute AUTOC & CI
# ---------------------------
norm_x = rate_df['Top Percentile'] / 100
AUTOC = np.trapz(rate_df['Average Treatment Effect'], norm_x)
rate_std = np.std(rate_df['Average Treatment Effect'])
rate_n = len(rate_df)
AUTOC_stderr = rate_std / np.sqrt(rate_n)
AUTOC_lower = AUTOC - 1.96 * AUTOC_stderr
AUTOC_upper = AUTOC + 1.96 * AUTOC_stderr

print("\nRATE Summary:")
print(rate_df)
print(f"\nAUTOC: {AUTOC:.4f} ± {AUTOC_stderr:.4f}  (95% CI: {AUTOC_lower:.4f} to {AUTOC_upper:.4f})")

# ---------------------------
# Overlay CATE by subgroup in top quantile
# ---------------------------
top_10_idx = sorted_indices[:int(0.1 * len(sorted_indices))]
top_10_df = df_test.iloc[top_10_idx]

subgroup_vars = ['age_group', 'tumor_size_group', 'cci_group', 'Gender', 'grade_clean']

print("\nSubgroup CATEs in Top 10% Priority Group:")
for var in subgroup_vars:
    if var in top_10_df.columns:
        subgroup_summary = top_10_df.groupby(var).agg(
            Mean_CATE=('CATE', 'mean'),
            Count=('CATE', 'count')
        ).sort_values(by='Mean_CATE', ascending=False)
        print(f"\n>>> {var}:")
        print(subgroup_summary)

# ---------------------------
# Plot TOC Curve
# ---------------------------
plt.figure(figsize=(10, 4))
plt.plot(rate_df['Top Percentile'], rate_df['Average Treatment Effect'], marker='o', label='TOC Curve')
plt.fill_between(
    rate_df['Top Percentile'],
    rate_df['Average Treatment Effect'] - 1.96 * AUTOC_stderr,
    rate_df['Average Treatment Effect'] + 1.96 * AUTOC_stderr,
    alpha=0.2, color='blue', label='95% Confidence Interval')
plt.title('TOC Curve with CATE Prioritization')
plt.xlabel('Top Percentile of Patients')
plt.ylabel('Average Treatment Effect')
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
