import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm
from global_config import datasets_dir, model_dir

X_val = np.load(os.path.join(datasets_dir, "X_test_corrected.npy"))
T_val = np.load(os.path.join(datasets_dir, "T_test.npy"))
Y_val = np.load(os.path.join(datasets_dir, "Y_test.npy"))
cf_model = joblib.load(os.path.join(model_dir, "cf_model.pkl"))
df_cf = pd.read_csv(os.path.join(datasets_dir, "dataset_with_cate.csv"))

B = 500  # Number of bootstrap replications
bootstrap_AUTOCs = []

print(f"\n Running {B} Bootstrap replications...")

for _ in tqdm(range(B)):

    # 1. Resample indices WITH replacement
    bootstrap_idx = np.random.choice(len(X_val), size=len(X_val), replace=True)

    # 2. Get bootstrap sample
    X_boot = X_val[bootstrap_idx]
    T_boot = T_val[bootstrap_idx]
    Y_boot = Y_val[bootstrap_idx]

    # 3. Predict CATEs on bootstrap sample
    priority_cate_boot = df_cf.loc[df_cf.index[-len(X_val):], "CATE"].values[bootstrap_idx]
    true_cate_boot = cf_model.effect(X_boot)

    # 4. Ranking and RATE calculation
    sorted_indices = np.argsort(-priority_cate_boot)

    k_list = np.linspace(0.1, 1.0, 10)
    avg_treatment_effects = []

    for k in k_list:
        top_k = int(k * len(priority_cate_boot))
        selected = sorted_indices[:top_k]
        avg_te = np.mean(true_cate_boot[selected])
        avg_treatment_effects.append(avg_te)

    normalized_top_percentile = (k_list * 100) / 100
    AUTOC = np.trapz(avg_treatment_effects, normalized_top_percentile)

    bootstrap_AUTOCs.append(AUTOC)

# ---------------------------
# Bootstrap Summary
# ---------------------------

bootstrap_AUTOCs = np.array(bootstrap_AUTOCs)

AUTOC_mean = np.mean(bootstrap_AUTOCs)
AUTOC_std = np.std(bootstrap_AUTOCs)
AUTOC_ci_lower = np.percentile(bootstrap_AUTOCs, 2.5)
AUTOC_ci_upper = np.percentile(bootstrap_AUTOCs, 97.5)

print("\n Bootstrap AUTOC Summary:")
print(f" Mean AUTOC: {AUTOC_mean:.4f}")
print(f" Std. Dev of AUTOC: {AUTOC_std:.4f}")
print(f" 95% Confidence Interval: ({AUTOC_ci_lower:.4f}, {AUTOC_ci_upper:.4f})")


# ---------------------------
# Plot: Compare Bootstrap vs Simulated AUTOC
# ---------------------------

# Simulated AUTOC (your original calculation)
AUTOC_simulated = AUTOC  # assuming you calculated it before bootstrap

plt.figure(figsize=(10,6))

# Plot histogram of bootstrapped AUTOC
plt.hist(bootstrap_AUTOCs, bins=30, edgecolor='black', alpha=0.7, label='Bootstrapped AUTOC', color='#ADD8E6')

# Add vertical lines
plt.axvline(AUTOC_simulated, color='green', linestyle='-', linewidth=2, label=f'Simulated AUTOC = {AUTOC_simulated:.4f}')
plt.axvline(np.mean(bootstrap_AUTOCs), color='blue', linestyle='--', linewidth=2, label=f'Bootstrap Mean AUTOC = {AUTOC_mean:.4f}')
plt.axvline(AUTOC_ci_lower, color='red', linestyle=':', linewidth=2, label='95% CI Lower Bound')
plt.axvline(AUTOC_ci_upper, color='red', linestyle=':', linewidth=2, label='95% CI Upper Bound')

# Titles and labels
plt.title('Comparison: Simulated vs Bootstrapped AUTOC', fontsize=14)
plt.xlabel('AUTOC Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()