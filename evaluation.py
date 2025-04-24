import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from run_causal_forest import get_test_results

from cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)

# Load results
df_test = get_test_results()

# ----------------------------
# RATE Calculation
# ----------------------------
treated = df_test[df_test[treatment_col] == 1]
untreated = df_test[df_test[treatment_col] == 0]

mean_cate_treated = treated['CATE'].mean()
mean_cate_untreated = untreated['CATE'].mean()

if mean_cate_untreated != 0:
    rate = (mean_cate_treated - mean_cate_untreated) / abs(mean_cate_untreated)
else:
    rate = float('inf') if mean_cate_treated > 0 else float('-inf')

print(f" RATE (Relative Average Treatment Effect): {rate:.3f}")
print(f"Mean CATE (Treated): {mean_cate_treated:.4f}")
print(f"Mean CATE (Untreated): {mean_cate_untreated:.4f}")

# ----------------------------
# Threshold-Based Recommendation Simulation
# ----------------------------
thresholds = [0.01, 0.05, 0.10]

print("\n📊 Simulated Recommendation Outcomes:")
for threshold in thresholds:
    recommended = df_test[df_test['CATE'] > threshold]
    survival_rate = recommended['survival_status_binary'].mean()
    print(f"Threshold > {threshold:.2f} → {len(recommended)} patients | Survival Rate: {survival_rate:.2%}")

# ----------------------------
# Plot CATE Distribution
# ----------------------------
plt.figure(figsize=(10, 5))
plt.hist(df_test['CATE'] * 100, bins=30, edgecolor='k')
plt.title("CATE Distribution on Test Set")
plt.xlabel("Estimated Change in Survival Probability (%)")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.show()
