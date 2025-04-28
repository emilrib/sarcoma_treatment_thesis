import numpy as np
import os
from tqdm import tqdm
from econml.dml import CausalForestDML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from global_config import datasets_dir

# ---------------------------
# Load data
# ---------------------------

X = np.load(os.path.join(datasets_dir, "X_test_corrected.npy"))
T = np.load(os.path.join(datasets_dir, "T_test.npy"))
Y = np.load(os.path.join(datasets_dir, "Y_test.npy"))

# ---------------------------
# Bagging Parameters
# ---------------------------

B = 50  # Number of causal forests to train
n_estimators_per_forest = 500  # number of trees inside each forest
cate_predictions_list = []

print(f"\n Running Bagging with {B} Causal Forests...")

# ---------------------------
# Bagging Loop
# ---------------------------
bootstrap_AUTOCs = []  # track AUTOC per bootstrap forest

for b in tqdm(range(B)):
    # 1. Bootstrap sample
    bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X[bootstrap_idx]
    T_boot = T[bootstrap_idx]
    Y_boot = Y[bootstrap_idx]

    # 2. Train Causal Forest
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=b),
        model_t=LogisticRegression(max_iter=1000),
        discrete_treatment=True,
        n_estimators=500,
        random_state=b,
        cv=3
    )
    cf.fit(Y_boot, T_boot, X=X_boot)

    # 3. Predict CATEs on original data
    cate_pred = cf.effect(X)
    cate_predictions_list.append(cate_pred)

    # 4. Compute AUTOC for this bootstrap
    sorted_indices = np.argsort(-cate_pred)
    k_list = np.linspace(0.1, 1.0, 10)
    avg_treatment_effects = []

    for k in k_list:
        top_k = int(k * len(cate_pred))
        selected = sorted_indices[:top_k]
        avg_te = np.mean(cate_pred[selected])
        avg_treatment_effects.append(avg_te)

    normalized_top_percentile = (k_list * 100) / 100
    AUTOC_boot = np.trapz(avg_treatment_effects, normalized_top_percentile)
    bootstrap_AUTOCs.append(AUTOC_boot)

# ---------------------------
# Aggregate Bagged Predictions
# ---------------------------

cate_predictions_array = np.array(cate_predictions_list)  # shape (B, n_samples)
cate_bagged = np.mean(cate_predictions_array, axis=0)

print("\n Bagging complete!")
print(f"Bagged CATE Predictions shape: {cate_bagged.shape}")

# ---------------------------
# Save Results
# ---------------------------

np.save(os.path.join(datasets_dir, "cate_bagged.npy"), cate_bagged)
print(" Saved bagged CATE predictions.")

# ---------------------------
# Plot Histogram
# ---------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(range(1, B+1), bootstrap_AUTOCs, color='darkred', marker='o')
plt.title('AUTOC Across Bagging Replications')
plt.xlabel('Bagging Iteration (B)')
plt.ylabel('AUTOC Value')
plt.grid(True)
plt.tight_layout()
plt.show()