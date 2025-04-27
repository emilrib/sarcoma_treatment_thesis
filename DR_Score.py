import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Setup paths
# ---------------------------
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

output_dir = os.path.join(current_dir)

# ---------------------------
# Load Data
# ---------------------------
df_test = pd.read_csv(os.path.join(output_dir, "dataset_with_cate.csv"))

# Load arrays (use the correct X_test, T_test, Y_test)
X_test = np.load(os.path.join(output_dir, "X_test.npy"))
T_test = np.load(os.path.join(output_dir, "T_test.npy"))
Y_test = np.load(os.path.join(output_dir, "Y_test.npy"))
# Load trained models
cf_model = joblib.load(os.path.join(output_dir, "cf_model.pkl"))
preprocessor = joblib.load(os.path.join(output_dir, "preprocessor.pkl"))


print(type(cf_model))

# ---------------------------
# 1. Estimate nuisance models on X_test
# ---------------------------
print("\nEstimating nuisance models...")

model_y = RandomForestRegressor()
model_t = LogisticRegression(max_iter=1000)

model_y.fit(X_test, Y_test)
model_t.fit(X_test, T_test)

mu = model_y.predict(X_test)
propensity = model_t.predict_proba(X_test)[:, 1]

W = T_test
Y = Y_test

# ---------------------------
# 2. Compute pseudo-outcome
# ---------------------------
pseudo_outcome = ((W - propensity) * (Y - mu)) / (propensity * (1 - propensity))

# Predict CATEs
cate_preds = cf_model.effect(X_test)

# Compute DR loss (proxy loss)
dr_loss = np.mean((pseudo_outcome - cate_preds) ** 2)

print(f"\n DR Score (lower is better): {dr_loss:.4f}")

# ---------------------------
# 3. Plot DR Score bands
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 2))

bands = [0.1, 0.3, 0.5, 1.0]
colors = ['green', 'yellow', 'orange', 'red']
labels = ['Excellent', 'Good', 'Moderate', 'Weak']

start = 0
for band, color, label in zip(bands, colors, labels):
    ax.axvspan(start, band, color=color, alpha=0.4, label=label)
    start = band

# Plot vertical line for DR Score
ax.axvline(dr_loss, color='blue', linestyle='--', linewidth=2)
plt.text(dr_loss + 0.01, 0.5, f'DR Score = {dr_loss:.3f}', verticalalignment='center', color='blue')

# Formatting
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel('DR Score (Lower is Better)')
ax.set_title('Model Reliability based on DR Score')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(True, axis='x')
plt.tight_layout()

# Show plot
plt.show()

