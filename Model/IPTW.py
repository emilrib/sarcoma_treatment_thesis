import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from global_config import datasets_dir, model_dir
from Model.cf_config import (numeric_cols, categorical_cols, treatment_col, outcome_col, covariate_cols)

# ---------------------------
# Load the imbalanced variables (SMD > 0.1)
# ---------------------------
imbalance_file = os.path.join(datasets_dir, "imbalance_vars_for_iptw.pkl")
with open(imbalance_file, 'rb') as f:
    imbalanced_cols = pickle.load(f)
print(f"Imbalanced Variables: {imbalanced_cols}")

# ---------------------------
# Load the dataset
# ---------------------------
input_file_groups = os.path.join(datasets_dir, "df_with_merged_groups.csv")
df = pd.read_csv(input_file_groups)

# ---------------------------
# Preprocessing Setup (Imputation only, as data is already encoded)
# ---------------------------
imbalanced_numeric = [col for col in imbalanced_cols if col in numeric_cols]
imbalanced_categorical = [col for col in imbalanced_cols if col in categorical_cols]

imbalanced_preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), imbalanced_numeric),
    ('cat', SimpleImputer(strategy='most_frequent'), imbalanced_categorical)
])

X_ps = imbalanced_preprocessor.fit_transform(df)
X_ps_df = pd.DataFrame(X_ps, columns=imbalanced_numeric + imbalanced_categorical)

if X_ps_df.isnull().any().any():
    print("Warning: Missing values (NaN) found after preprocessing!")
    print(X_ps_df.isnull().sum())
else:
    print("No missing values found after preprocessing.")

# ---------------------------
# Fit Logistic Regression for Propensity Scores
# ---------------------------
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_ps_df, df[treatment_col])
ps_scores = ps_model.predict_proba(X_ps_df)[:, 1]

# ---------------------------
# Calculate Weights for IPTW
# ---------------------------
treated = df[treatment_col].values
weights = np.where(treated == 1, 1 / ps_scores, 1 / (1 - ps_scores))
df['weights'] = weights

# ---------------------------
# Print Summary of IPTW Weights
# ---------------------------
print("\nSummary of IPTW Weights:")
print(f"Mean of Weights: {weights.mean():.4f}")
print(f"Standard Deviation of Weights: {weights.std():.4f}")
print(f"Minimum Weight: {weights.min():.4f}")
print(f"Maximum Weight: {weights.max():.4f}")

plt.figure(figsize=(8, 5))
plt.hist(weights, bins=30, edgecolor='k', alpha=0.7)
plt.title("Distribution of IPTW Weights")
plt.xlabel("IPTW Weight")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Save the weights DataFrame to a pickle file
# ---------------------------
id_col = 'Pat ID' if 'Pat ID' in df.columns else df.columns[0]
weights_file_pickle = os.path.join(datasets_dir, "weights.pkl")
df[[id_col, 'weights']].to_pickle(weights_file_pickle)
print("Weights have been saved to pickle file:", weights_file_pickle)

# ---------------------------
# Save IPTW Processed Data to CSV
# ---------------------------
output_file = os.path.join(datasets_dir, "df_with_iptw_weights.csv")
df.to_csv(output_file, index=False)
print(f"Processed Data with Weights saved to: {output_file}")
