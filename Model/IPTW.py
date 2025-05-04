import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from global_config import datasets_dir
from Model.cf_config import (numeric_cols, categorical_cols, treatment_col)

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

# ---------------------------
# Effective Sample Size (ESS)
# ---------------------------
def compute_effective_sample_size(weights):
    numerator = (np.sum(weights))**2
    denominator = np.sum(weights**2)
    return numerator / denominator

# Compute ESS
ess = compute_effective_sample_size(df['weights'].values)
print(f"\nEffective Sample Size (ESS): {ess:.2f} out of {len(df)} total samples")

# Cap weights at 99th percentile
threshold = np.percentile(df['weights'], 99)
df_trimmed = df[df['weights'] <= threshold]
print(f"Trimmed dataset size: {df_trimmed.shape[0]} (from original {df.shape[0]})")

ess_trimmed= compute_effective_sample_size(df_trimmed['weights'].values)
print(f"\nEffective Sample Size (ESS): {ess_trimmed:.2f} out of {len(df)} total samples")

# ---------------------------
# Cap weights at 99th percentile and plot trimmed weights
# ---------------------------
threshold = np.percentile(df['weights'], 99)
df_trimmed = df[df['weights'] <= threshold]

# ---------------------------
# Summary of Trimmed Weights
# ---------------------------
print("\nSummary of Trimmed IPTW Weights:")
print(f"Mean: {df_trimmed['weights'].mean():.4f}")
print(f"Standard Deviation: {df_trimmed['weights'].std():.4f}")
print(f"Min: {df_trimmed['weights'].min():.4f}")
print(f"Max: {df_trimmed['weights'].max():.4f}")

ess_trimmed = compute_effective_sample_size(df_trimmed['weights'].values)
print(f"Effective Sample Size (Trimmed): {ess_trimmed:.2f} out of {len(df)} total samples")

# ---------------------------
# Save the weights DataFrame to a pickle file
# ---------------------------

# Display for user

id_col = 'Pat ID' if 'Pat ID' in df.columns else df.columns[0]
weights_file_pickle = os.path.join(datasets_dir, "weights.pkl")
df_trimmed[[id_col, 'weights']].to_pickle(weights_file_pickle)
print("Weights have been saved to pickle file:", weights_file_pickle)

# ---------------------------
# Save IPTW Processed Data to CSV
# ---------------------------
output_file = os.path.join(datasets_dir, "df_with_iptw_weights.csv")
df_trimmed.to_csv(output_file, index=False)
print(f"Processed Data with Weights saved to: {output_file}")
