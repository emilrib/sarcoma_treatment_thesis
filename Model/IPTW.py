import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from global_config import datasets_dir
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from Model.cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols,
    subgroup_col
)

# Load the imbalanced variables (SMD > 0.1)
imbalance_file = os.path.join(datasets_dir, "imbalance_vars.pkl")

with open(imbalance_file, 'rb') as f:
    imbalanced_vars = pickle.load(f)
imbalanced_vars_filtered = imbalanced_vars[imbalanced_vars["SMD"] > 0.1]

# Extract imbalanced numeric and categorical covariates
numeric_imbalanced = [col for col in numeric_cols if col in imbalanced_vars["Covariate"].values]
categorical_imbalanced = [col for col in categorical_cols if col in imbalanced_vars["Covariate"].values]

# Load the dataset
input_file_groups = os.path.join(datasets_dir, "dataset_with_subgroups.csv")
df = pd.read_csv(input_file_groups)

# ---------------------------
# Propensity Score Weighting (IPTW)
# ---------------------------
#Preprocessor for imbalanced covariates (simple imputation)
imbalanced_preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), numeric_cols),  # Apply imputation for numeric columns
    ('cat', SimpleImputer(strategy='most_frequent'), subgroup_col)  # Use one-hot encoded subgroup columns
])

# Apply the preprocessor and create the design matrix X_ps
X_ps = imbalanced_preprocessor.fit_transform(df)
X_ps_df = pd.DataFrame(X_ps, columns=numeric_cols + subgroup_col)

# Ensure that all columns are numeric after encoding (just a precaution)
X_ps_df = X_ps_df.apply(pd.to_numeric, errors='coerce')

# Fit Logistic Regression for propensity scores
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_ps_df, df[treatment_col])

# Get the propensity scores (probabilities of being treated)
ps_scores = ps_model.predict_proba(X_ps_df)[:, 1]

# Calculate weights for IPTW
treated = df[treatment_col] == 1
weights = treated / ps_scores + (1 - treated) / (1 - ps_scores)

# Add the computed weights to the DataFrame (if needed for later analysis)
df['weights'] = weights

# Now you can proceed with further analysis or model fitting using the weights
print("Propensity Score Weights calculated and added to the DataFrame.")


# ---------------------------
# Print a Summary of IPTW
# ---------------------------

# Basic statistics of the weights
print("\nSummary of IPTW Weights:")
print(f"Mean of Weights: {weights.mean():.4f}")
print(f"Standard Deviation of Weights: {weights.std():.4f}")
print(f"Minimum Weight: {weights.min():.4f}")
print(f"Maximum Weight: {weights.max():.4f}")
print(f"Range of Weights: ({weights.min():.4f}, {weights.max():.4f})")

# Distribution of the weights (Histogram)
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
weights_file_pickle = os.path.join(datasets_dir, "weights.pkl")
df[['Pat ID', 'weights']].to_pickle(weights_file_pickle)

print("Weights have been saved to pickle file:", weights_file_pickle)

