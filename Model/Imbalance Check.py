import os
import pandas as pd
import numpy as np
from global_config import datasets_dir
import pickle
from Model.cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)

# Input/output locations
input_file_groups = os.path.join(datasets_dir, "df_with_groups.csv")

df = pd.read_csv(input_file_groups)


# ---------------------------
# Covariate Imbalance Check (Within Each Subgroup)
# ---------------------------
def compute_standardized_mean_diff(df, treatment_col, covariates):
    smd_results = []

    # Separate treated and untreated groups
    treated = df[df[treatment_col] == 1]
    untreated = df[df[treatment_col] == 0]

    for col in covariates:
        # Check if the covariate is categorical or numeric
        if df[col].dtype == 'object':  # Categorical covariate
            mode_t = treated[col].mode()[0] if not treated[col].mode().empty else None
            mode_u = untreated[col].mode()[0] if not untreated[col].mode().empty else None
            freq_t = (treated[col] == mode_t).mean() if mode_t else 0
            freq_u = (untreated[col] == mode_u).mean() if mode_u else 0
            smd = abs(freq_t - freq_u)
        else:  # Numeric covariates
            mean_t = treated[col].mean()
            mean_u = untreated[col].mean()
            std_pooled = np.sqrt((treated[col].var() + untreated[col].var()) / 2)
            smd = abs(mean_t - mean_u) / std_pooled if std_pooled > 0 else 0

        smd_results.append((col, smd))

    return pd.DataFrame(smd_results, columns=["Covariate", "SMD"]).sort_values(by="SMD", ascending=False)


# Perform imbalance check on all original covariates (including categorical and numeric)
imbalance_df = compute_standardized_mean_diff(df, treatment_col=treatment_col, covariates=covariate_cols)

# Filter covariates with SMD > 0.1 (can adjust threshold)
imbalanced_vars = imbalance_df[imbalance_df["SMD"] > 0.0]

# Print a summary of covariate imbalance
print("\nCovariate Imbalance Summary:")
print(imbalance_df)

# In the imbalance check script, after calculating the imbalance
#imbalanced_vars = imbalance_df[imbalance_df["SMD"] > 0.1]

# Save the imbalance DataFrame with the SMD scores to a pickle file
imbalance_file = os.path.join(datasets_dir, "imbalance_vars.pkl")

with open(imbalance_file, 'wb') as f:
    pickle.dump(imbalanced_vars, f)

print("SMD scores have been saved to pickle file:", imbalance_file)