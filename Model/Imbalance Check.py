import os
import pandas as pd
import numpy as np
import pickle
from global_config import datasets_dir
from Analysis.cf_config import (
    treatment_col,
    covariate_cols,
    numeric_cols
)

# Load dataset
input_file = os.path.join(datasets_dir, "df_with_merged_groups.csv")
df = pd.read_csv(input_file)

# ---------------------------
# Covariate Imbalance Check (Using Config Variables)
# ---------------------------
def compute_standardized_mean_diff(df, treatment_col, covariates, numeric_cols):
    smd_results = []
    treated = df[df[treatment_col] == 1]
    untreated = df[df[treatment_col] == 0]

    for col in covariates:
        if col in numeric_cols:
            mean_t = treated[col].mean()
            mean_u = untreated[col].mean()
            std_pooled = np.sqrt((treated[col].var() + untreated[col].var()) / 2)
            smd = abs(mean_t - mean_u) / std_pooled if std_pooled > 0 else 0
        else:
            prop_t = treated[col].mean()
            prop_u = untreated[col].mean()
            smd = abs(prop_t - prop_u)

        smd_results.append((col, smd))

    return pd.DataFrame(smd_results, columns=["Covariate", "SMD"]).sort_values(by="SMD", ascending=False)

# Compute SMD for covariates
imbalance_df = compute_standardized_mean_diff(df, treatment_col, covariate_cols, numeric_cols)

# Filter SMDs greater than threshold
imbalanced_vars = imbalance_df[imbalance_df["SMD"] > 0.1]
print("\nCovariate Imbalance Summary:")
print(imbalanced_vars)

# Save detailed SMD results
smd_output_file = os.path.join(datasets_dir, "imbalanced_vars.pkl")
with open(smd_output_file, 'wb') as f:
    pickle.dump(imbalanced_vars, f)
print("SMD scores saved to:", smd_output_file)

# Save list of variable names with SMD > 0.1
imbalanced_var_names = imbalanced_vars["Covariate"].values
target_iptw_file = os.path.join(datasets_dir, "imbalance_vars_for_iptw.pkl")
with open(target_iptw_file, 'wb') as f:
    pickle.dump(imbalanced_var_names, f)
print("Imbalanced variable names saved to:", target_iptw_file)
