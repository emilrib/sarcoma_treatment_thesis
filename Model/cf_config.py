import os.path
import pandas as pd
from global_config import datasets_dir

input_file = os.path.join(datasets_dir, "df_with_merged_groups.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df

df = import_file(input_file)

df_with_subgroups_without_patid = df.drop(df.columns[0], axis=1)

#print(df[['survival_status', 'Chemo_status']].value_counts())
#print(df.columns.tolist())


# ---------------------------
# Variables Definition for the Causal Forest when testing for overall survival
# ---------------------------

treatment_col = 'chemo_status'
outcome_col = 'survival_status'

numeric_cols = ['reoperation_label', 'metastasis_label', 'radiation_status']

# Define the columns you want to exclude from covariates
exclude_cols = ['Pat ID', treatment_col, outcome_col, 'age','cci_Timo', 'Tumor maximal size (mm)', 'Histological diagnosis', 'anatomic_region_label', 'Gender', 'grade_clean', 'Affected tissue',
                    'tumor_size_group', 'age_group', 'cci_group', numeric_cols]  # Columns to exclude from covariate_cols

# List all columns in the DataFrame
encoded_covariates = [col for col in df.columns if col not in exclude_cols and col not in numeric_cols]
categorical_cols = encoded_covariates
#print(categorical_cols)
covariate_cols = numeric_cols + categorical_cols
#print(covariate_cols)

