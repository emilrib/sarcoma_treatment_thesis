import os.path
import pandas as pd
from global_config import datasets_dir

input_file_covariates = os.path.join(datasets_dir, "df_with_groups.csv")
input_file_subgroups = os.path.join(datasets_dir, "dataset_with_subgroups.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df

df_with_covariates = import_file(input_file_covariates)
df_with_subgroups = import_file(input_file_subgroups)

df_with_subgroups_without_patid = df_with_subgroups.drop(df_with_subgroups.columns[0], axis=1)

#print(df[['survival_status', 'Chemo_status']].value_counts())
#print(df.columns.tolist())

"""
DEFINE VARIABLES FOR CAUSAL FOREST
"""

# model considering survival_status as outcome variable as well chemo status

treatment_col = 'Chemo_status'
outcome_col = 'survival_status'
covariate_cols = ['age_group', 'Gender', 'Histological diagnosis', 'cci_group', 'grade_clean','Affected tissue',
                  'reoperation_label', 'anatomic_region_label', 'metastasis_label', 'radiation_status', 'tumor_size_group']

# Split covariates into numeric and categorical
categorical_cols = ['Histological diagnosis', 'anatomic_region_label', 'Gender','grade_clean', 'Affected tissue',
                    'tumor_size_group','age_group', 'cci_group' ]
numeric_cols = ['reoperation_label', 'metastasis_label',
                 'radiation_status' ]

subgroup_col = list(df_with_subgroups_without_patid.columns)

