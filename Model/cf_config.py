import os.path
import pandas as pd
from global_config import datasets_dir

input_file = os.path.join(datasets_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

#print(df[['survival_status', 'Chemo_status']].value_counts())
#print(df.columns.tolist())

"""
DEFINE VARIABLES FOR CAUSAL FOREST
"""

# model considering survival_status as outcome variable as well chemo status

treatment_col = 'Chemo_status'
outcome_col = 'survival_status'
covariate_cols = ['age_group', 'Gender', 'Histological diagnosis', 'cci_Timo', 'grade_clean','Affected tissue',
                  'reoperation_label', 'anatomic_region_label', 'metastasis_label', 'radiation_status', 'tumor_size_group']

# Split covariates into numeric and categorical
categorical_cols = ['Histological diagnosis', 'anatomic_region_label', 'Gender','grade_clean', 'Affected tissue',
                    'tumor_size_group','age_group' ]
numeric_cols = ['reoperation_label', 'metastasis_label',
                 'radiation_status', 'cci_Timo' ]

