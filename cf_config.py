import os.path
import pandas as pd

if '__file__' in globals():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
else:
    # Fallback for interactive environments
    current_dir = os.getcwd()

output_directory = os.path.join(current_dir, "output")

input_file = os.path.join(current_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df[['survival_status_binary', 'Chemo_status']].value_counts())
#print(df.columns.tolist())

"""
DEFINE VARIABLES FOR CAUSAL FOREST
"""

# model considering survival_status as outcome variable as well chemo status

treatment_col = 'Chemo_status'
outcome_col = 'deceased_by_disease'
covariate_cols = ['age', 'gender_label', 'reoperation_label', 'anatomic_region_label', 'metastasis_label',
                  'metastasis_status', 'radiation_status', 'Tumor maximal size (mm)', 'anatomicregion_group_Timo', 'Histological diagnosis',
                  '(W) Other diagnoses?_Timo']

# Split covariates into numeric and categorical
categorical_cols = ['Histological diagnosis', '(W) Other diagnoses?_Timo', 'anatomic_region_label']
numeric_cols = ['age', 'gender_label', 'reoperation_label', 'metastasis_label',
                  'metastasis_status', 'radiation_status', 'Tumor maximal size (mm)', 'anatomicregion_group_Timo']

