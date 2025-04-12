import re
import os.path
import pandas as pd
from datetime import datetime
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

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

print(df.columns.tolist())

T = df[['Chemo_status', 'chemo_duration']]  # Treatment variable (chemo usage indicator)
Y = df[['deceased_by_disease', 'overall_survival_days', 'status_label_number']].mean(axis=1)  # Outcome of interest

# Covariates (X) as a list of relevant columns
X = df[['age', 'gender_label', 'Histological diagnosis',
        '(W) Other diagnoses?_Timo', 'Tumor maximal size (mm)', 'reoperation_label',
        'number_all_operation_Timo', 'status_label_number', 'chemo_duration',
        'radiation_status', 'Days_to_metastasis_discovery', 'metastasis_status',
        'metastasis_label','Reason for Chemotherapy', 'Anatomic side of lesion']]




"""

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# Define models for nuisance functions
model_y = RandomForestRegressor(n_estimators=100)
model_t = RandomForestRegressor(n_estimators=100)

# Causal forest estimator
est = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    n_estimators=100,
    min_samples_leaf=10,
    max_depth=10,
    verbose=1,
    random_state=42,
)

# Fit model
est.fit(Y_train, T_train, X=X_train)
"""