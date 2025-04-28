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

# Preprocessing - Convert categorical variables if needed
# For example, convert categorical columns to numeric
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X_encoded, T, Y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)
