import os.path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt

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

print(df.head(3))
#print(df.columns.tolist())

"""
DEFINE VARIABLES FOR CAUSAL FOREST
"""

# model considering survival_status as outcome variable as well chemo status

treatment_col = 'Chemo_status'
outcome_col = 'survival_status_binary'
covariate_cols = ['age', 'gender_label', 'reoperation_label', 'anatomic_region_label', 'metastasis_label',
                  'metastasis_status', 'radiation_status', 'Tumor maximal size (mm)', 'anatomicregion_group_Timo', 'Histological diagnosis',
                  '(W) Other diagnoses?_Timo', 'chemo_duration']

# Split covariates into numeric and categorical
categorical_cols = ['Histological diagnosis', '(W) Other diagnoses?_Timo', 'anatomic_region_label']
numeric_cols = ['age', 'gender_label', 'reoperation_label', 'metastasis_label',
                  'metastasis_status', 'radiation_status', 'Tumor maximal size (mm)', 'anatomicregion_group_Timo']

# Drop rows with missing T or Y
df_cf = df.dropna(subset=[treatment_col, outcome_col])

# Extract T and Y
T = df_cf[treatment_col].astype(int).values  # ensure binary
Y = df_cf[outcome_col].values

# Preprocess covariates
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numeric data
        ('scaler', StandardScaler())  # Standardize numerical features
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical data
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
    ]), categorical_cols)
])


# Transform X from DataFrame
X = preprocessor.fit_transform(df_cf[covariate_cols])
X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)

print(f"Shape of X after preprocessing: {X.shape}")
if X.shape[1] == 0 or X.ndim != 2:
    raise ValueError("Preprocessed X is empty or not 2D. Check your input data and preprocessing pipeline.")

# Fit Causal Forest model
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
    model_t=LogisticRegression(),
    discrete_treatment=True,
    n_estimators=1000,
    random_state=42
)

cf_model.fit(Y, T, X=X)

# Estimate ATE + confidence interval
ate = cf_model.ate(X)
ate_lb, ate_ub = cf_model.ate_interval(X)

print(f"\n Average Treatment Effect (ATE): {ate:.2f} survival days")
print(f"95% Confidence Interval: ({ate_lb:.2f}, {ate_ub:.2f})")

# Estimate CATE for individuals
cate = cf_model.effect(X)

# Plot CATE distribution
plt.figure(figsize=(10, 5))
plt.hist(cate, bins=30, edgecolor='k')
plt.title('Estimated Treatment Effect (CATE) Distribution')
plt.xlabel('Estimated Survival Days Gained from Chemotherapy')
plt.ylabel('Number of Patients')
plt.grid(True)
plt.tight_layout()
plt.show()