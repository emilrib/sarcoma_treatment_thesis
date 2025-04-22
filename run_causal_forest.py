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
from sklearn.model_selection import StratifiedKFold

# Load data
data_path = os.path.join(os.getcwd(), "dataset_labelled.csv")
df = pd.read_csv(data_path)

from cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)


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

print("Joint distribution (treatment vs outcome):")
print(pd.crosstab(df['Chemo_status'], df['survival_status_binary']))
print(df['Chemo_status'].value_counts(normalize=True))
print(df['survival_status_binary'].value_counts(normalize=True))

# Fit Causal Forest model
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    n_estimators=1000,
    min_samples_leaf=5,
    max_depth=10,
    random_state=42,
    cv=1
)
cf_model.fit(Y, T, X=X)


# Estimate CATE for individuals
cate = cf_model.effect(X, T0=0, T1=1)

cate_percent = cate * 100

# Plot CATE distribution as change in survival probability
plt.figure(figsize=(10, 5))
plt.hist(cate_percent, bins=30, edgecolor='k')
plt.title('Estimated Treatment Effect (CATE) Distribution')
plt.xlabel('Estimated Change in Survival Probability (%) from Chemotherapy')
plt.ylabel('Number of Patients')
plt.grid(True)
plt.tight_layout()
plt.show()