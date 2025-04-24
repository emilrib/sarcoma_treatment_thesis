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
from sklearn.model_selection import train_test_split

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

df_cf = df.dropna(subset=[treatment_col, outcome_col]).copy()

# Ensure all categoricals are strings
for col in categorical_cols:
    df_cf[col] = df_cf[col].astype(str)

# --------------------------------------
# Train/test split (stratified by treatment)
# --------------------------------------
df_train, df_test = train_test_split(
    df_cf, test_size=0.3, stratify=df_cf[treatment_col], random_state=42
)

# --------------------------------------
# Preprocessing pipeline
# --------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

# --------------------------------------
# Prepare training data
# --------------------------------------
T_train = df_train[treatment_col].astype(int).values.ravel()
Y_train = df_train[outcome_col].astype(int).values.ravel()
X_train = preprocessor.fit_transform(df_train[covariate_cols])
X_train = X_train.toarray() if hasattr(X_train, 'toarray') else np.asarray(X_train)

# --------------------------------------
# Fit Causal Forest on training data
# --------------------------------------
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    n_estimators=1000,
    min_samples_leaf=5,
    max_depth=10,
    random_state=42,
    cv=1  # Disable cross-fitting due to small and imbalanced dataset
)

cf_model.fit(Y_train, T_train, X=X_train)

# --------------------------------------
# Prepare test data
# --------------------------------------
T_test = df_test[treatment_col].astype(int).values.ravel()
Y_test = df_test[outcome_col].astype(int).values.ravel()
X_test = preprocessor.transform(df_test[covariate_cols])
X_test = X_test.toarray() if hasattr(X_test, 'toarray') else np.asarray(X_test)

# --------------------------------------
# Estimate CATE on test set
# --------------------------------------
cate = cf_model.effect(X_test, T0=0, T1=1)
df_test['CATE'] = cate
df_test['CATE_percent'] = cate * 100
df_test['treatment_recommendation'] = np.where(df_test['CATE'] > 0, 'Recommend Chemo', 'Do Not Recommend')

# --------------------------------------
# Plot CATE distribution on test set
# --------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(df_test['CATE'], bins=30, edgecolor='k')
plt.title('Estimated CATE Distribution on Test Set (Survival Probability Δ from Chemo)')
plt.xlabel('Estimated Change in Survival Probability (%)')
plt.ylabel('Number of Patients')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------
# (Optional) Export test results
# --------------------------------------
# df_test.to_csv("test_results_with_cate.csv", index=False)