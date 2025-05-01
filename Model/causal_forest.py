
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from sklearn.model_selection import KFold, ParameterGrid
from econml.dml import CausalForestDML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib

from global_config import datasets_dir, model_dir
from Model.cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)

# ---------------------------
# Load Data & IPTW Weights
# ---------------------------
df = pd.read_csv(os.path.join(datasets_dir, "df_with_merged_groups.csv"))
weights_df = pd.read_pickle(os.path.join(datasets_dir, "weights.pkl"))
df = df.merge(weights_df, on='Pat ID')

# Filter valid entries
df_cf = df.dropna(subset=[treatment_col, outcome_col, 'weights']).copy()

# Cast categories to string
for col in categorical_cols:
    df_cf[col] = df_cf[col].astype(str)

# Setup preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols)
])

# Preprocess all data
df_cf = df_cf.reset_index(drop=True)
X_all = preprocessor.fit_transform(df_cf[covariate_cols])
Y_all = df_cf[outcome_col].astype(int).values
T_all = df_cf[treatment_col].astype(int).values
W_all = df_cf['weights'].values

# ---------------------------
# Hyperparameter Tuning Setup
# ---------------------------
param_grid = {
    "n_estimators": [500, 1000],
    "min_samples_leaf": [5, 10],
    "max_depth": [5, 10, None],
    "max_samples": [0.3, 0.5]
}

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
best_params = None
best_mse = float('inf')

for params in ParameterGrid(param_grid):
    fold_mses = []
    print(f"\nTrying hyperparameters: {params}")

    for train_idx, val_idx in kf.split(X_all):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        Y_train, Y_val = Y_all[train_idx], Y_all[val_idx]
        T_train, T_val = T_all[train_idx], T_all[val_idx]
        W_train = W_all[train_idx]

        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=LogisticRegression(max_iter=1000),
            discrete_treatment=True,
            random_state=42,
            cv=3,
            **params
        )
        cf_model.fit(Y_train, T_train, X=X_train, sample_weight=W_train)
        cate_val = cf_model.effect(X_val)
        pseudo_outcome = cate_val * T_val
        mse = mean_squared_error(Y_val, pseudo_outcome)
        fold_mses.append(mse)

    avg_mse = np.mean(fold_mses)
    print(f"Avg Proxy MSE: {avg_mse:.4f}")
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_params = params

print(f"\nBest hyperparameters selected: {best_params}")

# ---------------------------
# Final Model Fit with Best Params
# ---------------------------
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    random_state=42,
    cv=3,
    **best_params
)
cf_model.fit(Y_all, T_all, X=X_all, sample_weight=W_all)

cate_preds = cf_model.effect(X_all)
cate_lower, cate_upper = cf_model.effect_interval(X_all)

# Attach results
df_cf["CATE"] = cate_preds
df_cf["CATE_lower"] = cate_lower
df_cf["CATE_upper"] = cate_upper

# ---------------------------
# Subgroup CATE Analysis Based on Covariates
# ---------------------------
subgroup_vars = [
    'age_group', 'tumor_size_group', 'cci_group',
    'Histological diagnosis', 'anatomic_region_label', 'Gender',
    'grade_clean', 'Affected tissue'
]

subgroup_results = []
print("\nSubgroup-specific CATE summary:")
for col in subgroup_vars:
    if col in df_cf.columns:
        summary = df_cf.groupby(col).agg(
            Mean_CATE=('CATE', 'mean'),
            Std_CATE=('CATE', 'std'),
            Count=('CATE', 'count')
        ).reset_index()
        summary['Z_Score'] = summary['Mean_CATE'] / summary['Std_CATE']
        summary['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(summary['Z_Score'])))
        summary['Significant'] = summary['p_value'] < 0.05
        print(f"\n>>> Subgroup analysis by: {col}")
        print(summary[[col, 'Mean_CATE', 'Std_CATE', 'Z_Score', 'p_value', 'Significant']])
        summary['Subgroup_Var'] = col
        subgroup_results.append(summary)

# ---------------------------
# Overall ATE
# ---------------------------
ate = df_cf["CATE"].mean()
ate_se = df_cf["CATE"].std() / np.sqrt(df_cf.shape[0])
print(f"\nOverall ATE: {ate:.4f} ± {ate_se:.4f}")

# ---------------------------
# Save Results
# ---------------------------
joblib.dump(cf_model, os.path.join(model_dir, "cf_model.pkl"))
joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
df_cf.to_csv(os.path.join(datasets_dir, "cf_results.csv"), index=False)

# ---------------------------
# Answering the Hypothesis
# ---------------------------
any_significant = any([df['Significant'].any() for df in subgroup_results])
if not any_significant:
    print("\nFail to reject the null hypothesis: No significant heterogeneity detected.")
else:
    print("\nReject the null hypothesis: Significant heterogeneity in treatment effect detected.")