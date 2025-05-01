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

# Input/output locations
input_file = os.path.join(datasets_dir, "df_with_merged_groups.csv")
df = pd.read_csv(input_file)

# ---------------------------
# Load Data
# ---------------------------
df_cf = df.dropna(subset=[treatment_col, outcome_col]).copy()

# Convert categorical columns to strings for consistency
for col in categorical_cols:
    df_cf[col] = df_cf[col].astype(str)

# ---------------------------
# Preprocessing Setup
# ---------------------------
# Setup for the preprocessor to handle numeric and categorical columns
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

# ---------------------------
# Import IPTW Weights
# ---------------------------
# Import previously calculated IPTW weights
weights_file_pickle = os.path.join(datasets_dir, "weights.pkl")
weights_df = pd.read_pickle(weights_file_pickle)
weights = weights_df['weights'].values

# ---------------------------
# Hyperparameter Tuning Setup
# ---------------------------
param_grid = {
    "n_estimators": [500, 1000],
    "min_samples_leaf": [5, 10],
    "max_depth": [5, 10, None],
    "max_samples": [0.3, 0.5]
}

# ---------------------------
# Out-of-Fold CATE Estimation with Hyperparameter Tuning
# ---------------------------
K = 5  # Number of folds
Q = 5  # Number of quantiles for ranking

kf = KFold(n_splits=K, shuffle=True, random_state=42)

df_cf = df_cf.reset_index(drop=True)
all_cate = np.zeros(df_cf.shape[0])
all_ci_lower = np.zeros(df_cf.shape[0])
all_ci_upper = np.zeros(df_cf.shape[0])

best_mse = float('inf')
best_params = None

# Hyperparameter tuning by training on K-1 folds and evaluating on hold-out
for param_set in ParameterGrid(param_grid):
    fold_mses = []
    print(f"\nTrying params: {param_set}")
    for train_idx, val_idx in kf.split(df_cf):
        df_train, df_val = df_cf.loc[train_idx], df_cf.loc[val_idx]

        # Apply preprocessing to both train and validation data
        X_train = np.asarray(preprocessor.fit_transform(df_train[covariate_cols]))
        X_val = np.asarray(preprocessor.transform(df_val[covariate_cols]))

        T_train = df_train[treatment_col].astype(int).values.ravel()
        Y_train = df_train[outcome_col].astype(int).values.ravel()
        T_val = df_val[treatment_col].astype(int).values.ravel()
        Y_val = df_val[outcome_col].astype(int).values.ravel()

        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=LogisticRegression(max_iter=1000),
            discrete_treatment=True,
            random_state=42,
            cv=3,
            **param_set
        )
        cf_model.fit(Y_train, T_train, X=X_train, sample_weight=weights[train_idx])

        cate_val = cf_model.effect(X_val)
        pseudo_outcome = cate_val * T_val
        mse = mean_squared_error(Y_val, pseudo_outcome)
        fold_mses.append(mse)

    avg_mse = np.mean(fold_mses)
    print(f"Average Proxy MSE: {avg_mse:.4f}")

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_params = param_set

# ---------------------------
# Train best model on full data (with best hyperparameters)
# ---------------------------
print(f"\nBest params selected: {best_params}")

# Train the model with the best hyperparameters
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    random_state=42,
    cv=3,
    **best_params
)
cf_model.fit(Y_train, T_train, X=X_train, sample_weight=weights)

# ---------------------------
# Save predictions and confidence intervals
# ---------------------------
cate_val = cf_model.effect(X_val)
ci_lower, ci_upper = cf_model.effect_interval(X_val)

all_cate[val_idx] = cate_val
all_ci_lower[val_idx] = ci_lower
all_ci_upper[val_idx] = ci_upper

df_cf["CATE"] = all_cate
df_cf["CATE_lower"] = all_ci_lower
df_cf["CATE_upper"] = all_ci_upper

# ---------------------------
# Interpretation with SingleTreeCateInterpreter
# ---------------------------
from econml.cate_interpreter import SingleTreeCateInterpreter

interpreter = SingleTreeCateInterpreter(max_depth=3)
interpreter.interpret(cf_model, X_val)

plt.figure(figsize=(12, 8))
interpreter.plot()
plt.title("Single Tree Approximation of CATE")
plt.tight_layout()
plt.show()

# ---------------------------
# Rank into Quantiles
# ---------------------------
df_cf['CATE_quantile'] = pd.qcut(df_cf['CATE'], q=Q, labels=[f"Q{i + 1}" for i in range(Q)])

# ---------------------------
# Analyze Quantile Groups
# ---------------------------
print("\n CATE by Quantile Group:")
grouped_summary = df_cf.groupby('CATE_quantile').agg(
    Estimate=('CATE', 'mean'),
    StdError=('CATE', 'std'),
    Count=('CATE', 'count')
).reset_index()

grouped_summary['Z_Score'] = grouped_summary['Estimate'] / grouped_summary['StdError']

# Determine significance based on Z-Score
grouped_summary['Significant'] = grouped_summary['Z_Score'].apply(lambda x: abs(x) > 1.96)

# Print the updated summary
print("\n CATE by Quantile Group with Significance (Based on Z-Score):")
print(grouped_summary[['CATE_quantile', 'Estimate', 'StdError', 'Z_Score', 'Significant', 'Count']])

# ---------------------------
# Calculate Overall ATE
# ---------------------------
ate = df_cf["CATE"].mean()
ate_se = df_cf["CATE"].std() / np.sqrt(df_cf.shape[0])

print(f"\n Overall ATE (Average Treatment Effect): {ate:.4f}")
print(f"Standard Error of ATE: {ate_se:.4f}")

# ---------------------------
# Hypothesis Testing for CATE Estimates
# ---------------------------
grouped_summary['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(grouped_summary['Z_Score'])))
grouped_summary['Significant'] = grouped_summary['p_value'] < 0.05

# Print the summary with p-values and significance
print("\nCATE by Quantile Group with p-values and Significance:")
print(grouped_summary[['CATE_quantile', 'Estimate', 'StdError', 'Z_Score', 'p_value', 'Significant']])

# ---------------------------
# Save Model and Results
# ---------------------------
# Save trained causal forest model
joblib.dump(cf_model, os.path.join(model_dir, "cf_model.pkl"))
joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))

# Save predictions to CSV
output_file = os.path.join(datasets_dir, "dataset_with_cate.csv")
df.to_csv(output_file, index=False)

print(f"Model and results saved successfully.")

