import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
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
    treatment_col, outcome_col, covariate_cols, categorical_cols, numeric_cols
)

# ---------------------------
# Load Data & IPTW Weights
# ---------------------------
df = pd.read_csv(os.path.join(datasets_dir, "df_with_merged_groups.csv"))
weights_df = pd.read_pickle(os.path.join(datasets_dir, "weights.pkl"))
df = df.merge(weights_df, on='Pat ID')

# Filter valid entries
df_cf = df.dropna(subset=[treatment_col, outcome_col, 'weights']).copy()
for col in categorical_cols:
    df_cf[col] = df_cf[col].astype(str)

# Split filtered data
train_df, test_df = train_test_split(df_cf, test_size=0.2, random_state=42)

# Preprocessing pipeline
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

# Fit-transform train, transform test
X_train = preprocessor.fit_transform(train_df[covariate_cols])
Y_train = train_df[outcome_col].astype(int).values
T_train = train_df[treatment_col].astype(int).values
W_train = train_df['weights'].values

X_test = preprocessor.transform(test_df[covariate_cols])
Y_test = test_df[outcome_col].astype(int).values
T_test = test_df[treatment_col].astype(int).values
W_test = test_df['weights'].values

# ---------------------------
# Hyperparameter Tuning Setup
# ---------------------------
param_grid = {
    "n_estimators": [1000],
    "min_samples_leaf": [3, 5, 9],
    "max_depth": [2, 3, 5],
    "max_samples": [0.3, 0.5]
}

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
best_params = None
best_mse = float('inf')

for params in ParameterGrid(param_grid):
    fold_mses = []
    print(f"\nTrying hyperparameters: {params}")

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]
        T_tr, T_val = T_train[train_idx], T_train[val_idx]
        W_tr = W_train[train_idx]

        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=LogisticRegression(max_iter=1000),
            discrete_treatment=True,
            random_state=42,
            cv=3,
            **params
        )
        cf_model.fit(Y_tr, T_tr, X=X_tr, sample_weight=W_tr)
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
# Final Model Fit
# ---------------------------
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    random_state=42,
    cv=3,
    **best_params
)
cf_model.fit(Y_train, T_train, X=X_train, sample_weight=W_train)

# Predict on test data
cate_test = cf_model.effect(X_test)
cate_lower_test, cate_upper_test = cf_model.effect_interval(X_test)
test_df = test_df.reset_index(drop=True)
test_df["CATE"] = cate_test
test_df["CATE_lower"] = cate_lower_test
test_df["CATE_upper"] = cate_upper_test

# ---------------------------
# Subgroup CATE Analysis (Test Set)
# ---------------------------
subgroup_vars = [
    'age_group', 'tumor_size_group', 'cci_group',
    'anatomic_region_label', 'Gender', 'grade_clean', 'Affected tissue', 'radiation_status', 'metastasis_label', 'reoperation_label'
]

subgroup_results_test = []
print("\nSubgroup-specific CATE summary (on test set):")
min_group_size = 10
for col in subgroup_vars:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)
        summary = test_df.groupby(col).agg(
            Mean_CATE=('CATE', 'mean'),
            Std_CATE=('CATE', 'std'),
            Count=('CATE', 'count')
        ).reset_index()
        summary = summary[summary['Count'] >= min_group_size]
        summary['Z_Score'] = summary['Mean_CATE'] / summary['Std_CATE']
        summary['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(summary['Z_Score'])))
        summary['p_value'] = summary['p_value'].clip(lower=1e-16)
        summary['Significant'] = summary['p_value'] < 0.05
        summary['p_value'] = summary['p_value'].apply(lambda x: f"{x:.2e}")

        print(f"\n>>> Subgroup analysis by: {col}")
        with pd.option_context('display.float_format', '{:0.2e}'.format):
            print(summary[[col, 'Mean_CATE', 'Std_CATE', 'Z_Score', 'p_value', 'Significant']])

        summary['Subgroup_Var'] = col
        subgroup_results_test.append(summary)

# ---------------------------
# Overall ATE on Test Set
# ---------------------------
ate = test_df["CATE"].mean()
ate_se = test_df["CATE"].std() / np.sqrt(test_df.shape[0])
print(f"\nTest Set ATE: {ate:.4f} ± {ate_se:.4f}")

# ---------------------------
# Plot: Subgroup Mean ± 2*SD Estimates
# ---------------------------
combined_subgroups_test = pd.concat(subgroup_results_test, ignore_index=True)
combined_subgroups_test['ranking'] = (
    combined_subgroups_test['Subgroup_Var'] + ": " + combined_subgroups_test[combined_subgroups_test.columns[0]].astype(str)
)
combined_subgroups_test = combined_subgroups_test.sort_values('Mean_CATE')

plt.figure(figsize=(12, 6))
plt.errorbar(
    x=combined_subgroups_test['ranking'],
    y=combined_subgroups_test['Mean_CATE'],
    yerr=2 * combined_subgroups_test['Std_CATE'],
    fmt='o', capsize=4, ecolor='gray', color='blue', label='Mean ± 2*SD'
)
plt.xticks(rotation=90)
plt.axhline(0, color='black', linestyle='--')
plt.title('Subgroup Mean CATE Estimates (Test Set) with ±2 SD')
plt.xlabel('Subgroup')
plt.ylabel('Estimated CATE')
plt.tight_layout()
plt.grid(True)
plt.legend(loc='lower right')
plt.show()

# ---------------------------
# Save Results
# ---------------------------
joblib.dump(cf_model, os.path.join(model_dir, "cf_model.pkl"))
joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
test_df.to_csv(os.path.join(datasets_dir, "cf_test_results.csv"), index=False)
combined_subgroups_test.to_csv(os.path.join(datasets_dir, "test_subgroup_summary.csv"), index=False)