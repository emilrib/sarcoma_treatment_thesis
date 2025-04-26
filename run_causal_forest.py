from cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)

import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from econml.dml import CausalForestDML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import seaborn as sns

data_path = os.path.join(os.getcwd(), "dataset_labelled.csv")
df = pd.read_csv(data_path)
# Load and preprocess data

from cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)

# ---------------------------
# Load Data
# ---------------------------
df_cf = df.dropna(subset=[treatment_col, outcome_col]).copy()
for col in categorical_cols:
    df_cf[col] = df_cf[col].astype(str)

# ---------------------------
# Preprocessing Setup
# ---------------------------
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
# Covariate Imbalance Check
# ---------------------------
def compute_standardized_mean_diff(df, treatment_col, covariates):
    treated = df[df[treatment_col] == 1]
    untreated = df[df[treatment_col] == 0]
    smd_results = []
    for col in covariates:
        if df[col].dtype == 'object':
            mode_t = treated[col].mode()[0] if not treated[col].mode().empty else None
            mode_u = untreated[col].mode()[0] if not untreated[col].mode().empty else None
            freq_t = (treated[col] == mode_t).mean() if mode_t else 0
            freq_u = (untreated[col] == mode_u).mean() if mode_u else 0
            smd = abs(freq_t - freq_u)
        else:
            mean_t = treated[col].mean()
            mean_u = untreated[col].mean()
            std_pooled = np.sqrt((treated[col].var() + untreated[col].var()) / 2)
            smd = abs(mean_t - mean_u) / std_pooled if std_pooled > 0 else 0
        smd_results.append((col, smd))
    return pd.DataFrame(smd_results, columns=["Variable", "SMD"]).sort_values(by="SMD", ascending=False)

imbalance_df = compute_standardized_mean_diff(df_cf, treatment_col, covariate_cols)
imbalanced_vars = imbalance_df[imbalance_df["SMD"] > 0.1]

# ---------------------------
# Propensity Score Weighting (IPTW)
# ---------------------------
numeric_imbalanced = [col for col in numeric_cols if col in imbalanced_vars["Variable"].values]
categorical_imbalanced = [col for col in categorical_cols if col in imbalanced_vars["Variable"].values]

imbalanced_preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), numeric_imbalanced),
    ('cat', SimpleImputer(strategy='most_frequent'), categorical_imbalanced)
])

X_ps = imbalanced_preprocessor.fit_transform(df_cf)
X_ps_df = pd.DataFrame(X_ps, columns=numeric_imbalanced + categorical_imbalanced)
X_ps_df = pd.get_dummies(X_ps_df)

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_ps_df, df_cf[treatment_col])
ps_scores = ps_model.predict_proba(X_ps_df)[:, 1]
treated = df_cf[treatment_col] == 1
weights = treated / ps_scores + (1 - treated) / (1 - ps_scores)

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

# Train best model on full data
print(f"\nBest params selected: {best_params}")

for train_idx, val_idx in kf.split(df_cf):
    df_train, df_val = df_cf.loc[train_idx], df_cf.loc[val_idx]

    X_train = np.asarray(preprocessor.fit_transform(df_train[covariate_cols]))
    X_val = np.asarray(preprocessor.transform(df_val[covariate_cols]))

    T_train = df_train[treatment_col].astype(int).values.ravel()
    Y_train = df_train[outcome_col].astype(int).values.ravel()

    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=LogisticRegression(max_iter=1000),
        discrete_treatment=True,
        random_state=42,
        cv=3,
        **best_params
    )
    cf_model.fit(Y_train, T_train, X=X_train, sample_weight=weights[train_idx])

    cate_fold = cf_model.effect(X_val)
    ci_lower, ci_upper = cf_model.effect_interval(X_val)

    all_cate[val_idx] = cate_fold
    all_ci_lower[val_idx] = ci_lower
    all_ci_upper[val_idx] = ci_upper

# Save predictions
df_cf["CATE"] = all_cate
df_cf["CATE_lower"] = all_ci_lower
df_cf["CATE_upper"] = all_ci_upper

# ---------------------------
# Rank into Quantiles
# ---------------------------
df_cf['CATE_quantile'] = pd.qcut(df_cf['CATE'], q=Q, labels=[f"Q{i+1}" for i in range(Q)])

# ---------------------------
# Analyze Quantile Groups
# ---------------------------
print("\n📊 CATE by Quantile Group:")
print(df_cf.groupby('CATE_quantile')['CATE'].agg(['mean', 'count']))

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='CATE_quantile', y='CATE', data=df_cf, estimator=np.mean, ci='sd')
plt.title('Average CATE by Quantile Group')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Export Function
# ---------------------------
def get_test_results():
    return df_cf.copy()
