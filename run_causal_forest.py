from cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)

import os.path
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from econml.dml import CausalForestDML
from econml.sklearn_extensions.metrics import r_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.path.join(os.getcwd(), "dataset_labelled.csv")
df = pd.read_csv(data_path)
# Load and preprocess data
df_cf = df.dropna(subset=[treatment_col, outcome_col]).copy()
for col in categorical_cols:
    df_cf[col] = df_cf[col].astype(str)

df_train, df_test = train_test_split(
    df_cf, test_size=0.3, stratify=df_cf[treatment_col], random_state=42
)

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

X_train = np.asarray(preprocessor.fit_transform(df_train[covariate_cols]))
X_test = np.asarray(preprocessor.transform(df_test[covariate_cols]))

T_train = df_train[treatment_col].astype(int).values.ravel()
Y_train = df_train[outcome_col].astype(int).values.ravel()
T_test = df_test[treatment_col].astype(int).values.ravel()
Y_test = df_test[outcome_col].astype(int).values.ravel()

print(df_train[outcome_col].value_counts(normalize=True).rename("Proportion"))

# Check covariate imbalance
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

    smd_df = pd.DataFrame(smd_results, columns=["Variable", "SMD"])
    return smd_df.sort_values(by="SMD", ascending=False)

imbalance_df = compute_standardized_mean_diff(df_train, treatment_col, covariate_cols)
print("\nCovariate Imbalance (SMD):")
print(imbalance_df)

threshold = 0.1
imbalanced_vars = imbalance_df[imbalance_df["SMD"] > threshold]
if not imbalanced_vars.empty:
    print("\n Warning: These covariates show notable imbalance (SMD > 0.1):")
    print(imbalanced_vars)
else:
    print("\n All covariates are reasonably balanced (SMD ≤ 0.1).")

numeric_imbalanced = [col for col in numeric_cols if col in imbalanced_vars["Variable"].values]
categorical_imbalanced = [col for col in categorical_cols if col in imbalanced_vars["Variable"].values]

imbalanced_preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_imbalanced),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_imbalanced)
    ],
    remainder='drop'
)

X_ps = imbalanced_preprocessor.fit_transform(df_train)
X_ps = pd.get_dummies(pd.DataFrame(X_ps, columns=numeric_imbalanced + categorical_imbalanced))

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X_ps, df_train[treatment_col])
ps_scores = ps_model.predict_proba(X_ps)[:, 1]
treated = df_train[treatment_col] == 1
weights = treated / ps_scores + (1 - treated) / (1 - ps_scores)

# Check covariate balance AFTER IPTW weighting
df_weighted = df_train.copy()
df_weighted["weight"] = weights

print("\n Weighted Covariate Means by Treatment Group (Post-IPTW):")
for col in covariate_cols:
    if df_weighted[col].dtype != 'object':
        try:
            treated_group = df_weighted[df_weighted[treatment_col] == 1]
            untreated_group = df_weighted[df_weighted[treatment_col] == 0]

            mean_t = np.average(treated_group[col], weights=treated_group["weight"])
            mean_u = np.average(untreated_group[col], weights=untreated_group["weight"])
            diff = abs(mean_t - mean_u)

            print(f"{col:<20} | Treated: {mean_t:.4f}, Untreated: {mean_u:.4f}, Δ = {diff:.4f}")
        except Exception as e:
            print(f" Error processing column {col}: {e}")

# Hyperparameter tuning for Causal Forest
param_grid = {
    "n_estimators": [500, 1000],
    "min_samples_leaf": [3, 5, 10],
    "max_depth": [5, 10, None],
    "max_samples": [0.8, 1.0]
}

best_model = None
best_score = float("inf")
best_params = None

print("\n🔧 Starting hyperparameter tuning...")

for params in ParameterGrid(param_grid):
    print(f"\n🔍 Trying: {params}")
    try:
        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=LogisticRegression(max_iter=1000),
            discrete_treatment=True,
            random_state=42,
            cv=3,
            **params
        )
        cf_model.fit(Y_train, T_train, X=X_train, sample_weight=weights)
        cate_train = cf_model.effect(X_train)
        pseudo_outcome = cate_train * T_train
        mse = mean_squared_error(Y_train, pseudo_outcome)
        print(f" Proxy MSE: {mse:.4f}")

        if mse < best_score:
            best_score = mse
            best_model = cf_model
            best_params = params
    except Exception as e:
        print(f"  Error: {e}")

# Evaluate best model
print("\n Best Parameters:", best_params)
print(" Best Proxy MSE:", best_score)

cate = best_model.effect(X_test)
ci_lower, ci_upper = best_model.effect_interval(X_test)
rate_score = r_score(Y_test, T_test, cate)

print(f"\n RATE Score (Test Set): {rate_score:.3f}")
print(f" Mean CATE (Treated): {cate[T_test == 1].mean():.4f}")
print(f" Mean CATE (Untreated): {cate[T_test == 0].mean():.4f}")

plt.figure(figsize=(8, 5))
sns.histplot(cate, bins=30, kde=True)
plt.title("CATE Distribution (Test Set)")
plt.xlabel("Estimated CATE")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

df_test = df_test.copy()
df_test["CATE"] = cate
