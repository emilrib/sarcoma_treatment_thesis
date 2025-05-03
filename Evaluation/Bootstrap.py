import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from econml.dml import CausalForestDML
import joblib
import os

from global_config import datasets_dir
from Model.cf_config import treatment_col, outcome_col, covariate_cols, categorical_cols, numeric_cols

# Load dataset
df = pd.read_csv(os.path.join(datasets_dir, "df_with_merged_groups.csv"))
weights_df = pd.read_pickle(os.path.join(datasets_dir, "weights.pkl"))
df = df.merge(weights_df, on='Pat ID')
df = df.dropna(subset=[treatment_col, outcome_col, 'weights']).copy()

# Cast categorical columns to strings
for col in categorical_cols:
    df[col] = df[col].astype(str)

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

# Prepare data
X_all = preprocessor.fit_transform(df[covariate_cols])
Y_all = df[outcome_col].astype(int).values
T_all = df[treatment_col].astype(int).values
W_all = df['weights'].values

# Bootstrap setup
B = 100
n = X_all.shape[0]
oob_errors = []
test_errors = []

for b in range(B):
    X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
        X_all, T_all, Y_all, W_all, test_size=0.2, random_state=42
    )

    # Bootstrap sample from training data
    bootstrap_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    oob_idx = np.setdiff1d(np.arange(len(X_train)), bootstrap_idx)

    X_bootstrap = X_train[bootstrap_idx]
    T_bootstrap = T_train[bootstrap_idx]
    Y_bootstrap = Y_train[bootstrap_idx]
    W_bootstrap = W_train[bootstrap_idx]

    X_oob = X_train[oob_idx]
    T_oob = T_train[oob_idx]
    Y_oob = Y_train[oob_idx]

    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=LogisticRegression(max_iter=1000),
        discrete_treatment=True,
        random_state=42,
        cv=3
    )
    model.fit(Y_bootstrap, T_bootstrap, X=X_bootstrap, sample_weight=W_bootstrap)

    # OOB error
    cate_oob = model.effect(X_oob)
    mse_oob = mean_squared_error(Y_oob, cate_oob * T_oob)
    oob_errors.append(mse_oob)

    # Test error
    cate_test = model.effect(X_test)
    mse_test = mean_squared_error(Y_test, cate_test * T_test)
    test_errors.append(mse_test)

# Combine results
error_df = pd.DataFrame({
    "Bootstrap_Iteration": list(range(1, B + 1)),
    "OOB_MSE": oob_errors,
    "Test_MSE": test_errors
})

plt.figure(figsize=(12, 6))
plt.plot(error_df['Bootstrap_Iteration'], error_df['OOB_MSE'], label='OOB MSE', marker='o')
plt.plot(error_df['Bootstrap_Iteration'], error_df['Test_MSE'], label='Test MSE', marker='x')
plt.title('OOB vs Test Proxy MSE Across Bootstrap Iterations')
plt.xlabel('Bootstrap Iteration')
plt.ylabel('MSE')

# Set Y-axis ticks (optional, if MSE is high, keep this)
y_min = 0
y_max = max(error_df[['OOB_MSE', 'Test_MSE']].max()) + 25
plt.yticks(np.arange(y_min, y_max + 1, 50))

# Set X-axis ticks in steps of 50
x_min = error_df['Bootstrap_Iteration'].min()
x_max = error_df['Bootstrap_Iteration'].max()
plt.xticks(np.arange(x_min, x_max + 1, 50))

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()