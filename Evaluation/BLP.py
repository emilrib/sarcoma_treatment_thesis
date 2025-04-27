import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm
from Model.cf_config import covariate_cols
import matplotlib.patches as mpatches
from global_config import  datasets_dir, model_dir
from Model.cf_config import covariate_cols


# ---------------------------
# Load Data
# ---------------------------
X_test = np.load(os.path.join(datasets_dir, "X_test_corrected.npy"))
cf_model = joblib.load(os.path.join(model_dir, "cf_model.pkl"))
preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))

# Load feature names safely
try:
    feature_names = preprocessor.get_feature_names_out()
    if len(feature_names) != X_test.shape[1]:
        raise ValueError(f"Feature names ({len(feature_names)}) do not match X_test columns ({X_test.shape[1]}).")
except Exception as e:
    print(f" Could not assign feature names properly: {e}")
    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

X_test_df = pd.DataFrame(X_test, columns=feature_names)
print(f" X_test shape: {X_test.shape}")

# ---------------------------
# Match Covariate Columns
# ---------------------------

matched_cols = [col for col in X_test_df.columns if any(base_col in str(col) for base_col in covariate_cols)]

if not matched_cols:
    raise ValueError(" No matched covariate columns found after feature name assignment.")

X_test_df = X_test_df[matched_cols]
print(f" Matched covariate columns ({len(matched_cols)}): {matched_cols}")

# ---------------------------
# Run Best Linear Projection (BLP)
# ---------------------------

print("\n Running Best Linear Projection (BLP)...")

X_blp = sm.add_constant(X_test_df)
y_blp = cf_model.effect(X_test)  # Predicted CATEs

blp_model = sm.OLS(y_blp, X_blp).fit(cov_type='HC3')

# ---------------------------
# Summarize Results
# ---------------------------

summary_df = pd.DataFrame({
    "Estimate": blp_model.params,
    "Std.Error": blp_model.bse,
    "t-value": blp_model.tvalues,
    "p-value": blp_model.pvalues
}).round(5)

summary_df["Significance"] = summary_df["p-value"].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
)

print("\n BLP Full Summary:")
print(summary_df)

# ---------------------------
# Group Coefficients by Original Covariates
# ---------------------------

def match_covariate(feature, bases):
    for base in bases:
        if base in feature:
            return base
    return "Other"

coef_df = summary_df.drop(index="const").copy()
coef_df["OriginalCovariate"] = coef_df.index.to_series().apply(lambda x: match_covariate(str(x), covariate_cols))

grouped_summary = coef_df.groupby("OriginalCovariate").agg(
    GroupedEstimate=('Estimate', 'mean'),
    GroupedStdError=('Std.Error', 'mean'),
    min_p_value=('p-value', 'min')
).reset_index()

grouped_summary["Significance"] = grouped_summary["min_p_value"].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else ''
)

print("\n Grouped BLP Summary:")
print(grouped_summary.round(5))

# ---------------------------
# Plot Grouped BLP
# ---------------------------

colors = ['#FFDAB9' if est > 0 else '#ADD8E6' for est in grouped_summary["GroupedEstimate"]]

plt.figure(figsize=(12, 7))
bars = plt.barh(grouped_summary["OriginalCovariate"], grouped_summary["GroupedEstimate"],
                xerr=grouped_summary["GroupedStdError"], capsize=5, color=colors, edgecolor='black')

# Highlight statistically significant bars
for idx, pval in enumerate(grouped_summary["min_p_value"]):
    if pval < 0.05:
        bars[idx].set_edgecolor('gold')
        bars[idx].set_linewidth(3)

# Add vertical line at zero
plt.axvline(0, color='black', linestyle='--')

# Titles and labels
plt.title('Grouped BLP Coefficients', fontsize=14)
plt.xlabel('Estimated Effect on CATE', fontsize=12)
plt.grid(True)

# Custom legend
legend_handles = [
    mpatches.Patch(color='#FFDAB9', label='Positive Effect'),
    mpatches.Patch(color='#ADD8E6', label='Negative Effect'),
    mpatches.Patch(edgecolor='gold', facecolor='white', linewidth=2, label='Statistically Significant (p < 0.05)')
]
plt.legend(handles=legend_handles, loc='best')

plt.tight_layout()
plt.show()