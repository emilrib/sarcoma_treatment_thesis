import os.path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from run_causal_forest import get_test_results, treatment_col, outcome_col
import pandas as pd
from cf_config import (
    treatment_col,
    outcome_col,
    covariate_cols,
    categorical_cols,
    numeric_cols
)


# Load results
df_test = get_test_results()

# Add age_grouping logic explicitly
if "age" in df_test.columns:
    df_test["age_group"] = pd.cut(df_test["age"], bins=[0, 40, 60, 80, 120], labels=["<40", "40-60", "60-80", "80+"])
    if "age_group" not in categorical_cols:
        categorical_cols.append("age_group")

def evaluate_model(df_test):
    treated = df_test[df_test[treatment_col] == 1]
    untreated = df_test[df_test[treatment_col] == 0]

    mean_treated = treated["CATE"].mean()
    mean_untreated = untreated["CATE"].mean()

    if mean_untreated != 0:
        rate = (mean_treated - mean_untreated) / abs(mean_untreated)
    else:
        rate = float('inf') if mean_treated > 0 else float('-inf')

    print(f"\n✅ Overall RATE: {rate:.3f}")
    print(f"Mean CATE (Treated): {mean_treated:.4f}")
    print(f"Mean CATE (Untreated): {mean_untreated:.4f}")

    # Subgroup analysis for categorical covariates
    cat_subgroups = [col for col in categorical_cols if col in df_test.columns and df_test[col].nunique() < 20]

    for var in cat_subgroups:
        print(f"\n📊 CATE by {var}:")
        cate_grouped = df_test.groupby(var)["CATE"].mean()
        ci_lower = df_test.groupby(var)["CATE_lower"].mean()
        ci_upper = df_test.groupby(var)["CATE_upper"].mean()
        print(cate_grouped)

        # RATE by subgroup category
        for level in df_test[var].dropna().unique():
            subset = df_test[df_test[var] == level]
            treated = subset[subset[treatment_col] == 1]
            untreated = subset[subset[treatment_col] == 0]
            m1 = treated["CATE"].mean()
            m0 = untreated["CATE"].mean()
            rate = (m1 - m0) / abs(m0) if m0 != 0 else float('inf')
            print(f"  - {var} = {level}: RATE = {rate:.3f} (treated: {m1:.4f}, untreated: {m0:.4f})")

        # Plot with confidence intervals
        plt.figure(figsize=(7, 4))
        plt.errorbar(cate_grouped.index.astype(str), cate_grouped.values,
                     yerr=[cate_grouped.values - ci_lower.values, ci_upper.values - cate_grouped.values],
                     fmt='o', capsize=5)
        plt.title(f'CATE by {var} (with CI)')
        plt.ylabel('CATE')
        plt.xlabel(var)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Analysis for numeric covariates using binned RATE and smoothed CATE plots
    for var in numeric_cols:
        if var in df_test.columns:
            print(f"\n📈 CATE and RATE by binned {var}:")
            df_test[f"{var}_bin"] = pd.qcut(df_test[var], q=4, duplicates='drop')
            grouped = df_test.groupby(f"{var}_bin")
            for bin_label, subset in grouped:
                treated = subset[subset[treatment_col] == 1]
                untreated = subset[subset[treatment_col] == 0]
                m1 = treated["CATE"].mean()
                m0 = untreated["CATE"].mean()
                rate = (m1 - m0) / abs(m0) if m0 != 0 else float('inf')
                print(f"  - {var} bin {bin_label}: RATE = {rate:.3f} (treated: {m1:.4f}, untreated: {m0:.4f})")

            # Smoothed plot
            plt.figure(figsize=(7, 4))
            sns.regplot(data=df_test, x=var, y="CATE", lowess=True, scatter_kws={'s': 10}, line_kws={'color': 'red'})
            plt.title(f"Smoothed CATE vs {var}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Plot CATE distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df_test["CATE"] * 100, bins=30, edgecolor='k')
    plt.title("Distribution of treatment effect estimation")
    plt.xlabel("CATE (%)")
    plt.ylabel("Patients")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Interpretation
    print("\n🧠 Interpretation:")
    print("- RATE compares treatment effect in treated vs. untreated.")
    print("- Categorical and numeric covariates reveal subgroup treatment effects.")
    print("- Confidence intervals and smooth curves show uncertainty and trends.")

# Run evaluation
evaluate_model(df_test)
