import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from global_config import datasets_dir

# ---------------------------
# Load Dataset with CATE Estimates
# ---------------------------
df_cf = pd.read_csv(os.path.join(datasets_dir, "cf_results.csv"))

# ---------------------------
# Plot CATE Distribution
# ---------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df_cf['CATE'], bins=30, kde=True, color='purple')
plt.axvline(0, color='black', linestyle='--')
plt.title('Distribution of Estimated CATEs')
plt.xlabel('Estimated Treatment Effect (CATE)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Survival Outcome by Treatment Group
# ---------------------------
if 'survival_status' in df_cf.columns and 'chemo_status' in df_cf.columns:
    treated = df_cf[df_cf['chemo_status'] == 1]
    untreated = df_cf[df_cf['chemo_status'] == 0]

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_cf, x='chemo_status', y='survival_status', errorbar=('ci', 95))
    plt.title('Average Survival Status by Treatment Group')
    plt.xlabel('Chemo Status (0=No, 1=Yes)')
    plt.ylabel('Mean Survival')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Required columns 'Chemo_status' and/or 'survival_status' not found in the dataset.")