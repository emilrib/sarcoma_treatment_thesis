import os.path
import pandas as pd
from datetime import datetime
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


if '__file__' in globals():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
else:
    # Fallback for interactive environments
    current_dir = os.getcwd()

output_directory = os.path.join(current_dir, "output")

input_file = os.path.join(current_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df.head(3))

"""
This section checks for SUTVA interference by anatomic region group
"""
group_anatomic = 'anatomicregion_group_Timo'

# Group statistics: mean, std, and count of survival by lesion side
group_stats = df.groupby(group_anatomic)['overall_survival_days'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("Group-level survival statistics by anatomic region:")
print(group_stats, "\n")

# ANOVA: test whether survival differs significantly across groups
group_labels = df[group_anatomic].dropna().unique()  # drop NaNs if present
group_values = [df[df[group_anatomic] == label]['overall_survival_days'].dropna() for label in group_labels]

# Only run ANOVA if we have at least 2 groups with data
if len(group_values) >= 2:
    f_stat, p_value = stats.f_oneway(*group_values)
    print(f"ANOVA Result ➤ F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference detected across groups — possible SUTVA interference.")
    else:
        print("No significant group-level differences detected — SUTVA may hold.")
else:
    print("Not enough groups with data to run ANOVA.")

# Visualization: Boxplot of survival days by lesion group
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x=group_anatomic, y='overall_survival_days')
plt.title('Overall Survival by Anatomic Region Group')
plt.xlabel('Anatomic Region Group')
plt.ylabel('Overall Survival Days')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Review treatment assignment consistency
print("\nChemo_status value counts:")
print(df['Chemo_status'].value_counts(dropna=False))

print("\nChemo_duration summary:")
print(df['chemo_duration'].describe())

"""
This section checks for SUTVA interference by gender
"""

group_gender = 'Gender'

# Group statistics: mean, std, and count of survival by lesion side
group_stats = df.groupby(group_gender)['overall_survival_days'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("Group-level survival statistics by anatomic region:")
print(group_stats, "\n")

# ANOVA: test whether survival differs significantly across groups
group_labels = df[group_gender].dropna().unique()  # drop NaNs if present
group_values = [df[df[group_gender] == label]['overall_survival_days'].dropna() for label in group_labels]

# Only run ANOVA if we have at least 2 groups with data
if len(group_values) >= 2:
    f_stat, p_value = stats.f_oneway(*group_values)
    print(f"ANOVA Result ➤ F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference detected across groups — possible SUTVA interference.")
    else:
        print("No significant group-level differences detected — SUTVA may hold.")
else:
    print("Not enough groups with data to run ANOVA.")

# Visualization: Boxplot of survival days by lesion group
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x=group_gender, y='overall_survival_days')
plt.title('Overall Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Overall Survival Days')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

"""
This section checks for SUTVA interference by histological diagnosis
"""
"""
group_histology = 'Histological diagnosis'

# Group statistics: mean, std, and count of survival by lesion side
group_stats = df.groupby(group_histology)['overall_survival_days'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print("Group-level survival statistics by anatomic region:")
print(group_stats, "\n")

# ANOVA: test whether survival differs significantly across groups
group_labels = df[group_histology].dropna().unique()  # drop NaNs if present
group_values = [df[df[group_histology] == label]['overall_survival_days'].dropna() for label in group_labels]

# Only run ANOVA if we have at least 2 groups with data
if len(group_values) >= 2:
    f_stat, p_value = stats.f_oneway(*group_values)
    print(f"ANOVA Result ➤ F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference detected across groups — possible SUTVA interference.")
    else:
        print("No significant group-level differences detected — SUTVA may hold.")
else:
    print("Not enough groups with data to run ANOVA.")

# Visualization: Boxplot of survival days by lesion group
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x=group_histology, y='overall_survival_days')
plt.title('Overall Survival by Histological Diagnosis')
plt.xlabel('Gender')
plt.ylabel('Overall Survival Days')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
"""
"""
if SUTVA holds means patients grouped by this variable don't seem to affect each other's outcomes
no significant group effect on outcomes (less likely interference (e.g. spillover effects, group contacgion)

"""

with open("README.md", "a") as file:
    file.write(
        "### STUVA Analysis\n"
        "The STUVA analysis uses ANOVA to determine if there is a significant STUVA interference amaong groups.\n"
        "- `1`: If p-value is less than 0.05: Significant difference detected across groups — possible SUTVA interference.\n"
        "- `2`: if p-value is higher than 0.05: No significant group-level differences detected — SUTVA may hold.\n"
    )