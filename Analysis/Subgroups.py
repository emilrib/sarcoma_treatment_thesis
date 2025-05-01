import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from sklearn.cluster import KMeans
from global_config import datasets_dir
import seaborn as sns

input_file = os.path.join(datasets_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df.head(3))

# ---------------------------
# Create Age subgroup
# ---------------------------
# Step 1: Prepare the age data for clustering
bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # Define the bin edges
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']  # Labels for each group

# Step 2: Assign age group labels based on the bins
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)  # right=False ensures bins are inclusive of the lower bound

# Step 3: Print the resulting subgroups and their mean age to inspect the subgroups
print("Mean age by age group:")
print(df.groupby('age_group')['age'].mean())
# Step 5: Save the dataframe with the age groups to a CSV file for later use
#df.to_csv('df_with_age_groups.csv', index=False)

# ---------------------------
# Create Tumor size subgroup
# ---------------------------

# Step 2: Create subgroups for 'Tumor maximal size (mm)' using manual binning
bins_tumor = [0, 50, 100, 150, 200, 250, float('inf')]  # The upper bound for each bin
labels_tumor = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-350']  # Labels for each group

# Step 2: Assign tumor size group labels based on the bins
df['tumor_size_group'] = pd.cut(df['Tumor maximal size (mm)'], bins=bins_tumor, labels=labels_tumor, right=False)

# Step 3: Print the resulting subgroups and their mean tumor size to inspect the subgroups
print("Mean tumor size by tumor size group:")
print(df.groupby('tumor_size_group')['Tumor maximal size (mm)'].mean())

df.to_csv(os.path.join(datasets_dir, "df_with_groups.csv"), index=False)
