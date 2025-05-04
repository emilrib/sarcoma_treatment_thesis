import os.path
import pandas as pd
import re
from global_config import datasets_dir

input_file = os.path.join(datasets_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df.head(3))


df = df[['Pat ID','survival_status', 'chemo_status','age', 'Gender', 'cci_Timo', 'grade_clean','Affected tissue',
                  'reoperation_label', 'anatomic_region_label', 'metastasis_label', 'radiation_status', 'Tumor maximal size (mm)']]

print(df.head(3))

# ---------------------------
# Create Age subgroup
# ---------------------------
# Step 1: Prepare the age data for clustering
bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # Define the bin edges
labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']  # Labels for each group

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
labels_tumor = ['0-49', '50-99', '100-149', '150-199', '200-249', '250+']  # Labels for each group

# Step 2: Assign tumor size group labels based on the bins
df['tumor_size_group'] = pd.cut(df['Tumor maximal size (mm)'], bins=bins_tumor, labels=labels_tumor, right=False)

# Step 3: Print the resulting subgroups and their mean tumor size to inspect the subgroups
print("Mean tumor size by tumor size group:")
print(df.groupby('tumor_size_group')['Tumor maximal size (mm)'].mean())

df.to_csv(os.path.join(datasets_dir, "df_with_groups.csv"), index=False)

# ---------------------------
# Create subgroups for numerical covariates
# ---------------------------
numeric_cols = ['reoperation_label', 'metastasis_label', 'radiation_status', 'cci_Timo']

bins_cci = [0, 1, 2, 3, 4, 5, 6, float('inf')]  # The upper bound for each bin
labels_cci = ['0', '1', '2', '3', '4', '5', '6']  # Labels for each group

# Step 2: Assign tumor size group labels based on the bins
df['cci_group'] = pd.cut(df['cci_Timo'], bins=bins_cci, labels=labels_cci, right=False)

print(df.head(5))

# ---------------------------
# Create subgroups for categorical covariates
# ---------------------------

categories = ['anatomic_region_label', 'Gender','grade_clean', 'Affected tissue',
                    'tumor_size_group','age_group', 'cci_group' ]
df_encoded = pd.get_dummies(df, columns=categories, drop_first=False)

# Print the resulting dataframe with one-hot encoded columns and cci_group
print(df_encoded.head())


columns_to_remove = ['age', 'cci_Timo', 'Tumor maximal size (mm)']

# Remove the selected columns using drop
df_encoded = df_encoded.drop(columns=columns_to_remove)

output_file = os.path.join(datasets_dir, "dataset_with_subgroups.csv")
df_encoded.to_csv(output_file, index=False)

output_file = os.path.join(datasets_dir, "df_with_groups.csv")
df.to_csv(output_file, index=False)

# ---------------------------
# Merge the original groups with the encoded subgroups
# ---------------------------
# Merge the original dataset with the encoded subgroups dataset based on 'Pat ID'
df_merged = pd.merge(df, df_encoded, on='Pat ID', how='left', suffixes=('_x', '_y'))

df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_x')]
df_merged.columns = [re.sub(r'_y$', '', col) for col in df_merged.columns]

output_file_merged = os.path.join(datasets_dir, "df_with_merged_groups.csv")
df_merged.to_csv(output_file_merged, index=False)

# ---------------------------
# Extract and Save One-Hot Encoded Column Names
# ---------------------------
one_hot_cols = [col for col in df_merged.columns if any(base in col for base in df_merged)]
one_hot_df = pd.DataFrame({'OneHotEncodedFeature': one_hot_cols})
one_hot_df.to_csv(os.path.join(datasets_dir, "one_hot_encoded_columns.txt"), index=False)
