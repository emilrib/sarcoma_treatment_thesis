import os.path
import re
import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from global_config import datasets_dir

# Read a dataset from /data/

input_file = os.path.join(datasets_dir, "dataset_clean.csv")

def import_file_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file_to_dataframe(input_file)

print(df.head(2))

"""
Calculation of age
"""

def calculate_age(year_of_birth):
    current_year = datetime.now().year
    return current_year - year_of_birth

df['age'] = df['Date of birth'].apply(calculate_age)

"""
classification label to determine patients that needed to be reoperated and impact of reoperation
"""
def reoperation(row):
    if row >= 1:
        return 1
    else:
        return 0

df['reoperation_label'] = df['number_all_operation_Timo'].apply(reoperation)

"""
Populate null values with the mode (most frequent value) for the anatomicregion_group_Timo 
Add label to Anatomic label group 

"""
mode_value = df['anatomicregion_group_Timo'].mode().iloc[0]

# Step 2: Fill missing values with the mode
df['anatomicregion_group_Timo'] = df['anatomicregion_group_Timo'].fillna(mode_value)


def anatomic_region_label(row):
    if row == 1:
        return 'Head and Neck'
    elif row == 2:
        return 'Upper Extremity'
    elif row == 3:
        return 'Lower Extremity'
    elif row == 3:
        return 'Lower Extremity'
    elif row == 4:
        return 'Trunk'
    elif row == 5:
        return 'Retroperitoneal'
    elif row == 6:
        return 'Vizeral and Intraperitoneal'
    else:
        return 'Others'

df['anatomic_region_label'] = df['anatomicregion_group_Timo'].apply(anatomic_region_label)

"""
classification of metastasis status during the treatment    
"""
def metastasis_label(row):
    if row == '-' or pd.isnull(row):
        return 0
    elif row == 0:
        return 0
    else:
        return 1

df['metastasis_label'] = df['number_metastasis_Timo'].apply(metastasis_label)

#print(df[['Pat ID', 'number_metastasis_Timo', 'metastasis_label', "date_metastasis_Timo", "date_first_patientcontact_Timo"]])
df["Date of last follow-up"] = pd.to_datetime(df["Date of last follow-up"], format="%Y-%m-%d", errors="coerce")
df["date_metastasis_Timo"] = pd.to_datetime(df["date_metastasis_Timo"], format="%Y-%m-%d", errors = "coerce")


# Compute mean
def days_of_metastasis(col1, col2, col3):
    if col1 == 1:
        return (col3 - col2).days
    else:
        return None

df["metastasis_days"] = df.apply(lambda row: days_of_metastasis(row['metastasis_label'],row['date_metastasis_Timo'],row['Date of last follow-up']), axis =1)


#label to heck if chemo was used or not
def chemo_indication_label(row):
    if row == "-" or pd.isnull(row):
        return 0
    else:
        return 1

df["Chemo_status"] = df['chemo_first_indication_Timo'].apply(chemo_indication_label)

# --- Count and percentage calculation ---
chemo_counts = df["Chemo_status"].value_counts().sort_index()
chemo_counts.index = ["No Chemotherapy", "Received Chemotherapy"]
percentages = (chemo_counts / chemo_counts.sum() * 100).round(1)

# --- Plot ---
plt.figure(figsize=(6, 4))
bars = plt.bar(chemo_counts.index, chemo_counts.values, color=["#A9CCE3", "#2E86C1"], edgecolor="black")
plt.title("Distribution of Chemotherapy Status")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# --- Add labels: absolute + percentage ---
for bar, count, pct in zip(bars, chemo_counts.values, percentages):
    height = bar.get_height()
    label = f"{count} ({pct:.1f}%)"
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, label, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


#calculate the duration of the first chemotherapy in days

df["Start date of line"] = pd.to_datetime(df["Start date of line"], format="%Y-%m-%d", errors="coerce")
df["Optional: End date of line"] = pd.to_datetime(df["Optional: End date of line"], format="%Y-%m-%d", errors = "coerce")

# Compute valid durations for mean estimation
valid_mask = pd.notna(df["Start date of line"]) & pd.notna(df["Optional: End date of line"])
durations = (df.loc[valid_mask, "Optional: End date of line"] - df.loc[valid_mask, "Start date of line"]).dt.days

fallback_mask = pd.notna(df["Start date of line"]) & df["Optional: End date of line"].isna()
fallback_durations = (pd.Timestamp.today().normalize() - df.loc[fallback_mask, "Start date of line"]).dt.days

# Combine all valid durations
#all_durations = pd.concat([durations, fallback_durations])
#mean_duration = round(all_durations.mean())

# Define function with mean fallback
def chemo_duration(start, end):
    if pd.notna(start) and pd.notna(end):
        return (end - start).days
    elif pd.notna(start) and pd.isna(end):
        return (pd.Timestamp.today().normalize() - start).days
    else:
        return None

df["chemo_duration"] = df.apply(lambda row: chemo_duration(row['Start date of line'],row['Optional: End date of line']), axis =1)

#extract the labels of the status column

def status_label(text):
    return re.search(r'\(([^)]+)\)', text).group(1) if isinstance(text, str) and '(' in text else None

df["survival_status_label"] = df['Status'].apply(status_label)

mode_value = df['survival_status_label'].mode()[0]  # mode() returns a Series; take the first value
df['survival_status_label'] = df['survival_status_label'].fillna(mode_value)

#label to find out the status of patients
def survival_status(text):
    if text == 'AWD' or text == 'NED':
        return 1
    else:
        return 0

df["survival_status"] = df['survival_status_label'].apply(survival_status)

awd_counts = df["survival_status"].value_counts().sort_index()
awd_counts.index = ["Not Alive", "Alive"]
percentages = (awd_counts / awd_counts.sum() * 100).round(1)

# --- Plot ---
plt.figure(figsize=(6, 4))
bars = plt.bar(awd_counts.index, awd_counts.values, color=["#AED6F1", "#2874A6"], edgecolor="black")
plt.title("Distribution of Alive Status")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# --- Add labels: absolute + percentage ---
for bar, count, pct in zip(bars, awd_counts.values, percentages):
    height = bar.get_height()
    label = f"{count} ({pct:.1f}%)"
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, label, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

#add label to distinguish patients that receive radiotherapy from the ones that did not
def extract_radiotherapy_number(text):
    val_str = str(text).strip().lower()
    match = re.match(r"\[?(\d+)\]?", val_str)
    if match:
        return int(match.group(1))
    else:
        return np.nan  # If no match, treat as missing for now

# Create a temporary column to hold extracted numbers
df["Indication for radiotherapy"] = df["Indication for radiotherapy"].apply(extract_radiotherapy_number)

# Step 2: Fill missing values with MODE (most common value)
mode_value = df["Indication for radiotherapy"].mode()[0]
df["Indication for radiotherapy"] = df["Indication for radiotherapy"].fillna(mode_value)

# Step 3: Now assign binary radiation status
def radiotherapy_status(number):
    if number == 0:
        return 0
    else:
        return 1

df["radiation_status"] = df["Indication for radiotherapy"].apply(radiotherapy_status)

#calculate the number of survival days a patient that passed away from the decease (from day of contact until the date of death)
df["date_death_Timo"] = pd.to_datetime(df["date_death_Timo"], format="%Y-%m-%d", errors = "coerce")
df["date_first_patientcontact_Timo"] = pd.to_datetime(df["date_first_patientcontact_Timo"], format="%Y-%m-%d", errors="coerce")

def survival_days(col1,col2, col3):
    if col1 == 0:
        return (col2 - col3).days
    return None

df["overall_survival_days"] = df.apply(lambda row: survival_days(row['survival_status'],row['date_death_Timo'], row['date_first_patientcontact_Timo']), axis =1)

#populate null values with mean values for the tumor size
df["Tumor maximal size (mm)"] = pd.to_numeric(df["Tumor maximal size (mm)"], errors="coerce")

mean_tumor_size = df["Tumor maximal size (mm)"].mean().round(0)
print(f' Mean tumour size (in mm): {mean_tumor_size}')
df["Tumor maximal size (mm)"] = df["Tumor maximal size (mm)"].fillna(mean_tumor_size)

#populate null values with mean values for the CCI
mode_value = df['cci_Timo'].mode()[0]  # mode() returns a Series; take the first value
df['cci_Timo'] = df['cci_Timo'].fillna(mode_value)


output_file = os.path.join(datasets_dir, "dataset_labelled.csv")
df.to_csv(output_file, index=False)

"""
Writing the translation of the binary code
"""
with open("../README.md", "a") as file:
    file.write(
        "#### Binary Codes Meaning\n"
        "- `0`: **No**\n"
        "- `1`: **Yes**\n"
    )



