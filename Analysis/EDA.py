import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

output_dir = os.path.join(os.getcwd())

input_file = os.path.join(output_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df.head(3))

"""
DEMOGRAPHIC DATA 
"""

df_demographics = df[['Pat ID','age', 'Gender', 'gender_label']]

#This chart displays the total amount of the sample as well as the distribution by gender
gender_counts = df_demographics.groupby('Gender')['Pat ID'].nunique().sort_index()

# Calculate the total number of patients
total_patients = df['Pat ID'].nunique()
print( f'Total Patients: {total_patients}')
# Calculate the percentage of each gender
total_count = gender_counts.sum()

gender_percentages = (gender_counts / total_count) * 100

# Plotting the bar chart
plt.figure(figsize=(5, 5))  # Set the figure size
bars = plt.bar(gender_counts.index, gender_counts.values, color=['pink', 'blue'])

# Adding count and percentage labels to each bar
for bar, percentage in zip(bars, gender_percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}\n({percentage:.1f}%)', ha='center', va='bottom')  # ha: horizontal alignment

# Display total above the bars
plt.text(0.4, max(gender_counts.values), f'Total: {total_patients}', ha='center', va='bottom', fontsize=12, color='green')

plt.xlabel('Gender')  # X-axis label
plt.ylabel('Number of Patients')  # Y-axis label
plt.title('Patient Distribution by Gender')  # Chart title
plt.ylim(0, max(gender_counts.values) * 1.2)  # Set y-axis limits to include text
plt.show()  # Display the chart

# The boxplot calculates the age of the patients and displays the ages in which sarcoma seems to appear the most. Outliers are also here identified and need to be verified with the SNN for data quality issues.
ages = df_demographics['age']
# Calculate statistics
median = round(ages.median())
mean = round(ages.mean())
quartiles = round(ages.quantile([0.25, 0.75]))
iqr = quartiles[0.75] - quartiles[0.25]
lower_whisker = quartiles[0.25] - 1.5 * iqr
upper_whisker = quartiles[0.75] + 1.5 * iqr
# Identify outliers
outlier_mask = (ages < lower_whisker) | (ages > upper_whisker)
outlier_values = ages[outlier_mask]


# Plot
plt.figure(figsize=(10, 8))

# Boxplot of patient ages
plt.boxplot(ages, widths=0.6, vert=True, patch_artist=True, showfliers=False)

# Highlight manually calculated outliers
if not outlier_values.empty:
    plt.scatter(np.ones(outlier_values.shape[0]), outlier_values, color='orange', s=50, label='Outliers')

# Display stats
plt.scatter(x=1, y=median, color='red', label=f'Median: {median:.2f}')
plt.scatter(x=1, y=mean, color='green', label=f'Mean: {mean:.2f}')
plt.axhline(y=quartiles[0.25], color='gray', linestyle='--', label=f'Q1: {quartiles[0.25]:.2f}')
plt.axhline(y=quartiles[0.75], color='gray', linestyle='--', label=f'Q3: {quartiles[0.75]:.2f}')

plt.title('Age Distribution of Sarcoma Patients')
plt.ylabel('Age')
plt.xticks([1], ['Patients'])
plt.legend()
plt.show()
print(df_demographics['age'].min())
print(df_demographics['age'].max())

"""
PATIENT HISTORY
"""

df_history = df[['Pat ID','date_first_patientcontact_Timo', 'Histological diagnosis', '(W) Other diagnoses?_Timo', 'cci_Timo', 'dignity_timo', 'anatomicregion_group_Timo', 'anatomic_region_label',  'Affected tissue', 'number_all_operation_Timo', 'Tumor maximal size (mm)', 'metastasis_status', 'metastasis_label', 'metastasis_label_description', 'reoperation_label']]

#Display of values of CCI
print(type(df_history['cci_Timo'][0]))

df_clean = df_history[df_history['cci_Timo'].notna()].copy()
df_clean['cci_Timo'] = df_clean['cci_Timo'].astype(int)

# Count and sort values by numeric order
cci_counts = df_clean['cci_Timo'].value_counts().sort_index(ascending=True)

# Plot
plt.figure(figsize=(10, 6))
ax = cci_counts.plot(kind='bar', color='#ADD8E6', edgecolor='black')

# Annotate each bar with count
for bar in ax.patches:
    ax.annotate(
        text=f'{int(bar.get_height())}',
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha='center', va='bottom', fontsize=10
    )

plt.xlabel('CCI Groups')
plt.ylabel('Number of Occurrences')
plt.title('Number of Occurrences by Different CCI Groups', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Identify the most common deceases in patients history


def extract_group_words(text):
    text = text.strip().lower()  # Convert to lowercase and remove surrounding spaces
    if text == '-':  # Ignore rows containing only '-'
        return []
    text = re.sub(r'\b\d{4}\b', '', text)  # Remove year dates
    group_words = re.split(r'[|,]', text)  # Split by '|' or ','
    group_words = [word.strip() for word in group_words if word.strip()]  # Remove extra spaces and empty strings
    return group_words

# Extract and flatten all group words from the column
other_diagnoses_group= df_history['(W) Other diagnoses?_Timo'].dropna().apply(extract_group_words).sum()
histological_diagnosis_group= df_history['Histological diagnosis'].dropna().apply(extract_group_words).sum()

# Count the frequency of group words
other_diagnoses_counts = Counter(other_diagnoses_group)
histological_diagnosis_counts = Counter(histological_diagnosis_group)

# Convert to DataFrame for better visualization
group_words_other_diagnosis = pd.DataFrame(other_diagnoses_counts.items(), columns=['Most occurred Other Diagnosis', 'Frequency']).sort_values(by='Frequency', ascending=False)
group_words_histoloical_diagnosis = pd.DataFrame(histological_diagnosis_counts.items(), columns=['Most occurred Histological Diagnosis', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Display the top 10 most common group words
print(group_words_other_diagnosis.head(10))
print(group_words_histoloical_diagnosis.head(10))

"""

#Display the total patient per anatomic region group
anatomic_grouping_counts = df_history['anatomic_region_label'].value_counts()

plt.figure(figsize=(10, 6))
anatomic_grouping_counts.plot(kind='barh', color='#ADD8E6', edgecolor='black')
for index, value in enumerate(anatomic_grouping_counts):
    plt.text(value + 0.1, index, str(value), va='center', fontsize=10)

plt.title('Total Sarcoma cases per Anatomic Region Group', fontsize=10, weight='bold')
plt.xlabel('Total', fontsize=12)
plt.ylabel('Group', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
"""
# Analyse the percentage of the data sample with metastasis and its resurfacing time
metastasis_counts = df_history['metastasis_label'].value_counts()

label_map = {1: 'No', 2: 'Yes'}
translated_index = metastasis_counts.index.map(label_map)

sizes = metastasis_counts.values
labels = translated_index

def autopct_func(pct):
    total = sum(sizes)
    count = int(round(pct * total / 100.0))
    return f'{pct:.1f}%\n({count})'

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct=autopct_func,
    startangle=140,
    colors=['#FFDAB9', '#ADD8E6', '#FFFFE0'],
    wedgeprops={'edgecolor': 'black'},
    textprops={'fontsize': 12}
)
plt.title('Presence of Metastasis by First Diagnosis')
plt.axis('equal')  # Ensures pie is a circle
plt.show()
print(metastasis_counts)

metastasis_counts_description = df_history['metastasis_label_description'].value_counts()

# Data
labels = metastasis_counts_description.index
values = metastasis_counts_description.values
total = values.sum()

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(
    labels, values,
    color=['#ACE1AF', '#FFDAB9', '#ADD8E6', '#FFFFE0'],
    edgecolor='black'
)

# Add absolute values & prepare percentage labels
percentages = []
for bar, label, value in zip(bars, labels, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f'{value}',
        ha='center', va='bottom', fontsize=10
    )
    percentages.append(f"{label}: {value / total:.1%}")

# Description box with percentages
description = "\n".join(percentages)
plt.gca().text(
    1.02, 0.95,
    description,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
)

# Titles and styling
plt.title('Metastasis Cases by Resurfacing Time', fontsize=12, weight='bold')
plt.xlabel('Resurfacing Time')
plt.ylabel('Number of Cases')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
