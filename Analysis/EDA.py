import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
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
# Demographic Data
# ---------------------------

df_demographics = df[['Pat ID','age', 'Gender']]

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
bars = plt.bar(gender_counts.index, gender_counts.values, color=['pink', '#ADD8E6'])

# Adding count and percentage labels to each bar
for bar, percentage in zip(bars, gender_percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}\n({percentage:.1f}%)', ha='center', va='bottom')  # ha: horizontal alignment

# Display total above the bars
plt.text(0.4, max(gender_counts.values), f'Total: {total_patients}', ha='center', va='bottom', fontsize=12, color='green')

plt.xlabel('Gender', fontsize=12)  # X-axis label
plt.ylabel('Number of Patients', fontsize=12)  # Y-axis label
plt.title('Patient Distribution by Gender', fontsize=12, weight='bold')  # Chart title
plt.ylim(0, max(gender_counts.values) * 1.2)  # Set y-axis limits to include text
plt.show()  # Display the chart

# The boxplot calculates the age of the patients and displays the ages in which sarcoma seems to appear the most. Outliers are also here identified and need to be verified with the SNN for data quality issues.
ages = df_demographics['age']
median = round(ages.median(), 2)
mean = round(ages.mean(), 2)
quartiles = ages.quantile([0.25, 0.75])
iqr = quartiles[0.75] - quartiles[0.25]
lower_whisker = max(ages.min(), quartiles[0.25] - 1.5 * iqr)
upper_whisker = min(ages.max(), quartiles[0.75] + 1.5 * iqr)

# Identify outliers
outlier_mask = (ages < lower_whisker) | (ages > upper_whisker)
outlier_values = ages[outlier_mask]

# Plot
plt.figure(figsize=(10, 8))

# Boxplot
box = plt.boxplot(
    ages,
    widths=0.6,
    vert=True,
    patch_artist=True,
    showfliers=False,
    showmeans=True,
    meanprops=dict(marker='o', markerfacecolor='green', markeredgecolor='black', markersize=8)
)

# Color the box
for element in ['boxes', 'whiskers', 'caps', 'medians']:
    for item in box[element]:
        item.set(color='black')
for patch in box['boxes']:
    patch.set_facecolor("#ADD8E6")

# Add manual median dot for legend
plt.scatter(x=1, y=median, color='red', s=50, zorder=10, label=f'Median: {median:.2f}')
# Outliers
if not outlier_values.empty:
    plt.scatter(np.ones(outlier_values.shape[0]), outlier_values, color='orange', s=50, label='Outliers')

# Add whisker lines as invisible dots (for legend only)
plt.scatter([], [], color='black', label=f'Lower Whisker: {lower_whisker:.2f}')
plt.scatter([], [], color='black', label=f'Upper Whisker: {upper_whisker:.2f}')
# Optional: add quartile lines
plt.axhline(y=quartiles[0.25], color='gray', linestyle='--', label=f'Q1: {quartiles[0.25]:.2f}')
plt.axhline(y=quartiles[0.75], color='gray', linestyle='--', label=f'Q3: {quartiles[0.75]:.2f}')

# Final plot formatting
plt.title('Age Distribution of Sarcoma Patients', fontsize=14, weight='bold')
plt.ylabel('Age', fontsize=12)
plt.xticks([1], ['Patients'])
plt.legend()
plt.tight_layout()
plt.show()

# Age range for debugging/confirmation
print("Min age:", df_demographics['age'].min())
print("Max age:", df_demographics['age'].max())

# ---------------------------
# Patient History
# ---------------------------

df_history = df[['Pat ID','Histological diagnosis', 'cci_Timo', 'grade_clean', 'Affected tissue',
                 'anatomic_region_label',  'reoperation_label', 'Tumor maximal size (mm)', 'radiation_status', 'metastasis_label',  'reoperation_label']]

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

plt.xlabel('CCI Groups', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.title('Occurrences by Charlson Comorbidity Index (CCI)', fontsize=14, weight='bold')
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
histological_diagnosis_group= df_history['Histological diagnosis'].dropna().apply(extract_group_words).sum()

# Count the frequency of group words
histological_diagnosis_counts = Counter(histological_diagnosis_group)

# Convert to DataFrame for better visualization
group_words_histolocial_diagnosis = pd.DataFrame(histological_diagnosis_counts.items(), columns=['Most occurred Histological Diagnosis', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Display the top 10 most common group words
print(group_words_histolocial_diagnosis.head(5))

#Display the total patient per anatomic region group
anatomic_grouping_counts = df_history['anatomic_region_label'].value_counts()

plt.figure(figsize=(10, 6))
anatomic_grouping_counts.plot(kind='barh', color='#ADD8E6', edgecolor='black')
for index, value in enumerate(anatomic_grouping_counts):
    plt.text(value + 0.1, index, str(value), va='center', fontsize=10)

plt.title('Total Sarcoma cases per Anatomic Region Group', fontsize=14, weight='bold')
plt.xlabel('Total', fontsize=12)
plt.ylabel('Anatomic Region', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Analyse the percentage of the data sample with metastasis and its resurfacing time
metastasis_counts = df_history['metastasis_label'].value_counts()

label_map = {0: 'No Metastasis', 1: 'Metastasis'}
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
    startangle=90,
    colors=['#FFDAB9', '#ADD8E6'],
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Presence of Metastasis', fontsize=14, weight='bold')
plt.axis('equal')  # Ensures pie is a circle
plt.show()
print(metastasis_counts)

#Display in a box plot how the tumor size is distributed
tumor_size = df_history["Tumor maximal size (mm)"]

# Compute statistics
mean_value = tumor_size.mean()
median_value = tumor_size.median()
q1 = tumor_size.quantile(0.25)
q3 = tumor_size.quantile(0.75)
iqr = q3 - q1
lower_whisker = max(tumor_size.min(), q1 - 1.5 * iqr)
upper_whisker = min(tumor_size.max(), q3 + 1.5 * iqr)

# --- Boxplot for Tumor Size ---
plt.figure(figsize=(8, 6))
sns.boxplot(
    y=tumor_size,
    color="#ADD8E6",
    width=0.5,
    flierprops=dict(marker='o', markerfacecolor='darkorange', markersize=8, linestyle='none')
)

# Plot mean and median points
plt.scatter(0, mean_value, color='red', s=80, zorder=10, label=f"Mean: {mean_value:.1f} mm")
plt.scatter(0, median_value, color='green', s=80, zorder=10, label=f"Median: {median_value:.1f} mm")

# Add invisible points to legend for quartiles and whiskers
plt.scatter([], [], color='black', label=f"Q1: {q1:.1f} mm")
plt.scatter([], [], color='black', label=f"Q3: {q3:.1f} mm")
plt.scatter([], [], color='darkorange', label=f"Lower Whisker: {lower_whisker:.1f} mm")
plt.scatter([], [], color='darkorange', label=f"Upper Whisker: {upper_whisker:.1f} mm")

# Formatting
plt.title('Distribution of Tumor Size (mm)', fontsize=14, weight='bold')
plt.ylabel('Tumor Size (mm)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Countplot for Grading

# Ensure correct order
grade_order = ['G1', 'G2', 'G3']

plt.figure(figsize=(8, 6))
sns.countplot(
    x="grade_clean",
    data=df_history,
    order=grade_order,  # Specify the desired order
    palette="pastel",
    edgecolor="black",
    hue="grade_clean",
    legend=False
)

plt.title('Number of Occurrences by FNCLCC Grading', fontsize=14, weight='bold')
plt.xlabel('FNCLCC Grading', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add counts above bars
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       fontsize=10, color='black',
                       xytext=(0, 10),
                       textcoords='offset points')

plt.tight_layout()
plt.show()

# ---------------------------
# Patient Treatment
# ---------------------------

##  PIe chart displaying the percentage of patients that received and did not receive chemotherapy
# Count the number of 0s and 1s

df_treatment = df[['Pat ID','chemo_status', 'reoperation_label', 'radiation_status']]

# Count 0s and 1s for each treatment variable
treatment_counts = df_treatment.drop(columns='Pat ID').apply(lambda col: col.value_counts()).T
treatment_counts = treatment_counts[[0, 1]].fillna(0).astype(int)  # Ensure both 0 and 1 exist

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
treatment_counts.plot(kind='bar', stacked=True, color=['#FFDAB9', '#ADD8E6'], edgecolor='black', ax=ax)

# Add text for individual bar segments
for i, col in enumerate(treatment_counts.index):
    total = treatment_counts.loc[col].sum()
    bottom = 0
    for j, value in enumerate(treatment_counts.loc[col]):
        # Centered label within the bar segment
        ax.text(i, bottom + value / 2, str(value), ha='center', va='center', fontsize=10)
        bottom += value
    # Total label above the stacked bar
#    ax.text(i, total + 2, f'Total: {total}', ha='center', va='bottom', fontsize=12)

# Labels and formatting
ax.set_title('Patient Counts by Treatment Type', fontsize=14, weight='bold')
ax.set_xlabel('Treatment Type', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_xticks(range(len(treatment_counts.index)))
ax.set_xticklabels(treatment_counts.index, rotation=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Move legend outside the plot
ax.legend(['Treatment not received', 'Treatment received'], title='Legend',
          bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend
plt.show()

# ---------------------------
# Patient Outcome
# ---------------------------

## Pie char dispaly the pertage of patient survival

# Count survival statuses (assuming 0 = Not Survived, 1 = Survived)
survival_counts = df["survival_status"].value_counts().sort_index()
total_survival = survival_counts.sum()

# Labels mapped manually
labels_survival = ['Not Survived', 'Survived']
colors_survival = ['#FFDAB9', '#ADD8E6']

# Custom autopct function
def autopct_format_survival(pct):
    count = int(round(pct * total_survival / 100.0))
    return f'{pct:.1f}%\n({count})'


plt.figure(figsize=(6, 6))
plt.pie(survival_counts,
        labels=labels_survival,
        autopct=autopct_format_survival,
        startangle=90,
        colors=colors_survival,
        wedgeprops={'edgecolor': 'black'})

# Title only (no legend)
plt.title('Percentage of Patient Survival', fontsize=14, weight='bold' )
plt.tight_layout()
plt.show()

##Cross-tabulation evaluating the combination of patient treatment and survival

# Crosstab
survival_chemo_ct = pd.crosstab(df["survival_status"], df["chemo_status"])

# Relabel the axes for clarity
survival_chemo_ct.index = ['Not Survived', 'Survived']
survival_chemo_ct.columns = ['No Chemotherapy', 'Chemotherapy']

print("\nSurvival vs Chemotherapy CrossTab:")
print(survival_chemo_ct)
# Plot

#Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    survival_chemo_ct,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    linewidths=0.5,
    linecolor='black',
    cbar_kws={"label": "Number of Patients"}
)

plt.title('Survival Status vs Chemotherapy', fontsize=14, weight='bold')
plt.xlabel('Chemotherapy Status', fontsize=12)
plt.ylabel('Survival Status', fontsize=12)
plt.tight_layout()
plt.show()