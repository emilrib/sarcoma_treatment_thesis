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

"""
DEMOGRAPHIC DATA 
"""

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

plt.xlabel('CCI Groups')
plt.ylabel('Number of Occurrences')
plt.title('Occurrences by Charlson Comorbidity Index (CCI)', fontsize=14)
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
group_words_histoloical_diagnosis = pd.DataFrame(histological_diagnosis_counts.items(), columns=['Most occurred Histological Diagnosis', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Display the top 10 most common group words
print(group_words_histoloical_diagnosis.head(10))



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

# Analyse the percentage of the data sample with metastasis and its resurfacing time
metastasis_counts = df_history['metastasis_label'].value_counts()

label_map = {0: 'No', 1: 'Yes'}
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


#Display in a box plot how the tumor size is distributed

plt.figure(figsize=(8, 6))
sns.boxplot(y=df_history["Tumor maximal size (mm)"], color="#ADD8E6", width=0.5)

plt.title('Distribution of Tumor Size (mm)', fontsize=14)
plt.ylabel('Tumor Size (mm)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add mean point
mean_value = df_history["Tumor maximal size (mm)"].mean()
plt.scatter(0, mean_value, color='red', label=f"Mean: {mean_value:.1f} mm", zorder=10)
plt.legend()

plt.tight_layout()
plt.show()

#Display boxplot for grade distribution

plt.figure(figsize=(8, 6))
sns.countplot(
    x="grade_clean",
    data=df_history,
    palette="pastel",
    edgecolor="black",
    hue="grade_clean",
    legend=False
)

plt.title('Number of Occurences by FNCLCC Grading', fontsize=14)
plt.xlabel('Grading', fontsize=12)
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

"""
PATIENT TREATMENT
"""
##  PIe chart displaying the percentage of patients that received and did not receive chemotherapy
# Count the number of 0s and 1s
chemo_counts = df["chemo_status"].value_counts().sort_index()
total = chemo_counts.sum()

# Labels mapped manually
labels = ['No Chemotherapy', 'Chemotherapy']

# Colors for the pie
colors = ['#ADD8E6', '#FFDAB9']  # pastel blue and peach

# Custom autopct function
def autopct_format(pct):
    count = int(round(pct * total / 100.0))
    return f'{pct:.1f}%\n({count})'

plt.figure(figsize=(7, 7))
plt.pie(chemo_counts, labels=labels, autopct=autopct_format, startangle=90, colors=colors,
        wedgeprops={'edgecolor': 'black'})

# Title and legend
plt.title('Percentage of Chemotherapy Use', fontsize=14)
plt.tight_layout()
plt.show()

"""
PATIENT OUTCOME
"""
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


plt.figure(figsize=(7, 7))
plt.pie(survival_counts, labels=labels_survival, autopct=autopct_format_survival, startangle=90, colors=colors_survival,
        wedgeprops={'edgecolor': 'black'})

# Title only (no legend)
plt.title('Percentage of Patient Survival', fontsize=14)
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

#heatmap
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

plt.title('Survival Status vs Chemotherapy (Heatmap)', fontsize=14)
plt.xlabel('Chemotherapy Status', fontsize=12)
plt.ylabel('Survival Status', fontsize=12)
plt.tight_layout()
plt.show()