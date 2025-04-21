import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#current_dir = os.path.dirname(os.path.abspath(__file__))
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

df_history = df[['Pat ID','date_first_patientcontact_Timo', 'Histological diagnosis', '(W) Other diagnoses?_Timo', 'cci_Timo', 'dignity_timo', 'anatomicregion_group_Timo', 'Affected tissue', 'number_all_operation_Timo', 'Tumor maximal size (mm)', 'metastasis_status', 'metastasis_label', 'reoperation_label']]

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
