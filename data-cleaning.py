import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords

def import_file_to_dataframe(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    #print(df.head())
    return df
df = import_file_to_dataframe('/Users/emiliaribeiro/Documents/Masters/Thesis/sarcoma_treatment_thesis/Timo dataset.xlsx')

dataset_raw = df[['id','year_of_birth', 'gender', 'date_first_patientcontact', 'OP_638 - Local situation', 'OP_648 - Presence of metastasis', 'OP_1115 - Specify current status', '(W) Other diagnoses?', 'cci', 'dignity', 'anatomicregion_group', 'anatomicregion_code', 'anatomicregion_code_grouping', 'date_radiologyexam', 'type_radiologyexam','date_first_radiologyexam', 'type_first_radiologyexam', 'date_biopsy','type_biopsy', 'biopsy_neoadjuvant', 'biopsy_grading', 'resection_diagnosis', 'number_all_operation', 'date_reoperation', '(all) Severty of reoperation (zB. Amputation)', 'chemo_indication', 'chemo_first_indication', 'date_chemo_start', 'date_chemo_end', 'chemo_substance', 'chemo_discontinuation', 'chemo_treatmentresponse', 'metastasis_initial', 'metastasis_followup', 'date_last_contact', 'status_last_contact', '(newest) Patient history (clinics, therapy) - latest to newest', 'date_metastasis', 'number_metastasis', 'date_death', '(newest) Patient history (clinics, therapy) - latest to newest', 'endpoint']]
#print(dataset_raw.head())

#check for duplicates
duplicates = dataset_raw[dataset_raw.duplicated(keep=False)]
#print(duplicates)

dataset_clean = dataset_raw[dataset_raw['anatomicregion_code_grouping'] != 1]

print(dataset_clean)

"""
def extract_group_words(text):
    text = text.strip().lower()  # Convert to lowercase and remove surrounding spaces
    if text == '-':  # Ignore rows containing only '-'
        return []
    text = re.sub(r'\b\d{4}\b', '', text)  # Remove year dates
    group_words = re.split(r'[|,]', text)  # Split by '|' or ','
    group_words = [word.strip() for word in group_words if word.strip()]  # Remove extra spaces and empty strings
    return group_words

# Extract and flatten all group words from the column
all_group_words = dataset_clean['(W) Other diagnoses?'].dropna().apply(extract_group_words).sum()

# Count the frequency of group words
group_word_counts = Counter(all_group_words)

# Convert to DataFrame for better visualization
group_words_other_diagnosis = pd.DataFrame(group_word_counts.items(), columns=['Group Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Display the top 10 most common group words
print(group_words_other_diagnosis.head(10))

"""

file_path_output = "/Users/emiliaribeiro/Documents/Masters/Thesis/sarcoma_treatment_thesis/dataset_clean.csv"
dataset_clean.to_csv(file_path_output, index=False)




