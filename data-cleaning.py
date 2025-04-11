import os.path
import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(current_dir, "output")

input_file = os.path.join(current_dir, "dataset.csv")


def import_file_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file_to_dataframe(input_file)

print(df.head(2))

#print(type(df["Start date of line"]))
#add new column to have the first chemo date and the same for the end of the first round of chemo

"""
def date_retrieval(row):
    if pd.isnull(row) or row == "NA" or row == None:
        return None
    elif str(row).strip().lower() == "open":
        return "open"
    else:
        return str(row).split("|")[0].strip()

df["first_round_chemo_start"] = df["Start date of line"].apply(date_retrieval)
df["first_round_chemo_end"] = df["Optional: End date of line"].apply(date_retrieval)
"""
#print(df[["Pat ID", "first_round_chemo_start", "first_round_chemo_end"]].head(20))
#print(df["first_round_chemo_start"].head(14))
#add new column to have the first chemo date and the same for the end of the first round of chemo


dataset_raw = df[['Pat ID','Date of birth', 'Gender', 'date_first_patientcontact_Timo', 'Date of histological diagnosis', 'Histological diagnosis', '(W) Other diagnoses?_Timo',
                  'Grading (FNCLCC)', 'cci_Timo', 'dignity_timo', 'anatomicregion_group_Timo', 'Affected tissue','resection_necrosis_timo', 'Anatomic side of lesion',
                  'Tumor maximal size before surgery', 'Type of index surgery', 'number_all_operation_Timo', 'Tumor maximal size (mm)',
                  '(all) Severty of reoperation (zB. Amputation)_Timo',
                  'Indication for radiotherapy','Reason for Chemotherapy','chemo_first_indication_Timo','Start date of line','Optional: End date of line', 'chemo_discontinuation_Timo','chemo_treatmentresponse_Timo'
                ,'metastasis_initial_Timo', 'metastasis_followup_Timo','date_metastasis_Timo', 'number_metastasis_Timo', 'date_death_Timo', 'Date of last follow-up', 'Status']]
#print(dataset_raw.head())

#check for duplicates
duplicates = dataset_raw[dataset_raw.duplicated(keep=False)]

#print(duplicates)


dataset_filtered = dataset_raw[dataset_raw['anatomicregion_group_Timo'] != 1]

dataset_clean = dataset_filtered[dataset_filtered['dignity_timo'] == 'malignant']




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
os.makedirs(output_directory, exist_ok=True)

output_file = os.path.join(output_directory, "dataset_clean.csv")

dataset_clean.to_csv(output_file, index=False)
print(f"File successfully saved to {output_file}")
