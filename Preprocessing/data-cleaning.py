import os.path
import pandas as pd
from global_config import datasets_dir

# Read a dataset from /data/

input_file = os.path.join(datasets_dir, "dataset.csv")

def import_file_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file_to_dataframe(input_file)

print(df.head(2))


dataset_raw = df[['Pat ID','Date of birth', 'Gender', 'date_first_patientcontact_Timo', 'Date of histological diagnosis', 'Histological diagnosis', '(W) Other diagnoses?_Timo',
                  'grade_clean', 'cci_Timo', 'dignity_timo', 'anatomicregion_group_Timo', 'Affected tissue','resection_necrosis_timo', 'Anatomic side of lesion',
                  'Tumor maximal size before surgery', 'Type of index surgery', 'number_all_operation_Timo', 'Tumor maximal size (mm)',
                  '(all) Severty of reoperation (zB. Amputation)_Timo',
                  'Indication for radiotherapy','Reason for Chemotherapy','chemo_first_indication_Timo','Start date of line','Optional: End date of line', 'chemo_discontinuation_Timo','chemo_treatmentresponse_Timo'
                ,'metastasis_initial_Timo', 'metastasis_followup_Timo','date_metastasis_Timo', 'number_metastasis_Timo', 'date_death_Timo', 'Date of last follow-up', 'Status']]
#print(dataset_raw.head())

#check for duplicates
duplicates = dataset_raw[dataset_raw.duplicated(keep=False)]


#dataset_filtered = dataset_raw[dataset_raw['anatomicregion_group_Timo'] != 1]

#test for change in total amount
total = dataset_raw['Pat ID'].nunique()
print( f'Total Patients: {total}')


dataset_filtered = dataset_raw.copy()

dataset_filtered.loc[:, 'Histological diagnosis'] = dataset_filtered['Histological diagnosis'].str.replace(r'^\d+(\.\d+)*\.\s*', '', regex=True)


output_file = os.path.join(datasets_dir, "dataset_clean.csv")
dataset_filtered.to_csv(output_file, index=False)
print(f"File successfully saved to {output_file}")