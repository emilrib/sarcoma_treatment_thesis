import re

import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords

file_path = "/Users/emiliaribeiro/Documents/Masters/Thesis/sarcoma_treatment_thesis/dataset_clean.csv"
def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(file_path)

print(df.head(3))

"""
classification label to determine patients that needed to be reoperated and impact of reoperation
"""
def reoperation(row):
    if row == 1:
        return 'No'
    elif row > 1:
        return 'Yes'
    else:
        return 'error'

df['reoperation_label'] = df['number_all_operation_Timo'].apply(reoperation)

"""
classification of metastasis status during the treatment    
"""
def metastasis_label(row):
    if row == '-':
        return 'No'
    elif row == 0:
        return 'No'
    else:
        return 'Yes'

df['metastasis_label'] = df['number_metastasis_Timo'].apply(metastasis_label)

print(df[['Pat ID', 'number_metastasis_Timo', 'metastasis_label', "date_metastasis_Timo", "date_first_patientcontact_Timo"]])

#print(type("date_metastasis_Timo"))
def classify_metastasis(col1,col2):
    if col2 == '-' or col2 == None:
        return 'Metastasis free'
    elif col1 == 1:
        return 'Metastasis existed'
    elif col1 == 0 and col2 != '-':
        return 'Metastasis Appeared'

df['metastasis_status'] = df.apply(lambda row: classify_metastasis(row['metastasis_initial_Timo'],row['number_metastasis_Timo']), axis =1)

#print(df['PAT ID', 'metastasis_status'])
#calculate the number of days have passed since the first patient contact and the date of metastasis discovery

df["date_first_patientcontact_Timo"] = pd.to_datetime(df["date_first_patientcontact_Timo"], format="%Y-%m-%d", errors="coerce")
df["date_metastasis_Timo"] = pd.to_datetime(df["date_metastasis_Timo"], format="%Y-%m-%d", errors = "coerce")

#print(df.columns)
#print(type(df[["date_metastasis"]]))
#print(df.dtypes)

def days_of_metastasis(col1, col2):
    if col1 != None:
        return (col2 - col1).days
    else:
        return "No Metastasis"

df["Days_to_metastasis_discovery"] = df.apply(lambda row: days_of_metastasis(row['date_metastasis_Timo'],row['date_first_patientcontact_Timo']), axis =1)

#print(df['PAT ID', 'metastasis_status', 'Days_to_metastasis_discovery'])
#check dates where days give a negative value, did the patients have metastasis when first contact was made?

"""
for index, row in df.iterrows():
    if pd.notna(row["date_metastasis"]):
        print(row[['id', 'metastasis_status', "date_first_patientcontact", "date_metastasis", "Days_to_metastasis_discovery"]])

"""


"""
classification of chemo 

"""

#check if chemo was used or not
def chemo_indication_label(row):
    if row == "-":
        return "No"
    else:
        return "Yes"

df["Chemo_status"] = df['chemo_first_indication_Timo'].apply(chemo_indication_label)


#calculate the durantion of the first chemotherapy in days

df["Start date of line"] = pd.to_datetime(df["Start date of line"], format="%Y-%m-%d", errors="coerce")
df["Optional: End date of line"] = pd.to_datetime(df["Optional: End date of line"], format="%Y-%m-%d", errors = "coerce")

def chemo_duration(col1, col2):
    if pd.notna(col1) and pd.notna(col2):
        return (col2 - col1).days
    if pd.notna(col1) and col2 == None:
        return (pd.Timestamp.today().normalize() - col1).days
    else:
        return None

df["chemo_duration"] = df.apply(lambda row: chemo_duration(row['Start date of line'],row['Optional: End date of line']), axis =1)

#extract the labels of the status column

def status_label(text):
    return re.search(r'\(([^)]+)\)', text).group(1) if isinstance(text, str) and '(' in text else None

df["status_label"] = df['Status'].apply(status_label)

def status_label_numbered(text):
    if not isinstance(text, str):
        return None
    match = re.match(r'\[?(\d+)\]?', text.strip())
    return int(match.group(1)) if match else None

df["status_label_number"] = df['Status'].apply(status_label_numbered)


print(df.head())
#print(df[["id","first_round_chemo_start", "first_round_chemo_end", "first_chemo_duration"]])

file_path_output = "/Users/emiliaribeiro/Documents/Masters/Thesis/sarcoma_treatment_thesis/dataset_labelled.csv"
df.to_csv(file_path_output, index=False)