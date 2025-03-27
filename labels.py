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
    if row['number_all_operation'] == 1:
        return 'No'
    elif row['number_all_operation'] > 1:
        return 'Yes'
    else:
        return 'error'

df['reoperation_label'] = df.apply(reoperation, axis=1)


"""
classification of metastasis status during the treatment
    
"""
def metastasis_label(row):
    if row['number_metastasis'] == '-':
        return 'No'
    elif row['number_metastasis'] == 0:
        return 'No'
    else:
        return 'Yes'

df['metastasis_label'] = df.apply(metastasis_label, axis=1)

print(df[['id', 'number_metastasis', 'metastasis_label', "date_metastasis", "date_first_patientcontact"]])

print(type("date_metastasis"))
def classify_metastasis(row):
    if row['metastasis_initial'] == row['metastasis_followup']:
        if row['metastasis_initial'] == 0:
            return 'Metastasis Free'
        elif row['metastasis_initial'] == 1:
            return 'Metastasis'
    else:
        return 'Metastasis Appeared'

df['metastasis_status'] = df.apply(classify_metastasis, axis=1)

#calculate the number of days have passed since the first patient contact and the date of metastasis discovery

df["date_first_patientcontact"] = pd.to_datetime(df["date_first_patientcontact"], format="%d.%m.%Y", errors="coerce")
df["date_metastasis"] = pd.to_datetime(df["date_metastasis"], format="%d.%m.%Y", errors = "coerce")

#print(df.columns)
#print(type(df[["date_metastasis"]]))
#print(df.dtypes)

def days_of_metastasis(row):
    if row["date_metastasis"] != "-":
        return (row["date_metastasis"] - row["date_first_patientcontact"]).days
    else:
        return "No Metastasis"

df["Days_to_metastasis_discovery"] = df.apply(days_of_metastasis, axis=1)

#check dates where days give a negative value, did the patients have metastisis when first contact was made?

"""
for index, row in df.iterrows():
    if pd.notna(row["date_metastasis"]):
        print(row[['id', 'metastasis_status', "date_first_patientcontact", "date_metastasis", "Days_to_metastasis_discovery"]])

"""
print(df.head())

"""
classification of chemo 

"""

#check if chemo was used or not
def chemo_indication_label(row):
    if row["chemo_first_indication"] == "-":
        return "No"
    else:
        return "Yes"

df["Chemo_status"] = df.apply(chemo_indication_label, axis=1)


#calculate the durantion of the first chemotherapy in days

df["first_round_chemo_start"] = pd.to_datetime(df["first_round_chemo_start"], format="%d.%m.%Y", errors="coerce")
df["first_round_chemo_end"] = pd.to_datetime(df["first_round_chemo_end"], format="%d.%m.%Y", errors = "coerce")

def first_chemo_duration(row):
    if pd.notna(row["first_round_chemo_start"]):
        if row["first_round_chemo_end"] != "open":
            return (row["first_round_chemo_end"] - row["first_round_chemo_start"]).days
        else:
            return (pd.Timestamp.today().normalize() - row["first_round_chemo_start"]).days
    else:
        return None

df["first_chemo_duration"] = df.apply(first_chemo_duration, axis=1)


#print(df[["id","first_round_chemo_start", "first_round_chemo_end", "first_chemo_duration"]])

file_path_output = "/Users/emiliaribeiro/Documents/Masters/Thesis/sarcoma_treatment_thesis/dataset.csv"
df.to_csv(file_path_output, index=False)