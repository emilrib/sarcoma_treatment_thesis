import os.path
import re
import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords

#current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.getcwd())

input_file = os.path.join(output_dir, "dataset_clean.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df.head(3))

"""
Calculation of age
"""

def calculate_age(year_of_birth):
    current_year = datetime.now().year
    return current_year - year_of_birth

df['age'] = df['Date of birth'].apply(calculate_age)

"""
Binary classification of the gender
"""
def gender_label(row):
    if row == "female":
        return 1
    elif row == "male":
        return 2
    else:
        return "error"

df['gender_label'] = df['Gender'].apply(gender_label)

"""
classification label to determine patients that needed to be reoperated and impact of reoperation
"""
def reoperation(row):
    if row >= 1:
        return 2
    else:
        return 1

df['reoperation_label'] = df['number_all_operation_Timo'].apply(reoperation)

"""
classification of metastasis status during the treatment    
"""
def metastasis_label(row):
    if row == '-':
        return 1
    elif row == 0:
        return 1
    else:
        return 2

df['metastasis_label'] = df['number_metastasis_Timo'].apply(metastasis_label)

print(df[['Pat ID', 'number_metastasis_Timo', 'metastasis_label', "date_metastasis_Timo", "date_first_patientcontact_Timo"]])

df["date_first_patientcontact_Timo"] = pd.to_datetime(df["date_first_patientcontact_Timo"], format="%Y-%m-%d", errors="coerce")
df["date_metastasis_Timo"] = pd.to_datetime(df["date_metastasis_Timo"], format="%Y-%m-%d", errors = "coerce")
#print(type("date_metastasis_Timo"))
def classify_metastasis(col1,col2, col3):
    if col1 == '-' or pd.isnull(col1):
        return 1
    elif col2 > col3:
        return 2
    elif col2 < col3:
        return 3
    else:
        return 0

df['metastasis_status'] = df.apply(lambda row: classify_metastasis(row['number_metastasis_Timo'],row['date_first_patientcontact_Timo'], row['date_metastasis_Timo']), axis =1)

#calculate the number of days from the first contact until from the date of discovery of metastasis for patients that did not have
def days_of_metastasis(col1, col2, col3):
    if col1 == 3:
        return (col2 - col3).days
    else:
        return None

df["Days_to_metastasis_discovery"] = df.apply(lambda row: days_of_metastasis(row['metastasis_status'],row['date_metastasis_Timo'],row['date_first_patientcontact_Timo']), axis =1)


#label to heck if chemo was used or not
def chemo_indication_label(row):
    if row == "-":
        return 1
    else:
        return 2

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

#label to find out the status of patients
def status_label_numbered(text):
    if not isinstance(text, str):
        return None
    match = re.match(r'\[?(\d+)\]?', text.strip())
    return int(match.group(1)) if match else None

df["status_label_number"] = df['Status'].apply(status_label_numbered)

#add label to distinguish patients that receive radiotherapy from the ones that did not
def radiotherapy_status(text):
    val_str = str(text).strip().lower()
    match = re.match(r"\[?(\d+)\]?", val_str)
    if match and int(match.group(1)) == 0 or pd.isnull(text):
        return 1
    if val_str == "none":
        return 1
    return 2

df["radiation_status"] = df["Indication for radiotherapy"].apply(radiotherapy_status)

#add label to distinguish patients that alive and patients that are not alive
def death_by_disease(row):
    if row == 3:
        return 2
    return 1

df["deceased_by_disease"] = df['status_label_number'].apply(death_by_disease)

#calculate the number of survival days a patient that passed away from the decease (from day of contact until the date of death)
df["date_death_Timo"] = pd.to_datetime(df["date_death_Timo"], format="%Y-%m-%d", errors = "coerce")
def survival_days(col1,col2, col3):
    if col1 == 3:
        return (col2 - col3).days
    return None

df["overall_survival_days"] = df.apply(lambda row: survival_days(row['status_label_number'],row['date_death_Timo'], row['date_first_patientcontact_Timo']), axis =1)

print(df.head())
#print(df[["id","first_round_chemo_start", "first_round_chemo_end", "first_chemo_duration"]])


os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "dataset_labelled.csv")
df.to_csv(output_file, index=False)
print(f"File successfully saved to {output_file}")

"""
Writing the translation of the binary code
"""

with open("README.md", "a") as file:
    file.write(
        "### Code Legends\n"
        "#### Gender\n"
        "- `1`: **Female**\n"
        "- `2`: **Male**\n\n"
        "#### Metastasis Classification\n"
        "- `1`: **Metastasis free** during treatment observation\n"
        "- `2`: Metastasis existed **before** first patient contact\n"
        "- `3`: Metastasis appeared **after** first patient contact\n"
        "- `0`: Metastasis present, discovery date unknown\n\n"
        "#### Patient Status\n"
        "- `1`: **NED** (No evidence of disease)\n"
        "- `2`: **AWD** (Alive with disease)\n"
        "- `3`: **DOD** (Dead of disease)\n\n"
        "#### Remaining Labels (Binary Codes)\n"
        "- `1`: **No**\n"
        "- `2`: **Yes**\n"
    )



