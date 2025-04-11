import re

import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords

file_path = "/Users/emiliaribeiro/Documents/Masters/Thesis/sarcoma_treatment_thesis/dataset_labelled.csv"
def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(file_path)

#print(df.head(3))

