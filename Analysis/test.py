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