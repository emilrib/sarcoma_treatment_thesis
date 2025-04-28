import os.path
import pandas as pd
from datetime import datetime
import numpy as np

if '__file__' in globals():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
else:
    # Fallback for interactive environments
    current_dir = os.getcwd()

output_directory = os.path.join(current_dir, "output")

input_file = os.path.join(current_dir, "dataset_labelled.csv")

def import_file(file_path):
    df = pd.read_csv(file_path)
    #print(df.head())
    return df
df = import_file(input_file)

print(df.head(3))

# Check if outcomes are correlated within possible clusters (e.g. lesion site)
group_col = 'Anatomic side of lesion'

grouped = df.groupby(group_col)['overall_survival_days'].agg(['mean', 'std', 'count'])
print(grouped)

# Optional: ANOVA to see if group affects outcome
import scipy.stats as stats
anova = stats.f_oneway(*(df[df[group_col] == g]['overall_survival_days'] for g in df[group_col].unique()))
print("ANOVA p-value:", anova.pvalue)
