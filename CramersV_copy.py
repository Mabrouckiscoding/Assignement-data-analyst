import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

file_path = r"C:\Users\mathi\Desktop\Assignement Data analyst\Assignement_réponse\Assignement-data-analyst\healthcare_dataset.csv"
df = pd.read_csv(file_path)
# Define function to calculate Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

# Select relevant columns
cat_cols = ['Gender', 'Blood Group Type', 'Medical Condition']

# Compute Cramér's V for each pair
cramers_v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

for col1 in cat_cols:
    for col2 in cat_cols:
        if col1 == col2:
            cramers_v_matrix.loc[col1, col2] = 1.0
        else:
            v = cramers_v(df[col1], df[col2])
            cramers_v_matrix.loc[col1, col2] = round(v, 3)

print(cramers_v_matrix)