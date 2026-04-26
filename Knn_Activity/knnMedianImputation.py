import numpy as np
import pandas as pd

df = pd.read_csv("diabetes-k-nn.csv")

cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for col in cols_with_zero:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)

df.to_csv("diabetes-k-nn-cleaned.csv", index=False) 