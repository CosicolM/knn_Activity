import pandas as pd

df = pd.read_csv("diabetes-k-nn.csv")

# Count zero values in each column
zero_counts = (df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum()
print(zero_counts)

