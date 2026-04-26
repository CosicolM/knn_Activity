import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes-k-nn.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for readability
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Before scaling:")
print(X.head())

print("\nAfter scaling:")
print(X_scaled.head())




