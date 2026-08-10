import pandas as pd

file_path = "clinical_dataset/cervical+cancer+risk+factors/risk_factors_cervical_cancer.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print()

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print(column)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())