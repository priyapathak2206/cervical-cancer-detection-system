import pandas as pd

file_path = "clinical_dataset/cervical+cancer+risk+factors/risk_factors_cervical_cancer.csv"

df = pd.read_csv(file_path)

targets = [
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Biopsy"
]

for column in targets:
    print("\n", column)
    print(df[column].value_counts())