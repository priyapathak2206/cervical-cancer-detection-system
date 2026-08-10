import pandas as pd

file_path = "clinical_dataset/cervical+cancer+risk+factors/risk_factors_cervical_cancer.csv"

df = pd.read_csv(file_path)

features = [
    "Age",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs:HPV",
    "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis"
]

for column in features:
    print("\n" + column)
    print(df[column].value_counts().head(10))