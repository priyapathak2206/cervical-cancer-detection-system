import pandas as pd
import os

input_file = (
    "clinical_dataset/"
    "cervical+cancer+risk+factors/"
    "risk_factors_cervical_cancer.csv"
)

output_file = "clinical_dataset/cleaned_clinical_data.csv"

# Load dataset
df = pd.read_csv(input_file)

# Features selected for risk assessment
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
    "STDs: Number of diagnosis"
]

target = "Dx:Cancer"

# Keep only selected columns
data = df[features + [target]].copy()

# Convert '?' to missing values
data = data.replace("?", pd.NA)

# Convert everything to numeric
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Fill missing values using median
for column in features:
    data[column] = data[column].fillna(data[column].median())

# Make sure target is integer
data[target] = data[target].astype(int)

# Create output directory
os.makedirs("clinical_dataset", exist_ok=True)

# Save cleaned dataset
data.to_csv(output_file, index=False)

print("Clinical dataset cleaned successfully!")
print()
print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nRemaining missing values:")
print(data.isnull().sum())

print("\nTarget distribution:")
print(data[target].value_counts())

print("\nSaved to:")
print(output_file)