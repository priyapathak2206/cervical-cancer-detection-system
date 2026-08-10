import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = "clinical_dataset/cleaned_clinical_data.csv"

df = pd.read_csv(input_file)

# Separate features and target
X = df.drop("Dx:Cancer", axis=1)
y = df["Dx:Cancer"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create folder
os.makedirs("clinical_dataset/split", exist_ok=True)

# Save training data
train_data = X_train.copy()
train_data["Dx:Cancer"] = y_train

train_data.to_csv(
    "clinical_dataset/split/train.csv",
    index=False
)

# Save testing data
test_data = X_test.copy()
test_data["Dx:Cancer"] = y_test

test_data.to_csv(
    "clinical_dataset/split/test.csv",
    index=False
)

print("Clinical dataset split completed!")
print()

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())