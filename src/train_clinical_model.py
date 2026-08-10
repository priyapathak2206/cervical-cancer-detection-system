import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load split data
train_file = "clinical_dataset/split/train.csv"
test_file = "clinical_dataset/split/test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Separate input and target
X_train = train.drop("Dx:Cancer", axis=1)
y_train = train["Dx:Cancer"]

X_test = test.drop("Dx:Cancer", axis=1)
y_test = test["Dx:Cancer"]

# Pipeline:
# 1. Standardize numerical features
# 2. Train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
])

# Train
print("Training clinical model...")

model.fit(X_train, y_train)

print("Clinical model trained successfully!")

# Predictions
y_pred = model.predict(X_test)

# Results
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Cancer", "Cancer"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(
    model,
    "models/clinical_logistic_model.pkl"
)

print("\nModel saved to:")
print("models/clinical_logistic_model.pkl")