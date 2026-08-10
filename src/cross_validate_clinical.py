import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load cleaned clinical data
file_path = "clinical_dataset/cleaned_clinical_data.csv"

df = pd.read_csv(file_path)

# Inputs and target
X = df.drop("Dx:Cancer", axis=1)
y = df["Dx:Cancer"]


# Clinical model
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


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Metrics
scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]


print("Starting 5-fold cross-validation...\n")


results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring
)


print("Cross-validation results:")
print()

print(
    "Accuracy:",
    results["test_accuracy"]
)

print(
    "Precision:",
    results["test_precision"]
)

print(
    "Recall:",
    results["test_recall"]
)

print(
    "F1-score:",
    results["test_f1"]
)


print("\nAverage results:")

print(
    "Average Accuracy:",
    results["test_accuracy"].mean()
)

print(
    "Average Precision:",
    results["test_precision"].mean()
)

print(
    "Average Recall:",
    results["test_recall"].mean()
)

print(
    "Average F1-score:",
    results["test_f1"].mean()
)