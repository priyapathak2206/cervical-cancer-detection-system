import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
train_path = "Dataset/processed/train"
validation_path = "Dataset/processed/validation"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_dataset = datasets.ImageFolder(
    train_path,
    transform=transform
)

validation_dataset = datasets.ImageFolder(
    validation_path,
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))

# Find class counts
targets = torch.tensor(train_dataset.targets)
class_counts = torch.bincount(targets)

print("Class counts:", class_counts)

# Give more sampling weight to minority class
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

# ResNet18
model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Training
epochs = 10
best_validation_accuracy = 0.0

os.makedirs("models", exist_ok=True)

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_accuracy = (correct / total) * 100

    # Validation
    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            validation_total += labels.size(0)
            validation_correct += (
                predictions == labels
            ).sum().item()

    validation_accuracy = (
        validation_correct / validation_total
    ) * 100

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {validation_accuracy:.2f}%"
    )

    # Save best model
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            "models/balanced_resnet18.pth"
        )

        print("Best balanced model saved!")

print("\nBalanced training completed!")
print(
    "Best validation accuracy:",
    f"{best_validation_accuracy:.2f}%"
)