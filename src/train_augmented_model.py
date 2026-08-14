import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Paths
# -----------------------------
train_path = "Dataset/processed/train"
validation_path = "Dataset/processed/validation"

# -----------------------------
# Data augmentation
# Training images only
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation images are NOT augmented
validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load datasets
# -----------------------------
train_dataset = datasets.ImageFolder(
    train_path,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    validation_path,
    transform=validation_transform
)

# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Classes:", train_dataset.classes)

# -----------------------------
# ResNet18
# -----------------------------
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

# -----------------------------
# Class-weighted loss
# -----------------------------
class_counts = torch.bincount(
    torch.tensor(train_dataset.targets)
)

weights = len(train_dataset) / (
    2 * class_counts.float()
)

weights = weights.to(device)

print("Class weights:", weights)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# -----------------------------
# Training
# -----------------------------
epochs = 10
best_validation_accuracy = 0.0

os.makedirs("models", exist_ok=True)

for epoch in range(epochs):

    model.train()

    correct = 0
    total = 0
    training_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()

    correct = 0
    total = 0
    validation_loss = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {training_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {validation_loss / len(validation_loader):.4f} "
        f"Val Acc: {validation_accuracy:.2f}%"
    )

    # -----------------------------
    # Save best model
    # -----------------------------
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            "models/best_augmented_resnet18.pth"
        )

        print("Best augmented model saved!")

print()
print("Augmented training completed!")
print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2f}%"
)