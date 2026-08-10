import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset paths
train_path = "Dataset/processed/train"
validation_path = "Dataset/processed/validation"
test_path = "Dataset/processed/test"


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

test_dataset = datasets.ImageFolder(
    test_path,
    transform=transform
)


# Create DataLoaders
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

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Print information
print("Dataset loaded successfully!")
print()

print("Classes:", train_dataset.classes)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Testing images:", len(test_dataset))

print()

# Get one batch
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)