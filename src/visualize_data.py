import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Dataset path
train_path = "Dataset/processed/train"


# Same preprocessing used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load dataset
train_dataset = datasets.ImageFolder(
    train_path,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


# Get one batch
images, labels = next(iter(train_loader))


# Undo normalization so images look normal
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

images = images * std + mean

images = torch.clamp(images, 0, 1)


# Display images
plt.figure(figsize=(12, 8))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    image = images[i].permute(1, 2, 0)

    plt.imshow(image)

    label = train_dataset.classes[labels[i]]

    plt.title(label)

    plt.axis("off")

plt.tight_layout()
plt.show()