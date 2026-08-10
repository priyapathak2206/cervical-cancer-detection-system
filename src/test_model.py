import torch
from torchvision import models

print("Loading ResNet18...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Change final layer from 1000 classes to 2 classes
model.fc = torch.nn.Linear(model.fc.in_features, 2)

print("Model ready!")
print(model.fc)