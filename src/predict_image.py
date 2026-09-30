import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import sys


# -----------------------------

# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Image transformation
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Load ResNet18
# -----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/best_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# -----------------------------
# Get image path
# -----------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python src\\predict_image.py <image_path>")
    sys.exit()

image_path = sys.argv[1]


# -----------------------------
# Load image
# -----------------------------

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("Could not open image.")
    print("Error:", e)
    sys.exit()


# -----------------------------
# Prepare image
# -----------------------------

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# -----------------------------
# Prediction
# -----------------------------

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        1
    )


# -----------------------------
# Class names
# -----------------------------

classes = [
    "abnormal",
    "normal"
]

predicted_class = classes[prediction.item()]

confidence_value = confidence.item() * 100


# -----------------------------
# Result
# -----------------------------

print("\n==============================")
print("CERVICAL CELL IMAGE ANALYSIS")
print("==============================")

print("Image:", image_path)

print("Prediction:", predicted_class)

print(
    f"Confidence: {confidence_value:.2f}%"
)

print("\nClass probabilities:")

print(
    f"Abnormal: {probabilities[0][0].item() * 100:.2f}%"
)

print(
    f"Normal:   {probabilities[0][1].item() * 100:.2f}%"
)

print("\nNote:")
print(
    "This is an AI screening prediction, "
    "not a medical diagnosis."
)