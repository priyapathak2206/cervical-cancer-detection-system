import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Image path
# -----------------------------
image_path = "Dataset/processed/test/abnormal/148497788-148497836-002.BMP"


# -----------------------------
# Load model
# -----------------------------
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/best_augmented_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Augmented ResNet18 loaded!")


# -----------------------------
# Image preprocessing
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
# Load image
# -----------------------------
image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0)
input_tensor = input_tensor.to(device)


# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

confidence = probabilities[0][predicted_class].item() * 100

class_names = ["abnormal", "normal"]

print()
print("==============================")
print("CERVICAL CELL ANALYSIS")
print("==============================")

print(
    "Prediction:",
    class_names[predicted_class]
)

print(
    f"Confidence: {confidence:.2f}%"
)


# -----------------------------
# Grad-CAM
# -----------------------------
target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [
    ClassifierOutputTarget(predicted_class)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]


# -----------------------------
# Prepare original image
# -----------------------------
rgb_image = image.resize((224, 224))

rgb_image = torch.tensor(
    __import__("numpy").array(rgb_image)
) / 255.0

rgb_image = rgb_image.numpy()


# -----------------------------
# Generate heatmap
# -----------------------------
visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# -----------------------------
# Save result
# -----------------------------
output_path = "models/gradcam_result.jpg"

Image.fromarray(visualization).save(
    output_path
)

print()
print("Grad-CAM generated successfully!")
print("Saved to:", output_path)