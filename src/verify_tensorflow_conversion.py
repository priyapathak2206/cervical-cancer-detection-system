import os
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
from torchvision.models import resnet18
from torchvision import transforms


BASE_DIR = r"D:\cervical_canceer"

PYTORCH_WEIGHTS = os.path.join(
    BASE_DIR,
    "models",
    "best_augmented_resnet18.pth"
)

TF_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_cervical_tf.keras"
)

TEST_IMAGE = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed",
    "test",
    "abnormal",
    "148497788-148497836-002.BMP"
)


# ------------------------------------------------------------
# CLASS MAPPING
# ------------------------------------------------------------

classes = {
    0: "Abnormal",
    1: "Normal"
}


# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# LOAD PYTORCH MODEL
# ------------------------------------------------------------

print("\nLoading PyTorch model...")

pytorch_model = resnet18(weights=None)

pytorch_model.fc = torch.nn.Linear(
    pytorch_model.fc.in_features,
    2
)

state_dict = torch.load(
    PYTORCH_WEIGHTS,
    map_location="cpu",
    weights_only=True
)

pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()

print("PyTorch model loaded.")


# ------------------------------------------------------------
# LOAD TENSORFLOW MODEL
# ------------------------------------------------------------

print("\nLoading TensorFlow model...")

tf_model = tf.keras.models.load_model(
    TF_MODEL
)

print("TensorFlow model loaded.")


# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

print("\nLoading test image:")

print(TEST_IMAGE)

image = Image.open(TEST_IMAGE).convert("RGB")


# ------------------------------------------------------------
# PYTORCH PREDICTION
# ------------------------------------------------------------

torch_input = pytorch_transform(image).unsqueeze(0)

with torch.no_grad():

    torch_output = pytorch_model(
        torch_input
    )

    torch_probabilities = torch.softmax(
        torch_output,
        dim=1
    )[0].numpy()

torch_prediction = int(
    np.argmax(torch_probabilities)
)


# ------------------------------------------------------------
# TENSORFLOW PREPROCESSING
# ------------------------------------------------------------

image_resized = image.resize(
    (224, 224)
)

image_array = np.asarray(
    image_resized,
    dtype=np.float32
)

image_array = image_array / 255.0

mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

image_array = (
    image_array - mean
) / std

tf_input = np.expand_dims(
    image_array,
    axis=0
)


# ------------------------------------------------------------
# TENSORFLOW PREDICTION
# ------------------------------------------------------------

tf_output = tf_model(
    tf_input,
    training=False
).numpy()[0]

# PyTorch model outputs logits.
# TensorFlow model also outputs logits.

tf_exp = np.exp(
    tf_output - np.max(tf_output)
)

tf_probabilities = (
    tf_exp / tf_exp.sum()
)

tf_prediction = int(
    np.argmax(tf_probabilities)
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n==========================================")
print("MODEL CONVERSION VERIFICATION")
print("==========================================")

print("\nPyTorch Prediction:")
print(
    classes[torch_prediction]
)

print(
    "Abnormal probability:",
    f"{torch_probabilities[0] * 100:.4f}%"
)

print(
    "Normal probability:",
    f"{torch_probabilities[1] * 100:.4f}%"
)


print("\nTensorFlow Prediction:")
print(
    classes[tf_prediction]
)

print(
    "Abnormal probability:",
    f"{tf_probabilities[0] * 100:.4f}%"
)

print(
    "Normal probability:",
    f"{tf_probabilities[1] * 100:.4f}%"
)


# ------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------

difference = np.abs(
    torch_probabilities -
    tf_probabilities
)

print("\n==========================================")
print("COMPARISON")
print("==========================================")

print(
    "PyTorch class:",
    classes[torch_prediction]
)

print(
    "TensorFlow class:",
    classes[tf_prediction]
)

print(
    "Maximum probability difference:",
    f"{difference.max():.6f}"
)

if torch_prediction == tf_prediction:

    print("\nPASS: Both models predict the same class.")

else:

    print("\nWARNING: Models predict different classes.")


if difference.max() < 0.01:

    print(
        "PASS: Probability difference is below 1%."
    )

else:

    print(
        "WARNING: Probability difference is above 1%."
    )

print("\n==========================================")