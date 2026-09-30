import os
import uuid

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# --------------------------------------------------
# Flask setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_augmented_resnet18.pth"
)

RESULT_DIR = os.path.join(BASE_DIR, "gradcam_outputs")
os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("Loading model:", MODEL_PATH)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Class names
# --------------------------------------------------

# ImageFolder used:
# normal = 1
# abnormal = 0

CLASS_NAMES = {
    0: "Abnormal",
    1: "Normal"
}


# --------------------------------------------------
# Grad-CAM target layer
# --------------------------------------------------

target_layers = [model.layer4[-1]]


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "best_augmented_resnet18",
        "device": str(device)
    })


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded. Use field name 'image'."
        }), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # ------------------------------------------
        # Read image
        # ------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")

        original_image = np.array(image)

        # ------------------------------------------
        # Resize image for model
        # ------------------------------------------

        resized_image = image.resize((224, 224))

        rgb_image = np.array(resized_image).astype(np.float32) / 255.0

        # ------------------------------------------
        # Prepare tensor
        # ------------------------------------------

        input_tensor = transform(image).unsqueeze(0).to(device)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

            predicted_class = torch.argmax(
                probabilities
            ).item()

            confidence = probabilities[
                predicted_class
            ].item()

        prediction = CLASS_NAMES[predicted_class]

        normal_probability = probabilities[1].item()
        abnormal_probability = probabilities[0].item()

        # ------------------------------------------
        # Grad-CAM
        # ------------------------------------------

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        grayscale_cam = cam(
            input_tensor=input_tensor
        )[0]

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        # ------------------------------------------
        # Save Grad-CAM image
        # ------------------------------------------

        result_filename = (
            f"gradcam_{uuid.uuid4().hex}.jpg"
        )

        result_path = os.path.join(
            RESULT_DIR,
            result_filename
        )

        cv2.imwrite(
            result_path,
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        # ------------------------------------------
        # Response
        # ------------------------------------------

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "normal_probability": round(
                normal_probability * 100,
                2
            ),
            "abnormal_probability": round(
                abnormal_probability * 100,
                2
            ),
            "gradcam_url": (
                f"/gradcam/{result_filename}"
            )
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Grad-CAM image endpoint
# --------------------------------------------------

@app.route(
    "/gradcam/<filename>",
    methods=["GET"]
)
def gradcam(filename):

    return send_from_directory(
        RESULT_DIR,
        filename
    )


# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("======================================")
    print("Cervical Cancer Detection API")
    print("======================================")
    print("Health:  http://127.0.0.1:5000/health")
    print("Predict: http://127.0.0.1:5000/predict")
    print("======================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )