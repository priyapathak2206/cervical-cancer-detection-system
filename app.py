import os
import uuid

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


# --------------------------------------------------
# Flask setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_cervical_tf.keras"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "gradcam_outputs"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

print("Using TensorFlow backend.")
print("Loading model:", MODEL_PATH)


# --------------------------------------------------
# TensorFlow Model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("TensorFlow model loaded successfully.")


# --------------------------------------------------
# Class names
# --------------------------------------------------

# Same mapping as the original PyTorch model:
#
# 0 = Abnormal
# 1 = Normal

CLASS_NAMES = {
    0: "Abnormal",
    1: "Normal"
}


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image):
    """
    Convert PIL image into the same normalized
    format used during PyTorch training.
    """

    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = (
        image_array - MEAN
    ) / STD

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# Softmax
# --------------------------------------------------

def softmax(logits):

    logits = logits - np.max(logits)

    exp_values = np.exp(logits)

    return exp_values / np.sum(exp_values)


# --------------------------------------------------
# TensorFlow Grad-CAM
# --------------------------------------------------

def generate_gradcam(
    input_tensor,
    predicted_class
):

    # layer4_1_relu2 is the final ReLU of the
    # final ResNet18 residual block.
    #
    # This is equivalent to the final convolutional
    # region used by the original PyTorch Grad-CAM.

    target_layer = model.get_layer(
        "layer4_1_relu2"
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False
        )

        class_score = predictions[
            :, predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling of gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    # Weight each feature map by its
    # corresponding gradient importance
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# --------------------------------------------------
# Create Grad-CAM visualization
# --------------------------------------------------

def create_gradcam_visualization(
    original_image,
    heatmap
):

    # Convert original image to 224x224
    resized_image = original_image.resize(
        (224, 224)
    )

    rgb_image = np.asarray(
        resized_image,
        dtype=np.uint8
    )

    # Resize Grad-CAM heatmap
    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # OpenCV uses BGR for color maps
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay heatmap on original image
    visualization = cv2.addWeighted(
        rgb_image,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    return visualization


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "ok",
        "model": "resnet18_cervical_tensorflow",
        "framework": "TensorFlow",
        "device": "CPU"
    })


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": (
                "No image uploaded. "
                "Use field name 'image'."
            )
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

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        # ------------------------------------------
        # Preprocess
        # ------------------------------------------

        input_array = preprocess_image(
            image
        )

        input_tensor = tf.convert_to_tensor(
            input_array,
            dtype=tf.float32
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        logits = model(
            input_tensor,
            training=False
        ).numpy()[0]

        probabilities = softmax(
            logits
        )

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        prediction = CLASS_NAMES[
            predicted_class
        ]

        abnormal_probability = float(
            probabilities[0]
        )

        normal_probability = float(
            probabilities[1]
        )

        # ------------------------------------------
        # Grad-CAM
        # ------------------------------------------

        heatmap = generate_gradcam(
            input_tensor,
            predicted_class
        )

        visualization = (
            create_gradcam_visualization(
                image,
                heatmap
            )
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

            "confidence": round(
                confidence * 100,
                2
            ),

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

        print(
            "Prediction error:",
            str(e)
        )

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
    print(
        "Framework: TensorFlow"
    )
    print(
        "Health:  http://127.0.0.1:5000/health"
    )
    print(
        "Predict: http://127.0.0.1:5000/predict"
    )
    print("======================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )