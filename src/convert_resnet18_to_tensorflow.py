import os
import numpy as np
import torch
import tensorflow as tf
from torchvision.models import resnet18


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\cervical_canceer"
PYTORCH_WEIGHTS = os.path.join(
    BASE_DIR, "models", "best_augmented_resnet18.pth"
)

TF_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "resnet18_cervical_tf.keras"
)

TF_WEIGHTS_PATH = os.path.join(
    BASE_DIR, "models", "resnet18_cervical_tf.weights.h5"
)


# ============================================================
# BUILD PYTORCH MODEL
# ============================================================

print("\nLoading PyTorch model...")

pytorch_model = resnet18(weights=None)

# Your model has 2 classes:
# 0 = Abnormal
# 1 = Normal
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

print("PyTorch weights loaded successfully.")


# ============================================================
# BUILD TENSORFLOW RESNET18
# ============================================================

def conv_bn_relu(x, filters, stride=1, name="block"):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        name=f"{name}_bn"
    )(x)

    x = tf.keras.layers.ReLU(
        name=f"{name}_relu"
    )(x)

    return x


def residual_block(
    x,
    filters,
    stride=1,
    downsample=False,
    name="block"
):
    shortcut = x

    # First convolution
    y = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)

    y = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        name=f"{name}_bn1"
    )(y)

    y = tf.keras.layers.ReLU(
        name=f"{name}_relu1"
    )(y)

    # Second convolution
    y = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2"
    )(y)

    y = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        name=f"{name}_bn2"
    )(y)

    # Downsample shortcut when required
    if downsample:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="valid",
            use_bias=False,
            name=f"{name}_downsample_conv"
        )(shortcut)

        shortcut = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            name=f"{name}_downsample_bn"
        )(shortcut)

    y = tf.keras.layers.Add(
        name=f"{name}_add"
    )([y, shortcut])

    y = tf.keras.layers.ReLU(
        name=f"{name}_relu2"
    )(y)

    return y


def build_tensorflow_resnet18():

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="input"
    )

    # PyTorch ResNet18 stem
    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1"
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        name="bn1"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu"
    )(x)

    x = tf.keras.layers.ZeroPadding2D(
        padding=1,
        name="maxpool_padding"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="valid",
        name="maxpool"
    )(x)

    # Layer 1
    x = residual_block(
        x, 64, 1, False, "layer1_0"
    )

    x = residual_block(
        x, 64, 1, False, "layer1_1"
    )

    # Layer 2
    x = residual_block(
        x, 128, 2, True, "layer2_0"
    )

    x = residual_block(
        x, 128, 1, False, "layer2_1"
    )

    # Layer 3
    x = residual_block(
        x, 256, 2, True, "layer3_0"
    )

    x = residual_block(
        x, 256, 1, False, "layer3_1"
    )

    # Layer 4
    x = residual_block(
        x, 512, 2, True, "layer4_0"
    )

    x = residual_block(
        x, 512, 1, False, "layer4_1"
    )

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="avgpool"
    )(x)

    # Two-class output
    outputs = tf.keras.layers.Dense(
        2,
        name="fc"
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet18_Cervical"
    )


tf_model = build_tensorflow_resnet18()

print("TensorFlow ResNet18 created.")


# ============================================================
# HELPER: COPY CONVOLUTION WEIGHTS
# ============================================================

def copy_conv_weights(tf_layer, torch_weight):

    weight = torch_weight.detach().cpu().numpy()

    # PyTorch:
    # [out_channels, in_channels, height, width]
    #
    # TensorFlow:
    # [height, width, in_channels, out_channels]

    weight = np.transpose(
        weight,
        (2, 3, 1, 0)
    )

    tf_layer.set_weights([weight])


# ============================================================
# HELPER: COPY BATCH NORMALIZATION WEIGHTS
# ============================================================

def copy_bn_weights(tf_layer, torch_bn):

    gamma = torch_bn.weight.detach().cpu().numpy()
    beta = torch_bn.bias.detach().cpu().numpy()
    moving_mean = torch_bn.running_mean.detach().cpu().numpy()
    moving_var = torch_bn.running_var.detach().cpu().numpy()

    tf_layer.set_weights([
        gamma,
        beta,
        moving_mean,
        moving_var
    ])


# ============================================================
# COPY STEM
# ============================================================

print("\nTransferring weights...")

copy_conv_weights(
    tf_model.get_layer("conv1"),
    state_dict["conv1.weight"]
)

copy_bn_weights(
    tf_model.get_layer("bn1"),
    pytorch_model.bn1
)


# ============================================================
# COPY RESIDUAL BLOCKS
# ============================================================

for layer_number in range(1, 5):

    for block_number in range(2):

        torch_prefix = f"layer{layer_number}.{block_number}"
        tf_prefix = f"layer{layer_number}_{block_number}"

        # Conv1
        copy_conv_weights(
            tf_model.get_layer(
                f"{tf_prefix}_conv1"
            ),
            state_dict[
                f"{torch_prefix}.conv1.weight"
            ]
        )

        # BN1
        copy_bn_weights(
            tf_model.get_layer(
                f"{tf_prefix}_bn1"
            ),
            getattr(
                getattr(
                    getattr(
                        pytorch_model,
                        f"layer{layer_number}"
                    ),
                    str(block_number)
                ),
                "bn1"
            )
        )

        # Conv2
        copy_conv_weights(
            tf_model.get_layer(
                f"{tf_prefix}_conv2"
            ),
            state_dict[
                f"{torch_prefix}.conv2.weight"
            ]
        )

        # BN2
        copy_bn_weights(
            tf_model.get_layer(
                f"{tf_prefix}_bn2"
            ),
            getattr(
                getattr(
                    getattr(
                        pytorch_model,
                        f"layer{layer_number}"
                    ),
                    str(block_number)
                ),
                "bn2"
            )
        )

        # Downsample
        if block_number == 0 and layer_number > 1:

            copy_conv_weights(
                tf_model.get_layer(
                    f"{tf_prefix}_downsample_conv"
                ),
                state_dict[
                    f"{torch_prefix}.downsample.0.weight"
                ]
            )

            downsample_bn = getattr(
                getattr(
                    getattr(
                        pytorch_model,
                        f"layer{layer_number}"
                    ),
                    str(block_number)
                ),
                "downsample"
            )[1]

            copy_bn_weights(
                tf_model.get_layer(
                    f"{tf_prefix}_downsample_bn"
                ),
                downsample_bn
            )


# ============================================================
# COPY FINAL FC LAYER
# ============================================================

fc_weight = state_dict["fc.weight"].detach().cpu().numpy()
fc_bias = state_dict["fc.bias"].detach().cpu().numpy()

# PyTorch Linear:
# [out_features, in_features]
#
# TensorFlow Dense:
# [in_features, out_features]

fc_weight = np.transpose(fc_weight)

tf_model.get_layer("fc").set_weights([
    fc_weight,
    fc_bias
])


print("All trained weights transferred successfully.")


# ============================================================
# SAVE TENSORFLOW MODEL
# ============================================================

tf_model.save(TF_MODEL_PATH)

tf_model.save_weights(TF_WEIGHTS_PATH)

print("\nTensorFlow model saved:")
print(TF_MODEL_PATH)

print("\nTensorFlow weights saved:")
print(TF_WEIGHTS_PATH)


# ============================================================
# BASIC MODEL CHECK
# ============================================================

print("\nTensorFlow model summary:")
tf_model.summary()

print("\nCONVERSION COMPLETED SUCCESSFULLY.")