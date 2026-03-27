"""
Custom ResNet50-based regression model for height prediction.

The architecture replicates the training setup:
- ResNet50 backbone (ImageNet weights, no top)
- Global Average Pooling
- Dense(512) + ReLU + Dropout(0.3)
- Dense(128) + ReLU + Dropout(0.2)
- Dense(1) linear output (regression)
Input shape: (256, 256, 3)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(256, 256, 3)):
    """Build and return the custom ResNet50 regression model.

    Args:
        input_shape: Tuple of (height, width, channels). Default (256, 256, 3).

    Returns:
        A compiled Keras Model with a single float regression output.
    """
    # weights=None because all weights (backbone + head) are loaded from the
    # .h5 weights file via model.load_weights().  Downloading ImageNet weights
    # here would be immediately overwritten and wastes time/bandwidth.
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="linear", name="output_height")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="height_regression_resnet50")
    return model
