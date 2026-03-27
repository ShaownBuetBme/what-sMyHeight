"""
Image preprocessing pipeline matching the training configuration.

Preprocessing steps:
  1. Open image and convert to RGB.
  2. Resize to (256, 256).
  3. Apply ResNet50 preprocess_input (scales pixels to [-1, 1] range using
     caffe-style mean subtraction / channel-wise normalization).
  4. Expand dims to produce a (1, 256, 256, 3) batch tensor.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

TARGET_SIZE = (256, 256)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess a PIL image for model inference.

    Args:
        image: A PIL.Image.Image object (any mode).

    Returns:
        A float32 numpy array of shape (1, 256, 256, 3) ready for inference.

    Raises:
        ValueError: If the image cannot be processed.
    """
    if image is None:
        raise ValueError("Received None instead of a PIL Image.")

    img = image.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
