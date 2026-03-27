"""
Inference helper for the height prediction model.
"""

import numpy as np


def predict(model, preprocessed_image: np.ndarray) -> float:
    """Run height prediction on a preprocessed image.

    Args:
        model: A loaded Keras model with a single float output.
        preprocessed_image: Float32 numpy array of shape (1, 256, 256, 3).

    Returns:
        Predicted height in centimetres as a Python float.

    Raises:
        RuntimeError: If the model inference fails.
    """
    try:
        prediction = model.predict(preprocessed_image, verbose=0)
        return float(np.squeeze(prediction))
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {exc}") from exc
