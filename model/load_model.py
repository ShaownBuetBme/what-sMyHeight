"""
Utility to load the trained ResNet50 regression model from a weights file.
"""

import os
import streamlit as st

from model.architecture import build_model


@st.cache_resource(show_spinner=False)
def get_model(weights_path: str):
    """Build the model architecture and load pre-trained weights.

    Args:
        weights_path: Path to the .h5 weights file.

    Returns:
        Loaded Keras model ready for inference.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError: If weights cannot be loaded into the model.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            "Please place 'fold00001_best.weights.h5' in the 'model/' directory."
        )

    model = build_model()
    try:
        model.load_weights(weights_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load weights from '{weights_path}': {exc}"
        ) from exc

    return model
