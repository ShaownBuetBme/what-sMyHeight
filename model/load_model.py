"""
Utility to load the trained ResNet50 regression model from a weights file.
"""

import os
import threading

from model.architecture import build_model

# Module-level cache: weights_path -> loaded Keras model
_model_cache: dict = {}
_cache_lock = threading.Lock()


def get_model(weights_path: str):
    """Build the model architecture and load pre-trained weights.

    The loaded model is cached in memory so that subsequent calls with the
    same weights path return the already-loaded instance without re-reading
    the file from disk.  Access to the cache is protected by a threading lock
    so that concurrent requests do not trigger redundant weight loads.

    Args:
        weights_path: Path to the .h5 weights file.

    Returns:
        Loaded Keras model ready for inference.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError: If weights cannot be loaded into the model.
    """
    with _cache_lock:
        if weights_path in _model_cache:
            return _model_cache[weights_path]

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

        _model_cache[weights_path] = model
        return model
