"""
In-memory history management for prediction records.
"""

import threading
from datetime import datetime

# Module-level list; persists for the lifetime of the server process.
_history: list = []
_history_lock = threading.Lock()


def add_record(image_name: str, prediction_cm: float) -> None:
    """Append a new prediction record to the in-memory history.

    Args:
        image_name: Original filename of the uploaded image.
        prediction_cm: Predicted height in centimetres.
    """
    with _history_lock:
        _history.append(
            {
                "Image": image_name,
                "Predicted Height (cm)": round(prediction_cm, 2),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


def get_history() -> list:
    """Return a thread-safe copy of the list of prediction records.

    Returns:
        A list of dicts with keys 'Image', 'Predicted Height (cm)', 'Timestamp'.
    """
    with _history_lock:
        return list(_history)


def clear_history() -> None:
    """Remove all prediction records from memory."""
    with _history_lock:
        _history.clear()
