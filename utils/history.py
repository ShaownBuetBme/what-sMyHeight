"""
Session-state history management for prediction records.
"""

from datetime import datetime
import streamlit as st

HISTORY_KEY = "prediction_history"


def _ensure_history():
    """Initialise the history list in session state if absent."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []


def add_record(image_name: str, prediction_cm: float) -> None:
    """Append a new prediction record to the session history.

    Args:
        image_name: Original filename of the uploaded image.
        prediction_cm: Predicted height in centimetres.
    """
    _ensure_history()
    st.session_state[HISTORY_KEY].append(
        {
            "Image": image_name,
            "Predicted Height (cm)": round(prediction_cm, 2),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def get_history() -> list:
    """Return the list of prediction records stored in the current session.

    Returns:
        A list of dicts with keys 'Image', 'Predicted Height (cm)', 'Timestamp'.
    """
    _ensure_history()
    return st.session_state[HISTORY_KEY]


def clear_history() -> None:
    """Remove all prediction records from the current session."""
    st.session_state[HISTORY_KEY] = []
