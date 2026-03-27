"""
What's My Height? — Gradio application entry point.

Run with:
    python app.py
"""

import os

import gradio as gr
from PIL import Image

from model.load_model import get_model
from utils.preprocessing import preprocess_image
from utils.inference import predict
from utils import history as hist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHTS_PATH = os.path.join("model", "fold00001_best.weights.h5")


# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------
def predict_height(image: Image.Image) -> str:
    """Run height prediction on an uploaded PIL image.

    Args:
        image: PIL Image provided by the Gradio Image component.

    Returns:
        A markdown string with the result or an error message.
    """
    if image is None:
        return "⚠️ Please upload an image first."

    try:
        model = get_model(WEIGHTS_PATH)
    except FileNotFoundError as exc:
        return f"❌ {exc}"
    except RuntimeError as exc:
        return f"❌ {exc}"

    try:
        img_tensor = preprocess_image(image)
        height_cm = predict(model, img_tensor)
        hist.add_record("uploaded_image", height_cm)
        return f"✅ **Predicted Height: {height_cm:.1f} cm**"
    except ValueError as exc:
        return f"❌ Image preprocessing error: {exc}"
    except RuntimeError as exc:
        return f"❌ Prediction error: {exc}"


def get_history_rows() -> list:
    """Return the current prediction history as row-wise values."""
    records = hist.get_history()
    return [
        [
            record["Image"],
            record["Predicted Height (cm)"],
            record["Timestamp"],
        ]
        for record in records
    ]


def clear_and_refresh() -> list:
    """Clear all history records and return empty table rows."""
    hist.clear_history()
    return get_history_rows()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="What's My Height? 📏") as demo:
    gr.Markdown("# 📏 What's My Height?")
    gr.Markdown(
        "AI-based height estimation from a front-view photograph using a "
        "ResNet50 regression model."
    )

    with gr.Tabs():
        # ── Home tab ──────────────────────────────────────────────────────
        with gr.Tab("🏠 Home"):
            gr.Markdown(
                "Upload a front-view image of a person (with a checkerboard "
                "pattern in the background for best results) and click "
                "**Predict Height**."
            )
            image_input = gr.Image(label="Upload Image", type="pil")
            predict_btn = gr.Button("🔍 Predict Height", variant="primary")
            result_output = gr.Markdown()

            predict_btn.click(
                fn=predict_height,
                inputs=image_input,
                outputs=result_output,
            )

        # ── History tab ───────────────────────────────────────────────────
        with gr.Tab("📋 History"):
            gr.Markdown("Predictions made during the current server session.")
            history_table = gr.DataFrame(
                headers=["Image", "Predicted Height (cm)", "Timestamp"],
                value=get_history_rows,
                label="Prediction History",
                interactive=False,
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh")
                clear_btn = gr.Button("🗑️ Clear History", variant="stop")

            refresh_btn.click(fn=get_history_rows, outputs=history_table)
            clear_btn.click(fn=clear_and_refresh, outputs=history_table)

        # ── About tab ─────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
## AI-based Anthropometric Measurement Using Deep Learning

This application estimates a person's height (in centimetres) from a
front-view photograph using a custom deep-learning regression model.

### Model Architecture
| Component | Details |
|-----------|---------|
| Backbone | **ResNet50** (pre-trained on ImageNet) |
| Head | GlobalAveragePooling → Dense(512, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(1, linear) |
| Input size | 256 × 256 × 3 |
| Output | Single float — predicted height in **cm** |
| Framework | TensorFlow / Keras |

### How It Works
1. Upload a front-view image.
2. The image is resized to 256 × 256 and preprocessed using ResNet50's
   standard normalisation.
3. The model produces a single continuous value representing the
   predicted height.

### Author
**ShaownBuetBme** — BUET BME  
*Project: What's My Height? — height regression from human images*

---
> **Tip:** For best accuracy, photograph the subject standing upright in
> front of a flat surface with a checkerboard pattern visible in the
> background.
                """
            )

if __name__ == "__main__":
    server_port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=server_port)
