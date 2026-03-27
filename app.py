"""
What's My Height? — Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from PIL import Image

from model.load_model import get_model
from utils.preprocessing import preprocess_image
from utils.inference import predict
from utils import history as hist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHTS_PATH = os.path.join("model", "fold00001_best.weights.h5")

st.set_page_config(
    page_title="What's My Height?",
    page_icon="📏",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📏 What's My Height?")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📋 History", "ℹ️ About"])

# ---------------------------------------------------------------------------
# Page: Home — prediction
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📏 Height Prediction")
    st.write(
        "Upload a front-view image of a person (with a checkerboard pattern "
        "in the background for best results) and the model will estimate their height."
    )

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("❌ Could not open the uploaded file. Please upload a valid JPG or PNG image.")
            st.stop()

        st.image(image, caption=uploaded_file.name, use_column_width=True)

        if st.button("🔍 Predict Height"):
            # Load model (cached after first call)
            model_loaded = True
            model = None
            try:
                with st.spinner("Loading model…"):
                    model = get_model(WEIGHTS_PATH)
            except FileNotFoundError as exc:
                st.error(f"❌ {exc}")
                model_loaded = False
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
                model_loaded = False

            if model_loaded and model is not None:
                try:
                    with st.spinner("Running inference…"):
                        img_tensor = preprocess_image(image)
                        height_cm = predict(model, img_tensor)

                    st.success(f"**Predicted Height: {height_cm:.1f} cm**")
                    hist.add_record(uploaded_file.name, height_cm)
                except ValueError as exc:
                    st.error(f"❌ Image preprocessing error: {exc}")
                except RuntimeError as exc:
                    st.error(f"❌ Prediction error: {exc}")

# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------
elif page == "📋 History":
    st.title("📋 Prediction History")

    records = hist.get_history()
    if records:
        st.dataframe(records, use_container_width=True)
        if st.button("🗑️ Clear History"):
            hist.clear_history()
            st.rerun()
    else:
        st.info("No predictions yet. Go to the **Home** page to get started.")

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown(
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
