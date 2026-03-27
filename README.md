# What's My Height? 📏

> AI-based anthropometric measurement using deep learning — height regression from human images.

A production-ready **Streamlit** web application that estimates a person's height (in centimetres) from a front-view photograph using a custom **ResNet50**-based regression model trained with TensorFlow / Keras.

---

## Project Overview

The model takes a front-view image of a person (ideally with a checkerboard pattern in the background) and outputs a single continuous value representing the predicted height in centimetres.

| Detail | Value |
|--------|-------|
| Model backbone | ResNet50 |
| Task | Regression (single float output) |
| Input size | 256 × 256 × 3 |
| Output | Height in **cm** |
| Framework | TensorFlow / Keras |
| Weights file | `fold00001_best.weights.h5` (~262 MB) |

---

## Repository Structure

```
.
├── app.py                   # Streamlit application entry point
├── model/
│   ├── __init__.py
│   ├── architecture.py      # Custom ResNet50 regression model definition
│   └── load_model.py        # Weight loading with st.cache_resource
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py     # Image resize + ResNet50 normalisation
│   ├── inference.py         # Model.predict wrapper
│   └── history.py           # Session-state prediction history
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ShaownBuetBme/what-sMyHeight.git
cd what-sMyHeight
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the model weights file

Download `fold00001_best.weights.h5` (~262 MB) and place it inside the `model/` directory:

```
model/
└── fold00001_best.weights.h5
```

> **Note:** The weights file is excluded from version control via `.gitignore` because of its large size. You must obtain it separately.

---

## How to Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## App Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Upload an image and get an instant height prediction |
| 📋 **History** | View all predictions made in the current session |
| ℹ️ **About** | Project description and model details |

---

## Performance Notes

- The model runs entirely on **CPU** — no GPU required.
- The model is loaded once and cached via `@st.cache_resource` to avoid repeated disk I/O.

---

## Author

**ShaownBuetBme** — BUET BME
