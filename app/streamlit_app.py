"""Streamlit interface for BrainScan image classification."""

from pathlib import Path
import tempfile

import joblib
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

from src.config import DEPLOYED_MODEL_PATH, LABEL_ENCODER_PATH
from src.predict import predict_image


@st.cache_resource
def load_artifacts():
    """Load the saved model and label encoder once per Streamlit process."""
    if not DEPLOYED_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {DEPLOYED_MODEL_PATH}. "
            "Train and save the model first."
        )
    if not LABEL_ENCODER_PATH.is_file():
        raise FileNotFoundError(
            f"Label encoder not found at {LABEL_ENCODER_PATH}. "
            "Save it after preparing the dataset."
        )
    return load_model(DEPLOYED_MODEL_PATH), joblib.load(LABEL_ENCODER_PATH)


st.set_page_config(page_title="BrainScan", page_icon="🧠")
st.title("BrainScan - Brain MRI Classifier")
st.write("Upload an MRI image to obtain the model prediction.")

try:
    model, label_encoder = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width="stretch")

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        temporary_file.flush()
        label, probabilities = predict_image(
            temporary_file.name,
            model,
            label_encoder,
        )

    st.success(f"Predicted class: {label}")
    st.subheader("Class probabilities")
    st.bar_chart(
        {
            str(class_name): float(probability)
            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities,
            )
        }
    )
