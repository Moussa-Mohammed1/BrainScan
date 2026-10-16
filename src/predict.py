"""Single-image inference helpers."""

from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

from src.config import IMAGE_SIZE


def predict_image(
    image_path: str | Path,
    model: keras.Model,
    label_encoder: LabelEncoder,
) -> tuple[str, np.ndarray]:
    """Predict the class of one image and return its label and probabilities."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    label = str(label_encoder.inverse_transform([class_index])[0])
    return label, probabilities