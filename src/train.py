"""Model training helpers."""

import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import BATCH_SIZE, BEST_MODEL_PATH, EPOCHS, LABEL_ENCODER_PATH
from src.model import build_cnn, compile_cnn
from src.utils import load_processed_data


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calculate balanced class weights for integer encoded labels."""
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }


def create_checkpoint_callback(
    model_path: Path = BEST_MODEL_PATH,
) -> keras.callbacks.ModelCheckpoint:
    """Create a callback that keeps the model with the lowest validation loss."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )


def save_label_encoder(
    label_encoder: object,
    output_path: Path = LABEL_ENCODER_PATH,
) -> Path:
    """Persist the label encoder used to map model outputs to class names."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_path)
    return output_path


def train_model(
    model: keras.Model | None = None,
    data: dict[str, np.ndarray] | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    checkpoint_path: Path = BEST_MODEL_PATH,
) -> tuple[keras.Model, keras.callbacks.History, float]:
    """Train the CNN and return it, its history, and duration in seconds."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be greater than zero")

    arrays = load_processed_data() if data is None else data
    required_keys = {"X_train", "y_train"}
    missing_keys = required_keys - arrays.keys()
    if missing_keys:
        raise ValueError(f"Training data is missing keys: {sorted(missing_keys)}")

    if model is None:
        model = compile_cnn(build_cnn())
    elif getattr(model, "optimizer", None) is None:
        compile_cnn(model)

    start_time = time.perf_counter()
    history = model.fit(
        arrays["X_train"],
        arrays["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=calculate_class_weights(arrays["y_train"]),
        callbacks=[create_checkpoint_callback(checkpoint_path)],
        verbose=1,
    )
    return model, history, time.perf_counter() - start_time


def main() -> None:
    """Train the default model from the processed dataset."""
    _, _, duration = train_model()
    print(f"Training duration: {duration / 60:.2f} minutes")


if __name__ == "__main__":
    main()