"""CNN architecture and compilation helpers."""

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from src.config import (
    IMG_CHANNELS,
    IMAGE_SIZE,
    LEARNING_RATE,
    DEPLOYED_MODEL_PATH,
    MODEL_PLOT_PATH,
    NUM_CLASSES,
)


def build_cnn(
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, IMG_CHANNELS),
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Build the convolutional classifier used by the project."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="brainscan_cnn",
    )


def compile_cnn(
    model: keras.Model,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """Compile a CNN with the sparse-label loss used by data preparation."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model_plot(
    model: keras.Model,
    output_path: Path = MODEL_PLOT_PATH,
) -> Path:
    """Save a visual representation of the model architecture."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_model(
        model,
        to_file=str(output_path),
        show_shapes=True,
        show_layer_names=True,
        show_dtype=True,
        dpi=120,
    )
    return output_path


def save_trained_model(
    model: keras.Model,
    output_path: Path = DEPLOYED_MODEL_PATH,
) -> Path:
    """Save a trained Keras model in HDF5 format for deployment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return output_path