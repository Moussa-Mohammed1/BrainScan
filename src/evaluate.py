"""Evaluation helpers for trained BrainScan models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def _integer_labels(labels: np.ndarray) -> np.ndarray:
    """Convert sparse or one-hot labels to integer class IDs."""
    if labels.ndim > 1:
        return np.argmax(labels, axis=1)
    return labels.astype(int)


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a model and return named test metrics."""
    values = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    return {name: float(value) for name, value in values.items()}


def get_classification_report(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """Return a text classification report for the test set."""
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return classification_report(
        y_test,
        predictions,
        target_names=class_names,
        zero_division=0,
    )


def get_confusion_matrix(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Return the confusion matrix for model predictions."""
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return confusion_matrix(y_test, predictions)


def plot_training_history(
    history: keras.callbacks.History,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot training/validation accuracy and loss."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    history_values = history.history
    axes[0].plot(history_values["accuracy"], label="train")
    axes[0].plot(history_values["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(history_values["loss"], label="train")
    axes[1].plot(history_values["val_loss"], label="validation")
    axes[1].set_title("Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
    return figure


def plot_prediction_examples(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    max_examples: int = 10,
    output_path: Path | None = None,
) -> plt.Figure:
    """Display correct and incorrect predictions from a labeled image set."""
    if max_examples <= 0:
        raise ValueError("max_examples must be greater than zero")
    if len(images) != len(labels):
        raise ValueError("images and labels must contain the same number of items")

    true_labels = _integer_labels(labels)
    probabilities = model.predict(images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    correct_indices = np.flatnonzero(predicted_labels == true_labels)[:max_examples]
    incorrect_indices = np.flatnonzero(predicted_labels != true_labels)[:max_examples]
    selected_indices = np.concatenate((correct_indices, incorrect_indices))

    if len(selected_indices) == 0:
        raise ValueError("No predictions were available to display")

    columns = min(5, len(selected_indices))
    rows = int(np.ceil(len(selected_indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes_array = np.atleast_1d(axes).ravel()

    def label_for(class_id: int) -> str:
        if class_names is None:
            return str(class_id)
        return class_names[class_id]

    for axis, index in zip(axes_array, selected_indices):
        image = images[index]
        axis.imshow(np.clip(image, 0, 1) if image.dtype.kind == "f" else image)
        is_correct = predicted_labels[index] == true_labels[index]
        status = "Correct" if is_correct else "Incorrect"
        axis.set_title(
            f"{status}\n"
            f"True: {label_for(int(true_labels[index]))}\n"
            f"Pred: {label_for(int(predicted_labels[index]))}"
        )
        axis.axis("off")

    for axis in axes_array[len(selected_indices):]:
        axis.axis("off")

    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
    return figure