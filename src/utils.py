"""Utilities shared by training and inference code."""

from pathlib import Path

import numpy as np

from src.config import PROCESSED_DATA_DIR


def load_processed_data(
    data_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, np.ndarray]:
    """Load the arrays produced by the exploration/data-preparation notebook."""
    required_files = ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")
    missing_files = [name for name in required_files if not (data_dir / name).is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Processed dataset is missing from {data_dir}: {missing}. "
            "Run the data preparation step first."
        )

    return {
        "X_train": np.load(data_dir / "X_train.npy"),
        "X_test": np.load(data_dir / "X_test.npy"),
        "y_train": np.load(data_dir / "y_train.npy"),
        "y_test": np.load(data_dir / "y_test.npy"),
    }