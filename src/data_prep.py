from collections import Counter

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    RANDOM_STATE,
    SUPPORTED_EXTENSIONS,
    TEST_SIZE,
)

def load_dataset():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    class_dirs = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise ValueError(f"No class directories found in {DATA_DIR}")
    images = []
    labels = []
    unreadable_images = []
    unsupported_files = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
        )
        for path in image_paths:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported_files.append(path)
                continue
            try:
                image = cv2.imread(str(path))
                if image is None:
                    unreadable_images.append(path)
                    continue

                image = cv2.resize(
                    image,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA,
                )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                images.append(image)
                labels.append(class_name)
            except (cv2.error, OSError, ValueError) as error:
                unreadable_images.append(path)
                print(f"Could not process {path}: {error}")
        
    if not images:
        raise ValueError("No readable images were loaded.")

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)

    return X, y

def prepare_dataset():

    X, y_text = load_dataset()
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(np.unique(y_text)))

    y = label_encoder.transform(y_text)
    counts = Counter(y_text)


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
    }

