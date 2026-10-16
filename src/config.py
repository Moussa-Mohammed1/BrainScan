from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RANDOM_STATE = 42
TEST_SIZE = 0.20

IMG_CHANNELS = 3
NUM_CLASSES = 4
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 32
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_PLOT_PATH = PROJECT_ROOT / "models" / "02_cnn_architecture.png"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "brainscan_best.keras"
DEPLOYED_MODEL_PATH = PROJECT_ROOT / "models" / "brainscan_model.h5"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
