import os
import gzip
import struct
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "mnist_data.csv")

TRAIN_IMAGES_PATH = os.path.join(RAW_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS_PATH = os.path.join(RAW_DIR, "train-labels-idx1-ubyte.gz")

NUM_SAMPLES = 5000
NUM_FEATURES = 784
NUM_CLASSES = 10

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def load_raw_mnist():
    """Unpacks raw IDX .gz binary files."""
    if not (os.path.exists(TRAIN_IMAGES_PATH) and os.path.exists(TRAIN_LABELS_PATH)):
        return None, None

    try:
        print("[INFO] Raw MNIST .gz files detected. Unpacking binary files...")
        with gzip.open(TRAIN_IMAGES_PATH, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            buffer = f.read(num_images * rows * cols)
            X = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            X = X.reshape(num_images, rows * cols) / 255.0

        with gzip.open(TRAIN_LABELS_PATH, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            buffer = f.read(num_labels)
            y = np.frombuffer(buffer, dtype=np.uint8)

        return X[:NUM_SAMPLES], y[:NUM_SAMPLES]
    except Exception as e:
        print(f"[WARNING] Failed to parse raw .gz files ({e}).")
        return None, None

def generate_synthetic_data():
    """Generates structured synthetic benchmark data if raw data is unavailable."""
    print("[WARNING] Raw .gz dataset missing or corrupt.")
    print(f"[INFO] Generating synthetic dataset: {NUM_SAMPLES} samples, {NUM_FEATURES} features...")
    
    np.random.seed(42)
    y = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES)
    X = np.random.uniform(0.0, 0.2, size=(NUM_SAMPLES, NUM_FEATURES)).astype(np.float32)
    
    # Inject class-specific deterministic features so the ANN can learn
    for i in range(NUM_SAMPLES):
        label = y[i]
        feature_start = label * (NUM_FEATURES // NUM_CLASSES)
        feature_end = feature_start + 40
        X[i, feature_start:feature_end] += np.random.uniform(0.6, 0.8, size=(feature_end - feature_start))

    X = np.clip(X, 0.0, 1.0)
    return X, y

# Execution flow
X_data, y_data = load_raw_mnist()

if X_data is None or y_data is None:
    X_data, y_data = generate_synthetic_data()

# Combine: Column 0 = Label, Columns 1 to 784 = Normalized Pixels
dataset = np.column_stack((y_data, X_data))
np.savetxt(OUTPUT_FILE, dataset, delimiter=",", fmt="%.4f")
print(f"[SUCCESS] Dataset ready at '{OUTPUT_FILE}' with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
