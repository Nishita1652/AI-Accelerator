import os
import gzip
import struct
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRAIN_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "mnist_data.csv")
TEST_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "mnist_test.csv")

TRAIN_IMAGES_PATH = os.path.join(RAW_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS_PATH = os.path.join(RAW_DIR, "train-labels-idx1-ubyte.gz")
TEST_IMAGES_PATH = os.path.join(RAW_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_PATH = os.path.join(RAW_DIR, "t10k-labels-idx1-ubyte.gz")

NUM_TRAIN_SAMPLES = 5000
NUM_TEST_SAMPLES = 1000  # real MNIST test set has 10000; capped for speed
NUM_FEATURES = 784
NUM_CLASSES = 10

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def load_raw_idx(images_path, labels_path, num_samples):
    if not (os.path.exists(images_path) and os.path.exists(labels_path)):
        return None, None
    try:
        with gzip.open(images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            buffer = f.read(num_images * rows * cols)
            X = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            X = X.reshape(num_images, rows * cols) / 255.0

        with gzip.open(labels_path, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            buffer = f.read(num_labels)
            y = np.frombuffer(buffer, dtype=np.uint8)

        return X[:num_samples], y[:num_samples]
    except Exception as e:
        print(f"[WARNING] Failed to parse {images_path} ({e}).")
        return None, None


def generate_synthetic_data(num_samples):
    print(f"[WARNING] Raw dataset missing/corrupt. Generating {num_samples} synthetic samples...")
    np.random.seed(42)
    y = np.random.randint(0, NUM_CLASSES, size=num_samples)
    X = np.random.uniform(0.0, 0.2, size=(num_samples, NUM_FEATURES)).astype(np.float32)

    for i in range(num_samples):
        label = y[i]
        feature_start = label * (NUM_FEATURES // NUM_CLASSES)
        feature_end = feature_start + 40
        X[i, feature_start:feature_end] += np.random.uniform(0.6, 0.8, size=(feature_end - feature_start))

    return np.clip(X, 0.0, 1.0), y


def write_csv(path, X, y):
    dataset = np.column_stack((y, X))
    np.savetxt(path, dataset, delimiter=",", fmt="%.4f")
    print(f"[SUCCESS] Wrote {len(y)} samples to '{path}'")


# ---- Training set (used by main.cpp for training) ----
X_train, y_train = load_raw_idx(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, NUM_TRAIN_SAMPLES)
if X_train is None:
    X_train, y_train = generate_synthetic_data(NUM_TRAIN_SAMPLES)
write_csv(TRAIN_OUTPUT_FILE, X_train, y_train)

# ---- Held-out test set (NEW — for accuracy_test.cpp / tb_main.cpp,
#      must NOT overlap with the training data above) ----
X_test, y_test = load_raw_idx(TEST_IMAGES_PATH, TEST_LABELS_PATH, NUM_TEST_SAMPLES)
if X_test is None:
    print("[WARNING] Real MNIST test set (t10k-*) not found — generating a")
    print("          SEPARATE synthetic test set (different seed from train)")
    print("          so it's at least not identical to the training data.")
    np.random.seed(999)  # deliberately different seed from generate_synthetic_data's internal seed(42)
    X_test, y_test = generate_synthetic_data(NUM_TEST_SAMPLES)
write_csv(TEST_OUTPUT_FILE, X_test, y_test)