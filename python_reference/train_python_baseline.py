"""
train_python_baseline.py — Item #1 for your professor: the Python-trained
ANN accuracy, as a genuine baseline against the C++ (#2) and HLS/INT8 (#3)
numbers.

This is a DELIBERATE, faithful numpy port of ann_model.cpp's exact
architecture and training procedure — same 784->32->10 shape, same ReLU,
same softmax+cross-entropy, same plain-SGD update rule, same learning
rate, same He-style initialization — NOT a from-scratch "better" model
using Adam/PyTorch conveniences. The point of this comparison is
isolating "same model, different language" — introducing a different
optimizer or framework here would confound the comparison your professor
is actually asking for.

Usage:
    python3 train_python_baseline.py

Expects data/processed/mnist_data.csv (training) and
data/processed/mnist_test.csv (held-out test) to already exist —
run scripts/prepare_mnist_gz.py first if they don't.
"""
import numpy as np

INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
LEARNING_RATE = 0.005
EPOCHS = 25

TRAIN_CSV = "data/processed/mnist_data.csv"
TEST_CSV = "data/processed/mnist_test.csv"


def load_csv(path):
    labels, pixels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            labels.append(int(float(parts[0])))
            pixels.append([float(p) for p in parts[1:INPUT_SIZE + 1]])
    return np.array(pixels, dtype=np.float32), np.array(labels, dtype=np.int64)


def relu(x):
    return np.maximum(0.0, x)


def relu_deriv(x):
    return (x > 0).astype(np.float32)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


class LightweightANN:
    """Mirrors ann_model.cpp's LightweightANN class exactly."""

    def __init__(self, seed=1337):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / INPUT_SIZE), (INPUT_SIZE, HIDDEN_SIZE)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / HIDDEN_SIZE), (HIDDEN_SIZE, OUTPUT_SIZE)).astype(np.float32)
        self.b2 = np.zeros(OUTPUT_SIZE, dtype=np.float32)

    def forward(self, x):
        hidden_raw = x @ self.W1 + self.b1
        hidden = relu(hidden_raw)
        output_raw = hidden @ self.W2 + self.b2
        probs = softmax(output_raw)
        return hidden, probs

    def train_sample(self, x, target_label):
        x = x.reshape(1, -1)
        hidden, probs = self.forward(x)

        d_output = probs.copy()
        d_output[0, target_label] -= 1.0

        d_hidden = (d_output @ self.W2.T) * relu_deriv(hidden)

        self.W2 -= LEARNING_RATE * (hidden.T @ d_output)
        self.b2 -= LEARNING_RATE * d_output.flatten()
        self.W1 -= LEARNING_RATE * (x.T @ d_hidden)
        self.b1 -= LEARNING_RATE * d_hidden.flatten()

    def predict(self, x):
        _, probs = self.forward(x.reshape(1, -1))
        return int(np.argmax(probs))

    def predict_batch(self, X):
        _, probs = self.forward(X)
        return np.argmax(probs, axis=1)


def main():
    print("=" * 56)
    print("  Python Reference ANN — faithful port of ann_model.cpp")
    print("=" * 56)

    X_train, y_train = load_csv(TRAIN_CSV)
    X_test, y_test = load_csv(TEST_CSV)
    print(f"[INFO] Loaded {len(y_train)} training samples, {len(y_test)} held-out test samples.")

    model = LightweightANN()

    print(f"Starting Python training loop ({EPOCHS} epochs)...")
    for epoch in range(1, EPOCHS + 1):
        for i in range(len(X_train)):
            model.train_sample(X_train[i], y_train[i])

        if epoch % 5 == 0 or epoch == 1:
            train_preds = model.predict_batch(X_train)
            train_acc = (train_preds == y_train).mean() * 100.0
            print(f"Epoch [{epoch}/{EPOCHS}] -> Training Accuracy: {train_acc:.2f}%")

    # This is the number that matters — held-out, not training accuracy.
    test_preds = model.predict_batch(X_test)
    test_acc = (test_preds == y_test).mean() * 100.0

    print("\n" + "=" * 56)
    print(f"[RESULT] Python (FP32) held-out test accuracy: {test_acc:.2f}%")
    print(f"         ({len(y_test)} samples, {TEST_CSV})")
    print("=" * 56)


if __name__ == "__main__":
    main()
