#include <iostream>
#include <vector>
#include "ann_model.h"

int main() {
    std::cout << "========================================================\n";
    std::cout << "   Hardware Pathfinder ANN - Phase 1 Software Baseline  \n";
    std::cout << "========================================================\n";

    std::vector<std::vector<float>> X;
    std::vector<int> y;

    const std::string dataset_path = "data/processed/mnist_data.csv";
    load_or_fallback_data(dataset_path, X, y);

    LightweightANN model;

    int epochs = 25;
    std::cout << "Starting C++ training loop (" << epochs << " epochs)...\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (size_t s = 0; s < X.size(); ++s) {
            model.train_sample(X[s], y[s]);
        }

        if (epoch % 5 == 0 || epoch == 1) {
            int correct = 0;
            for (size_t s = 0; s < X.size(); ++s) {
                if (model.predict(X[s]) == y[s]) correct++;
            }
            float acc = (float)correct / X.size() * 100.0f;
            std::cout << "Epoch [" << epoch << "/" << epochs << "] -> Training Accuracy: " << acc << "%\n";
        }
    }

    std::cout << "\n[SUCCESS] Phase 1 software training baseline complete!\n";

    // ============================================================
    // Item #2 for your professor: TRUE held-out FP32 C++ accuracy.
    // Deliberately separate from ann_inference() (the INT8 kernel) —
    // this uses the model's own predict(), still floating-point,
    // no quantization involved. Loads mnist_test.csv independently
    // from the training data above.
    // ============================================================
    std::vector<std::vector<float>> X_test;
    std::vector<int> y_test;
    const std::string test_path = "data/processed/mnist_test.csv";
    load_or_fallback_data(test_path, X_test, y_test);

    int test_correct = 0;
    for (size_t s = 0; s < X_test.size(); ++s) {
        if (model.predict(X_test[s]) == y_test[s]) test_correct++;
    }
    float test_acc = X_test.empty() ? 0.0f : (float)test_correct / X_test.size() * 100.0f;

    std::cout << "\n========================================================\n";
    std::cout << "[RESULT] C++ (FP32) held-out test accuracy: " << test_acc << "%\n";
    std::cout << "         (" << X_test.size() << " samples, " << test_path << ")\n";
    std::cout << "========================================================\n";

    // Export trained FP32 weights so quantize_weights.cpp can pick them up.
    model.export_for_quantization("float_weights.txt");

    return 0;
}
