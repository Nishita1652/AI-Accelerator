#include <iostream>
#include <vector>
#include "ann_model.h"

int main() {
    std::cout << "========================================================\n";
    std::cout << "   Hardware Pathfinder ANN - Phase 1 Software Baseline  \n";
    std::cout << "========================================================\n";

    std::vector<std::vector<float>> X;
    std::vector<int> y;

    // Load real dataset from processed directory or automatically fallback to synthetic data
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
            std::cout << "Epoch [" << epoch << "/" << epochs << "] -> Accuracy: " << acc << "%\n";
        }
    }

    std::cout << "\n[SUCCESS] Phase 1 software training baseline complete!\n";
    return 0;
}

