// Accuracy test harness against real MNIST data — originally written
// by Pragya inside ann_inference.cpp, extracted here since the kernel
// file itself is Member 1's owned deliverable per INTERFACE_CONTRACT.md.
// Adapted to call ann_inference()'s actual signature (void, with
// output_scores[]/predicted_label as outputs, biases included in the
// accumulation) rather than the modified version that dropped both.
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include "ann_inference.h"

int main() {
    std::ifstream file("data/processed/mnist_data.csv");

    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open mnist_data.csv\n";
        return 1;
    }

    std::string line;
    int total = 0;
    int correct = 0;

    // Test first 10 images
    while (std::getline(file, line) && total < 10) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        float value;

        // First value = label
        ss >> value;
        int actual_label = static_cast<int>(value);

        // Remaining 784 values = pixels
        data_t input[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            char comma;
            ss >> comma;
            ss >> value;

            // CSV pixels are normalized 0.0 to 1.0.
            // Convert to signed INT8 range.
            int pixel = static_cast<int>(value * 127.0f);
            if (pixel > 127) pixel = 127;
            if (pixel < 0) pixel = 0;
            input[i] = static_cast<data_t>(pixel);
        }

        // Run inference — real kernel signature: void, with
        // output_scores[] and predicted_label as outputs.
        acc_t output_scores[OUTPUT_SIZE];
        int predicted_label = -1;
        ann_inference(input, output_scores, predicted_label);

        if (predicted_label == actual_label) correct++;
        total++;

        std::cout << "Image " << total
                   << ": Actual = " << actual_label
                   << ", Predicted = " << predicted_label;
        std::cout << (predicted_label == actual_label ? "  [OK]" : "  [WRONG]") << "\n";
    }

    file.close();

    std::cout << "\n========================================\n";
    std::cout << "INT8 MNIST TEST\n";
    std::cout << "========================================\n";
    std::cout << "Correct: " << correct << " / " << total << "\n";

    if (total > 0) {
        float accuracy = (static_cast<float>(correct) / total) * 100.0f;
        std::cout << "Accuracy: " << accuracy << "%\n";
    }
    std::cout << "========================================\n";

    return 0;
}
