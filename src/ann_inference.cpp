#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include "weights.h"

// =========================================================
// ReLU
// =========================================================
static int32_t relu(int32_t x)
{
    return (x > 0) ? x : 0;
}


// =========================================================
// ANN INFERENCE
// =========================================================
int ann_inference(const int8_t input[INPUT_SIZE])
{
    int32_t hidden[HIDDEN_SIZE];

    // -----------------------------------------------------
    // Layer 1: 784 -> 32
    // -----------------------------------------------------
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
        int32_t acc = 0;

        for (int i = 0; i < INPUT_SIZE; ++i)
        {
            int index = i * HIDDEN_SIZE + j;

            acc += static_cast<int32_t>(input[i]) *
                   static_cast<int32_t>(W1_QUANT[index]);
        }

        hidden[j] = relu(acc);
    }


    // -----------------------------------------------------
    // Layer 2: 32 -> 10
    // -----------------------------------------------------
    int32_t output[OUTPUT_SIZE];

    for (int k = 0; k < OUTPUT_SIZE; ++k)
    {
        int32_t acc = 0;

        for (int j = 0; j < HIDDEN_SIZE; ++j)
        {
            int index = j * OUTPUT_SIZE + k;

            acc += hidden[j] *
                   static_cast<int32_t>(W2_QUANT[index]);
        }

        output[k] = acc;
    }


    // -----------------------------------------------------
    // Argmax
    // -----------------------------------------------------
    int predicted_class = 0;
    int32_t max_value = output[0];

    for (int k = 1; k < OUTPUT_SIZE; ++k)
    {
        if (output[k] > max_value)
        {
            max_value = output[k];
            predicted_class = k;
        }
    }

    return predicted_class;
}


// =========================================================
// MAIN
// Read MNIST CSV and test 10 images
// =========================================================
int main()
{
    std::ifstream file("data/processed/mnist_data.csv");

    if (!file.is_open())
    {
        std::cerr << "[ERROR] Could not open mnist_data.csv\n";
        return 1;
    }

    std::string line;

    int total = 0;
    int correct = 0;

    // -----------------------------------------------------
    // Test first 10 images
    // -----------------------------------------------------
    while (std::getline(file, line) && total < 10)
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);

        float value;

        // -------------------------------------------------
        // First value = label
        // -------------------------------------------------
        ss >> value;

        int actual_label = static_cast<int>(value);

        // -------------------------------------------------
        // Remaining 784 values = pixels
        // -------------------------------------------------
        int8_t input[INPUT_SIZE];

        for (int i = 0; i < INPUT_SIZE; ++i)
        {
            char comma;

            ss >> comma;
            ss >> value;

            // CSV pixels are normalized 0.0 to 1.0.
            // Convert to signed INT8 range.
            int pixel = static_cast<int>(value * 127.0f);

            if (pixel > 127)
                pixel = 127;

            if (pixel < 0)
                pixel = 0;

            input[i] = static_cast<int8_t>(pixel);
        }

        // -------------------------------------------------
        // Run inference
        // -------------------------------------------------
        int prediction = ann_inference(input);

        if (prediction == actual_label)
            correct++;

        total++;

        std::cout
            << "Image " << total
            << ": Actual = " << actual_label
            << ", Predicted = " << prediction;

        if (prediction == actual_label)
            std::cout << "  [OK]";
        else
            std::cout << "  [WRONG]";

        std::cout << "\n";
    }

    file.close();

    // -----------------------------------------------------
    // Accuracy
    // -----------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "INT8 MNIST TEST\n";
    std::cout << "========================================\n";

    std::cout << "Correct: "
              << correct
              << " / "
              << total
              << "\n";

    if (total > 0)
    {
        float accuracy =
            (static_cast<float>(correct) / total) * 100.0f;

        std::cout << "Accuracy: "
                  << accuracy
                  << "%\n";
    }

    std::cout << "========================================\n";

    return 0;
}