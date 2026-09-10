#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>

// Pull in the shared hardware types and function prototype
#include "src/ann_inference.h"

// Number of test vectors for C/RTL Co-Simulation
// (Keep between 50 and 200 to prevent multi-hour RTL simulation times)
constexpr int COSIM_SAMPLES = 100;

// Minimum accuracy threshold required to pass the HLS verification gate
constexpr float ACCURACY_THRESHOLD = 80.0f;

int main() {
    // 1. Locate dataset relative to the run directory
    // Reads mnist_test.csv (the real MNIST t10k-* held-out test set),
    // NOT mnist_data.csv (the training set) — testing against training
    // data would inflate this gate's accuracy number and defeat its
    // whole purpose as a real verification check.
    std::string csv_path = "../data/processed/mnist_test.csv";
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        csv_path = "data/processed/mnist_test.csv";
        file.open(csv_path);
    }

    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not locate mnist_test.csv!" << std::endl;
        return 1; // Non-zero return halts HLS flow on missing data
    }

    std::cout << "[HLS TB] Feeding vectors from: " << csv_path << std::endl;

    // Static buffers matching the hardware interface contracts
    data_t hw_input_buffer[INPUT_SIZE];     // int8_t[784]
    acc_t  hw_output_scores[OUTPUT_SIZE];   // int32_t[10] raw logits
    int    hw_predicted_label = 0;          // Output from hardware comparator tree

    std::string line;
    int samples_processed = 0;
    int correct_predictions = 0;

    // 2. Stream test vectors
    while (std::getline(file, line) && samples_processed < COSIM_SAMPLES) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        // Extract true label (first column)
        if (!std::getline(ss, token, ',')) continue;
        int true_label = std::stoi(token);

        // Parse and scale pixels: CSV stores normalized floats [0.0, 1.0]
        // (from prepare_mnist_gz.py), NOT raw [0,255] integers — must
        // parse as float, not int, or virtually all pixel information
        // is silently discarded (std::stoi("0.5692") truncates to 0).
        int pixel_count = 0;
        while (std::getline(ss, token, ',') && pixel_count < INPUT_SIZE) {
            float pixel_norm = std::stof(token);
            int scaled = static_cast<int>(pixel_norm * 127.0f);
            if (scaled > 127) scaled = 127;
            if (scaled < 0) scaled = 0;
            hw_input_buffer[pixel_count++] = static_cast<data_t>(scaled);
        }

        if (pixel_count != INPUT_SIZE) {
            continue; // Skip malformed rows
        }

        // 3. Drive hardware accelerator (Nishita's synthesized kernel)
        ann_inference(hw_input_buffer, hw_output_scores, hw_predicted_label);

        // TEMPORARY diagnostic — remove once the 7%-vs-96% discrepancy
        // is resolved. Prints the first 10 predictions so they can be
        // compared directly against accuracy_test.cpp's known-good
        // output on the same file.
        if (samples_processed < 10) {
            std::cout << "  [DEBUG] Row " << samples_processed
                      << ": true=" << true_label
                      << " predicted=" << hw_predicted_label
                      << " first_pixel=" << (int)hw_input_buffer[0]
                      << " logit0=" << hw_output_scores[0] << "\n";
        }

        // 4. Validate output response using hardware argmax
        if (hw_predicted_label == true_label) {
            correct_predictions++;
        }

        samples_processed++;
    }

    file.close();

    // 5. Generate validation report
    float accuracy = (samples_processed > 0)
        ? (static_cast<float>(correct_predictions) / samples_processed) * 100.0f
        : 0.0f;

    std::cout << "\n================ HLS CO-SIMULATION REPORT ================\n";
    std::cout << "Target Accelerator:  INT8 Feedforward ANN\n";
    std::cout << "Vectors Evaluated:   " << samples_processed << "\n";
    std::cout << "Hardware Matches:    " << correct_predictions << "\n";
    std::cout << "Observed Accuracy:   " << accuracy << "%\n";
    std::cout << "Accuracy Threshold:  " << ACCURACY_THRESHOLD << "%\n";
    std::cout << "==========================================================\n";

    if (accuracy < ACCURACY_THRESHOLD) {
        std::cerr << "[FAIL] Observed accuracy below target threshold.\n";
        return 1;
    }

    std::cout << "[PASS] Hardware validation succeeded!\n";
    return 0;
}