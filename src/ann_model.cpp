#include "ann_model.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>

// Hardware Activation Functions
float relu(float x) {
    return std::max(0.0f, x);
}

float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] /= sum;
    }
    return result;
}

// Synthetic Dataset Fallback Generator (Upgraded with Gaussian Noise)
void generate_synthetic_data(std::vector<std::vector<float>>& X, std::vector<int>& y, int num_samples) {
    std::cout << "[WARN] CSV unreadable. Generating " << num_samples << " highly-randomized synthetic samples...\n";
    
    std::default_random_engine rand_eng(42); // Fixed seed for reproducibility
    
    // Gaussian distribution for background noise (mean 0.05, std_dev 0.05)
    std::normal_distribution<float> background_noise(0.05f, 0.05f); 
    
    // Gaussian distribution for the "digit strokes" (mean 0.7, std_dev 0.15)
    std::normal_distribution<float> feature_intensity(0.7f, 0.15f);
    
    std::uniform_int_distribution<int> label_dist(0, OUTPUT_SIZE - 1);

    X.assign(num_samples, std::vector<float>(INPUT_SIZE, 0.0f));
    y.assign(num_samples, 0);

    for (int s = 0; s < num_samples; ++s) {
        y[s] = label_dist(rand_eng);
        
        // Define a unique "stroke cluster" for each class to mimic structural data
        // For example, class 0 gets pixels 0-78, class 1 gets 78-156, etc.
        int cluster_size = INPUT_SIZE / OUTPUT_SIZE;
        int cluster_start = y[s] * cluster_size;
        
        // Add some random shift to the cluster size to make it imperfect (like handwriting)
        std::uniform_int_distribution<int> length_variance(-15, 15);
        int cluster_end = cluster_start + 45 + length_variance(rand_eng);

        for (int i = 0; i < INPUT_SIZE; ++i) {
            if (i >= cluster_start && i < cluster_end) {
                // Inject structural "digit" data with high intensity and noise
                float pixel_val = feature_intensity(rand_eng);
                X[s][i] = std::clamp(pixel_val, 0.0f, 1.0f);
            } else {
                // Inject random background noise across the rest of the image
                float noise_val = background_noise(rand_eng);
                X[s][i] = std::clamp(noise_val, 0.0f, 1.0f);
            }
        }
    }
    std::cout << "[INFO] Realistic noisy synthetic dataset created successfully.\n";
}

// Unified Data Loader with Fallback
void load_or_fallback_data(const std::string& filename, std::vector<std::vector<float>>& X, std::vector<int>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        generate_synthetic_data(X, y);
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        
        if (std::getline(ss, val, ',')) {
            y.push_back(std::stoi(val));
        }

        std::vector<float> row;
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stof(val));
        }
        if (row.size() == INPUT_SIZE) {
            X.push_back(row);
        }
    }

    if (X.empty()) {
        generate_synthetic_data(X, y);
    } else {
        std::cout << "[INFO] Successfully loaded " << X.size() 
                  << " real samples from " << filename << ".\n";
    }
}

// Artificial Neural Network Constructor
LightweightANN::LightweightANN() {
    std::default_random_engine gen(1337);
    std::normal_distribution<float> d1(0.0f, std::sqrt(2.0f / INPUT_SIZE));
    std::normal_distribution<float> d2(0.0f, std::sqrt(2.0f / HIDDEN_SIZE));

    W1.assign(INPUT_SIZE, std::vector<float>(HIDDEN_SIZE));
    b1.assign(HIDDEN_SIZE, 0.0f);
    W2.assign(HIDDEN_SIZE, std::vector<float>(OUTPUT_SIZE));
    b2.assign(OUTPUT_SIZE, 0.0f);

    for (int i = 0; i < INPUT_SIZE; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            W1[i][j] = d1(gen);

    for (int i = 0; i < HIDDEN_SIZE; ++i)
        for (int j = 0; j < OUTPUT_SIZE; ++j)
            W2[i][j] = d2(gen);
}

// Forward Propagation
std::vector<float> LightweightANN::forward(const std::vector<float>& input, std::vector<float>& hidden_out) {
    hidden_out.assign(HIDDEN_SIZE, 0.0f);
    std::vector<float> output_raw(OUTPUT_SIZE, 0.0f);

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float sum = b1[j];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            sum += input[i] * W1[i][j];
        }
        hidden_out[j] = relu(sum);
    }

    for (int k = 0; k < OUTPUT_SIZE; ++k) {
        float sum = b2[k];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            sum += hidden_out[j] * W2[j][k];
        }
        output_raw[k] = sum;
    }

    return softmax(output_raw);
}

// Backpropagation & Weight Update
void LightweightANN::train_sample(const std::vector<float>& input, int target_label) {
    std::vector<float> hidden_out;
    std::vector<float> probs = forward(input, hidden_out);

    std::vector<float> d_output(OUTPUT_SIZE, 0.0f);
    for (int k = 0; k < OUTPUT_SIZE; ++k) {
        d_output[k] = probs[k] - (k == target_label ? 1.0f : 0.0f);
    }

    std::vector<float> d_hidden(HIDDEN_SIZE, 0.0f);
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float error = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            error += d_output[k] * W2[j][k];
        }
        d_hidden[j] = error * relu_derivative(hidden_out[j]);
    }

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            W2[j][k] -= LEARNING_RATE * d_output[k] * hidden_out[j];
        }
    }
    for (int k = 0; k < OUTPUT_SIZE; ++k) b2[k] -= LEARNING_RATE * d_output[k];

    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            W1[i][j] -= LEARNING_RATE * d_hidden[j] * input[i];
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; ++j) b1[j] -= LEARNING_RATE * d_hidden[j];
}

// Inference
int LightweightANN::predict(const std::vector<float>& input) {
    std::vector<float> hidden_out;
    std::vector<float> output = forward(input, hidden_out);
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}
