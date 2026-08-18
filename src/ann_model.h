#ifndef ANN_MODEL_H
#define ANN_MODEL_H

#include <vector>
#include <string>

// Hardware-friendly Model Dimensions
constexpr int INPUT_SIZE   = 784; // 28x28 flattened pixels
constexpr int HIDDEN_SIZE  = 32;  // Small hidden layer for FPGA Block RAM fitting
constexpr int OUTPUT_SIZE  = 10;  // 10 output classes (digits 0-9)
constexpr float LEARNING_RATE = 0.005f;

// Hardware Activation Functions
float relu(float x);
float relu_derivative(float x);
std::vector<float> softmax(const std::vector<float>& input);

// Data Loaders with Synthetic Fallback
void generate_synthetic_data(std::vector<std::vector<float>>& X, std::vector<int>& y, int num_samples = 1000);
void load_or_fallback_data(const std::string& filename, std::vector<std::vector<float>>& X, std::vector<int>& y);

// Hardware-Aware Artificial Neural Network Class
class LightweightANN {
public:
    std::vector<std::vector<float>> W1; // [INPUT_SIZE x HIDDEN_SIZE]
    std::vector<float> b1;              // [HIDDEN_SIZE]
    std::vector<std::vector<float>> W2; // [HIDDEN_SIZE x OUTPUT_SIZE]
    std::vector<float> b2;              // [OUTPUT_SIZE]

    LightweightANN();
    std::vector<float> forward(const std::vector<float>& input, std::vector<float>& hidden_out);
    void train_sample(const std::vector<float>& input, int target_label);
    int predict(const std::vector<float>& input);
};

#endif // ANN_MODEL_H
