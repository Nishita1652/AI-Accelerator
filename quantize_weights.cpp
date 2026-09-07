// ============================================================
//  quantize_weights.cpp — Member 2 (Pragya) deliverable.
//
//  Pipeline:
//    1. Read trained FP32 weights from float_weights.txt
//       (produced by ann_model.cpp's export_for_quantization()).
//    2. Calibrate: run the FP32 forward pass (Layer 1 only) over
//       real samples from mnist_data.csv to find the true dynamic
//       range of hidden-layer activations. Naive clamping without
//       this step saturates real trained weights — this calibration
//       is what makes the fixed-point rescale below correct instead
//       of a guess.
//    3. Quantize W1/W2 to signed INT8 (symmetric, per-tensor scale,
//       bounded [-127, 127]). Quantize b1/b2 to INT32 (biases need
//       accumulator-domain precision — INT8 biases lose too much
//       resolution once added to a wide INT32 MAC sum).
//    4. Derive a fixed-point requantization multiplier (integer
//       multiply + right-shift) that rescales Layer 1's raw
//       accumulator output back into the INT8 domain Layer 2
//       expects. Layer 2's output does NOT need this treatment —
//       argmax is invariant to a shared positive scale, so raw
//       logits are fine as-is.
//    5. Emit quantized_weights.txt (human-readable audit trail) and
//       weights.h (the real deliverable — exact array names/types
//       ann_inference.cpp expects).
//
//  Usage:  g++ -O2 -o quantize_weights quantize_weights.cpp
//          ./quantize_weights
//  (run from the project root — paths below are relative to it)
// ============================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>

static const std::string FLOAT_WEIGHTS_PATH = "float_weights.txt";
static const std::string CALIBRATION_CSV_PATH = "data/processed/mnist_data.csv";
static const std::string QUANTIZED_TXT_PATH = "quantized_weights.txt";
static const std::string WEIGHTS_H_PATH = "weights.h";

static const int RESCALE_SHIFT_BITS = 16;   // Q16 fixed-point for the L1->L2 rescale
static const int MAX_CALIBRATION_SAMPLES = 500;

// ---- Load W1/b1/W2/b2 from float_weights.txt ----
struct FloatWeights {
    int input_size, hidden_size, output_size;
    std::vector<float> W1; // flattened [input_size][hidden_size]
    std::vector<float> b1;
    std::vector<float> W2; // flattened [hidden_size][output_size]
    std::vector<float> b2;
};

static bool load_float_weights(const std::string& path, FloatWeights& fw) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Could not open " << path << "\n";
        return false;
    }
    in >> fw.input_size >> fw.hidden_size >> fw.output_size;

    fw.W1.resize((size_t)fw.input_size * fw.hidden_size);
    for (auto& v : fw.W1) in >> v;

    fw.b1.resize(fw.hidden_size);
    for (auto& v : fw.b1) in >> v;

    fw.W2.resize((size_t)fw.hidden_size * fw.output_size);
    for (auto& v : fw.W2) in >> v;

    fw.b2.resize(fw.output_size);
    for (auto& v : fw.b2) in >> v;

    if (!in) {
        std::cerr << "[ERROR] " << path << " is malformed or truncated.\n";
        return false;
    }
    return true;
}

// ---- Calibration: run FP32 Layer 1 forward pass over real data,
//      track the maximum abs(ReLU(hidden)) activation observed.
//      Falls back to a conservative default if no CSV is found,
//      so the pipeline never silently produces garbage scales. ----
static float calibrate_hidden_range(const FloatWeights& fw) {
    std::ifstream file(CALIBRATION_CSV_PATH);
    if (!file.is_open()) {
        std::cerr << "[WARN] Could not open " << CALIBRATION_CSV_PATH
                  << " for calibration. Using a conservative default hidden range (1.0).\n"
                  << "       Real accuracy depends on calibrating against real data —\n"
                  << "       re-run once mnist_data.csv is available.\n";
        return 1.0f;
    }

    float max_abs_hidden = 1e-6f; // avoid divide-by-zero downstream
    std::string line;
    int samples_used = 0;

    while (std::getline(file, line) && samples_used < MAX_CALIBRATION_SAMPLES) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        float value;

        ss >> value; // label, discarded — calibration only needs pixels

        std::vector<float> input(fw.input_size);
        bool row_ok = true;
        for (int i = 0; i < fw.input_size; ++i) {
            char comma;
            ss >> comma >> value;
            if (!ss) { row_ok = false; break; }
            input[i] = value; // already normalized [0,1] in the CSV
        }
        if (!row_ok) continue;

        for (int j = 0; j < fw.hidden_size; ++j) {
            float acc = fw.b1[j];
            for (int i = 0; i < fw.input_size; ++i) {
                acc += input[i] * fw.W1[(size_t)i * fw.hidden_size + j];
            }
            float activated = std::max(0.0f, acc);
            max_abs_hidden = std::max(max_abs_hidden, activated);
        }
        samples_used++;
    }

    if (samples_used == 0) {
        std::cerr << "[WARN] " << CALIBRATION_CSV_PATH << " had no usable rows. "
                  << "Using conservative default hidden range (1.0).\n";
        return 1.0f;
    }

    std::cout << "[INFO] Calibrated hidden-activation range over "
              << samples_used << " real samples: max = " << max_abs_hidden << "\n";
    return max_abs_hidden;
}

// ---- Symmetric per-tensor INT8 quantization ----
static float compute_scale(const std::vector<float>& values) {
    float max_abs = 1e-9f;
    for (float v : values) max_abs = std::max(max_abs, std::fabs(v));
    return max_abs / 127.0f;
}

static int8_t quantize_int8(float value, float scale) {
    int q = (int)std::lround(value / scale);
    q = std::max(-127, std::min(127, q));
    return (int8_t)q;
}

static int32_t quantize_int32_bias(float value, float scale) {
    return (int32_t)std::lround(value / scale);
}

static void write_int8_array(std::ofstream& out, const std::string& name,
                              const std::vector<int8_t>& arr) {
    out << "static const weight_t " << name << "[" << arr.size() << "] = {\n    ";
    for (size_t i = 0; i < arr.size(); ++i) {
        out << (int)arr[i];
        if (i + 1 < arr.size()) out << ", ";
        if ((i + 1) % 20 == 0) out << "\n    ";
    }
    out << "\n};\n\n";
}

static void write_int32_array(std::ofstream& out, const std::string& name,
                               const std::vector<int32_t>& arr) {
    out << "static const acc_t " << name << "[" << arr.size() << "] = {\n    ";
    for (size_t i = 0; i < arr.size(); ++i) {
        out << arr[i];
        if (i + 1 < arr.size()) out << ", ";
        if ((i + 1) % 20 == 0) out << "\n    ";
    }
    out << "\n};\n\n";
}

int main() {
    FloatWeights fw;
    if (!load_float_weights(FLOAT_WEIGHTS_PATH, fw)) return 1;

    std::cout << "[INFO] Loaded FP32 weights: " << fw.input_size << " -> "
              << fw.hidden_size << " -> " << fw.output_size << "\n";

    // ---- Scales ----
    const float S_in = 1.0f / 127.0f; // input pixels quantized [0,1] -> [0,127]
    const float S_w1 = compute_scale(fw.W1);
    const float S_w2 = compute_scale(fw.W2);
    const float max_hidden = calibrate_hidden_range(fw);
    const float S_hidden = max_hidden / 127.0f;

    std::cout << "[INFO] Scales: S_in=" << S_in << " S_w1=" << S_w1
              << " S_w2=" << S_w2 << " S_hidden=" << S_hidden << "\n";

    // ---- Quantize weights (INT8) ----
    std::vector<int8_t> W1_QUANT(fw.W1.size());
    for (size_t i = 0; i < fw.W1.size(); ++i) W1_QUANT[i] = quantize_int8(fw.W1[i], S_w1);

    std::vector<int8_t> W2_QUANT(fw.W2.size());
    for (size_t i = 0; i < fw.W2.size(); ++i) W2_QUANT[i] = quantize_int8(fw.W2[i], S_w2);

    // ---- Quantize biases (INT32, accumulator-domain scale) ----
    std::vector<int32_t> B1_QUANT(fw.b1.size());
    for (size_t i = 0; i < fw.b1.size(); ++i)
        B1_QUANT[i] = quantize_int32_bias(fw.b1[i], S_in * S_w1);

    std::vector<int32_t> B2_QUANT(fw.b2.size());
    for (size_t i = 0; i < fw.b2.size(); ++i)
        B2_QUANT[i] = quantize_int32_bias(fw.b2[i], S_hidden * S_w2);

    // ---- Derive the Layer1->Layer2 fixed-point rescale multiplier ----
    // Real value after Layer 1's accumulate+ReLU is (acc * S_in * S_w1).
    // We need hidden[j] (int8, scale S_hidden) such that
    // hidden[j] * S_hidden ~= acc * S_in * S_w1
    // => hidden[j] ~= acc * (S_in * S_w1 / S_hidden) = acc * M
    double M = (double)(S_in * S_w1) / (double)S_hidden;
    int64_t L1_RESCALE_MULT = (int64_t)std::llround(M * (1LL << RESCALE_SHIFT_BITS));

    std::cout << "[INFO] L1 rescale: M=" << M
              << " -> L1_RESCALE_MULT=" << L1_RESCALE_MULT
              << " RESCALE_SHIFT=" << RESCALE_SHIFT_BITS << "\n";

    // ---- quantized_weights.txt: human-readable audit trail ----
    {
        std::ofstream out(QUANTIZED_TXT_PATH);
        out << "# Quantization summary (audit trail, not consumed by the kernel)\n";
        out << "S_in=" << S_in << "\nS_w1=" << S_w1 << "\nS_w2=" << S_w2
            << "\nS_hidden=" << S_hidden << " (calibrated max_hidden=" << max_hidden << ")\n";
        out << "L1_RESCALE_MULT=" << L1_RESCALE_MULT
            << " RESCALE_SHIFT=" << RESCALE_SHIFT_BITS << "\n\n";

        out << "W1_QUANT (int8, " << W1_QUANT.size() << " values):\n";
        for (size_t i = 0; i < W1_QUANT.size(); ++i) out << (int)W1_QUANT[i] << " ";
        out << "\n\nB1_QUANT (int32, " << B1_QUANT.size() << " values):\n";
        for (size_t i = 0; i < B1_QUANT.size(); ++i) out << B1_QUANT[i] << " ";
        out << "\n\nW2_QUANT (int8, " << W2_QUANT.size() << " values):\n";
        for (size_t i = 0; i < W2_QUANT.size(); ++i) out << (int)W2_QUANT[i] << " ";
        out << "\n\nB2_QUANT (int32, " << B2_QUANT.size() << " values):\n";
        for (size_t i = 0; i < B2_QUANT.size(); ++i) out << B2_QUANT[i] << " ";
        out << "\n";
        std::cout << "[SUCCESS] Wrote " << QUANTIZED_TXT_PATH << "\n";
    }

    // ---- weights.h: the real deliverable ----
    {
        std::ofstream out(WEIGHTS_H_PATH);
        out << "#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n";
        out << "#include \"src/ann_inference.h\"\n\n";
        out << "// ============================================================\n";
        out << "// Real INT8-quantized weights, generated by quantize_weights.cpp\n";
        out << "// from a real trained model. DO NOT hand-edit -- regenerate via\n";
        out << "// the quantize_weights pipeline if the model is retrained.\n";
        out << "//\n";
        out << "// Scales used (see quantized_weights.txt for the full audit trail):\n";
        out << "//   S_in=" << S_in << " S_w1=" << S_w1 << " S_w2=" << S_w2
            << " S_hidden=" << S_hidden << "\n";
        out << "// ============================================================\n\n";

        out << "#define RESCALE_SHIFT " << RESCALE_SHIFT_BITS << "\n";
        out << "#define L1_RESCALE_MULT " << L1_RESCALE_MULT << "LL\n\n";

        write_int8_array(out, "W1_QUANT", W1_QUANT);
        write_int32_array(out, "B1_QUANT", B1_QUANT);
        write_int8_array(out, "W2_QUANT", W2_QUANT);
        write_int32_array(out, "B2_QUANT", B2_QUANT);

        out << "#endif // WEIGHTS_H\n";
        std::cout << "[SUCCESS] Wrote " << WEIGHTS_H_PATH << "\n";
    }

    return 0;
}
