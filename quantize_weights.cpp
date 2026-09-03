#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdint>

constexpr int INPUT_SIZE  = 784;
constexpr int HIDDEN_SIZE = 32;
constexpr int OUTPUT_SIZE = 10;

// Signed 8-bit quantization range
constexpr int8_t Q_MIN = -127;
constexpr int8_t Q_MAX = 127;


// =========================================================
// Quantize Float32 -> signed int8
// =========================================================
int8_t quantize(float value, float scale)
{
    if (scale == 0.0f)
        return 0;

    int q = static_cast<int>(std::round(value / scale));

    if (q > Q_MAX)
        q = Q_MAX;

    if (q < Q_MIN)
        q = Q_MIN;

    return static_cast<int8_t>(q);
}


int main()
{
    // =========================================================
    // Open Phase 1 Float32 weights
    // =========================================================

    std::ifstream input("float_weights.txt");

    if (!input.is_open())
    {
        std::cerr << "[ERROR] Could not open float_weights.txt\n";
        return 1;
    }


    // =========================================================
    // Open verification output
    // =========================================================

    std::ofstream txt_output("quantized_weights.txt");

    if (!txt_output.is_open())
    {
        std::cerr << "[ERROR] Could not create quantized_weights.txt\n";
        return 1;
    }


    // =========================================================
    // Open final hardware header
    // =========================================================

    std::ofstream header("weights.h");

    if (!header.is_open())
    {
        std::cerr << "[ERROR] Could not create weights.h\n";
        return 1;
    }


    std::string section;


    // =========================================================
    // W1: 784 x 32
    // Total = 25,088 weights
    // =========================================================

    input >> section;

    float W1[INPUT_SIZE * HIDDEN_SIZE];

    float W1_max_abs = 0.0f;

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i)
    {
        input >> W1[i];

        float abs_value = std::fabs(W1[i]);

        if (abs_value > W1_max_abs)
            W1_max_abs = abs_value;
    }

    float W1_scale = W1_max_abs / 127.0f;


    // ---------------------------------------------------------
    // Write W1 to verification text file
    // ---------------------------------------------------------

    txt_output << "W1\n";


    // ---------------------------------------------------------
    // Write header beginning
    // ---------------------------------------------------------

    header << "#ifndef WEIGHTS_H\n";
    header << "#define WEIGHTS_H\n\n";

    header << "#include <cstdint>\n\n";

    header << "constexpr int INPUT_SIZE = 784;\n";
    header << "constexpr int HIDDEN_SIZE = 32;\n";
    header << "constexpr int OUTPUT_SIZE = 10;\n\n";

    header << "// Symmetric signed int8 quantization scale\n";
    header << "constexpr float W1_SCALE = "
           << W1_scale << "f;\n\n";


    // ---------------------------------------------------------
    // W1 flattened static array
    // ---------------------------------------------------------

    header << "// W1: 784 x 32 = 25088 signed 8-bit values\n";

    header << "static const int8_t "
           << "W1_QUANT[INPUT_SIZE * HIDDEN_SIZE] = {\n";


    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i)
    {
        int value =
            static_cast<int>(quantize(W1[i], W1_scale));

        txt_output << value << " ";

        header << value;

        if (i != INPUT_SIZE * HIDDEN_SIZE - 1)
            header << ", ";

        if ((i + 1) % 16 == 0)
            header << "\n";
    }

    txt_output << "\n";

    header << "};\n\n";


    // =========================================================
    // b1: 32
    // =========================================================

    input >> section;

    float b1[HIDDEN_SIZE];

    for (int i = 0; i < HIDDEN_SIZE; ++i)
        input >> b1[i];


    float b1_max_abs = 0.0f;

    for (int i = 0; i < HIDDEN_SIZE; ++i)
    {
        float abs_value = std::fabs(b1[i]);

        if (abs_value > b1_max_abs)
            b1_max_abs = abs_value;
    }

    float b1_scale = b1_max_abs / 127.0f;


    txt_output << "b1\n";


    header << "// b1: 32 signed 8-bit bias values\n";

    header << "constexpr float B1_SCALE = "
           << b1_scale << "f;\n\n";

    header << "static const int8_t "
           << "B1_QUANT[HIDDEN_SIZE] = {\n";


    for (int i = 0; i < HIDDEN_SIZE; ++i)
    {
        int value =
            static_cast<int>(quantize(b1[i], b1_scale));

        txt_output << value << " ";

        header << value;

        if (i != HIDDEN_SIZE - 1)
            header << ", ";
    }

    txt_output << "\n";

    header << "\n};\n\n";


    // =========================================================
    // W2: 32 x 10
    // Total = 320 weights
    // =========================================================

    input >> section;

    float W2[HIDDEN_SIZE * OUTPUT_SIZE];

    float W2_max_abs = 0.0f;


    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i)
    {
        input >> W2[i];

        float abs_value = std::fabs(W2[i]);

        if (abs_value > W2_max_abs)
            W2_max_abs = abs_value;
    }


    float W2_scale = W2_max_abs / 127.0f;


    txt_output << "W2\n";


    header << "// W2: 32 x 10 = 320 signed 8-bit values\n";

    header << "constexpr float W2_SCALE = "
           << W2_scale << "f;\n\n";

    header << "static const int8_t "
           << "W2_QUANT[HIDDEN_SIZE * OUTPUT_SIZE] = {\n";


    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i)
    {
        int value =
            static_cast<int>(quantize(W2[i], W2_scale));

        txt_output << value << " ";

        header << value;

        if (i != HIDDEN_SIZE * OUTPUT_SIZE - 1)
            header << ", ";

        if ((i + 1) % 16 == 0)
            header << "\n";
    }

    txt_output << "\n";

    header << "};\n\n";


    // =========================================================
    // b2: 10
    // =========================================================

    input >> section;

    float b2[OUTPUT_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; ++i)
        input >> b2[i];


    float b2_max_abs = 0.0f;

    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        float abs_value = std::fabs(b2[i]);

        if (abs_value > b2_max_abs)
            b2_max_abs = abs_value;
    }


    float b2_scale = b2_max_abs / 127.0f;


    txt_output << "b2\n";


    header << "// b2: 10 signed 8-bit bias values\n";

    header << "constexpr float B2_SCALE = "
           << b2_scale << "f;\n\n";

    header << "static const int8_t "
           << "B2_QUANT[OUTPUT_SIZE] = {\n";


    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        int value =
            static_cast<int>(quantize(b2[i], b2_scale));

        txt_output << value << " ";

        header << value;

        if (i != OUTPUT_SIZE - 1)
            header << ", ";
    }

    txt_output << "\n";

    header << "\n};\n\n";


    // =========================================================
    // Finish weights.h
    // =========================================================

    header << "#endif // WEIGHTS_H\n";


    // =========================================================
    // Close files
    // =========================================================

    input.close();
    txt_output.close();
    header.close();


    // =========================================================
    // Status information
    // =========================================================

    std::cout << "[SUCCESS] Quantization complete!\n";

    std::cout << "[INFO] W1 scale: "
              << W1_scale << "\n";

    std::cout << "[INFO] b1 scale: "
              << b1_scale << "\n";

    std::cout << "[INFO] W2 scale: "
              << W2_scale << "\n";

    std::cout << "[INFO] b2 scale: "
              << b2_scale << "\n";

    std::cout << "[INFO] Quantized text output: "
              << "quantized_weights.txt\n";

    std::cout << "[INFO] Hardware header output: "
              << "weights.h\n";


    return 0;
}