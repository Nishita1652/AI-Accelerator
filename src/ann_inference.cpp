#include "ann_inference.h"
#include "../weights.h"

// Hardware ReLU: synthesizes directly into a comparator + mux
static inline acc_t relu_hw(acc_t x) {
    return (x > 0) ? x : 0;
}

void ann_inference(
    data_t input_image[INPUT_SIZE],
    acc_t  output_scores[OUTPUT_SIZE],
    int    &predicted_label
) {
    // ------------------------------------------------------------
    // Hardware communication protocol
    // ------------------------------------------------------------
#pragma HLS INTERFACE m_axi     port=input_image    offset=slave bundle=gmem0 depth=784
#pragma HLS INTERFACE m_axi     port=output_scores  offset=slave bundle=gmem1 depth=10
#pragma HLS INTERFACE s_axilite port=predicted_label bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return          bundle=CTRL

    // ------------------------------------------------------------
    // On-Chip BRAM / Register Buffers
    // ------------------------------------------------------------
    data_t local_input[INPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_input cyclic factor=UNROLL_FACTOR dim=1

    data_t hidden[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1

    acc_t local_output[OUTPUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_output complete dim=1

    // Step A: Burst-read image into fast on-chip BRAM (1 read per clock)
    Load_Input: for (int i = 0; i < INPUT_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        local_input[i] = input_image[i];
    }

    // ------------------------------------------------------------
    // Layer 1: Input (784) -> Hidden (32)
    // ------------------------------------------------------------
    Layer1_Neurons: for (int j = 0; j < HIDDEN_SIZE; j++) {
        // Shift int8 bias into the 24-bit product domain
        acc_t acc = static_cast<acc_t>(B1_QUANT[j]) << 7;

        Layer1_MACs: for (int i = 0; i < INPUT_SIZE; i++) {
#pragma HLS PIPELINE II=1
            int w_idx = i * HIDDEN_SIZE + j; // Row-major
            acc += static_cast<acc_t>(local_input[i]) * static_cast<acc_t>(W1_QUANT[w_idx]);
        }

        acc_t relu_val = relu_hw(acc);

        // Fixed-point requantization
        int64_t scaled = (static_cast<int64_t>(relu_val) * static_cast<int64_t>(L1_RESCALE_MULT)) >> RESCALE_SHIFT;
        hidden[j] = (scaled > 127) ? static_cast<data_t>(127) : static_cast<data_t>(scaled);
    }

    // ------------------------------------------------------------
    // Layer 2: Hidden (32) -> Output (10)
    // ------------------------------------------------------------
    Layer2_Neurons: for (int k = 0; k < OUTPUT_SIZE; k++) {
#pragma HLS PIPELINE II=1
        acc_t acc = static_cast<acc_t>(B2_QUANT[k]) << 7;

        Layer2_MACs: for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS UNROLL
            int w_idx = j * OUTPUT_SIZE + k; // Row-major
            acc += static_cast<acc_t>(hidden[j]) * static_cast<acc_t>(W2_QUANT[w_idx]);
        }

        local_output[k] = acc;
        output_scores[k] = acc; // Stream out to bus
    }

    // ------------------------------------------------------------
    // Hardware ArgMax
    // ------------------------------------------------------------
    acc_t max_val = local_output[0];
    int   max_idx = 0;

    Argmax: for (int k = 1; k < OUTPUT_SIZE; k++) {
#pragma HLS PIPELINE II=1
        if (local_output[k] > max_val) {
            max_val = local_output[k];
            max_idx = k;
        }
    }

    predicted_label = max_idx;
}