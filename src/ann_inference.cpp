#include "ann_inference.h"
#include "../weights.h"

// Hardware ReLU: a single compare/mux, no library calls -> synthesizes trivially.
static inline acc_t relu_hw(acc_t x) {
    return (x > 0) ? x : 0;
}

void ann_inference(
    data_t input_image[INPUT_SIZE],
    acc_t  output_scores[OUTPUT_SIZE],
    int    &predicted_label
) {
    // ------------------------------------------------------------
    // Hardware communication protocol:
    //   - input_image / output_scores: m_axi, streamed data.
    //   - predicted_label / block control: s_axilite, control registers.
    // Weights are compile-time constants from weights.h, mapped into
    // BRAM — they never touch an AXI bus.
    // ------------------------------------------------------------
#pragma HLS INTERFACE m_axi     port=input_image    bundle=gmem0 depth=784
#pragma HLS INTERFACE m_axi     port=output_scores  bundle=gmem1 depth=10
#pragma HLS INTERFACE s_axilite port=predicted_label bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return           bundle=CTRL

    // ------------------------------------------------------------
    // Partition the weight ROMs into UNROLL_FACTOR parallel banks.
    // Without this, UNROLL below is a no-op: a single-port BRAM can
    // only serve one read per cycle, so parallel MACs would just
    // stall on memory access (a classic HLS bring-up bug).
    // ------------------------------------------------------------
#pragma HLS ARRAY_PARTITION variable=W1_QUANT cyclic factor=UNROLL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=W2_QUANT cyclic factor=UNROLL_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=B1_QUANT complete dim=1
#pragma HLS ARRAY_PARTITION variable=B2_QUANT complete dim=1

    data_t hidden[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1

    // ---------------- Layer 1: Input(784) -> Hidden(32), ReLU + rescale ----------------
    Layer1_Neurons: for (int j = 0; j < HIDDEN_SIZE; j++) {
        acc_t acc = B1_QUANT[j];

        Layer1_MACs: for (int i = 0; i < INPUT_SIZE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_FACTOR
            acc += (acc_t)input_image[i] * (acc_t)W1_QUANT[i * HIDDEN_SIZE + j];
        }

        acc_t relu_val = relu_hw(acc);

        // Fixed-point requantization: bring the wide accumulator (in
        // input_scale*weight1_scale units) back into the int8 domain
        // Layer 2's weights were quantized against (hidden_scale units).
        // int64_t intermediate avoids overflow: relu_val can be up to
        // ~12.6M, L1_RESCALE_MULT up to ~65536 (2^16), product up to
        // ~8.3e11, which overflows int32 but fits comfortably in int64.
        int64_t scaled = ((int64_t)relu_val * (int64_t)L1_RESCALE_MULT) >> RESCALE_SHIFT;
        hidden[j] = (scaled > 127) ? (data_t)127 : (data_t)scaled;
    }

    // ---------------- Layer 2: Hidden(32) -> Output(10), raw logits ----------------
    // No rescale needed here: per-tensor scaling is uniform across all
    // 10 output channels, so it can't change which one argmax picks.
    Layer2_Neurons: for (int k = 0; k < OUTPUT_SIZE; k++) {
        acc_t acc = B2_QUANT[k];

        Layer2_MACs: for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=UNROLL_FACTOR
            acc += (acc_t)hidden[j] * (acc_t)W2_QUANT[j * OUTPUT_SIZE + k];
        }

        output_scores[k] = acc;
    }

    // ---------------- Argmax over raw logits (softmax skipped, see header) ----------------
    acc_t max_val = output_scores[0];
    int   max_idx = 0;
    Argmax: for (int k = 1; k < OUTPUT_SIZE; k++) {
#pragma HLS PIPELINE II=1
        if (output_scores[k] > max_val) {
            max_val = output_scores[k];
            max_idx = k;
        }
    }
    predicted_label = max_idx;
}
