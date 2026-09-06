#ifndef ANN_INFERENCE_H
#define ANN_INFERENCE_H

#include <stdint.h>

// ============================================================
//  Network dimensions — MUST stay in sync with Phase 1's
//  ann_model.h (INPUT_SIZE / HIDDEN_SIZE / OUTPUT_SIZE).
// ============================================================
#define INPUT_SIZE   784   // 28x28 flattened MNIST pixels
#define HIDDEN_SIZE  32    // Hidden layer neurons
#define OUTPUT_SIZE  10    // Output classes (digits 0-9)

// How many MACs are computed per cycle in each dot-product loop.
// Must evenly divide INPUT_SIZE and HIDDEN_SIZE. Raise this for more
// throughput / more DSP+BRAM-port usage, lower it to save area.
#define UNROLL_FACTOR 8

// ============================================================
//  Data types
//  - weight_t / data_t: int8_t, signed, bounded to [-127, 127]
//    per spec (matches Pragya's INT8 weight quantization).
//  - acc_t: int32_t accumulator AND bias type. Biases are stored
//    as int32_t (not int8_t) — accumulator-domain precision is
//    needed once they're summed against a wide INT32 MAC total;
//    int8 biases lose too much resolution to matter.
//    Worst case a single MAC-sum is 784*127*127 ~= 12.6M, well
//    inside +/-2.147B, so no overflow risk even at full-scale
//    INT8 inputs.
// ============================================================
typedef int8_t  weight_t;
typedef int8_t  data_t;
typedef int32_t acc_t;

// ============================================================
//  Top-level HLS kernel (synthesis entry point).
//
//  weights.h (included by ann_inference.cpp) supplies, as
//  compile-time static const arrays mapped into BRAM:
//    weight_t W1_QUANT[INPUT_SIZE*HIDDEN_SIZE]   -- row-major [INPUT_SIZE][HIDDEN_SIZE]
//    acc_t    B1_QUANT[HIDDEN_SIZE]              -- INT32, accumulator-domain scale
//    weight_t W2_QUANT[HIDDEN_SIZE*OUTPUT_SIZE]  -- row-major [HIDDEN_SIZE][OUTPUT_SIZE]
//    acc_t    B2_QUANT[OUTPUT_SIZE]              -- INT32, accumulator-domain scale
//    #define L1_RESCALE_MULT / RESCALE_SHIFT     -- fixed-point Q(RESCALE_SHIFT)
//        multiplier that rescales Layer 1's raw accumulator+ReLU
//        output back into the INT8 domain Layer 2 expects. This
//        replaces naive saturate-to-127 clamping, which silently
//        saturates real trained weights instead of representing
//        them faithfully — the multiplier is derived from real
//        calibration data by quantize_weights.cpp.
//        Layer 2's output is NOT rescaled: argmax is invariant to
//        a shared positive scale, so raw INT32 logits are fine.
//
//  Softmax is intentionally NOT computed in hardware: it's monotonic,
//  so it never changes which class has the highest score. We argmax
//  the raw logits directly and skip the exp()/division cost entirely.
// ============================================================
void ann_inference(
    data_t input_image[INPUT_SIZE],
    acc_t  output_scores[OUTPUT_SIZE],
    int    &predicted_label
);

#endif // ANN_INFERENCE_H