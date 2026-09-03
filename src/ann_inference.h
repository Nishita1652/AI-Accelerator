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
//    per spec (matches Pragya's INT8 quantization output).
//  - acc_t: int32_t accumulator. Worst case a single MAC-sum is
//    784 * 127 * 127 ~= 12.6M, comfortably inside +/-2.147B, so no
//    overflow risk even at full-scale INT8 inputs.
// ============================================================
typedef int8_t  weight_t;
typedef int8_t  data_t;
typedef int32_t acc_t;

// ============================================================
//  Top-level HLS kernel (synthesis entry point).
//
//  NOTE: weights are NOT function parameters. This kernel
//  #include "weights.h" directly (see ann_inference.cpp) so
//  W1_QUANT/B1_QUANT/W2_QUANT/B2_QUANT are compile-time static
//  const arrays that the HLS tool maps straight into on-chip
//  BRAM/ROM. Only the per-sample input image and the per-sample
//  results cross the function boundary at runtime.
//
//  Weight layout convention (Pragya's weights.h must match this):
//    W1_QUANT[i * HIDDEN_SIZE + j]  -- row-major, [INPUT_SIZE][HIDDEN_SIZE]
//    W2_QUANT[j * OUTPUT_SIZE + k]  -- row-major, [HIDDEN_SIZE][OUTPUT_SIZE]
//  i.e. same orientation as Phase 1's W1[INPUT_SIZE][HIDDEN_SIZE] and
//  W2[HIDDEN_SIZE][OUTPUT_SIZE], just flattened to 1D.
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
