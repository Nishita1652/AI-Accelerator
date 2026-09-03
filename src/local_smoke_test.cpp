// Not part of the HLS deliverable — a quick local g++ sanity check only.
// Sanskriti's tb_main.cpp is the real, official testbench.
#include <cstdio>
#include "ann_inference.h"

int main() {
    data_t input_image[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) input_image[i] = 1;

    acc_t output_scores[OUTPUT_SIZE];
    int predicted_label = -1;

    ann_inference(input_image, output_scores, predicted_label);

    printf("Output logits: ");
    for (int k = 0; k < OUTPUT_SIZE; k++) printf("%d ", output_scores[k]);
    printf("\nPredicted label: %d\n", predicted_label);

    // Hand-calc with placeholder weights.h (all W=1, all B=0):
    //   hidden[j] = relu(784) clamped to 127 -> 127, for every j
    //   output[k] = sum of 32 hidden values   -> 32*127 = 4064, for every k
    // All 10 logits tie, so argmax deterministically picks class 0.
    bool ok = (predicted_label == 0) && (output_scores[0] == 32 * 127);
    printf(ok ? "[PASS] matches hand-calculated expected result\n"
              : "[FAIL] does not match expected result\n");
    return ok ? 0 : 1;
}
