// Not part of the HLS deliverable — a quick local g++ sanity check only.
// Sanskriti's tb_main.cpp is the real, official testbench.
#include <cstdio>
#include "ann_inference.h"

int main() {
    data_t input_image[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) input_image[i] = 64; // arbitrary mid-range test input

    acc_t output_scores[OUTPUT_SIZE];
    int predicted_label = -1;

    ann_inference(input_image, output_scores, predicted_label);

    printf("Output logits: ");
    for (int k = 0; k < OUTPUT_SIZE; k++) printf("%d ", output_scores[k]);
    printf("\nPredicted label: %d\n", predicted_label);

    // No hardcoded pass/fail here — with real quantized weights the
    // "correct" answer depends on training data, not a hand-calculable
    // constant. This just confirms the kernel runs and the logits are
    // NOT degenerate (not all identical, not all clamped to one value),
    // which would indicate a saturation/rescale bug.
    bool all_same = true;
    for (int k = 1; k < OUTPUT_SIZE; k++)
        if (output_scores[k] != output_scores[0]) { all_same = false; break; }

    if (all_same) {
        printf("[WARN] All logits identical — check weights.h isn't degenerate "
               "(all-zero weights, or a rescale bug saturating everything).\n");
    } else {
        printf("[OK] Logits vary across classes — rescale/quantization looks non-degenerate.\n");
    }
    return 0;
}
