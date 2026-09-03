# Kernel Interface Contract — `ann_inference`

## Repo layout (confirmed against Pragya's tree)

```
AI-Accelerator/
├── src/
│   ├── ann_model.cpp / ann_model.h / main.cpp   (Phase 1)
│   ├── ann_inference.h                           (Member 1)
│   └── ann_inference.cpp                         (Member 1)
├── scripts/prepare_mnist_gz.py
├── data/raw/  data/processed/mnist_data.csv
├── float_weights.txt          (Pragya — intermediate FP32 dump)
├── quantize_weights.cpp       (Pragya — quantization/export program)
├── quantized_weights.txt      (Pragya — intermediate INT8 text dump)
├── weights.h                  (Pragya — final header, PROJECT ROOT)
└── README.md
```

⚠️ **Real footgun caught and fixed:** `weights.h` sits at the project
root, but `ann_inference.cpp`/`.h` live in `src/`. A plain
`#include "weights.h"` does NOT resolve across that boundary — tested
it, it fails with `fatal error: weights.h: No such file or directory`.
Fixed with relative includes instead of relying on compiler `-I`
flags (which would need to be set consistently in every build path —
g++, Vitis HLS, and Bambu — and are easy to forget in one of them):

- `src/ann_inference.cpp` now does `#include "../weights.h"`
- `weights.h` (at root) now does `#include "src/ann_inference.h"`
  (needed since it reuses `INPUT_SIZE`/`HIDDEN_SIZE`/`OUTPUT_SIZE`/
  `weight_t` rather than hardcoding them)

Verified this actually compiles and runs correctly from both the
project root and from inside `src/` — both delivered files already
have this fix applied, no action needed unless you move files around
later, in which case these two include paths need to move with them.

**Updated:** weights are no longer passed as function arguments. This
changed after the detailed spec confirmed weights must be static
compile-time arrays the HLS tool maps directly into BRAM, included via
`weights.h`. Re-read this even if you saw the earlier version.

## The exact function signature (do not change without telling Member 1)

```cpp
void ann_inference(
    data_t input_image[INPUT_SIZE],   // int8_t[784] — the only per-sample input
    acc_t  output_scores[OUTPUT_SIZE],// int32_t[10] — RAW logits, not probabilities
    int    &predicted_label           // 0-9, hardware argmax result
);
```

- `weight_t` = `int8_t`, `data_t` = `int8_t`, `acc_t` = `int32_t`.
- **No softmax exists in hardware.** `predicted_label` is the number to
  grade accuracy against, not `output_scores`.
- Hidden-layer activations are clamped to `[0, 127]` inside the kernel
  (simple saturation after ReLU) before Layer 2.
- **Interface pragmas are now in place**, per spec: `input_image` and
  `output_scores` use `m_axi` (streamed data), `predicted_label` and
  the block-level control signals use `s_axilite` (control registers).

## ⚠️ #1 thing that changed: weights.h is now a hard, name-exact contract

`ann_inference.cpp` does `#include "weights.h"` directly and references
these globals by name inside its loops:

```cpp
static const weight_t W1_QUANT[INPUT_SIZE * HIDDEN_SIZE];  // row-major [INPUT_SIZE][HIDDEN_SIZE]
static const weight_t B1_QUANT[HIDDEN_SIZE];
static const weight_t W2_QUANT[HIDDEN_SIZE * OUTPUT_SIZE]; // row-major [HIDDEN_SIZE][OUTPUT_SIZE]
static const weight_t B2_QUANT[OUTPUT_SIZE];
```

This is **stricter than the old parameter-passing approach**: before,
any correctly-ordered/typed arguments worked regardless of name. Now
Pragya's `weights.h` must match these exact array names, sizes, and
`static const weight_t` type, or the kernel won't compile. A
placeholder `weights.h` (all 1s/0s) is included in this delivery as
the format Pragya's real file needs to match — it's a literal
drop-in replacement, same filename, same array names.

⚠️ **Still unresolved, needs a real conversation, not just this doc:**
the kernel does raw INT8×INT8→INT32 accumulation with no rescaling
between layers. That only produces sane predictions if Pragya's weight
scale factor keeps the raw sums meaningful on its own. If proper
per-layer requantization is needed, the kernel needs a small addition
— bring your chosen scale factor(s) to a sync before finalizing real
weights.

---

## Pragya (INT8 Quantization Engine)

**Do:**
- Quantize weights to signed `int8_t`, bounded `[-127, 127]` per spec.
- Export as `static const weight_t` arrays named exactly `W1_QUANT`,
  `B1_QUANT`, `W2_QUANT`, `B2_QUANT`, in a file named exactly
  `weights.h`, formatted as literal initializer lists (see the
  placeholder file for the exact format to match).
- Flatten in **row-major** order, matching Phase 1's orientation —
  literally flatten your existing `W1`/`W2` vectors, no transpose:
  `W1_QUANT[i * HIDDEN_SIZE + j]`, `W2_QUANT[j * OUTPUT_SIZE + k]`.
- Have `weights.h` `#include "ann_inference.h"` to reuse `INPUT_SIZE`/
  `HIDDEN_SIZE`/`OUTPUT_SIZE`/`weight_t` rather than hardcoding numbers.
- Decide and document the quantization scale factor(s) you used —
  Sanskriti needs the same convention for quantizing test images, and
  Member 1 needs it to know whether raw-sum accumulation (no rescale)
  is accurate enough for this design.
- Modify `ann_model.h`/`.cpp`/`main.cpp` to auto-export `weights.h` as
  part of the training run, as your task calls for.

**Don't:**
- Don't use different array names, a different filename, or a
  different array type — the kernel references these by exact name,
  it will not compile with anything else.
- Don't transpose `W1`/`W2` relative to Phase 1's orientation.
- Don't calibrate against a probability output — there isn't one.
- Don't have any other `.cpp` file in the project also
  `#include "weights.h"` — it's only included by `ann_inference.cpp`.
  (The arrays are declared `static`, so accidental double-inclusion
  won't break the *build*, but it will silently duplicate storage and
  invite confusion — just don't do it.)

---

## Sanskriti (HLS C/RTL Testbench)

**Do:**
- Write `tb_main.cpp` to stream normalized MNIST test samples from
  `mnist_data.csv`, quantize each to `int8_t` using the **same scale
  convention Pragya used for weights**, then call
  `ann_inference(input_image, output_scores, predicted_label)`.
- Compare `predicted_label` against the true label from the CSV to
  compute accuracy — same pattern as `local_smoke_test.cpp`, scaled
  to the full test set.
- Load `ann_inference.cpp`, Pragya's `weights.h`, and your
  `tb_main.cpp` into the same Vitis HLS (or Bambu) project.
- Run **C Simulation first** to verify numerical correctness of the
  quantized logic, **then C/RTL Co-Simulation** to confirm the
  synthesized Verilog matches the C model bit-for-bit — both steps
  are required per spec, not just csim. `run_hls.tcl` runs both.

**Don't:**
- Don't use the placeholder `weights.h` for accuracy numbers — it's
  all-1s/0s, for structural/compile testing only.
- Don't expect a confidence score or probability vector — only
  `predicted_label` is meaningful for grading.
- Don't quantize test images with your own arbitrary scale — it must
  match Pragya's weight-quantization scale, or numbers will compile
  and run but accuracy will be silently wrong.
- Don't skip cosim and call it verified after csim alone — csim only
  proves the C++ math is right, not that the generated hardware
  matches it.
