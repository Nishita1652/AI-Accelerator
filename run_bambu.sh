# ============================================================
#  run_bambu.sh — Reference invocation for the open-source path
#  (Bambu HLS, Politecnico di Milano), targeting the same
#  ZCU7EV device family used on the ZCU104.
#
#  Usage (from the AI-Accelerator/ project root):
#      bash run_bambu.sh
#
#  NOTE: Unlike Vitis HLS, Bambu is driven from the command line
#  rather than a Tcl project script — no separate project setup
#  step is needed. weights.h resolves automatically via
#  src/ann_inference.cpp's own relative #include "../weights.h" —
#  no extra include-path flags required, same as the g++ build.
#
#  VERIFY BEFORE RELYING ON THIS: Bambu's own docs show it
#  recognizing the same "#pragma HLS ..." syntax we used for
#  Vitis (confirmed for #pragma HLS interface; PIPELINE/UNROLL/
#  ARRAY_PARTITION are the universal HLS pragma set most tools
#  implement identically). But pragma coverage can vary by Bambu
#  release, so run this once Bambu is installed and check the
#  build log for any "ignored/unsupported pragma" warnings before
#  trusting the synthesized result. Compatibility notes:
#  https://github.com/ferrandi/PandA-bambu/wiki/Accelerator-Interfaces
# ============================================================

bambu \
  src/ann_inference.cpp \
  --top-fname=ann_inference \
  --device=xczu7ev-ffvc1156-2-e \
  --clock-period=10 \
  --generate-interface=INFER \
  -v3 \
  --print-dot

# For a full csim-equivalent run using the local smoke test as the
# driver:
#
# bambu src/ann_inference.cpp src/local_smoke_test.cpp \
#   --top-fname=main --device=xczu7ev-ffvc1156-2-e --clock-period=10
