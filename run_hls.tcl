# ============================================================
#  run_hls.tcl — Vitis HLS project automation for ann_inference
#
#  Usage (from the AI-Accelerator/ project root):
#      vitis_hls -f run_hls.tcl
#
#  Confirms the kernel compiles (csim, using the placeholder
#  weights.h), synthesizes (csynth) cleanly with the required
#  INTERFACE/PIPELINE/UNROLL/ARRAY_PARTITION directives, and
#  RTL-cosimulates bit-exactly, targeting the ZCU104's actual part.
#
#  Paths below match the project layout: weights.h at the repo
#  root, kernel sources in src/.
# ============================================================

open_project ann_kernel_proj -reset
set_top ann_inference

# weights.h is #include'd by src/ann_inference.cpp (as "../weights.h"),
# so the preprocessor picks it up automatically — it's still added
# explicitly here so it shows up in the project's file list and gets
# version-tracked.
add_files src/ann_inference.cpp
add_files weights.h
add_files -tb src/local_smoke_test.cpp

open_solution "solution1" -flow_target vivado

# XCZU7EV-2FFVC1156, the exact device on the ZCU104 board
set_part {xczu7ev-ffvc1156-2-e}

# 10 ns = 100 MHz target clock. Conservative starting point;
# tighten later once real timing/area numbers are in.
create_clock -period 10 -name default

# 1) C simulation — confirms the kernel compiles & runs in software
csim_design

# 2) C synthesis — confirms the loops/pragmas synthesize to RTL
csynth_design

# 3) C/RTL co-simulation — confirms the synthesized Verilog matches
#    the C model bit-for-bit. Runs fine even against the placeholder
#    weights.h; swap in Pragya's real weights.h and Sanskriti's real
#    tb_main.cpp (replacing local_smoke_test.cpp above) once ready.
cosim_design

exit
