# Custom AI Hardware Acceleration Pipeline

A hardware-software co-design project targeting an end-to-end custom neural network accelerator—progressing from a software baseline to C++ High-Level Synthesis (HLS), FPGA prototyping, open-source EDA synthesis, and ASIC tapeout.

---
## Phase 1: C++ Hardware Pathfinder Neural Network

Part of the Custom AI Hardware Acceleration Pipeline. Phase 1 establishes a software-level, dependency-free C++ baseline model (MLP/ANN) trained on the MNIST handwritten digit benchmark dataset to prepare for High-Level Synthesis (HLS) and FPGA/ASIC implementation.

---

## 👥 Team Members

* **Member 1 (Nishita)
* **Member 2 (Sanskriti Dasila)
* **Member 3 (Pragya Baruah)

---

## 📁 Repository Directory Structure

```text
├── data/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── scripts/
│   └── prepare_mnist_gz.py      # Unpacks binary datasets into normalized CSV format
├── src/
│   └── main.cpp                  # Pure C++ ANN forward/backward pass & inference
├── .gitignore
└── README.md
