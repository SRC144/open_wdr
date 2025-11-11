# Project Plan: Hybrid Python/C++ WDR Image Coder

## 🧑‍💻 Developer Guidelines

  * **Code Authority:** You must act as an expert C++ and Python developer, adhering to modern best practices.
  * **Modernization:** The C code provided in `aac_algorithm.md` is from 1987. You must **not** reproduce it literally. You will **modernize** it into a clean, encapsulated C++17 class.
  * **Mathematical Equivalence:** The core logic (e.g., integer-based interval narrowing, bit-shifting, and underflow handling) must remain **mathematically equivalent** to the original paper (as summarized in `aac_algorithm.md`) to ensure the compressed output is identical.
  * **Best Practices:** Use modern C++ features:
      * Proper class encapsulation (e.g., `ArithmeticCoder`, `WDRCompressor`).
      * Standard library containers (e.g., `std::vector`, `std::string`).
      * Clear, well-named variables and functions.
      * Avoid raw C-style pointers and manual memory management where `std::vector` or smart pointers are appropriate.

## 🎯 Project Goal

Create a functional, modular, hybrid Python/C++ image compression pipeline implementing the Wavelet Difference Reduction (WDR) algorithm.

The pipeline will use Python for high-level tasks (image I/O, wavelet transform orchestration) and a compiled C++ module for the low-level, computationally-intensive compression and decompression.

## 🏛️ Architecture Overview

1.  **Python Layer (Frontend):**

      * Loads the image.
      * Performs the 2D Discrete Wavelet Transform (DWT) using `PyWavelets`.
      * **"Flattens"** the 2D coefficient matrices into a single 1D array, using the specific WDR scanning order (specified in `wdr_algorithm.md`).
      * Passes the 1D array and an output file path to the C++ module for compression.
      * Receives a 1D decompressed array from the C++ module by providing an input file path.
      * **"Un-flattens"** the 1D array back into the 2D DWT matrices.
      * Performs the 2D Inverse DWT (IDWT) to reconstruct the image.

2.  **C++ Layer (Backend Module):**

      * Compiled as a Python module (e.g., using `pybind11`).
      * Implements the **WDR compression algorithm** (Sorting Pass, Refinement Pass).
      * Implements the **Witten-Neal-Cleary Adaptive Arithmetic Coder** (from C source).
      * Exposes a `compress()` function that takes the coefficient array and an output filepath, writing the compressed data (including `T` and the `bitstream`) to a **`.wdr`** file.
      * Exposes a `decompress()` function that takes an input **`.wdr`** filepath, reads the data, and returns the 1D reconstructed coefficient array.

-----

## 📚 Context (To be provided)

This plan relies on two external context documents:

  * `wdr_algorithm.md`: A detailed step-by-step summary of the WDR algorithm logic (passes, lists, bitstream generation).
  * `aac_algorithm.md`: The C source code and summary for the Witten-Neal-Cleary adaptive arithmetic coder.

-----

## 🛠️ Build & Environment Setup

  * **C++:** A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
  * **Build System:** `CMake` will be used to manage the C++ build process.
  * **Bindings:** `pybind11` (to be included as a `git submodule` or fetched by `CMake`).
  * **Python:** Python 3.8+
  * **Python Dependencies:** A `requirements.txt` file will specify:
      * `numpy` (for data-passing and array manipulation)
      * `pywavelets` (for DWT/IDWT)
      * `Pillow` (for image I/O)
      * `pybind11` (for building)
  * **Reproducibility:** The entire project will be buildable as a Python package using a `setup.py` or `pyproject.toml` file that invokes `CMake`. This allows for an editable install (`pip install -e .`) for development.

-----

## 📋 Task Breakdown

### 🐍 Python Workflow (main.py & `wdr_helpers.py`)

#### Task 1: Python Helper Library (`wdr_helpers.py`)

  * Create a file `wdr_helpers.py` that will contain the Python-side logic for the library.
  * **`load_image(filepath)`:** Uses `Pillow` to load an image, **assuming it is single-channel**, and returns a NumPy array.
  * **`save_image(filepath, img_array)`:** Uses `Pillow` to save a NumPy array as an image file.
  * **`do_dwt(img_array, scales)`:** Uses `pywt.wavedec2` to perform the DWT (e.g., with 'bior9/7' filter) and returns the `pywt` coefficient structure.
  * **`flatten_coeffs(coeffs)`:**
      * Iterates through subbands in the **WDR coarse-to-fine order** ($LL_N \rightarrow HL_N \rightarrow LH_N \rightarrow HH_N \rightarrow \dots \rightarrow HH_1$) as defined in `wdr_algorithm.md`.
      * Flattens each subband using the **WDR scanning heuristic** (row-by-row, except for **column-by-column** for $HL$ bands).
      * Appends all coefficients to a single 1D NumPy array.
      * Returns the `flat_array` and a `shape_metadata` object (e.g., a list of `(subband_name, shape)`) needed for reversal.
  * **`unflatten_coeffs(flat_array, shape_metadata)`:**
      * The exact inverse of `flatten_coeffs`. It uses the `shape_metadata` to read from the `flat_array` and reconstruct the `pywt` coefficient structure, applying the same scanning heuristic in reverse.
  * **`do_idwt(coeffs)`:** Uses `pywt.waverec2` to perform the inverse DWT and return the reconstructed image array.

#### Task 2: C++ Module Interaction (Python Side)

  * The Python code will import the compiled C++ module (e.g., `import wdr_coder`).
  * **Compression Call:** `wdr_coder.compress(flat_coeffs, "output_image.wdr")`
  * **Decompression Call:** `reconstructed_flat_coeffs = wdr_coder.decompress("output_image.wdr")`

#### Task 3: Example Usage (`main.py`)

  * Create an example script `main.py` that demonstrates the full pipeline as shown in the **"Final Module API & Python Usage"** section.
  * This script will show both use cases:
    1.  **Visualization:** Compressing and decompressing an image, then saving the result.
    2.  **Metrics:** (Placeholder) Show where one would compare the original `img_array` with the `reconstructed_img` to calculate metrics (the metric functions themselves are out of scope).

-----

### 🚀 C++ Module Workflow (wdr\_cpp\_module)

#### Task 4: C++ Bindings & Public API (pybind11)

  * Set up the `CMake` build system to compile the C++ code into a Python module.
  * Create the `pybind11` binding file. This file defines the **public Python-facing API**.
  * Define a `compress` function that accepts a NumPy array (`py::array_t`) and a `std::string` (for the output filepath).
  * Define a `decompress` function that accepts a `std::string` (for the input filepath) and returns a NumPy array (`py::array_t`).

#### Task 5: Adaptive Arithmetic Coder

  * Implement a modern C++ `ArithmeticCoder` class based on the logic in `aac_algorithm.md`.
  * **Adhere to the Developer Guidelines** for modernization (encapsulation, C++17).
  * The class must encapsulate the state (`low`, `high`, `bits_to_follow`) and logic (`encode_symbol`, `decode_symbol`, etc.).
  * It should be designed to work with `std::ostream` (for encoding) and `std::istream` (for decoding), rather than handling file I/O directly.
  * **Include a comment header** in the file(s) attributing the code to its original authors (Witten, Neal, Cleary, ACM 1987).

#### Task 6: WDR Pipeline

  * Implement the full WDR algorithm (Sorting Pass, Refinement Pass, list management) in C++, as detailed in `wdr_algorithm.md`.
  * This will be the core logic, likely in a `WDRCompressor` class.
  * The `compress` function (from Task 4) will orchestrate this, calling the `ArithmeticCoder` to output bits to a file stream.
  * The `decompress` function (from Task 4) will run the inverse logic, calling the `ArithmeticCoder` to read bits from a file stream.

#### Task 7: `.wdr` File Handling

  * Define a simple binary file format for `.wdr`.
  * **On Compress:**
    1.  Open an output file stream (`std::ofstream`) to the specified filepath.
    2.  Write `Initial_T` to the file (e.g., as a `uint32_t`).
    3.  Pass the `ofstream` to the `ArithmeticCoder`, which will write the compressed `bitstream` byte by byte.
  * **On Decompress:**
    1.  Open an input file stream (`std::ifstream`) from the specified filepath.
    2.  Read `Initial_T` from the file.
    3.  Pass the `ifstream` to the `ArithmeticCoder`, which will read the bitstream byte by byte to decode symbols.

-----

## 📦 Final Module API & Python Usage

The final product will be a standard Python package that can be installed via `pip`. The user will interact with it as a library.

### Python-Side Helpers (Provided by `wdr_helpers.py`):

  * `load_image(filepath) -> np.array`
  * `save_image(filepath, img_array)`
  * `do_dwt(img_array, scales) -> coeffs`
  * `do_idwt(coeffs) -> img_array`
  * `flatten_coeffs(coeffs) -> (flat_array, shape_metadata)`
  * `unflatten_coeffs(flat_array, shape_metadata) -> coeffs`

### C++ Module API (The compiled `wdr_coder` library):

  * `wdr_coder.compress(flat_array, output_filepath)`
  * `wdr_coder.decompress(input_filepath) -> flat_array`

### Example Workflow (Contents of `main.py`):

```python
import numpy as np
import wdr_coder
import wdr_helpers as hlp

# --- Define Helper for Metrics (Out of scope, but shows use case) ---
def calculate_psnr(original, compressed):
    # (Implementation for PSNR)
    pass

# --- 1. COMPRESSION ---
print("Compressing image...")
original_img = hlp.load_image("my_image.png")
wavelet_coeffs = hlp.do_dwt(original_img, scales=6)
flat_coeffs, shape_data = hlp.flatten_coeffs(wavelet_coeffs)

wdr_coder.compress(flat_coeffs, "compressed_file.wdr")
print("Compression complete: compressed_file.wdr")

# --- 2. DECOMPRESSION ---
print("Decompressing image...")
decompressed_flat_coeffs = wdr_coder.decompress("compressed_file.wdr")
decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_data)
reconstructed_img = hlp.do_idwt(decompressed_coeffs)

hlp.save_image("reconstructed.png", reconstructed_img)
print("Reconstruction complete: reconstructed.png")

# --- 3. (Optional) METRICS ---
# psnr = calculate_psnr(original_img, reconstructed_img)
# print(f"PSNR: {psnr} dB")
```