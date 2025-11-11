# Hybrid Python/C++ Wavelet Difference Reduction (WDR) Image Coder

[English](README.md) | [Español](README.es.md)

## Introduction

This project provides a complete implementation of the Wavelet Difference Reduction (WDR) algorithm for embedded image compression. WDR is an efficient image compression technique that combines discrete wavelet transforms with progressive transmission and adaptive arithmetic coding to achieve high compression ratios while supporting lossless reconstruction.

**Note**: This implementation is actively being optimized. Parameters such as quantization step size, number of passes, and compression strategies are still being tuned for optimal performance.

### Key Features

- **Embedded Bitstream**: Fully embedded compression allows stopping at any point to meet target bit rate or distortion
- **Progressive Transmission**: Images can be transmitted and displayed progressively with improving quality
- **Lossless Compression**: With sufficient passes, achieves lossless compression
- **Python/C++ Hybrid**: Python interface for ease of use, C++ core for performance
- **No Training Required**: Adaptive arithmetic coding requires no prior training

## Core Features

- **WDR Compression Algorithm**: Complete implementation of the Wavelet Difference Reduction algorithm
- **Adaptive Arithmetic Coding**: Implementation of the Witten-Neal-Cleary (1987) arithmetic coding algorithm
- **Python/C++ Hybrid Architecture**: Python for image I/O and DWT, C++ for compression core
- **Progressive Image Transmission Support**: Embedded bitstream supports progressive decoding
- **Lossless Compression Capability**: Can achieve lossless compression with sufficient passes

## How It Works

The WDR compression pipeline consists of three main stages:

1. **Discrete Wavelet Transform (DWT)**: Transforms the image into frequency domain using wavelets
2. **WDR Compression**: Encodes significant coefficients using differential coding, binary reduction, and bit-plane transmission
3. **Adaptive Arithmetic Coding**: Compresses the symbol stream using adaptive arithmetic coding

### Algorithm Overview

```
Input Image
    ↓
DWT (Discrete Wavelet Transform)
    ↓
Flatten Coefficients (WDR Scanning Order)
    ↓
WDR Compression
    ├─ Sorting Pass: Find & encode significant coefficients
    ├─ Refinement Pass: Refine existing coefficients
    └─ Adaptive Arithmetic Coding: Final compression
    ↓
.wdr File
```

### Scanning Order

The algorithm processes wavelet coefficients in a "coarse-to-fine" scanning order:

- **Order**: LL_N → HL_N → LH_N → HH_N → HL_{N-1} → ... → HH_1
- **HL subbands**: Scanned column-by-column (vertically)
- **LL, LH, HH subbands**: Scanned row-by-row (horizontally)

For a detailed theoretical explanation, see [docs/theory.md](docs/theory.md).

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CMake**: 3.15 or higher
- **C++ Compiler**: C++17 compatible compiler (GCC, Clang, or MSVC)
- **Python Packages**: NumPy, PyWavelets, Pillow, pybind11

### Step-by-Step Installation

1. **Clone the repository** (or download the source code)

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build and install the package**:
   ```bash
   pip install -e .
   ```

This will:
- Build the C++ extension module using CMake
- Compile the WDR compression core
- Install the Python package with the `wdr_coder` module

### Build from Source

If you need to build manually:

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build .

# The Python module will be built in the project root
```

### Troubleshooting

**Issue**: CMake not found
- **Solution**: Install CMake from https://cmake.org/download/

**Issue**: Python development headers not found
- **Solution**: Install Python development packages:
  - Ubuntu/Debian: `sudo apt-get install python3-dev`
  - macOS: Python development headers are included with Xcode command line tools
  - Windows: Install Python from python.org with development tools

**Issue**: NumPy not found during build
- **Solution**: Ensure NumPy is installed: `pip install numpy`

## Quick Start

### Python API

**Basic usage (without quantization)**:
```python
import numpy as np
import wdr_coder
import wdr_helpers as hlp

# Load image
original_img = hlp.load_image("input.png")

# Perform DWT (default: 2 scales, recommended: 2-3)
# Note: scales=6+ introduces boundary artifacts as warned by PyWavelets
wavelet_coeffs = hlp.do_dwt(original_img, scales=2, wavelet='bior4.4')

# Flatten coefficients
flat_coeffs, shape_data = hlp.flatten_coeffs(wavelet_coeffs)

# Compress (default: 26 passes for high precision)
wdr_coder.compress(flat_coeffs, "compressed.wdr", num_passes=26)

# Decompress
decompressed_flat_coeffs = wdr_coder.decompress("compressed.wdr")

# Unflatten coefficients
decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_data)

# Perform IDWT
reconstructed_img = hlp.do_idwt(decompressed_coeffs)
```

**With quantization (optional, for better compression)**:
```python
# Quantize coefficients before compression (optional)
# Quantization creates redundancy for better compression efficiency
quantization_step = hlp.calculate_quantization_step(flat_coeffs, num_passes=26, method='threshold_based')
quantized_coeffs, _ = hlp.quantize_coeffs(flat_coeffs, quantization_step)

# Compress quantized coefficients
wdr_coder.compress(quantized_coeffs, "compressed.wdr", num_passes=26)

# Decompress
decompressed_flat_coeffs = wdr_coder.decompress("compressed.wdr")

# Dequantize (if quantization was used)
dequantized_coeffs = hlp.dequantize_coeffs(decompressed_flat_coeffs, quantization_step)

# Unflatten and reconstruct
decompressed_coeffs = hlp.unflatten_coeffs(dequantized_coeffs, shape_data)
reconstructed_img = hlp.do_idwt(decompressed_coeffs)

# Save reconstructed image
hlp.save_image("reconstructed.png", reconstructed_img)
```

### Command-Line Usage

```bash
# Compress an image (basic usage)
python main.py input.png output.wdr

# Compress with custom settings
python main.py input.png output.wdr --scales 2 --wavelet bior4.4 --num-passes 16

# Compress with quantization (optional, for better compression)
python main.py input.png output.wdr --scales 2 --num-passes 16 --reconstructed recon.png

# Disable quantization explicitly
python main.py input.png output.wdr --quantization-step 0

# Use custom quantization step
python main.py input.png output.wdr --quantization-step 0.5
```

**Note**: 
- Default scales is 2 (2-3 recommended). Using scales=6+ introduces boundary artifacts as warned by PyWavelets.
- Quantization is optional but recommended for better compression ratios (1.5-2.5x typical).
- Set `--quantization-step 0` to disable quantization for lossless compression (may result in larger files).

## Project Structure

```
wdr_compression_pipeline/
├── src/                    # C++ source code
│   ├── arithmetic_coder.*  # Adaptive arithmetic coder (Witten-Neal-Cleary)
│   ├── adaptive_model.*    # Adaptive probability model
│   ├── bit_stream.*        # Bit-level I/O
│   ├── wdr_compressor.*    # WDR compression core
│   ├── wdr_file_format.*   # File format definitions
│   └── bindings.cpp        # Python bindings (pybind11)
├── wdr_helpers.py          # Python helper functions (DWT, I/O, flattening)
├── main.py                 # Example command-line script
├── tests/                  # Test files
│   ├── test_wdr_helpers.py # Python helper tests
│   ├── test_wdr_coder.py   # Integration tests
│   └── test_cpp/           # C++ unit tests (GTest)
├── docs/                   # Documentation
│   ├── theory.md           # Theoretical explanation
│   └── theory.es.md        # Theoretical explanation (Spanish)
├── CMakeLists.txt          # CMake build configuration
├── setup.py                # Python package setup
└── requirements.txt        # Python dependencies
```

### Key Files

- **`src/wdr_compressor.*`**: Core WDR compression algorithm implementation
- **`src/arithmetic_coder.*`**: Adaptive arithmetic coding implementation (Witten-Neal-Cleary 1987)
- **`wdr_helpers.py`**: Python functions for image I/O, DWT, and coefficient flattening
- **`main.py`**: Example script demonstrating the compression pipeline
- **`docs/theory.md`**: Comprehensive theoretical explanation of the algorithms

## Testing

### Python Tests

Run the Python test suite:

```bash
python -m pytest tests/
```

Or run specific test files:

```bash
python -m pytest tests/test_wdr_helpers.py
python -m pytest tests/test_wdr_coder.py
```

### C++ Tests

Build and run C++ unit tests:

```bash
cd build
cmake ..
cmake --build .
ctest
```

Or run tests with verbose output:

```bash
cd build
ctest --verbose
```

## Documentation

### Theory Documentation

For a comprehensive theoretical explanation of the WDR algorithm and adaptive arithmetic coding, see:

- **[docs/theory.md](docs/theory.md)**: Detailed algorithm explanation (English)
- **[docs/theory.es.md](docs/theory.es.md)**: Detailed algorithm explanation (Spanish)

### API Documentation

The Python API is documented in the source code. Key functions:

- **`wdr_coder.compress(coeffs, output_file, num_passes=26)`**: Compress coefficients to a .wdr file
- **`wdr_coder.decompress(input_file)`**: Decompress coefficients from a .wdr file
- **`wdr_helpers.load_image(filepath)`**: Load an image file
- **`wdr_helpers.save_image(filepath, img_array)`**: Save an image file
- **`wdr_helpers.do_dwt(img_array, scales=2, wavelet='bior4.4')`**: Perform DWT
- **`wdr_helpers.do_idwt(coeffs, wavelet='bior4.4')`**: Perform IDWT
- **`wdr_helpers.flatten_coeffs(coeffs)`**: Flatten wavelet coefficients
- **`wdr_helpers.unflatten_coeffs(flat_coeffs, shape_data)`**: Unflatten wavelet coefficients
- **`wdr_helpers.quantize_coeffs(coeffs, step_size)`**: Quantize coefficients (optional, for better compression)
- **`wdr_helpers.dequantize_coeffs(quantized_coeffs, step_size)`**: Dequantize coefficients
- **`wdr_helpers.calculate_quantization_step(coeffs, num_passes, method)`**: Calculate quantization step size (optional)

## Configuration & Optimization

### Compression Parameters

The WDR compression pipeline supports several configurable parameters that affect compression ratio, quality, and speed:

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `scales` | 2 | Number of wavelet decomposition levels | More scales = better compression but more artifacts |
| `wavelet` | 'bior4.4' | Wavelet basis function | Affects compression efficiency and quality |
| `num_passes` | 26 | Number of bit-plane passes | More passes = higher precision, larger file |
| `quantization_step` | auto | Quantization step size (optional) | Larger step = better compression, lower quality |
| `quantization_method` | 'threshold_based' | Quantization calculation method | Affects how quantization step is determined |

### Quantization (Optional)

Quantization is **optional** but highly recommended for compression efficiency. Without quantization, float64 coefficients have high precision with mostly unique values, preventing arithmetic coding from finding redundancy. This can result in compressed files larger than the original.

**With quantization**:
- Creates redundancy in coefficient values
- Enables arithmetic coding to achieve better compression
- Typical compression ratios: 1.5-2.5x
- Acceptable quality loss (PSNR > 40 dB typical)

**Without quantization**:
- Lossless compression (no quality loss)
- May result in compressed files larger than original
- Suitable for applications requiring perfect reconstruction

### Performance Optimization

**Note**: This implementation is actively being optimized. The following areas are under development:
- Quantization step size optimization
- Adaptive quantization strategies
- Compression ratio vs. quality trade-offs
- Performance optimizations for large images

## Credits & References

### Proper Attribution

This implementation is based on the following theoretical sources:

#### WDR Algorithm

This implementation is based on the Wavelet Difference Reduction algorithm for embedded image compression. The algorithm combines discrete wavelet transforms with efficient index coding and progressive transmission.

**Full Citation:**
[WDR paper citation - to be filled with actual paper details]

#### Adaptive Arithmetic Coding

The adaptive arithmetic coding implementation is based on the algorithm from:

**Witten, I.H., Neal, R.M., & Cleary, J.G. (1987).** "Arithmetic coding for data compression." *Communications of the ACM*, 30(6), 520-540.

This paper presents the adaptive arithmetic coding algorithm used in the final compression stage of WDR. The C++ implementation maintains mathematical equivalence to the original algorithm while using modern C++17 features.

### Implementation Statement

This implementation is based on the theoretical sources cited above. Full credit is given to the original authors and researchers who developed these algorithms. The code comments and documentation include proper attribution to ensure academic integrity and give credit where it is due.

**Status**: This implementation is actively being optimized. Parameters, quantization strategies, and performance optimizations are under continuous development.

### Additional Resources

- **PyWavelets**: Python library for discrete wavelet transforms (https://pywavelets.readthedocs.io/)
- **NumPy**: Numerical computing library for Python (https://numpy.org/)
- **Pillow**: Python Imaging Library for image I/O (https://pillow.readthedocs.io/)
- **pybind11**: Seamless operability between C++11 and Python (https://pybind11.readthedocs.io/)

## Contributing

Contributions are welcome! Please follow these guidelines:

### Code Style

- **C++**: Follow modern C++17 best practices, use meaningful variable names, add comments for complex logic
- **Python**: Follow PEP 8 style guide, use type hints where appropriate, add docstrings to all functions

### Testing

- Add tests for new features
- Ensure all tests pass before submitting
- Test with both Python and C++ test suites

### Documentation

- Update documentation for new features
- Add examples for new functionality
- Keep theoretical documentation accurate

## License

[License information to be added]

## Language Selection

- **[English](README.md)**: This document
- **[Español](README.es.md)**: Documentación en español

---

For detailed theoretical explanations, see [docs/theory.md](docs/theory.md).
