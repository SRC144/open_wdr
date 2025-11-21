# Hybrid Python/C++ Wavelet Difference Reduction (WDR) Image Coder

[English](README.md) | [Español](README.es.md)

## Overview

Wavelet Difference Reduction (WDR) combines discrete wavelet transforms, progressive bit-plane coding, and adaptive arithmetic coding to deliver high compression ratios with optional lossless reconstruction. The Python surface API keeps workflows simple, while the C++ core (built with pybind11 + CMake) provides performance.

## Install & Build (All Platforms)

1. **Clone**
   ```bash
   git clone <repo>
   cd wdr_compression_pipeline
   ```
2. **(Optional) Create virtual environment**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate  # Linux/macOS
   python -m venv .venv && .venv\Scripts\activate      # Windows
   ```
3. **Install tools**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
4. **Build + install**
   ```bash
   pip install -e .
   ```

`pip install -e .` configures CMake, builds the native `wdr.coder` module, and exposes it as a Python package. If the build fails, confirm you have Python ≥3.8, CMake ≥3.15, and a C++17 compiler. Detailed diagnostics, platform notes, and alternative setups live in `TROUBLESHOOTING.md`.

## Sample Assets & Outputs

- Source fixtures live in `assets/` (e.g., `assets/lenna.png`, `assets/pattern.png`) for quick demos.
- Generated artifacts go under `compressed/` (CLI runs) and `compressed/tests/` (test suite). When you pass only a filename to `main.py`, it automatically saves the `.wdr` and reconstructed image inside `compressed/`.

## Development & Testing (optional)

### Python tests

```bash
pip install pytest
python -m pytest tests/
```

### Native build + C++ tests

```bash
pip install pybind11
PYBIND11_DIR=$(python3 -c 'import pybind11, pathlib; print(pathlib.Path(pybind11.get_cmake_dir()))')
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build -j && ctest --test-dir build --output-on-failure
```

Need to inspect the native build output? Re-run the first line without `-DBUILD_TESTING=ON` to build only the extension, or add `-DCMAKE_VERBOSE_MAKEFILE=ON` for more logs. The compiled module lands inside the package as `wdr/coder.cpython-<ver>-<platform>.so|pyd`.

For a quick map of what each suite validates (and how to extend them), see `tests/README.md`.

## Example Usage

### Python API

```python
from wdr import coder as wdr_coder
from wdr.utils import helpers as hlp

img = hlp.load_image("input.png")
coeffs = hlp.do_dwt(img, scales=2, wavelet="bior4.4")
flat_coeffs, shape_data = hlp.flatten_coeffs(coeffs)

wdr_coder.compress(flat_coeffs, "output.wdr", num_passes=26)
decoded = wdr_coder.decompress("output.wdr")
unflat = hlp.unflatten_coeffs(decoded, shape_data)
reconstructed = hlp.do_idwt(unflat)
hlp.save_image("reconstructed.png", reconstructed)
```

Quantization helpers (`calculate_quantization_step`, `quantize_coeffs`, `dequantize_coeffs`) live in `wdr.utils.helpers` and remain optional—set `--quantization-step 0` (or skip quantization) for lossless workflows.

### CLI

```bash
python main.py input.png output.wdr \
  --scales 2 \
  --wavelet bior4.4 \
  --num-passes 26 \
  --reconstructed recon.png  # optional file for decoded image
```

If you omit directory components in `output.wdr` or `--reconstructed`, the script writes them into `compressed/` automatically; provide full paths when you want a different destination.

### Benchmarking

`main.py` prints two compression ratios:
- **Algorithm CR**: bytes(coeff array) / bytes(.wdr)
- **True System CR**: bytes(raw pixels) / bytes(.wdr)

Quantization (optional) can substantially improve both; disable with `--quantization-step 0` for strictly lossless flows.

## Documentation

- `TROUBLESHOOTING.md`: platform notes, clean-build recipes, Docker/Conda tips.
- `tests/README.md`: explains what every Python/C++ test covers and how to extend them.
- `README.es.md`: Spanish version of this quick guide.

Enjoy the pipeline—and open an issue if anything in these quick steps goes stale.
