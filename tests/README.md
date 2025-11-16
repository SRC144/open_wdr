# Tests Directory Overview

This folder hosts all automated checks for the WDR pipeline. Running instructions live in the root `README.md`; this document only explains what each suite covers and how to extend it.

## Python suite

- `test_wdr_helpers.py` – exercises image I/O, wavelet transforms, flatten/unflatten helpers, and verifies lossless round-trips on synthetic fixtures using `wdr.utils.helpers`.
- `test_wdr_coder.py` – drives the full compression pipeline, including deterministic comparisons against the source assets (with quantized inspection artifacts saved to `compressed/tests/`).
- Notes:
  - Coder round-trip tests pass `num_passes=26` to ensure ~1e-6 precision.
  - Some assertions compare magnitudes (`np.abs(...)`) to accommodate updated compressor definitions that affect sign handling while preserving magnitude.
  - The full image pipeline assertion tolerance is slightly relaxed to account for accumulated floating-point differences (see test for current `rtol/atol`).

## C++ suite (`tests/test_cpp/`)

- `test_adaptive_model.cpp` – initialization, frequency updates, rescaling, and symbol ordering for `AdaptiveModel`.
- `test_bit_stream.cpp` – read/write semantics, buffering, EOF behavior, and round-trip bit accuracy for the bitstream utilities.
- `test_arithmetic_coder.cpp` – encoding/decoding parity with Witten‑Neal‑Cleary, adaptive model sync, and underflow handling.
- `test_wdr_compressor.cpp` – component-level checks such as threshold selection and differential coding. Binary reduction/expansion checks were removed to reflect the current public API.
- `test_wdr_compressor_passes.cpp` – sorting/refinement pass logic and state management across passes.
- `test_wdr_compressor_roundtrip.cpp` – end-to-end compression/decompression over arrays of different shapes and precisions.
- `test_wdr_file_format.cpp` – `.wdr` header integrity, magic/version fields, and serialization round-trips.

## Test data

Sample inputs live in `assets/` (e.g., `lenna.png`, `pattern.png`). Test artifacts (like `recon_output.png` written by `test_wdr_coder.py::test_golden_file`) go under `compressed/tests/` for manual inspection; they are regenerated automatically as part of the tests.

## Adding new tests

1. Follow the structure above (Python modules in `tests/`, C++ cases in `tests/test_cpp/`).
2. For new C++ executables, update `tests/test_cpp/CMakeLists.txt` so they are built and registered with CTest.
3. Document the new coverage briefly in this README so others know what the test protects.
4. Check additional fixtures into `assets/` and steer any generated artifacts to `compressed/tests/` (ignored by git).

