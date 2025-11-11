# WDR Compression Pipeline Tests

## Overview

This directory contains comprehensive tests for the WDR compression pipeline, including C++ unit tests (using Google Test) and Python integration tests.

## Test Structure

### C++ Unit Tests (`test_cpp/`)

- **test_adaptive_model.cpp**: Tests for AdaptiveModel class
  - Initial state
  - Frequency updates
  - Rescaling
  - Symbol re-sorting
  - Internal index mapping

- **test_bit_stream.cpp**: Tests for BitStream classes
  - Single bit read/write
  - Multiple bits read/write
  - Buffer flushing
  - EOF handling
  - Round-trip tests

- **test_arithmetic_coder.cpp**: Tests for ArithmeticCoder class
  - Mathematical equivalence with Witten-Neal-Cleary algorithm
  - Round-trip encoding/decoding
  - Adaptive model synchronization
  - Underflow handling

- **test_wdr_compressor.cpp**: Tests for WDRCompressor component functions
  - Initial threshold calculation
  - Binary reduction/expansion
  - Differential coding
  - Round-trip correctness

- **test_wdr_compressor_passes.cpp**: Tests for WDR pass logic
  - Sorting pass encoding/decoding
  - Refinement pass encoding/decoding
  - Full pass integration
  - State management

- **test_wdr_compressor_roundtrip.cpp**: Tests for full compression/decompression
  - Simple arrays
  - Edge cases
  - Multiple passes
  - Progressive precision

- **test_wdr_file_format.cpp**: Tests for file format
  - Header read/write
  - Magic number validation
  - Version validation
  - Round-trip correctness

### Python Integration Tests

- **test_wdr_helpers.py**: Tests for Python helper functions
  - Image I/O
  - DWT/IDWT
  - Flatten/unflatten coefficients
  - Round-trip correctness

- **test_wdr_coder.py**: Tests for full pipeline
  - Compression/decompression round-trip
  - Edge cases
  - Full image pipeline
  - Golden file testing

## Running Tests

### C++ Tests

```bash
# Build tests
mkdir -p build
cd build
cmake ..
make

# Run all C++ tests
ctest

# Run specific test
./tests/test_cpp/test_adaptive_model
./tests/test_cpp/test_bit_stream
./tests/test_cpp/test_arithmetic_coder
./tests/test_cpp/test_wdr_compressor
./tests/test_cpp/test_wdr_compressor_passes
./tests/test_cpp/test_wdr_compressor_roundtrip
./tests/test_cpp/test_wdr_file_format
```

### Python Tests

```bash
# Run all Python tests
pytest tests/ -v

# Run specific test file
pytest tests/test_wdr_helpers.py -v
pytest tests/test_wdr_coder.py -v

# Run with coverage
pytest tests/ --cov=wdr_helpers --cov=wdr_coder
```

## Test Data

Test data files are located in `tests_data/`:

- **lenna_small.png**: Test image for integration tests
- **test_pattern.png**: Simple test pattern with vertical and horizontal lines
- **golden_recon.png**: Golden file for regression testing (generated on first successful run)

## Golden File Testing

The golden file test (`test_golden_file`) works as follows:

1. On first run, it generates `recon_output.png` from the test image
2. The developer manually inspects `recon_output.png` to verify correctness
3. If correct, the developer renames it to `golden_recon.png`
4. On subsequent runs, the test compares newly generated output with the golden file
5. If they don't match, the test fails (indicating a regression)

## Expected Test Outcomes

### C++ Tests

All C++ unit tests should pass with 100% success rate. These tests verify:
- Mathematical correctness of algorithms
- Round-trip fidelity
- Edge case handling
- State management

### Python Tests

All Python integration tests should pass. These tests verify:
- Full pipeline correctness
- Image reconstruction quality
- File I/O correctness
- Integration between Python and C++ components

## Troubleshooting

### C++ Tests Fail to Build

- Verify CMake can find Google Test
- Check that C++17 compiler is available
- Ensure all source files compile without errors

### Python Tests Fail to Import wdr_coder

- Build the C++ module first: `pip install -e .`
- Verify the module is in Python path
- Check that pybind11 bindings are correct

### Round-Trip Tests Fail

- Check that decompression logic is complete
- Verify adaptive models are synchronized
- Check file format encoding/decoding
- Verify ICS state management

### Golden File Test Fails

- If first run: Manually verify `recon_output.png` and rename to `golden_recon.png`
- If subsequent run: Check for algorithm changes or bugs
- Verify test image hasn't changed

## Test Coverage Goals

- **Unit Test Coverage**: > 90% for C++ code
- **Integration Test Coverage**: All major code paths
- **Edge Case Coverage**: All identified edge cases
- **Round-Trip Fidelity**: 100% for test cases (within float precision)

## Adding New Tests

When adding new tests:

1. **C++ Tests**: Add to appropriate test file in `tests/test_cpp/`
2. **Python Tests**: Add to appropriate test file in `tests/`
3. **Update CMakeLists.txt**: Add new test executable if needed
4. **Update this README**: Document new tests

## Test Maintenance

- Run tests regularly during development
- Fix failing tests immediately
- Update golden files when algorithm changes are intentional
- Keep test data files in version control
- Document any test-specific behavior or limitations

