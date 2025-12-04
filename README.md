# Open WDR: A hybrid Python/C++ Wavelet Difference Reduction (WDR) Image Coder

[English](README.md) | [Español](README.es.md)

## Overview

Wavelet Difference Reduction (WDR) combines discrete wavelet transforms, progressive bit-plane coding, and adaptive arithmetic coding to deliver high compression ratios with optional lossless reconstruction. The Python surface API keeps workflows simple, while the C++ core (built with pybind11 + CMake) provides performance.

The library uses a tiled architecture for memory-efficient processing of large and gigapixel images. Images are processed in fixed-size tiles (default 512×512), enabling compression of images larger than available RAM while maintaining consistent quality across tile boundaries through a global threshold mechanism.

## Install & Build (All Platforms)

1. **Clone**
   ```bash
   git clone <repo>
   cd wdr_compression_pipeline
   ```
2. **(Optional) Create virtual environment**
   ```bash
   # Linux/macOS
   python3 -m venv .venv && source .venv/bin/activate  

   # Windows
   python -m venv .venv 
   .venv\Scripts\activate      
   ```
3. **Install tools**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
   This includes OpenSlide bindings (`openslide-python` + `openslide-bin`) for Whole Slide Image format support (NDPI, SVS, Philips TIFF).

4. **Build + install**
   ```bash
   pip install -e .
   ```

`pip install -e .` configures CMake, builds the native `wdr.coder` module, and exposes it as a Python package. If the build fails, confirm you have Python ≥3.8, CMake ≥3.15, and a C++17 compiler. Detailed diagnostics, platform notes, and alternative setups live in `TROUBLESHOOTING.md`.


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

## Core API

### `wdr.io.compress()`

Compresses a 2D image array into a `.wdr` archive file using a tiled approach. The function tiles the image, applies DWT per tile, compresses each tile with WDR, and writes to archive, processing tiles sequentially to maintain O(1) memory usage. This enables compression of gigapixel images without loading the entire image coefficients into RAM (Billions for gigapixel images). The global threshold parameter ensures consistent bit-plane alignment across tiles, preventing visual artifacts that could occur if each tile used a different quantization level.

**Parameters:**
- `image_source`: 2D numpy array (single channel, dtype float64)
- `output_path`: Destination `.wdr` file path
- `global_T`: Global threshold calculated from the entire image, ensuring bit-plane alignment across tiles
- `tile_size`: Size of compression tiles (default 512)
- `scales`: DWT decomposition levels
- `wavelet`: Wavelet type (default `'bior4.4'`)
- `num_passes`: Number of bit-plane passes (quality/size tradeoff)
- `quant_step`: Optional quantization step size (None or 0 for lossless)

**Returns:** None (writes `.wdr` archive file to disk)

### `wdr.io.decompress()`

Streams decompressed tiles from a `.wdr` archive, returning a generator that yields 2D numpy arrays. The function reads archive metadata, decompresses tiles one at a time, applies inverse DWT, and yields tiles in row-major order. This memory-efficient approach holds only one tile in RAM at a time, enabling reconstruction of images larger than available memory.

**Parameters:**
- `wdr_path`: Path to `.wdr` file
- `progress_callback`: Optional callback function accepting float (0.0-1.0) for progress tracking (e.g Progress indicators on CLI)

**Returns:** Generator yielding 2D numpy arrays (reconstructed tiles, dtype float64)

### Key Helpers (`wdr.utils.helpers`)

- **`scan_for_max_coefficient()`**: Scans all tiles to find the global maximum coefficient value. Required before compression to calculate `global_T`.
- **`calculate_global_T()`**: Computes global threshold from maximum coefficient. Ensures all tiles use the same bit-plane alignment during compression.
- **`do_dwt()` / `do_idwt()`**: Forward and inverse discrete wavelet transform. Converts image tiles to/from wavelet coefficients.
- **`flatten_coeffs()` / `unflatten_coeffs()`**: Converts DWT coefficient tuples (multi-level subbands) to/from flat arrays for WDR encoding.
- **`quantize_coeffs()` / `dequantize_coeffs()`**: Optional quantization for lossy compression. Set `quant_step=0` for lossless workflows.
- **`yield_tiles()`**: Generator that yields image tiles with edge padding. Handles both RAM arrays and memory-mapped files.

## Tiling Architecture

The library processes images in fixed-size tiles (default 512×512) rather than processing the entire image at once. This approach provides several benefits:

1. **Memory Efficiency**: Only one tile is processed at a time, maintaining O(1) memory usage regardless of image size. This enables handling of gigapixel images that exceed available RAM.

2. **Global Threshold**: All tiles share the same global threshold (`global_T`) calculated from the entire image. This ensures bit-plane alignment across tile boundaries, preventing visual artifacts that could occur if each tile used a different quantization level.

3. **Edge Handling**: Edge tiles are padded to maintain consistent tile size, with padding removed during reconstruction. The padding uses edge replication to minimize boundary artifacts.

4. **Streaming**: Both compression and decompression stream tiles sequentially, making the library suitable for disk-backed images (e.g., memory-mapped TIFF files) and large datasets.

## Example Usage

### Python API

Complete workflow showing compression and decompression with tile reassembly:

```python
import numpy as np
from PIL import Image
from wdr import io as wdr_io
from wdr.utils import helpers as hlp

# Load image (must be single channel)
img = np.array(Image.open("input.png").convert("L"), dtype=np.float64)
height, width = img.shape

# Step 1: Calculate global threshold
# This scans all tiles to find the maximum coefficient, ensuring consistent
# bit-plane alignment across tiles during compression.
global_max = hlp.scan_for_max_coefficient(img, tile_size=512, scales=2, wavelet="bior4.4")
global_T = hlp.calculate_global_T(global_max)

# Step 2: Compress (tiles processed internally, streamed to disk)
wdr_io.compress(
    image_source=img,
    output_path="output.wdr",
    global_T=global_T,
    tile_size=512,
    scales=2,
    wavelet="bior4.4",
    num_passes=16,
    quant_step=0  # 0 = lossless
)

# Step 3: Decompress and reassemble tiles
tiles = wdr_io.decompress("output.wdr")
reconstructed = np.zeros((height, width), dtype=np.float64)

tile_size = 512
tile_idx = 0
for r in range((height + tile_size - 1) // tile_size):
    for c in range((width + tile_size - 1) // tile_size):
        tile = next(tiles)
        
        # Calculate valid region (crop edge padding)
        y_start = r * tile_size
        x_start = c * tile_size
        y_end = min(y_start + tile_size, height)
        x_end = min(x_start + tile_size, width)
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        # Place tile in reconstructed image
        reconstructed[y_start:y_end, x_start:x_end] = tile[:tile_h, :tile_w]
        tile_idx += 1

# Save reconstructed image
Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8)).save("reconstructed.png")
```

### Example Application: Medical Whole Slide Imaging

The `scripts/` directory includes a complete implementation of the Color Wavelet Difference Reduction (CWDR) based workflow for medical imaging, as described in Zerva et al. (2023). This example demonstrates how to apply the WDR library to whole slide images (WSI) from pathology scanners (models supported by openslide [https://openslide.org/formats/]), handling RGB-to-YUV colorspace conversion, independent WDR compression per channel (Y, U, V), reconstruction, and quality evaluation. The tools integrate OpenSlide for reading proprietary formats (NDPI, SVS, Philips TIFF).

#### WSI Compression/Extraction Pipeline

`scripts/wdr_wsi_pipeline.py` is the primary tool for compressing and extracting whole slide images. It handles format detection, streaming tile extraction via OpenSlide, automatic YCbCr colorspace conversion, and cleanup of intermediate files.

```bash
# Compress a whole slide image
python scripts/wdr_wsi_pipeline.py compress CMU-1.svs cmu1 --tile-size 512 --scales 2 --wavelet bior4.4 --passes 16

# Extract back to RGB BigTIFF
python scripts/wdr_wsi_pipeline.py extract results/ cmu1 reconstructed.tiff
```

The pipeline produces three channel files (`_Y.wdr`, `_U.wdr`, `_V.wdr`) for efficient storage. Use `--keep-temp` to preserve intermediate TIFF files for debugging.

#### Slide Metadata Inspection

`scripts/wsi_info.py` quickly inspects slide metadata without loading the full image. Useful for verifying dimensions and format before processing.

```bash
python scripts/wsi_info.py CMU-1.svs
```

#### Quality Metrics Evaluation

`scripts/wsi_metrics.py` calculates PSNR and SSIM between reconstructed and original slides using streaming processing to avoid memory exhaustion on gigapixel images.

```bash
python scripts/wsi_metrics.py reconstructed.tiff CMU-1.svs --tile-size 2048
```

These tools serve as reference implementations showing practical WSI workflows. Adapt them for your specific needs or build custom pipelines using the core API directly.

## Reference

This implementation is based on:

**Wavelet Difference Reduction (WDR) Algorithm:**  
Tian, J., Wells, R.O. (2002). Embedded Image Coding Using Wavelet Difference Reduction. In: Topiwala, P.N. (eds) Wavelet Image and Video Compression. The International Series in Engineering and Computer Science, vol 450. Springer, Boston, MA. https://doi.org/10.1007/0-306-47043-8_17

**Color WDR (CWDR) for Medical Imaging:**  
Zerva, M.C.H., Christou, V., Giannakeas, N., Tzallas, A.T., & Kondi, L.P. (2023). "An Improved Medical Image Compression Method Based on Wavelet Difference Reduction." IEEE Access, vol. 11, pp. 18026-18037. https://doi.org/10.1109/ACCESS.2023.3246948

**Adaptive Arithmetic Coding:**  
Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding for data compression." Communications of the ACM, 30(6), 520-540.

## Documentation

- `TROUBLESHOOTING.md`: platform notes, clean-build recipes, Docker/Conda tips.
- `tests/README.md`: explains what every Python/C++ test covers and how to extend them.
- `README.es.md`: Spanish version of this quick guide.
