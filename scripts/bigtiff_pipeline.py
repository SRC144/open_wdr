#!/usr/bin/env python3
"""
Gigapixel TIFF WDR Workflow Tool

Command-line tool for compressing and extracting large TIFF images using the WDR library.
Designed for memory-efficient processing of gigapixel images using memory-mapped file access.

COMMANDS:

compress
    Compress a TIFF image to WDR format.

    Usage:
        python scripts/converters.py compress <input.tiff> <output.wdr> [OPTIONS]

    Arguments:
        input.tiff          Input TIFF image file (single channel)

    Options:
        --tile-size N       Tile size for processing (default: 512)
        --scales N          Number of DWT decomposition levels (default: 2)
        --wavelet NAME      Wavelet type from pywt (default: bior4.4)
        --passes N          Number of bit-plane passes (default: 16)
                            Higher values improve quality but increase file size
        --qstep N           Quantization step size (default: None)
                            Use 0 or omit for lossless compression
        --quiet             Suppress progress and statistics output

    Output:
        - Creates output.wdr archive file
        - Prints compression statistics (raw size, compressed size, ratio, time)

extract
    Extract a WDR archive to TIFF format.

    Usage:
        python scripts/converters.py extract <input.wdr> <output.tiff> [OPTIONS]

    Arguments:
        input.wdr           WDR archive file to decompress

    Options:
        --original-image PATH   Path to original image for PSNR calculation
                                If provided, calculates and displays MSE and PSNR metrics
        --quiet                 Suppress progress and statistics output

    Output:
        - Creates output.tiff image file
        - Prints decompression statistics (time, optional PSNR metrics)

FEATURES:
    - Memory-efficient streaming for gigapixel images
    - Compression ratio (CR) metrics
    - PSNR calculation for quality assessment
    - Progress reporting
    - Supports memory-mapped TIFF files

EXAMPLES:
    # Lossless compression
    python scripts/converters.py compress image.tiff compressed.wdr --passes 16 --qstep 0

    # Lossy compression with custom settings
    python scripts/converters.py compress image.tiff compressed.wdr --passes 20 --qstep 2.5

    # Extract without metrics
    python scripts/converters.py extract compressed.wdr reconstructed.tiff

    # Extract with PSNR calculation
    python scripts/converters.py extract compressed.wdr reconstructed.tiff --original-image image.tiff
"""
import argparse
import time
import math
import os
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from wdr import io as wdr_io
from wdr.utils import helpers as hlp
from wdr.container import WDRTileReader

# --- Metrics ---
class StreamMetrics:
    """
    Accumulates error statistics (SSE) for streaming image comparison.
    Memory Usage: O(1). Math is optimized for in-place operations.
    """
    def __init__(self, max_val=255.0):
        self.sse = 0.0
        self.total_pixels = 0
        self.max_val = max_val

    def update(self, original_tile: np.ndarray, reconstructed_tile: np.ndarray):
        # 1. Handle Edge Padding (Crop to match smallest)
        if original_tile.shape != reconstructed_tile.shape:
            h = min(original_tile.shape[0], reconstructed_tile.shape[0])
            w = min(original_tile.shape[1], reconstructed_tile.shape[1])
            # Views (no copy)
            original_tile = original_tile[:h, :w]
            reconstructed_tile = reconstructed_tile[:h, :w]

        # 2. Memory Optimized Calculation
        # We cast 'original' to float64 creating one new array
        diff = original_tile.astype(np.float64)
        
        # Subtract in-place 
        diff -= reconstructed_tile 
        
        # Square in-place
        np.square(diff, out=diff)
        
        # Accumulate
        self.sse += np.sum(diff)
        self.total_pixels += original_tile.size

    def get_results(self):
        if self.total_pixels == 0: return 0.0, float('inf')
        
        mse = self.sse / self.total_pixels
        if mse == 0: return 0.0, float('inf')
            
        psnr = 20 * math.log10(self.max_val / math.sqrt(mse))
        return mse, psnr

def get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0

def format_size(size_bytes: int) -> str:
    if size_bytes == 0: return "0 B"
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {suffixes[i]}"

# --- Image Loading ---
# Increase PIL limit just in case we fall back to it for a medium-sized image

Image.MAX_IMAGE_PIXELS = None 

def load_image(filepath: str) -> np.ndarray:
    """
    Strict Image Loader.
    Loads image but enforces Single-Channel (2D) requirement.
    Raises ValueError if image is Multi-Channel (RGB/3D).
    """
    path = Path(filepath)
    img_array = None
    
    # 1. Load Strategy (Memmap or PIL)
    if path.suffix.lower() in ['.tif', '.tiff', '.btf']:
        try:
            # Load image as memory-mapped array
            # This is necessary for large images since
            # it doesn't load the entire image into memory at once.
            img_array = tifffile.memmap(filepath, mode='r')
        except Exception:
            pass

    if img_array is None:
        img = Image.open(filepath)
        img_array = np.array(img)

    # 2. Strict Validation (Fail Fast)
    if img_array.ndim != 2:
        raise ValueError(
            f"WDR Library Error: Image '{filepath}' is not 2D ({img_array.shape}).\n"
            "The WDR core strictly requires single-channel input.\n"
            "Please handle color conversion or channel splitting in your application logic."
        )
        
    return img_array

def save_image(filepath: str, img_array: np.ndarray) -> None:
    img_array = np.clip(img_array, 0, 255)
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img_array)
    img.save(filepath)

# --- Compression ---

def compress_tiff(input_path, output_path, tile_size=512, scales=2,
                  wavelet="bior4.4", num_passes=16, quantization_step=None, 
                  verbose=True):
    """Compresses a TIFF to WDR and returns the time taken and compression ratio as dictionary."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    start_time = time.time()
    
    if verbose:
        print("="*60)
        print("WDR Compression")
        print("="*60)
        print(f"Input: {input_path}")
    
    # Load image (handles TIFF memmap and PIL fallback, validates 2D)
    image_source = load_image(input_path)
    h, w = image_source.shape[:2]
    raw_size = image_source.nbytes
    
    if verbose:
        print(f"Dimensions: {w} x {h}")
        print(f"Raw Size: {format_size(raw_size)}")
        print("\n[Pass 1] Analysis (Calculating Global Threshold)...")
    
    # Calculate global threshold
    global_max = hlp.scan_for_max_coefficient(image_source, tile_size, scales, wavelet)
    global_T = hlp.calculate_global_T(global_max)
    
    # Set quantization step if provided, otherwise set to 0 (no quantization)
    if quantization_step is not None:
        quant_step = quantization_step
    else:
        quant_step = 0
        if verbose:
            print("  No quantization step provided, using 0 (no quantization)")
    
    if verbose:
        print(f"  Global T: {global_T:.4f}")
        print(f"\n[Pass 2] Compressing to {output_path}...")
    
    # Progress callback for verbose mode
    def progress_cb(progress):
        if verbose:
            print(f"  Progress: {progress*100:.1f}%", end='\r')
    
    # Compress using wdr_io.compress
    wdr_io.compress(
        image_source=image_source,
        output_path=output_path,
        global_T=global_T,
        tile_size=tile_size,
        scales=scales,
        wavelet=wavelet,
        num_passes=num_passes,
        quant_step=quant_step,
        progress_callback=progress_cb if verbose else None
    )
    
    elapsed_time = time.time() - start_time
    
    # Calculate compression statistics locally
    compressed_size = get_file_size(output_path)
    compression_ratio = raw_size / compressed_size if compressed_size > 0 else 0
    
    if verbose:
        print()  # New line after progress
        print("-" * 40)
        print(f"Uncompressed Data: {format_size(raw_size)}")
        print(f"WDR Archive Size:  {format_size(compressed_size)}")
        print(f"Compression Ratio: {compression_ratio:.2f}x")
        print(f"Time Taken: {elapsed_time:.2f}s")
        print("-" * 40)
        print("="*60)
    
    return {
        "time_taken": elapsed_time,
        "compression_ratio": compression_ratio,
        "raw_size": raw_size,
        "compressed_size": compressed_size
    }
def extract_wdr(input_path, output_path, original_image=None, verbose=True):
    """Extracts a WDR to TIFF and returns the metrics dictionary if original_image is provided."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    start_time = time.time()
    
    if verbose:
        print("="*60)
        print("WDR Extraction")
        print("="*60)
        print(f"Input: {input_path}")
    
    # Open reader to get dimensions (will be reopened by wdr_io.decompress)
    reader = WDRTileReader(input_path)
    width = reader.width
    height = reader.height
    tile_size = reader.tile_size
    rows = reader.rows
    cols = reader.cols
    reader.close()  # Close early since decompress will open its own
    
    if verbose:
        print(f"Dimensions: {width} x {height}")
        print(f"Tile Size: {tile_size}")
        print("\nDecompressing...")
    
    # Load reference image if provided for metrics (uses memmap for TIFF, so memory efficient)
    reference_img = None
    if original_image:
        if not os.path.exists(original_image):
            raise FileNotFoundError(f"Reference image not found: {original_image}")
        reference_img = load_image(original_image)
        if reference_img.shape[:2] != (height, width):
            raise ValueError(
                f"Reference image dimensions {reference_img.shape[:2]} "
                f"do not match WDR dimensions ({height}, {width})"
            )
    
    # Initialize metrics if reference provided (O(1) memory - just SSE accumulator)
    metrics_obj = StreamMetrics(max_val=255.0) if reference_img is not None else None
    
    # Progress callback for verbose mode
    def progress_cb(progress):
        if verbose:
            print(f"  Progress: {progress*100:.1f}%", end='\r')
    
    # Allocate output image (required for saving, but we process tile-by-tile)
    output_image = np.zeros((height, width), dtype=np.float64)
    
    # Stream tiles and reassemble tile-by-tile (memory efficient)
    # Only one tile in memory at a time during decompression
    tile_gen = wdr_io.decompress(
        wdr_path=input_path,
        progress_callback=progress_cb if verbose else None
    )
    
    tile_idx = 0
    for r in range(rows):
        for c in range(cols):
            # Get next tile from generator (streaming, not all in memory)
            tile = next(tile_gen)
            
            # Calculate position in output image
            y_start = r * tile_size
            x_start = c * tile_size
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            
            # Calculate valid region of tile (crop edge padding)
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # Extract valid region from tile
            tile_valid = tile[:tile_h, :tile_w]
            
            # Place tile in output (crop to valid region)
            output_image[y_start:y_end, x_start:x_end] = tile_valid
            
            # Calculate metrics tile-by-tile (memory efficient - only processes one tile)
            if metrics_obj is not None and reference_img is not None:
                # Extract corresponding region from reference (memmap slice is efficient)
                tile_orig = reference_img[y_start:y_end, x_start:x_end]
                # Update metrics with this tile pair (O(1) memory accumulation)
                metrics_obj.update(tile_orig, tile_valid)
            
            tile_idx += 1
    
    if verbose:
        print()  # New line after progress
    
    # Save image
    if verbose:
        print(f"Saving to {output_path}...")
    save_image(output_path, output_image)
    
    elapsed_time = time.time() - start_time
    
    # Get metrics results if calculated
    metrics = None
    if metrics_obj is not None:
        mse, psnr = metrics_obj.get_results()
        metrics = {"mse": mse, "psnr": psnr}
        
        if verbose:
            print("-" * 40)
            print(f"MSE:  {mse:.4f}")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"Time Taken: {elapsed_time:.2f}s")
            print("-" * 40)
    elif verbose:
        print(f"Time Taken: {elapsed_time:.2f}s")
    
    if verbose:
        print("="*60)
    
    return metrics

# ==========================================
# CLI Handler
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="WDR Gigapixel Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Command: compress ---
    cmd_c = subparsers.add_parser("compress", help="Convert TIFF to WDR")
    cmd_c.add_argument("input", help="Input TIFF")
    cmd_c.add_argument("output", help="Output WDR")
    cmd_c.add_argument("--tile-size", type=int, default=512, help="Tile size")
    cmd_c.add_argument("--scales", type=int, default=2, help="Number of dwt scales")
    cmd_c.add_argument("--wavelet", type=str, default="bior4.4" , help="pywt wavelet name")
    cmd_c.add_argument("--passes", type=int, default=16, dest="num_passes", help="Bitplane passes")
    cmd_c.add_argument("--qstep", type=float, default=None, help="Quantization step")
    cmd_c.add_argument("--quiet", action="store_true", help="Suppress output")

    # --- Command: extract ---
    cmd_x = subparsers.add_parser("extract", help="Convert WDR to TIFF")
    cmd_x.add_argument("input", help="Input WDR")
    cmd_x.add_argument("output", help="Output TIFF")
    cmd_x.add_argument("--original-image", type=str, default=None, help="Original image for metrics")
    cmd_x.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.command == "compress":
        compress_tiff(
            args.input, args.output, tile_size=args.tile_size,
            scales=args.scales, wavelet=args.wavelet,
            num_passes=args.num_passes, quantization_step=args.qstep,
            verbose=not args.quiet
        )
    elif args.command == "extract":
        extract_wdr(args.input, args.output, original_image=args.original_image, verbose=not args.quiet)

if __name__ == "__main__":
    main()