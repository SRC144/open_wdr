#!/usr/bin/env python3
"""
Gigapixel WSI WDR Pipeline (Unified)

A standalone tool for compressing Whole Slide Images (NDPI, SVS, Philips TIFF)
using the WDR algorithm.

This script implements the Color Wavelet Difference Reduction (CWDR) workflow
for medical imaging, as described in:

    Zerva, M.C.H., Christou, V., Giannakeas, N., Tzallas, A.T., & Kondi, L.P. (2023).
    "An Improved Medical Image Compression Method Based on Wavelet Difference Reduction."
    IEEE Access, vol. 11, pp. 18026-18037.
    https://doi.org/10.1109/ACCESS.2023.3246948

FEATURES:
- Universal OpenSlide Reader (Handles Headers, Sparse Tiles, Metadata)
- Digital YCbCr Colorspace (Prevents color clipping)
- Automatic Cleanup of intermediate Gigapixel TIFFs
- Live Progress Reporting
- Output Directory Management

DEPENDENCIES:
    - numpy
    - tifffile
    - openslide-python (and the OpenSlide binary)
    - wdr (Your custom library)
"""

import argparse
import os
import time
import math
import sys
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image

# --- Dependency Check ---
try:
    import openslide
    from openslide import deepzoom
except ImportError:
    print("CRITICAL ERROR: 'openslide-python' not found.")
    print("Please install: pip install openslide-python")
    sys.exit(1)

# Import your custom WDR library
try:
    from wdr import io as wdr_io
    from wdr.utils import helpers as hlp
    from wdr.container import WDRTileReader
except ImportError:
    print("CRITICAL ERROR: 'wdr' library not found in Python path.")
    sys.exit(1)

Image.MAX_IMAGE_PIXELS = None

# ==========================================
# PART 1: Core Utilities
# ==========================================

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

def load_image_strict(filepath: str) -> np.ndarray:
    """Loads a single-channel image, enforcing 2D shape."""
    try:
        img_array = tifffile.memmap(filepath, mode='r')
    except Exception:
        img = Image.open(filepath)
        img_array = np.array(img)

    if img_array.ndim != 2:
        if img_array.ndim == 3:
             # Fallback for unexpected 3D saves
             return img_array[:,:,0]
        raise ValueError(f"Image '{filepath}' must be 2D single-channel.")
        
    return img_array

def resolve_output_path(base_name, out_dir=None):
    """
    Combines output directory and filename, creating the directory if needed.
    """
    # 1. Handle explicit directory
    if out_dir:
        filename = os.path.basename(base_name)
        full_path = os.path.join(out_dir, filename)
    else:
        full_path = base_name

    # 2. Strip extension if user typed "output.wdr" or "output.tiff"
    if full_path.lower().endswith(('.wdr', '.tiff', '.tif', '.svs', '.ndpi')):
        full_path = os.path.splitext(full_path)[0]

    # 3. Create parent directory
    parent_dir = os.path.dirname(full_path)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            print(f"  [Init] Created output directory: {parent_dir}")
        except OSError as e:
            print(f"  [Error] Could not create directory {parent_dir}: {e}")
            sys.exit(1)
            
    return full_path

# ==========================================
# PART 2: WDR Compression Logic
# ==========================================

def compress_channel(input_path, output_path, args, chan_name):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing channel file: {input_path}")
    
    start_time = time.time()
    
    image_source = load_image_strict(input_path)
    raw_size = image_source.nbytes
    
    # 1. Analyze
    print(f"    -> Obtaining global threshold T for {chan_name}...", end='\n')
    global_max = hlp.scan_for_max_coefficient(
        image_source, args.tile_size, args.scales, args.wavelet
    )
    global_T = hlp.calculate_global_T(global_max)
    print(f" Done (T={global_T:.4f})")
    
    # 2. Compress
    quant_step = args.qstep if args.qstep is not None else 0
    
    def progress_cb(progress):
        print(f"    -> Compressing {chan_name}: {progress*100:.1f}%", end='\r')

    wdr_io.compress(
        image_source=image_source,
        output_path=output_path,
        global_T=global_T,
        tile_size=args.tile_size,
        scales=args.scales,
        wavelet=args.wavelet,
        num_passes=args.num_passes,
        quant_step=quant_step,
        progress_callback=progress_cb
    )
    
    # Final cleanup of the line
    print(f"    -> Compressing {chan_name}: 100.0%    ")
    
    elapsed = time.time() - start_time
    comp_size = get_file_size(output_path)
    
    return {"raw_size": raw_size, "compressed_size": comp_size, "time": elapsed}

def decompress_channel(wdr_path, tiff_path, chan_name):
    if not os.path.exists(wdr_path):
        raise FileNotFoundError(f"Missing WDR file: {wdr_path}")

    reader = WDRTileReader(wdr_path)
    h, w = reader.height, reader.width
    tile_size = reader.tile_size
    rows, cols = reader.rows, reader.cols
    reader.close()

    output_image = tifffile.memmap(tiff_path, shape=(h, w), dtype=np.uint8, bigtiff=True)
    tile_gen = wdr_io.decompress(wdr_path, progress_callback=None)
    
    total_tiles = rows * cols
    processed = 0

    print(f"    -> Decompressing {chan_name}: 0.0%", end='\r')

    for r in range(rows):
        for c in range(cols):
            tile = next(tile_gen)
            y_start = r * tile_size
            x_start = c * tile_size
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            tile_valid = tile[:(y_end - y_start), :(x_end - x_start)]
            output_image[y_start:y_end, x_start:x_end] = tile_valid
            
            processed += 1
            if processed % 10 == 0:
                print(f"    -> Decompressing {chan_name}: {processed/total_tiles*100:.1f}%", end='\r')

    print(f"    -> Decompressing {chan_name}: 100.0%    ")
    output_image.flush()
    del output_image

# ==========================================
# PART 3: Streaming Color Transformations
# ==========================================

def stream_slide_to_yuv(input_path, output_base, tile_size):
    print(f"  [1/3] Splitting WSI -> Y/U/V (OpenSlide Mode)...")
    
    slide = openslide.OpenSlide(input_path)
    
    # Overlap=0 is critical for distinct WDR tiles
    dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=True)
    max_level = dz.level_count - 1
    w, h = slide.dimensions
    cols, rows = dz.level_tiles[max_level]
    
    print(f"    Format: {slide.detect_format(input_path).upper()}")
    print(f"    Source: {w}x{h}")
    
    path_y = f"{output_base}_Y.tiff"
    path_u = f"{output_base}_U.tiff"
    path_v = f"{output_base}_V.tiff"

    y_img = tifffile.memmap(path_y, shape=(h, w), dtype=np.uint8, bigtiff=True)
    u_img = tifffile.memmap(path_u, shape=(h, w), dtype=np.uint8, bigtiff=True)
    v_img = tifffile.memmap(path_v, shape=(h, w), dtype=np.uint8, bigtiff=True)

    processed_pixels = 0
    start_time = time.time()

    for r in range(rows):
        for c in range(cols):
            try:
                pil_img = dz.get_tile(max_level, (c, r))
            except Exception as e:
                print(f"    Error tile {c},{r}: {e}")
                pil_img = Image.new("RGB", (tile_size, tile_size), (0,0,0))

            y_start = r * tile_size
            x_start = c * tile_size
            valid_w, valid_h = pil_img.size
            y_end = min(y_start + valid_h, h)
            x_end = min(x_start + valid_w, w)

            arr = np.array(pil_img).astype(np.float32)
            if arr.shape[2] == 4: arr = arr[..., :3]

            R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

            # JFIF (Digital) YCbCr
            Y = (0.29900 * R) + (0.58700 * G) + (0.11400 * B)
            U = (-0.16874 * R) - (0.33126 * G) + (0.50000 * B) + 128.0
            V = (0.50000 * R) - (0.41869 * G) - (0.08131 * B) + 128.0

            y_img[y_start:y_end, x_start:x_end] = np.clip(Y, 0, 255).astype(np.uint8)
            u_img[y_start:y_end, x_start:x_end] = np.clip(U, 0, 255).astype(np.uint8)
            v_img[y_start:y_end, x_start:x_end] = np.clip(V, 0, 255).astype(np.uint8)

            processed_pixels += valid_w * valid_h
            if time.time() - start_time > 1.0:
                 print(f"    Progress: {processed_pixels/(w*h)*100:.1f}%", end='\r')
                 start_time = time.time()

    del y_img, u_img, v_img
    slide.close()
    print(f"    Progress: 100.0%    ")
    
    return {'y': path_y, 'u': path_u, 'v': path_v}, (h, w, 3)

def stream_yuv_to_rgb(base_path, output_path, chunk_size=1024):
    print(f"  [3/3] Merging Channels -> RGB...")

    path_y = f"{base_path}_Y.tiff"
    path_u = f"{base_path}_U.tiff"
    path_v = f"{base_path}_V.tiff"

    y_img = tifffile.memmap(path_y, mode='r')
    u_img = tifffile.memmap(path_u, mode='r')
    v_img = tifffile.memmap(path_v, mode='r')
    h, w = y_img.shape

    rgb_out = tifffile.memmap(output_path, shape=(h, w, 3), dtype=np.uint8, bigtiff=True)

    total_pixels = h * w
    processed = 0
    start_time = time.time()

    for i in range(0, h, chunk_size):
        end_i = min(i + chunk_size, h)

        Y = y_img[i:end_i].astype(np.float32)
        U = u_img[i:end_i].astype(np.float32) - 128.0
        V = v_img[i:end_i].astype(np.float32) - 128.0

        # Inverse JFIF
        R = Y + (1.40200 * V)
        G = Y - (0.34414 * U) - (0.71414 * V)
        B = Y + (1.77200 * U)

        rgb_out[i:end_i] = np.dstack((
            np.clip(R, 0, 255).astype(np.uint8),
            np.clip(G, 0, 255).astype(np.uint8),
            np.clip(B, 0, 255).astype(np.uint8)
        ))

        processed += (end_i - i) * w
        if time.time() - start_time > 1.0:
             print(f"    Progress: {processed/total_pixels*100:.1f}%", end='\r')
             start_time = time.time()

    del rgb_out, y_img, u_img, v_img
    print(f"    Progress: 100.0%    ")

# ==========================================
# PART 4: Pipeline Execution (Robust)
# ==========================================

def cleanup_files(file_list):
    print("\n  [Cleanup] Removing intermediate temp files...")
    for p in file_list:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"    Warning: Could not delete {p}: {e}")

def run_compress(args):
    start_total = time.time()
    input_path = args.input
    
    # Resolve Output Path (Handle --out-dir)
    output_base = resolve_output_path(args.output_wdr_base, args.out_dir)
    
    # Track temp files for cleanup
    temp_files = []

    print("="*60)
    print(f"WDR WSI COMPRESSION")
    print(f"Input:  {os.path.basename(input_path)}")
    print(f"Output: {output_base}_[Y/U/V].wdr")
    print("="*60)

    try:
        # 1. Stream & Split
        tiff_paths, dims = stream_slide_to_yuv(input_path, output_base, args.tile_size)
        temp_files.extend(tiff_paths.values())
        
        raw_size_total = dims[0] * dims[1] * dims[2]

        # 2. Compress Channels
        print(f"\n  [2/3] Compressing Channels...")
        channel_sizes = {}
        
        for chan in ['y', 'u', 'v']:
            tiff_in = tiff_paths[chan]
            wdr_out = f"{output_base}_{chan.upper()}.wdr"
            
            # Determine passes for this channel
            if args.half_chroma_passes and chan in ['u', 'v']:
                original_passes = args.num_passes
                args.num_passes = max(1, args.num_passes // 2)
                print(f"    (Using {args.num_passes} passes for chroma channel {chan.upper()})")
                stats = compress_channel(tiff_in, wdr_out, args, chan.upper())
                args.num_passes = original_passes  # Restore for next iteration
            else:
                stats = compress_channel(tiff_in, wdr_out, args, chan.upper())
            
            channel_sizes[chan] = stats['compressed_size']

        total_comp = sum(channel_sizes.values())
        cr = raw_size_total / total_comp if total_comp > 0 else 0
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Input:             {input_path}")
        print(f"Dimensions:        {dims[1]} x {dims[0]}")
        print(f"Raw Size:          {format_size(raw_size_total)}")
        print(f"Compressed Size:   {format_size(total_comp)}")
        print(f"Compression Ratio: {cr:.2f}x")
        print("-" * 30)
        print(f"Y Size:            {format_size(channel_sizes['y'])}")
        print(f"U Size:            {format_size(channel_sizes['u'])}")
        print(f"V Size:            {format_size(channel_sizes['v'])}")
        print("-" * 30)
        print(f"Total Time:        {time.time() - start_total:.2f}s")
        print("="*60)

    finally:
        # SAFE CLEANUP: Runs even if compression fails
        if not args.keep_temp:
            cleanup_files(temp_files)
        else:
            print(f"\n  [Cleanup] Skipped (--keep-temp active).")

def run_extract(args):
    start_total = time.time()
    
    # 1. --- Input WDR Base Resolution ---
    # input_base is now explicitly joined from dir and name.
    input_base = os.path.join(args.wdr_input_dir, args.wdr_base_name)
    
    # 2. Resolve Path for the Final TIFF Output
    output_tiff_base = resolve_output_path(args.output_tiff, args.out_dir)
    if not output_tiff_base.lower().endswith(('.tiff', '.tif')):
        output_tiff = output_tiff_base + '.tiff'
    else:
        output_tiff = output_tiff_base
        output_tiff_base = os.path.splitext(output_tiff)[0]
    
    # 3. Determine Temp File Location (Next to the final output TIFF)
    temp_base = output_tiff_base
    
    temp_files = []

    print("="*60)
    print(f"WDR WSI EXTRACTION")
    print(f"WDR Source: {input_base}_[Y/U/V].wdr")
    print(f"Output:     {output_tiff}")
    print("="*60)

    try:
        channels = ['Y', 'U', 'V']
        # WDR files are located using the full input_base path
        wdr_files = {c: f"{input_base}_{c}.wdr" for c in channels}
        # Temp tiffs go to output location
        tiff_files = {c: f"{temp_base}_{c}.tiff" for c in channels}
        
        # Verify inputs
        for c, f in wdr_files.items():
            if not os.path.exists(f):
                print(f"CRITICAL: Missing component {f}")
                print(f"Checked path: {os.path.abspath(f)}")
                return

        # 1. Decompress
        print(f"  [1/3] Decompressing Channels...")
        for c in channels:
            decompress_channel(wdr_files[c], tiff_files[c], c)
            temp_files.append(tiff_files[c])

        # 3. Merge
        stream_yuv_to_rgb(temp_base, output_tiff)

        print("\n" + "="*60)
        print(f"Extraction Complete: {output_tiff}")
        print(f"Total Time: {time.time() - start_total:.2f}s")
        print("="*60)

    finally:
        # SAFE CLEANUP
        if not args.keep_temp:
            cleanup_files(temp_files)
        else:
            print(f"\n  [Cleanup] Skipped (--keep-temp active).")

# ==========================================
# PART 5: CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="WDR Whole Slide Image (WSI) Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Compress ---
    cmd_c = subparsers.add_parser("compress", help="WSI (NDPI/SVS/TIFF) -> WDR")
    cmd_c.add_argument("input", help="Input WSI file")
    cmd_c.add_argument("output_wdr_base", help="Output base filename (e.g. 'cmu1')")
    cmd_c.add_argument("--out-dir", help="Optional output directory")
    cmd_c.add_argument("--tile-size", type=int, default=512)
    cmd_c.add_argument("--scales", type=int, default=2)
    cmd_c.add_argument("--wavelet", type=str, default="bior4.4")
    cmd_c.add_argument("--passes", type=int, default=16, dest="num_passes")
    cmd_c.add_argument("--qstep", type=float, default=None, help="Quantization step (0=Lossless)")
    cmd_c.add_argument("--keep-temp", action="store_true", help="Keep intermediate Y/U/V TIFFs")
    cmd_c.add_argument("--half-chroma-passes", action="store_true",
                       help="Use half the passes for U/V chroma channels")

    # --- Extract ---
    cmd_x = subparsers.add_parser("extract", help="WDR -> RGB BigTIFF")
    cmd_x.add_argument("wdr_input_dir", help="Directory containing WDR channel files (e.g., 'results')")
    cmd_x.add_argument("wdr_base_name", help="Base filename (prefix) of the WDR files (e.g., 'cmu1')")
    cmd_x.add_argument("output_tiff", help="Output RGB BigTIFF filename (e.g. 'reconstructed.tiff')")
    cmd_x.add_argument("--out-dir", help="Optional output directory for the final TIFF")
    cmd_x.add_argument("--keep-temp", action="store_true", help="Keep intermediate Y/U/V TIFFs")

    args = parser.parse_args()

    if args.command == "compress":
        run_compress(args)
    elif args.command == "extract":
        run_extract(args)

if __name__ == "__main__":
    main()