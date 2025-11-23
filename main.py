#!/usr/bin/env python3
"""
WDR Gigapixel Compression Pipeline.

Workflow:
1. Analysis: Scan for Global Threshold.
2. Compression: Stream tiles to .wdr archive.
3. Verification: (Optional) Stream decompression to calculate PSNR/CR.
"""

import argparse
import math
import numpy as np
import time
from wdr import coder as wdr_coder
from wdr.utils import helpers as hlp
from wdr.utils import metrics as met
from wdr.container import WDRTileWriter, WDRTileReader

TILE_SIZE = 512

def calculate_global_T(max_abs: float) -> float:
    if max_abs == 0.0: return 1.0
    return 2.0 ** math.floor(math.log2(max_abs))

def get_dummy_shape_metadata(tile_size, scales, wavelet):
    """Generates shape metadata constant for all tiles of same size."""
    dummy_tile = np.zeros((tile_size, tile_size), dtype=np.float64)
    coeffs = hlp.do_dwt(dummy_tile, scales=scales, wavelet=wavelet)
    flat, metadata = hlp.flatten_coeffs(coeffs)
    return len(flat), metadata

def main():
    parser = argparse.ArgumentParser(description="WDR Gigapixel Compressor")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_file", help="Path to output .wdr archive")
    
    # Compression Params
    parser.add_argument("--scales", type=int, default=2, help="DWT Scales")
    parser.add_argument("--wavelet", type=str, default="bior4.4")
    parser.add_argument("--num-passes", type=int, default=16, help="Quality (Bitplanes)")
    parser.add_argument("--quantization-step", type=float, default=None)
    
    # Verification
    parser.add_argument("--verify", action="store_true", help="Run streaming PSNR verification")
    
    args = parser.parse_args()

    # --- 1. Setup ---
    print("="*60)
    print("WDR Gigapixel Pipeline")
    print("="*60)
    
    start_time = time.time()
    
    # Smart Load: Uses memmap for TIFFs (0 RAM usage, virtual pointer)
    original_img = hlp.load_image(args.input_image)
    
    # Handle potential 3D shape (H, W, Channels) for dimensions
    h, w = original_img.shape[:2]
    
    # Accurate Raw Size Calculation:
    # .nbytes calculates exactly (Height * Width * Channels * BytesPerPixel)
    # This is the true "Uncompressed Size" in memory.
    raw_size_bytes = original_img.nbytes
    
    print(f"Input: {args.input_image}")
    print(f"Dimensions: {w} x {h}")
    print(f"Raw Size: {met.format_size(raw_size_bytes)} (Uncompressed)")

    # --- 2. Global Analysis ---
    print("\n[Pass 1] Analysis (Calculating Global Threshold)...")
    global_max = hlp.scan_for_max_coefficient(original_img, TILE_SIZE, args.scales, args.wavelet)
    global_T = calculate_global_T(global_max)
    
    # Determine Quantization
    # Explicit check for None to allow 0.0 (disabled)
    if args.quantization_step is not None:
        quant_step = args.quantization_step
    else:
        quant_step = global_T / 64.0 # Auto default
        print(f"  Auto-Quantization Step: {quant_step:.6f}")
    
    print(f"  Global T: {global_T:.4f}")

    # --- 3. Compression ---
    print(f"\n[Pass 2] Compressing to {args.output_file}...")
    
    writer = WDRTileWriter(
        args.output_file, width=w, height=h, tile_size=TILE_SIZE,
        global_T=global_T, scales=args.scales, wavelet=args.wavelet, 
        quant_step=quant_step, num_passes=args.num_passes
    )

    compressor = wdr_coder.WDRCompressor(args.num_passes)
    total_tiles = writer.total_tiles
    processed = 0
    
    # Streaming Loop: Disk -> RAM(Tile) -> CPU -> Disk(WDR) -> Free RAM
    for tile_data in hlp.yield_tiles(original_img, TILE_SIZE):
        coeffs = hlp.do_dwt(tile_data, scales=args.scales, wavelet=args.wavelet)
        flat, _ = hlp.flatten_coeffs(coeffs)
        
        if quant_step > 0:
            flat, _ = hlp.quantize_coeffs(flat, quant_step)
            
        compressed_vec = compressor.compress(flat, global_T)
        writer.add_tile(bytes(compressed_vec))
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed}/{total_tiles} tiles...", end='\r')

    writer.close()
    comp_time = time.time() - start_time
    print(f"  Processed {processed}/{total_tiles} tiles. Done in {comp_time:.2f}s.")

    # --- 4. Stats (Corrected CR) ---
    compressed_size_bytes = met.get_file_size(args.output_file)
    
    # CR = Raw Bytes / Compressed Bytes
    cr = raw_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
    
    print("-" * 40)
    print(f"Uncompressed Data: {met.format_size(raw_size_bytes)}")
    print(f"WDR Archive Size:  {met.format_size(compressed_size_bytes)}")
    print(f"Compression Ratio: {cr:.2f}x")
    print("-" * 40)

    # --- 5. Verification (PSNR) ---
    if args.verify:
        print("\n[Verification] Streaming Decompression...")
        
        reader = WDRTileReader(args.output_file)
        decompressor = wdr_coder.WDRCompressor(reader.num_passes)
        num_coeffs, shape_meta = get_dummy_shape_metadata(TILE_SIZE, args.scales, args.wavelet)
        
        metrics = met.StreamMetrics(max_val=255.0)
        rec_processed = 0
        
        for r in range(reader.rows):
            for c in range(reader.cols):
                # Decode Tile
                blob = reader.get_tile_bytes(r, c)
                
                # We know 512x512 tiles result in fixed coefficient count
                # (Exact count is width*height for standard DWT)
                flat_len = TILE_SIZE * TILE_SIZE 
                
                flat_recon = decompressor.decompress(blob, reader.global_T, flat_len)
                flat_recon = np.array(flat_recon, dtype=np.float64)
                
                if reader.quant_step > 0:
                    flat_recon = hlp.dequantize_coeffs(flat_recon, reader.quant_step)
                
                coeffs_recon = hlp.unflatten_coeffs(flat_recon, shape_meta)
                tile_recon = hlp.do_idwt(coeffs_recon, wavelet=reader.wavelet)
                tile_recon = tile_recon[:TILE_SIZE, :TILE_SIZE] # Crop DWT padding

                # Fetch Original (Smart Slice)
                y, x = r * TILE_SIZE, c * TILE_SIZE
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                tile_orig = original_img[y:y_end, x:x_end]
                
                # Crop reconstruction to valid area
                tile_recon_valid = tile_recon[:tile_orig.shape[0], :tile_orig.shape[1]]
                
                # Accumulate Error (Pixels discarded immediately after)
                metrics.update(tile_orig, tile_recon_valid)
                
                rec_processed += 1
                if rec_processed % 20 == 0:
                    print(f"  Verified {rec_processed}/{total_tiles} tiles...", end='\r')
        
        reader.close()
        print(f"  Verified {rec_processed}/{total_tiles} tiles.      ")
        
        mse, psnr = metrics.get_results()
        print(f"MSE:  {mse:.4f}")
        print(f"PSNR: {psnr:.2f} dB")

    print("="*60)

if __name__ == "__main__":
    main()