#!/usr/bin/env python3
"""
WSI Reconstruction Metrics (PSNR & SSIM)

Calculates quality metrics for Gigapixel images by streaming tiles
to avoid memory exhaustion. Supports 8-bit and 16-bit depths.

Usage:
    python wsi_metrics.py <reconstructed.tiff> <reference.svs/ndpi/tiff>
"""

import argparse
import math
import time
import numpy as np
import tifffile
from PIL import Image

try:
    import openslide
except ImportError:
    print("Error: openslide-python is required.")
    sys.exit(1)

# Increase PIL limit for safety
Image.MAX_IMAGE_PIXELS = None

class StreamingMetrics:
    def __init__(self, bit_depth=8):
        self.mse_sum = 0.0
        self.ssim_sum = 0.0
        self.total_pixels = 0
        self.total_windows = 0
        
        # Adaptive Dynamic Range (L)
        # 8-bit -> 255, 16-bit -> 65535
        self.L = float(2**bit_depth - 1)
        
        # SSIM Constants (Standard)
        k1 = 0.01
        k2 = 0.03
        self.c1 = (k1 * self.L) ** 2
        self.c2 = (k2 * self.L) ** 2
        
    def update(self, img_recon, img_ref):
        """
        Update metrics with a pair of image tiles (H, W, Channels).
        Tiles must be the same shape.
        """
        # Ensure float computation
        x = img_recon.astype(np.float64)
        y = img_ref.astype(np.float64)
        
        # --- MSE Calculation ---
        diff = x - y
        self.mse_sum += np.sum(diff ** 2)
        self.total_pixels += x.size # Total values (Pixels * Channels)
        
        # --- SSIM Calculation (Simplified Block Approach) ---
        # We treat the entire tile as one large "window" for averaging.
        # For rigorous MSSIM, one would slide a Gaussian window, but for 
        # Gigapixel tiles (e.g. 1024x1024), block averaging is a standard approximation.
        
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        
        # Covariance approximation
        cov_xy = np.mean((x - mu_x) * (y - mu_y))
        
        # SSIM Formula
        numerator = (2 * mu_x * mu_y + self.c1) * (2 * cov_xy + self.c2)
        denominator = (mu_x**2 + mu_y**2 + self.c1) * (var_x + var_y + self.c2)
        
        ssim_val = numerator / denominator
        
        self.ssim_sum += ssim_val
        self.total_windows += 1

    def get_results(self):
        if self.total_pixels == 0:
            return 0.0, 0.0, 0.0
            
        # Global MSE
        mse = self.mse_sum / self.total_pixels
        
        # PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * math.log10((self.L ** 2) / mse)
            
        # Mean SSIM
        mssim = self.ssim_sum / self.total_windows if self.total_windows > 0 else 0.0
        
        return mse, psnr, mssim

def calculate_metrics(reconstructed_path, reference_path, tile_size=2048):
    print(f"Comparing:")
    print(f"  Reconstructed: {reconstructed_path}")
    print(f"  Reference:     {reference_path}")
    
    # 1. Open Reference (OpenSlide for SVS/NDPI support)
    ref_slide = openslide.OpenSlide(reference_path)
    ref_w, ref_h = ref_slide.dimensions
    
    # 2. Open Reconstructed (Memmap for Speed)
    try:
        recon_img = tifffile.memmap(reconstructed_path, mode='r')
    except Exception as e:
        print(f"Error opening reconstructed TIFF: {e}")
        return

    # Verify Dimensions
    # Note: Recon might be (H,W,C) or (H,W)
    rec_h, rec_w = recon_img.shape[:2]
    if (rec_w, rec_h) != (ref_w, ref_h):
        print("WARNING: Dimension Mismatch!")
        print(f"  Ref:   {ref_w} x {ref_h}")
        print(f"  Recon: {rec_w} x {rec_h}")
        # We will proceed using the intersection area
        proc_w = min(rec_w, ref_w)
        proc_h = min(rec_h, ref_h)
    else:
        proc_w, proc_h = ref_w, ref_h

    # Detect Bit Depth from Reconstruction
    if recon_img.dtype == np.uint16:
        print("Mode: 16-bit evaluation")
        bit_depth = 16
    else:
        print("Mode: 8-bit evaluation")
        bit_depth = 8
        
    metrics = StreamingMetrics(bit_depth)
    
    # 3. Stream Processing
    rows = (proc_h + tile_size - 1) // tile_size
    cols = (proc_w + tile_size - 1) // tile_size
    total_tiles = rows * cols
    processed = 0
    
    print(f"\nProcessing {total_tiles} tiles ({tile_size}x{tile_size})...")
    start_time = time.time()

    for r in range(rows):
        for c in range(cols):
            y = r * tile_size
            x = c * tile_size
            w = min(tile_size, proc_w - x)
            h = min(tile_size, proc_h - y)
            
            # A. Get Reference Patch (OpenSlide)
            # read_region returns RGBA PIL image
            ref_pil = ref_slide.read_region((x, y), 0, (w, h))
            ref_patch = np.array(ref_pil.convert("RGB")) # Drop Alpha
            
            # B. Get Recon Patch (Memmap Slice)
            recon_patch = recon_img[y:y+h, x:x+w]
            
            # Handle potential shape mismatch (e.g. RGB vs RGBA vs Grayscale)
            # If recon is 3D and ref is 3D, ensure same channel count
            if recon_patch.ndim == 3 and recon_patch.shape[2] == 3:
                pass # OK
            elif recon_patch.ndim == 2:
                # Recon is grayscale, convert Ref to gray for fair comparison?
                # Or broadcast recon? Let's assume user wants color-to-color
                # but if recon is gray, we take mean of ref.
                ref_patch = np.mean(ref_patch, axis=2).astype(recon_patch.dtype)
            
            metrics.update(recon_patch, ref_patch)
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Progress: {processed/total_tiles*100:.1f}%", end='\r')

    print(f"  Progress: 100.0%    ")
    
    # 4. Report
    mse, psnr, ssim = metrics.get_results()
    elapsed = time.time() - start_time
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"MSE:       {mse:.4f}")
    print(f"PSNR:      {psnr:.4f} dB")
    print(f"SSIM:      {ssim:.4f}")
    print(f"Time:      {elapsed:.2f}s")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reconstructed", help="Path to reconstructed BigTIFF")
    parser.add_argument("reference", help="Path to original WSI (SVS/NDPI/TIFF)")
    parser.add_argument("--tile-size", type=int, default=2048, help="Processing block size")
    args = parser.parse_args()
    
    calculate_metrics(args.reconstructed, args.reference, args.tile_size)