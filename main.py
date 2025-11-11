#!/usr/bin/env python3
"""
Main example script for WDR image compression pipeline.

This script demonstrates the full compression and decompression pipeline:
1. Load an image
2. Perform Discrete Wavelet Transform (DWT)
3. Flatten coefficients using WDR scanning order
4. Compress using WDR algorithm
5. Decompress using WDR algorithm
6. Unflatten coefficients
7. Perform Inverse Discrete Wavelet Transform (IDWT)
8. Save reconstructed image

The script supports command-line arguments for customization:
- Input image file path
- Output .wdr file path
- Number of wavelet decomposition scales (default: 2, recommended: 2-3)
- Wavelet name (default: bior4.4)
- Optional reconstructed image output path

Example usage:
    python main.py input.png output.wdr
    python main.py input.png output.wdr --scales 2 --reconstructed recon.png

Note: Using scales=6+ introduces boundary artifacts as warned by PyWavelets,
      so scales=2-3 is recommended for practical use.
"""

import argparse
import sys
import os
import numpy as np
import wdr_coder
import wdr_helpers as hlp


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    PSNR is a metric used to measure the quality of reconstructed images.
    Higher PSNR values indicate better quality. For lossless compression,
    PSNR should be infinite (or very high).
    
    Formula: PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
    where MSE is the Mean Squared Error between original and compressed images.
    
    Args:
        original: Original image array (2D NumPy array, dtype float64)
        compressed: Compressed/reconstructed image array (2D NumPy array, dtype float64)
        
    Returns:
        PSNR value in dB. Returns float('inf') if images are identical (MSE = 0).
        
    Note:
        The function assumes images are in the range [0, 255] with max_pixel = 255.0.
    """
    # Calculate MSE
    mse = np.mean((original - compressed) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def main():
    parser = argparse.ArgumentParser(
        description="WDR Image Compression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.png output.wdr
  %(prog)s input.png output.wdr --scales 2
  %(prog)s input.png output.wdr --scales 2 --reconstructed reconstructed.png
  %(prog)s input.png output.wdr --scales 3 --wavelet bior4.4 --reconstructed recon.png

Note: Default scales is 2 (recommended: 2-3). Using scales=6+ introduces
      boundary artifacts as warned by PyWavelets, so it is not recommended
      for practical use.
        """
    )
    
    parser.add_argument(
        "input_image",
        help="Input image file path"
    )
    
    parser.add_argument(
        "output_wdr",
        help="Output .wdr compressed file path"
    )
    
    parser.add_argument(
        "--scales",
        type=int,
        default=2,
        help="Number of wavelet decomposition scales (default: 2, recommended: 2-3. Note: scales=6+ introduces boundary artifacts as warned by PyWavelets)"
    )
    
    parser.add_argument(
        "--reconstructed",
        type=str,
        default=None,
        help="Path to save reconstructed image (optional)"
    )
    
    parser.add_argument(
        "--wavelet",
        type=str,
        default="bior4.4",
        help="Wavelet name (default: bior4.4)"
    )
    
    parser.add_argument(
        "--num-passes",
        type=int,
        default=26,
        help="Number of bit-plane passes (default: 26 for high precision)"
    )
    
    parser.add_argument(
        "--quantization-step",
        type=float,
        default=None,
        help="Quantization step size (optional, default: auto-calculate). Set to 0 to disable quantization. Quantization improves compression efficiency by creating redundancy in coefficient values."
    )
    
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="threshold_based",
        choices=["threshold_based", "fixed_precision"],
        help="Method for calculating quantization step size when auto-calculating (default: threshold_based). Only used if --quantization-step is not specified."
    )
    
    args = parser.parse_args()
    
    # --- 1. COMPRESSION ---
    print("=" * 60)
    print("WDR Image Compression Pipeline")
    print("=" * 60)
    print(f"Input image: {args.input_image}")
    print(f"Output .wdr file: {args.output_wdr}")
    print(f"Wavelet scales: {args.scales}")
    print(f"Wavelet: {args.wavelet}")
    print()
    
    try:
        # Load image
        print("Step 1: Loading image...")
        original_img = hlp.load_image(args.input_image)
        print(f"  Image size: {original_img.shape}")
        print(f"  Image dtype: {original_img.dtype}")
        print(f"  Image range: [{original_img.min():.2f}, {original_img.max():.2f}]")
        print()
        
        # Perform DWT
        print("Step 2: Performing DWT...")
        wavelet_coeffs = hlp.do_dwt(original_img, scales=args.scales, wavelet=args.wavelet)
        print(f"  DWT completed with {args.scales} scales")
        print()
        
        # Flatten coefficients
        print("Step 3: Flattening coefficients...")
        flat_coeffs, shape_data = hlp.flatten_coeffs(wavelet_coeffs)
        print(f"  Flattened array size: {flat_coeffs.shape}")
        print(f"  Number of coefficients: {len(flat_coeffs)}")
        print(f"  Coefficient range: [{flat_coeffs.min():.2f}, {flat_coeffs.max():.2f}]")
        print(f"  Unique values: {len(np.unique(flat_coeffs))}")
        print()
        
        # Quantize coefficients (optional, improves compression efficiency)
        quantization_step = args.quantization_step
        if quantization_step is None:
            # Auto-calculate quantization step (compression-focused for better CR)
            # Quantization is optional but recommended for better compression ratios
            quantization_step = hlp.calculate_quantization_step(
                flat_coeffs, 
                num_passes=args.num_passes, 
                method=args.quantization_method,
                compression_focused=True  # Use larger step for better compression
            )
            print(f"Step 4: Quantizing coefficients (optional, auto-calculated step: {quantization_step:.6f})...")
            print("  Note: Quantization is optional but improves compression efficiency")
        elif quantization_step == 0:
            # Disable quantization explicitly
            print("Step 4: Quantization disabled (--quantization-step=0)...")
            quantized_coeffs = flat_coeffs
            quantization_step = None
        else:
            print(f"Step 4: Quantizing coefficients (optional, step: {quantization_step:.6e})...")
            print("  Note: Quantization is optional but improves compression efficiency")
        
        if quantization_step is not None and quantization_step > 0:
            quantized_coeffs, _ = hlp.quantize_coeffs(flat_coeffs, quantization_step)
            unique_before = len(np.unique(flat_coeffs))
            unique_after = len(np.unique(quantized_coeffs))
            reduction = ((unique_before - unique_after) / unique_before * 100) if unique_before > 0 else 0
            print(f"  Unique values before: {unique_before}")
            print(f"  Unique values after: {unique_after}")
            print(f"  Reduction: {reduction:.2f}%")
            
            # Calculate quantization error
            mse = np.mean((flat_coeffs - quantized_coeffs) ** 2)
            rmse = np.sqrt(mse)
            print(f"  Quantization MSE: {mse:.6e}")
            print(f"  Quantization RMSE: {rmse:.6e}")
        else:
            quantized_coeffs = flat_coeffs
            print("  Warning: No quantization applied. Compression ratio may be poor (compressed file may be larger than original).")
        print()
        
        # Compress
        print("Step 5: Compressing with WDR...")
        wdr_coder.compress(quantized_coeffs, args.output_wdr, num_passes=args.num_passes)
        
        # Calculate compression ratio
        original_size = os.path.getsize(args.input_image)
        compressed_size = os.path.getsize(args.output_wdr)
        cr = original_size / compressed_size if compressed_size > 0 else 0
        print(f"  Compression complete: {args.output_wdr}")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Compressed size: {compressed_size:,} bytes")
        print(f"  Compression ratio: {cr:.3f}x")
        if cr < 1:
            print(f"  ⚠️  WARNING: Compressed file is {((1-cr)*100):.1f}% LARGER than original!")
        print()
        
        # --- 2. DECOMPRESSION ---
        if args.reconstructed:
            print("Step 6: Decompressing with WDR...")
            decompressed_flat_coeffs = wdr_coder.decompress(args.output_wdr)
            print(f"  Decompressed array size: {decompressed_flat_coeffs.shape}")
            print(f"  Number of coefficients: {len(decompressed_flat_coeffs)}")
            print()
            
            # Dequantize coefficients (if quantization was used)
            if quantization_step is not None and quantization_step > 0:
                print("Step 7: Dequantizing coefficients...")
                dequantized_coeffs = hlp.dequantize_coeffs(decompressed_flat_coeffs, quantization_step)
                print(f"  Dequantization complete (step: {quantization_step:.6e})")
                print()
            else:
                dequantized_coeffs = decompressed_flat_coeffs
            
            # Unflatten coefficients
            print("Step 8: Unflattening coefficients...")
            decompressed_coeffs = hlp.unflatten_coeffs(dequantized_coeffs, shape_data)
            print("  Unflattening complete")
            print()
            
            # Perform IDWT
            print("Step 9: Performing IDWT...")
            reconstructed_img = hlp.do_idwt(decompressed_coeffs, wavelet=args.wavelet)
            print(f"  Reconstructed image size: {reconstructed_img.shape}")
            print()
            
            # Save reconstructed image
            print("Step 10: Saving reconstructed image...")
            hlp.save_image(args.reconstructed, reconstructed_img)
            print(f"  Reconstructed image saved: {args.reconstructed}")
            print()
            
            # Calculate metrics
            print("Step 11: Calculating metrics...")
            # Clip reconstructed image to valid range for comparison
            reconstructed_clipped = np.clip(reconstructed_img, 0, 255)
            psnr = calculate_psnr(original_img, reconstructed_clipped)
            
            # Calculate MSE
            mse = np.mean((original_img - reconstructed_clipped) ** 2)
            rmse = np.sqrt(mse)
            
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  PSNR: {psnr:.2f} dB")
            
            if quantization_step is not None and quantization_step > 0:
                print(f"  Quantization step: {quantization_step:.6e}")
            print()
        
        print("=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid value: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

