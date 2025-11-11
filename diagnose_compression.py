#!/usr/bin/env python3
"""
Diagnostic script for WDR compression pipeline.

This script analyzes the compression pipeline to identify issues with:
- DWT coefficients (precision, uniqueness)
- Quantization (if applied)
- List uniqueness (ICS/SCS)
- Compression ratio and bitrate
- Coefficient distribution

Usage:
    python diagnose_compression.py input.png [output.wdr]
"""

import argparse
import sys
import os
import numpy as np
import wdr_helpers as hlp
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import wdr_coder
try:
    import wdr_coder
    WDR_CODER_AVAILABLE = True
except ImportError:
    WDR_CODER_AVAILABLE = False
    print("Warning: wdr_coder module not available. Compression analysis will be skipped.")


def analyze_dwt_coeffs(coeffs, name="DWT Coefficients"):
    """Analyze DWT coefficients."""
    print(f"\n{'='*60}")
    print(f"{name} Analysis")
    print(f"{'='*60}")
    
    # Analyze coefficient structure
    if isinstance(coeffs, (list, tuple)):
        print(f"Structure: pywt coefficient structure ({len(coeffs)} levels)")
        # Flatten for analysis
        flat_coeffs, _ = hlp.flatten_coeffs(coeffs)
    else:
        flat_coeffs = coeffs
    
    print(f"Shape: {flat_coeffs.shape}")
    print(f"Dtype: {flat_coeffs.dtype}")
    print(f"Min: {flat_coeffs.min():.6f}")
    print(f"Max: {flat_coeffs.max():.6f}")
    print(f"Mean: {flat_coeffs.mean():.6f}")
    print(f"Std: {flat_coeffs.std():.6f}")
    print(f"Total coefficients: {flat_coeffs.size}")
    
    # Count unique values
    unique_values = np.unique(flat_coeffs)
    print(f"Unique values: {len(unique_values)}")
    print(f"Uniqueness ratio: {len(unique_values) / flat_coeffs.size * 100:.2f}%")
    
    # Analyze near-zero coefficients
    near_zero = np.abs(flat_coeffs) < 1e-6
    print(f"Near-zero coefficients (|x| < 1e-6): {np.sum(near_zero)} ({np.sum(near_zero)/flat_coeffs.size*100:.2f}%)")
    
    # Analyze coefficient distribution
    nonzero = flat_coeffs[flat_coeffs != 0]
    if len(nonzero) > 0:
        print(f"Non-zero coefficients: {len(nonzero)}")
        print(f"Non-zero min: {np.abs(nonzero).min():.6f}")
        print(f"Non-zero max: {np.abs(nonzero).max():.6f}")
        print(f"Non-zero mean: {np.abs(nonzero).mean():.6f}")
    
    return flat_coeffs


def test_quantization(coeffs, step_size):
    """Test quantization with given step size."""
    print(f"\n{'='*60}")
    print(f"Quantization Test (step_size={step_size})")
    print(f"{'='*60}")
    
    # Quantize
    quantized = np.round(coeffs / step_size) * step_size
    
    # Analyze quantization results
    unique_before = len(np.unique(coeffs))
    unique_after = len(np.unique(quantized))
    
    print(f"Unique values before quantization: {unique_before}")
    print(f"Unique values after quantization: {unique_after}")
    print(f"Reduction: {unique_before - unique_after} ({((unique_before - unique_after) / unique_before * 100):.2f}%)")
    
    # Calculate MSE
    mse = np.mean((coeffs - quantized) ** 2)
    print(f"MSE after quantization: {mse:.12f}")
    print(f"RMSE: {np.sqrt(mse):.12f}")
    
    # Calculate max error
    max_error = np.max(np.abs(coeffs - quantized))
    print(f"Max error: {max_error:.12f}")
    print(f"Max error / step_size: {max_error / step_size:.2f} (should be <= 0.5)")
    
    # Verify reversibility (for uniform quantization, it's identity)
    reconstructed = quantized  # Uniform quantization is reversible
    mse_recon = np.mean((coeffs - reconstructed) ** 2)
    print(f"MSE after reconstruction: {mse_recon:.12f}")
    
    return quantized, unique_before, unique_after, mse


def analyze_compression_ratio(original_file, compressed_file):
    """Analyze compression ratio."""
    print(f"\n{'='*60}")
    print("Compression Ratio Analysis")
    print(f"{'='*60}")
    
    if not os.path.exists(compressed_file):
        print(f"Error: Compressed file not found: {compressed_file}")
        return None
    
    # Get file sizes
    original_size = os.path.getsize(original_file)
    compressed_size = os.path.getsize(compressed_file)
    
    print(f"Original file size: {original_size:,} bytes")
    print(f"Compressed file size: {compressed_size:,} bytes")
    print(f"Size difference: {compressed_size - original_size:,} bytes ({((compressed_size - original_size) / original_size * 100):+.2f}%)")
    
    # Calculate compression ratio
    if compressed_size > 0:
        cr = original_size / compressed_size
        print(f"Compression Ratio (CR): {cr:.3f}")
        if cr < 1:
            print("⚠️  WARNING: CR < 1 means compressed file is LARGER than original!")
        elif cr > 1:
            print(f"✓ Compression successful: {cr:.1f}x smaller")
    else:
        cr = 0
        print("Error: Compressed file is empty")
    
    return cr, original_size, compressed_size


def analyze_bitrate(original_img, compressed_file):
    """Analyze bitrate (bits per pixel)."""
    print(f"\n{'='*60}")
    print("Bitrate Analysis")
    print(f"{'='*60}")
    
    if not os.path.exists(compressed_file):
        print(f"Error: Compressed file not found: {compressed_file}")
        return None
    
    # Calculate bitrate
    num_pixels = original_img.size
    original_bits = original_img.size * 8  # 8 bits per pixel for grayscale
    compressed_bits = os.path.getsize(compressed_file) * 8
    
    bitrate = compressed_bits / num_pixels
    
    print(f"Number of pixels: {num_pixels:,}")
    print(f"Original bits: {original_bits:,} ({original_bits / num_pixels:.2f} bpp)")
    print(f"Compressed bits: {compressed_bits:,} ({bitrate:.2f} bpp)")
    print(f"Bitrate: {bitrate:.2f} bits per pixel")
    
    if bitrate > 8:
        print("⚠️  WARNING: Bitrate > 8 bpp means compressed file is LARGER than original!")
    elif bitrate < 8:
        print(f"✓ Compression successful: {8 / bitrate:.2f}x compression")
    
    return bitrate


def test_differential_coding(indices, name="Differential Coding"):
    """Test differential coding on indices."""
    print(f"\n{'='*60}")
    print(f"{name} Test")
    print(f"{'='*60}")
    
    if len(indices) == 0:
        print("No indices to analyze")
        return
    
    # Show first 10 values
    print(f"Original indices (first 10): {indices[:10]}")
    
    # Apply differential coding
    diff_indices = [indices[0]]
    for i in range(1, len(indices)):
        diff_indices.append(indices[i] - indices[i-1])
    
    print(f"Differential indices (first 10): {diff_indices[:10]}")
    
    # Analyze differences
    if len(diff_indices) > 1:
        differences = diff_indices[1:]  # Skip first element
        print(f"Mean difference: {np.mean(differences):.2f}")
        print(f"Std difference: {np.std(differences):.2f}")
        print(f"Max difference: {np.max(differences)}")
        print(f"Min difference: {np.min(differences)}")
        
        # Count small differences (good for compression)
        small_diffs = [d for d in differences if d <= 10]
        print(f"Small differences (<=10): {len(small_diffs)} ({len(small_diffs)/len(differences)*100:.2f}%)")
    
    # Verify reversibility
    reconstructed = [diff_indices[0]]
    for i in range(1, len(diff_indices)):
        reconstructed.append(reconstructed[-1] + diff_indices[i])
    
    if reconstructed == indices:
        print("✓ Differential coding is reversible")
    else:
        print("✗ ERROR: Differential coding is NOT reversible!")
        print(f"First mismatch at index {next((i for i, (a, b) in enumerate(zip(reconstructed, indices)) if a != b), None)}")
    
    return diff_indices


def plot_histograms(coeffs_before, coeffs_after, step_size, output_dir=None):
    """Plot histograms of coefficients before and after quantization."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram of coefficients before quantization
        axes[0, 0].hist(coeffs_before, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Coefficients Before Quantization')
        axes[0, 0].set_xlabel('Coefficient Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of coefficients after quantization
        axes[0, 1].hist(coeffs_after, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title(f'Coefficients After Quantization (step={step_size})')
        axes[0, 1].set_xlabel('Coefficient Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of quantization error
        error = coeffs_before - coeffs_after
        axes[1, 0].hist(error, bins=50, alpha=0.7, edgecolor='black', color='red')
        axes[1, 0].set_title('Quantization Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: before vs after
        # Sample for performance
        sample_size = min(10000, len(coeffs_before))
        indices = np.random.choice(len(coeffs_before), sample_size, replace=False)
        axes[1, 1].scatter(coeffs_before[indices], coeffs_after[indices], alpha=0.5, s=1)
        axes[1, 1].plot([coeffs_before.min(), coeffs_before.max()], 
                       [coeffs_before.min(), coeffs_before.max()], 'r--', linewidth=2)
        axes[1, 1].set_title('Before vs After Quantization')
        axes[1, 1].set_xlabel('Before Quantization')
        axes[1, 1].set_ylabel('After Quantization')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'quantization_analysis.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nHistogram saved to: {output_path}")
        else:
            plt.savefig('quantization_analysis.png', dpi=150, bbox_inches='tight')
            print(f"\nHistogram saved to: quantization_analysis.png")
        
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create histogram: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose WDR compression pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_image",
        help="Input image file path"
    )
    
    parser.add_argument(
        "output_wdr",
        nargs="?",
        default=None,
        help="Output .wdr file path (optional, for compression analysis)"
    )
    
    parser.add_argument(
        "--scales",
        type=int,
        default=2,
        help="Number of wavelet decomposition scales (default: 2)"
    )
    
    parser.add_argument(
        "--wavelet",
        type=str,
        default="bior4.4",
        help="Wavelet name (default: bior4.4)"
    )
    
    parser.add_argument(
        "--quantization-step",
        type=float,
        default=None,
        help="Quantization step size to test (default: auto-calculate based on threshold)"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate histogram plots"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: current directory)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WDR Compression Diagnostic Tool")
    print("=" * 60)
    print(f"Input image: {args.input_image}")
    print(f"Scales: {args.scales}")
    print(f"Wavelet: {args.wavelet}")
    print()
    
    try:
        # 1. Load image
        print("Step 1: Loading image...")
        original_img = hlp.load_image(args.input_image)
        print(f"  Image size: {original_img.shape}")
        print(f"  Image dtype: {original_img.dtype}")
        print(f"  Image range: [{original_img.min():.2f}, {original_img.max():.2f}]")
        
        # 2. Perform DWT
        print("\nStep 2: Performing DWT...")
        wavelet_coeffs = hlp.do_dwt(original_img, scales=args.scales, wavelet=args.wavelet)
        
        # 3. Analyze DWT coefficients
        flat_coeffs = analyze_dwt_coeffs(wavelet_coeffs, "DWT Coefficients")
        
        # 4. Test quantization
        if args.quantization_step:
            step_size = args.quantization_step
        else:
            # Auto-calculate based on initial threshold
            max_abs = np.max(np.abs(flat_coeffs))
            initial_T = 2.0 ** np.floor(np.log2(max_abs))
            # Use a step size that aligns with WDR's threshold approach
            step_size = initial_T / (2 ** 16)  # Start with fine quantization
            print(f"\nAuto-calculated quantization step: {step_size:.6f} (based on initial_T={initial_T:.6f})")
        
        quantized_coeffs, unique_before, unique_after, mse = test_quantization(flat_coeffs, step_size)
        
        # 5. Test differential coding on quantized coefficients
        # Simulate finding significant coefficients at a threshold
        T = np.max(np.abs(quantized_coeffs)) / 2.0
        significant_indices = [i for i, coeff in enumerate(quantized_coeffs) if np.abs(coeff) >= T]
        
        if len(significant_indices) > 0:
            print(f"\nFound {len(significant_indices)} significant coefficients at T={T:.6f}")
            test_differential_coding(significant_indices[:100], "Differential Coding (first 100 indices)")
        
        # 6. Compression analysis (if wdr_coder is available and output file provided)
        if WDR_CODER_AVAILABLE and args.output_wdr:
            print("\nStep 3: Testing compression...")
            
            # Test compression WITHOUT quantization
            print("\n--- Compression WITHOUT Quantization ---")
            temp_file_no_quant = args.output_wdr.replace('.wdr', '_no_quant.wdr') if args.output_wdr.endswith('.wdr') else args.output_wdr + '_no_quant.wdr'
            try:
                wdr_coder.compress(flat_coeffs, temp_file_no_quant, num_passes=26)
                cr_no_quant, orig_size, comp_size_no_quant = analyze_compression_ratio(args.input_image, temp_file_no_quant)
                bitrate_no_quant = analyze_bitrate(original_img, temp_file_no_quant)
            except Exception as e:
                print(f"Error during compression (no quantization): {e}")
                cr_no_quant = None
                bitrate_no_quant = None
            
            # Test compression WITH quantization
            print("\n--- Compression WITH Quantization ---")
            temp_file_quant = args.output_wdr.replace('.wdr', '_quant.wdr') if args.output_wdr.endswith('.wdr') else args.output_wdr + '_quant.wdr'
            try:
                wdr_coder.compress(quantized_coeffs, temp_file_quant, num_passes=26)
                cr_quant, _, comp_size_quant = analyze_compression_ratio(args.input_image, temp_file_quant)
                bitrate_quant = analyze_bitrate(original_img, temp_file_quant)
            except Exception as e:
                print(f"Error during compression (with quantization): {e}")
                cr_quant = None
                bitrate_quant = None
            
            # Compare results
            if cr_no_quant is not None and cr_quant is not None:
                print(f"\n{'='*60}")
                print("Compression Comparison")
                print(f"{'='*60}")
                print(f"CR without quantization: {cr_no_quant:.3f}")
                print(f"CR with quantization: {cr_quant:.3f}")
                print(f"Improvement: {((cr_quant - cr_no_quant) / cr_no_quant * 100):+.2f}%")
                print(f"Bitrate without quantization: {bitrate_no_quant:.2f} bpp")
                print(f"Bitrate with quantization: {bitrate_quant:.2f} bpp")
                print(f"Bitrate reduction: {bitrate_no_quant - bitrate_quant:.2f} bpp")
            
            # Clean up temporary files
            for temp_file in [temp_file_no_quant, temp_file_quant]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        elif not WDR_CODER_AVAILABLE:
            print("\n⚠️  Skipping compression analysis (wdr_coder not available)")
        elif not args.output_wdr:
            print("\n⚠️  Skipping compression analysis (no output file specified)")
        
        # 7. Generate plots
        if args.plot:
            plot_histograms(flat_coeffs, quantized_coeffs, step_size, args.output_dir)
        
        # 8. Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Unique values before quantization: {unique_before}")
        print(f"Unique values after quantization: {unique_after}")
        print(f"Reduction: {unique_before - unique_after} ({((unique_before - unique_after) / unique_before * 100):.2f}%)")
        print(f"Quantization MSE: {mse:.12f}")
        
        if unique_after >= unique_before * 0.9:
            print("\n⚠️  WARNING: Quantization did not significantly reduce unique values!")
            print("   Consider using a larger quantization step size.")
        else:
            print(f"\n✓ Quantization successfully reduced unique values by {((unique_before - unique_after) / unique_before * 100):.2f}%")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

