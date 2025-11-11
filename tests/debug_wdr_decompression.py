#!/usr/bin/env python3
"""
Debug script for WDR decompression issue.

This script compresses and decompresses a simple array with debug output enabled
to trace the encoder/decoder state and identify where the bug occurs.
"""

import numpy as np
import wdr_coder
import sys
import tempfile
import os

def main():
    # Test with simple array
    print("=" * 80)
    print("WDR Decompression Debug Script")
    print("=" * 80)
    print()
    
    # Test array
    test_coeffs = np.array([100.0, -42.0, 10.0, 0.0, 3.0], dtype=np.float64)
    print(f"Original array: {test_coeffs}")
    print(f"Array shape: {test_coeffs.shape}")
    print(f"Array dtype: {test_coeffs.dtype}")
    print()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        print("=" * 80)
        print("COMPRESSION (with debug output)")
        print("=" * 80)
        print()
        
        # Compress with debug enabled
        wdr_coder.compress(test_coeffs, temp_path, num_passes=26)
        
        print()
        print("=" * 80)
        print("DECOMPRESSION (with debug output)")
        print("=" * 80)
        print()
        
        # Decompress with debug enabled
        decompressed_coeffs = wdr_coder.decompress(temp_path)
        
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        print(f"Decompressed array: {decompressed_coeffs}")
        print(f"Decompressed shape: {decompressed_coeffs.shape}")
        print(f"Decompressed dtype: {decompressed_coeffs.dtype}")
        print()
        print(f"Difference: {test_coeffs - decompressed_coeffs}")
        print(f"Max error: {np.max(np.abs(test_coeffs - decompressed_coeffs))}")
        print(f"All close (1e-6): {np.allclose(test_coeffs, decompressed_coeffs, atol=1e-6)}")
        print()
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        print(f"Compressed file size: {file_size} bytes")
        print()
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    main()

