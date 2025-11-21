"""
Integration tests for WDR coder (requires compiled C++ module).
Updated for Memory-Based Tiling Architecture.
"""

import numpy as np
import pytest
import math
from wdr.utils import helpers as hlp

# Try to import the compiled coder - skip tests if not available
try:
    from wdr import coder as wdr_coder
    WDR_CODER_AVAILABLE = True
except ImportError:
    WDR_CODER_AVAILABLE = False

pytestmark = [
    pytest.mark.filterwarnings("ignore:Level value of .*boundary effects.:UserWarning"),
    pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
]

def calculate_test_T(coeffs):
    """Helper to calculate threshold for tests."""
    max_abs = np.max(np.abs(coeffs))
    if max_abs == 0: return 1.0
    return 2.0 ** math.floor(math.log2(max_abs))

def test_compress_decompress_round_trip_simple():
    """Test basic round trip with a known small array."""
    test_coeffs = np.array([100.0, -42.0, 10.0, 0.0, 3.0], dtype=np.float64)
    global_T = calculate_test_T(test_coeffs)
    
    # Instantiate Class
    compressor = wdr_coder.WDRCompressor(num_passes=16)
    
    # Compress (Returns list/vector of ints)
    compressed_data = compressor.compress(test_coeffs, global_T)
    
    # Verify we got bytes back
    assert len(compressed_data) > 0
    assert isinstance(compressed_data, (list, tuple, bytes))
    
    # Decompress (Requires explicit T and length)
    # Note: Convert to bytes object for the binding
    compressed_bytes = bytes(compressed_data)
    recon = compressor.decompress(compressed_bytes, global_T, len(test_coeffs))
    
    # Verify
    recon = np.array(recon)
    assert recon.shape == test_coeffs.shape
    # Check values (lossy compression, so allow small tolerance)
    np.testing.assert_allclose(recon, test_coeffs, atol=1.0)

def test_compress_decompress_random():
    """Test round trip with random data."""
    test_coeffs = np.random.randn(1000).astype(np.float64) * 100.0
    global_T = calculate_test_T(test_coeffs)
    
    compressor = wdr_coder.WDRCompressor(num_passes=20)
    
    compressed = compressor.compress(test_coeffs, global_T)
    recon = compressor.decompress(bytes(compressed), global_T, len(test_coeffs))
    
    np.testing.assert_allclose(np.array(recon), test_coeffs, atol=0.5)

def test_compress_empty_array():
    """Test handling of empty input."""
    empty_coeffs = np.array([], dtype=np.float64)
    compressor = wdr_coder.WDRCompressor(num_passes=16)
    
    # Should return empty bytes, not crash
    compressed = compressor.compress(empty_coeffs, 1.0)
    assert len(compressed) == 0
    
    # Decompressing empty bytes should return empty array (if size 0 requested)
    recon = compressor.decompress(bytes(), 1.0, 0)
    assert len(recon) == 0

def test_decompress_garbage_data():
    """Test resilience against garbage input."""
    garbage = b'\x00\xff\xab\x12' # Random junk
    compressor = wdr_coder.WDRCompressor(num_passes=16)
    
    # Should probably raise RuntimeError from C++ or return zeros/garbage without crashing
    try:
        compressor.decompress(garbage, 128.0, 100)
    except RuntimeError:
        pass # crashing with detailed error is acceptable for garbage
    except Exception:
        pass

def test_full_pipeline_integration():
    """Test the DWT -> Compress -> Decompress -> IDWT chain."""
    # 1. Create Image
    test_img = np.random.rand(64, 64) * 255
    
    # 2. DWT
    coeffs = hlp.do_dwt(test_img, scales=2, wavelet='bior4.4')
    flat_coeffs, meta = hlp.flatten_coeffs(coeffs)
    
    # 3. Global T
    global_T = calculate_test_T(flat_coeffs)
    
    # 4. Compress
    compressor = wdr_coder.WDRCompressor(num_passes=24)
    compressed = compressor.compress(flat_coeffs, global_T)
    
    # 5. Decompress
    recon_flat = compressor.decompress(bytes(compressed), global_T, len(flat_coeffs))
    recon_flat = np.array(recon_flat)
    
    # 6. IDWT
    recon_coeffs = hlp.unflatten_coeffs(recon_flat, meta)
    recon_img = hlp.do_idwt(recon_coeffs, wavelet='bior4.4')
    
    # 7. Verify
    assert recon_img.shape == test_img.shape
    # High pass count = high quality
    np.testing.assert_allclose(recon_img, test_img, atol=2.0)

def test_edge_cases_zeros():
    """Test compression of all-zero array."""
    zeros = np.zeros(100, dtype=np.float64)
    compressor = wdr_coder.WDRCompressor(num_passes=16)
    
    # T=1.0 default for zeros
    compressed = compressor.compress(zeros, 1.0)
    recon = compressor.decompress(bytes(compressed), 1.0, 100)
    
    np.testing.assert_array_equal(np.array(recon), zeros)

def test_edge_cases_negatives():
    """Test compression of negative values."""
    neg = np.array([-50.0, -10.0, -0.5], dtype=np.float64)
    global_T = calculate_test_T(neg)
    
    compressor = wdr_coder.WDRCompressor(num_passes=20)
    compressed = compressor.compress(neg, global_T)
    recon = compressor.decompress(bytes(compressed), global_T, len(neg))
    
    np.testing.assert_allclose(np.array(recon), neg, atol=0.1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])