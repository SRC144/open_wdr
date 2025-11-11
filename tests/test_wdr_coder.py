"""
Integration tests for WDR coder (requires compiled C++ module).
"""

import numpy as np
import pytest
import tempfile
import os
import wdr_helpers as hlp

# Try to import wdr_coder - skip tests if not available
try:
    import wdr_coder
    WDR_CODER_AVAILABLE = True
except ImportError:
    WDR_CODER_AVAILABLE = False
    pytestmark = pytest.mark.skip("wdr_coder module not available")


@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_compress_decompress_round_trip_simple():
    """Test that compress and decompress are inverse operations with simple known array."""
    # Create simple, known array
    test_coeffs = np.array([100.0, -42.0, 10.0, 0.0, 3.0], dtype=np.float64)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        # Compress
        wdr_coder.compress(test_coeffs, temp_path)
        
        # Check that file exists
        assert os.path.exists(temp_path)
        
        # Decompress
        decompressed_coeffs = wdr_coder.decompress(temp_path)
        
        # Check properties
        assert decompressed_coeffs.shape == test_coeffs.shape
        assert decompressed_coeffs.dtype == np.float64
        
        # Verify reconstruction (within quantization precision)
        # With 26 passes, we achieve ~1e-6 precision, so use atol=1e-6
        np.testing.assert_allclose(decompressed_coeffs, test_coeffs, rtol=1e-6, atol=1e-6)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_compress_decompress_round_trip():
    """Test that compress and decompress are inverse operations."""
    # Create test coefficients
    test_coeffs = np.random.randn(1000).astype(np.float64)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        # Compress
        wdr_coder.compress(test_coeffs, temp_path)
        
        # Check that file exists
        assert os.path.exists(temp_path)
        
        # Decompress
        decompressed_coeffs = wdr_coder.decompress(temp_path)
        
        # Check properties
        assert decompressed_coeffs.shape == test_coeffs.shape
        assert decompressed_coeffs.dtype == np.float64
        
        # Verify reconstruction (should be very close)
        np.testing.assert_allclose(decompressed_coeffs, test_coeffs, rtol=1e-6, atol=1e-6)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_compress_empty_array():
    """Test error handling for empty coefficient array."""
    # Create empty array
    empty_coeffs = np.array([])
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        # Should raise an error
        with pytest.raises(Exception):
            wdr_coder.compress(empty_coeffs, temp_path)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_decompress_nonexistent_file():
    """Test error handling for nonexistent file."""
    # Try to decompress a nonexistent file
    with pytest.raises(Exception):
        wdr_coder.decompress("nonexistent_file.wdr")


@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_full_pipeline():
    """Test full compression/decompression pipeline."""
    # Create a test image
    test_img = np.random.rand(32, 32) * 255
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_img, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        # Compress
        wdr_coder.compress(flat_coeffs, temp_path)
        
        # Decompress
        decompressed_flat_coeffs = wdr_coder.decompress(temp_path)
        
        # Unflatten coefficients
        decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_metadata)
        
        # Perform IDWT
        reconstructed_img = hlp.do_idwt(decompressed_coeffs, wavelet='bior4.4')
        
        # Check that shapes match
        assert reconstructed_img.shape == test_img.shape
        
        # Verify reconstruction quality
        # With DWT/IDWT and quantization, allow slightly larger tolerance
        np.testing.assert_allclose(reconstructed_img, test_img, rtol=1e-5, atol=1e-5)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_edge_cases():
    """Test edge cases for compression/decompression."""
    # Test: All zeros
    zero_coeffs = np.zeros(100, dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    try:
        wdr_coder.compress(zero_coeffs, temp_path)
        decompressed = wdr_coder.decompress(temp_path)
        np.testing.assert_allclose(decompressed, zero_coeffs, atol=1e-10)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test: All negative
    negative_coeffs = np.array([-100.0, -50.0, -25.0], dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    try:
        wdr_coder.compress(negative_coeffs, temp_path)
        decompressed = wdr_coder.decompress(temp_path)
        np.testing.assert_allclose(decompressed, negative_coeffs, rtol=1e-6, atol=1e-6)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test: Single element
    single_coeff = np.array([42.0], dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    try:
        wdr_coder.compress(single_coeff, temp_path)
        decompressed = wdr_coder.decompress(temp_path)
        np.testing.assert_allclose(decompressed, single_coeff, rtol=1e-6, atol=1e-6)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def create_test_pattern():
    """Create a test pattern image with vertical and horizontal lines."""
    # Create 16x16 image
    img = np.zeros((16, 16), dtype=np.uint8)
    
    # Add vertical line at column 8
    img[:, 8] = 255
    
    # Add horizontal line at row 8
    img[8, :] = 255
    
    return img

@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_test_pattern():
    """Test compression/decompression with test pattern image."""
    # Create test pattern
    test_pattern = create_test_pattern().astype(np.float64)
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_pattern, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Create test data directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
    os.makedirs(test_data_dir, exist_ok=True)
    test_pattern_path = os.path.join(test_data_dir, 'test_pattern.png')
    
    # Save test pattern
    hlp.save_image(test_pattern_path, test_pattern)
    
    # Compress and decompress
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        wdr_coder.compress(flat_coeffs, temp_path)
        decompressed_flat_coeffs = wdr_coder.decompress(temp_path)
        
        # Unflatten and reconstruct
        decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_metadata)
        reconstructed_img = hlp.do_idwt(decompressed_coeffs, wavelet='bior4.4')
        
        # Verify reconstruction
        # With DWT/IDWT and quantization, allow slightly larger tolerance
        np.testing.assert_allclose(reconstructed_img, test_pattern, rtol=1e-5, atol=1e-5)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_golden_file():
    """Test golden file comparison for deterministic behavior."""
    # Use lenna_small.png if it exists, otherwise skip
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
    lenna_path = os.path.join(test_data_dir, 'lenna_small.png')
    golden_path = os.path.join(test_data_dir, 'golden_recon.png')
    
    if not os.path.exists(lenna_path):
        pytest.skip("Test image not found")
    
    # Load original image
    original_img = hlp.load_image(lenna_path)
    
    # Perform DWT
    coeffs = hlp.do_dwt(original_img, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Compress with fixed settings
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        wdr_coder.compress(flat_coeffs, temp_path)
        
        # Decompress
        decompressed_flat_coeffs = wdr_coder.decompress(temp_path)
        
        # Unflatten and reconstruct
        decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_metadata)
        reconstructed_img = hlp.do_idwt(decompressed_coeffs, wavelet='bior4.4')
        
        # Save reconstructed image
        recon_output_path = os.path.join(test_data_dir, 'recon_output.png')
        hlp.save_image(recon_output_path, reconstructed_img)
        
        # If golden file exists, compare
        if os.path.exists(golden_path):
            golden_img = hlp.load_image(golden_path)
            np.testing.assert_array_equal(reconstructed_img, golden_img)
        else:
            # First run - golden file doesn't exist yet
            # User should manually verify and rename recon_output.png to golden_recon.png
            pytest.skip("Golden file not found - this is the first run. Please verify recon_output.png and rename to golden_recon.png")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@pytest.mark.skipif(not WDR_CODER_AVAILABLE, reason="wdr_coder module not available")
def test_real_image_round_trip():
    """Test full round-trip with real image."""
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
    lenna_path = os.path.join(test_data_dir, 'lenna_small.png')
    
    if not os.path.exists(lenna_path):
        pytest.skip("Test image not found")
    
    # Load original image
    original_img = hlp.load_image(lenna_path)
    
    # Perform DWT
    coeffs = hlp.do_dwt(original_img, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Compress
    with tempfile.NamedTemporaryFile(suffix='.wdr', delete=False) as f:
        temp_path = f.name
    
    try:
        wdr_coder.compress(flat_coeffs, temp_path)
        
        # Decompress
        decompressed_flat_coeffs = wdr_coder.decompress(temp_path)
        
        # Unflatten and reconstruct
        decompressed_coeffs = hlp.unflatten_coeffs(decompressed_flat_coeffs, shape_metadata)
        reconstructed_img = hlp.do_idwt(decompressed_coeffs, wavelet='bior4.4')
        
        # Verify shapes match
        assert reconstructed_img.shape == original_img.shape
        
        # Verify reconstruction quality
        # With DWT/IDWT and quantization, allow slightly larger tolerance
        np.testing.assert_allclose(reconstructed_img, original_img, rtol=1e-5, atol=1e-5)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

